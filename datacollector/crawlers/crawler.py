"""A crawler that crawls (okay, and scrapes) the web."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Pattern, Tuple
from urllib.parse import urljoin, urlparse
import asyncio
import json
import re
import posixpath

import httpx
from parsel import Selector
from pydantic import BaseModel, Field, validator
from tldextract import tldextract
from w3lib.url import canonicalize_url
from trafilatura import extract
from datasets import Dataset
from rich.progress import Progress
from rich.console import Console

from random_user_agent.user_agent import UserAgent
from random_user_agent.params import SoftwareName, OperatingSystem


class JsonlSaver:
    def __init__(self, output_path: str, batch_size: int = 100):
        self.output_path = output_path
        self.batch_size = batch_size
        self.results = []
        self.counter = 0

        Path(self.output_path).mkdir(parents=True, exist_ok=True)

    def save_to_file(self):
        file_name = (
            f"{self.output_path}/results_{self.counter // self.batch_size}.jsonl"
        )
        with open(file_name, "w") as outfile:
            for result in self.results:
                json.dump(result, outfile)
                outfile.write("\n")

    def add_result(self, result: Dict):
        self.results.append(result)
        self.counter += 1

        if self.counter % self.batch_size == 0:
            self.save_to_file()
            self.results = []

    def flush(self):
        if self.results:
            self.save_to_file()


class AbstractCrawler(ABC):
    @abstractmethod
    async def __aenter__(self):
        pass

    @abstractmethod
    async def __aexit__(self, *args, **kwargs):
        pass

    @abstractmethod
    async def scrape(self, urls: List[str]) -> List[httpx.Response]:
        pass

    @abstractmethod
    async def run(self, start_urls: List[str], max_depth: int) -> None:
        pass


def extract_urls(response: httpx.Response) -> List[str]:
    """Extract urls from a response."""
    tree = Selector(text=response.text)
    urls = tree.xpath("//a/@href").getall()
    urls = tree.css("a::attr(href)").getall()
    urls = [urljoin(str(response.url), url.strip()) for url in urls]
    return urls


class UrlFilter(BaseModel):
    domains: List[Tuple[str, List[str]]] = Field(default_factory=list)
    follow: List[Pattern] = Field(default_factory=list)
    seen: set = Field(default_factory=set)
    IGNORED_EXTENSIONS = [
        # archives
        "7z",
        "7zip",
        "bz2",
        "rar",
        "tar",
        "tar.gz",
        "xz",
        "zip",
        # images
        "mng",
        "pct",
        "bmp",
        "gif",
        "jpg",
        "jpeg",
        "png",
        "pst",
        "psp",
        "tif",
        "tiff",
        "ai",
        "drw",
        "dxf",
        "eps",
        "ps",
        "svg",
        "cdr",
        "ico",
        # audio
        "mp3",
        "wma",
        "ogg",
        "wav",
        "ra",
        "aac",
        "mid",
        "au",
        "aiff",
        # video
        "3gp",
        "asf",
        "asx",
        "avi",
        "mov",
        "mp4",
        "mpg",
        "qt",
        "rm",
        "swf",
        "wmv",
        "m4a",
        "m4v",
        "flv",
        "webm",
        # office suites
        "xls",
        "xlsx",
        "ppt",
        "pptx",
        "pps",
        "doc",
        "docx",
        "odt",
        "ods",
        "odg",
        "odp",
        # other
        "css",
        "pdf",
        "exe",
        "bin",
        "rss",
        "dmg",
        "iso",
        "apk",
    ]

    def add_domain(self, domain: str, subdomains: List[str] = None):
        self.domains.append((domain, subdomains or []))

    def is_valid_ext(self, url):
        """Ignore non-crawlable documents"""
        return (
            posixpath.splitext(urlparse(url).path)[1].lower()
            not in self.IGNORED_EXTENSIONS
        )

    def is_valid_scheme(self, url):
        """Ignore non http/s links"""
        return urlparse(url).scheme in ["https", "http"]

    def is_valid_domain(self, url):
        """Ignore offsite urls"""
        parsed = tldextract.extract(url)
        for domain, subdomains in self.domains:
            if parsed.registered_domain == domain and (
                not subdomains or parsed.subdomain in subdomains
            ):
                return True
        return False

    def is_valid_path(self, url):
        """Ignore urls of undesired paths"""
        if not self.follow:
            return True
        path = urlparse(url).path
        for pattern in self.follow:
            if pattern.match(path):
                return True
        return False

    def is_new(self, url):
        """Ignore visited urls (in canonical form)"""
        return canonicalize_url(url) not in self.seen

    def filter(self, urls: List[str]) -> List[str]:
        """Filter list of urls"""
        found = []
        for url in urls:
            if not self.is_valid_scheme(url):
                continue
            if not self.is_valid_domain(url):
                continue
            if not self.is_valid_ext(url):
                continue
            if not self.is_valid_path(url):
                continue
            if not self.is_new(url):
                continue
            self.seen.add(canonicalize_url(url))
            found.append(url)
        return found


class HttpxCrawler(AbstractCrawler):
    """
    An asynchronous web crawler using the httpx library, with support for random User-Agent headers and JSONL output.
    """

    def __init__(
        self,
        url_filter: UrlFilter,
        callbacks: Optional[Dict[str, Callable]] = None,
        delay: float = 0.3,
        output_path: str = "output",
        robots_txt: bool = True,
    ) -> None:
        self.url_filter = url_filter
        self.callbacks = callbacks or {}
        self.delay = delay
        self.saver = JsonlSaver(output_path)
        self.robots_txt = robots_txt
        self.robots_rules = {}

        self.console = Console()

        software_names = [SoftwareName.CHROME.value, SoftwareName.FIREFOX.value]
        operating_systems = [OperatingSystem.WINDOWS.value, OperatingSystem.LINUX.value]
        self.user_agent_rotator = UserAgent(
            software_names=software_names,
            operating_systems=operating_systems,
            limit=100,
        )

    @staticmethod
    def _get_headers(user_agent_rotator: UserAgent) -> Dict[str, str]:
        return {
            "user-agent": user_agent_rotator.get_random_user_agent(),
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "accept-language": "it-IT,it;q=0.9,en-US;en;q=0.8",
            "accept-encoding": "gzip, deflate, br",
        }

    async def __aenter__(self):
        self.session = await httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            limits=httpx.Limits(max_connections=5),
        ).__aenter__()
        return self

    async def __aexit__(self, *args, **kwargs):
        await self.session.__aexit__(*args, **kwargs)

    def parse_robots_txt(self, content: str) -> Dict[str, List[str]]:
        user_agent = "*"
        rules = {}
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = re.split(r":\s*", line, maxsplit=1)
            key = key.lower()
            if key == "user-agent":
                user_agent = value
            elif key == "disallow":
                if user_agent not in rules:
                    rules[user_agent] = []
                rules[user_agent].append(value)
        return rules

    async def fetch_robots_txt(self, url: str) -> Dict[str, List[str]]:
        domain = urlparse(url).netloc
        if domain not in self.robots_rules:
            robots_url = urljoin(url, "/robots.txt")
            try:
                response = await self.session.get(robots_url)
                if response.status_code == 200:
                    content = response.text
                    self.robots_rules[domain] = self.parse_robots_txt(content)
            except Exception:
                pass
            if domain not in self.robots_rules:
                self.robots_rules[domain] = {}
        return self.robots_rules[domain]

    def is_allowed(self, url: str, rules: Dict[str, List[str]]) -> bool:
        if not rules:
            return True
        user_agent = self.user_agent_rotator.get_random_user_agent()
        user_agents = [user_agent, "*"]
        for ua in user_agents:
            if ua in rules:
                for rule in rules[ua]:
                    if re.match(rule, urlparse(url).path):
                        return False
        return True

    async def scrape(self, urls: List[str]) -> List[httpx.Response]:
        async def _scrape(url):
            headers = self._get_headers(self.user_agent_rotator)
            await asyncio.sleep(self.delay)  # be nice
            domain = urlparse(url).netloc

            if self.robots_txt and domain not in self.robots_rules:
                await self.fetch_robots_txt(url)

            if not self.robots_txt or self.is_allowed(url, self.robots_rules[domain]):
                return await self.session.get(
                    url, headers=headers, follow_redirects=True
                )
            else:
                return None

        tasks = [_scrape(url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def run(self, start_urls: List[str], max_depth=2) -> None:
        """Crawl target to maximum depth or until no more urls are found"""
        url_pool = start_urls
        depth = 0

        with Progress(console=self.console) as progress:
            task_id = progress.add_task("[cyan]Crawling...", total=max_depth)

            while url_pool and depth <= max_depth:
                responses = await self.scrape(url_pool)
                responses = [
                    response
                    for response in responses
                    if response is not None and not isinstance(response, Exception)
                ]
                progress.update(task_id, advance=1, description=f"[cyan]Depth {depth}")

                url_pool = self.parse(responses)
                await self.callback(responses)
                depth += 1

        self.saver.flush()

    def parse(self, responses: List[httpx.Response]) -> List[str]:
        """Finds valid urls in responses"""
        all_uniques = set()
        for response in responses:
            sel = Selector(response.text, base_url=str(response.url))
            _urls_in_page = set(
                urljoin(str(response.url), url.strip())
                for url in sel.xpath("//a/@href").getall()
            )
            all_uniques |= _urls_in_page

        return self.url_filter.filter(all_uniques)

    async def callback(self, responses):
        saved_pages = 0
        for response in responses:
            for pattern, fn in self.callbacks.items():
                if pattern.match(str(response.url)):
                    fn(response=response)
            try:
                json_response = json.loads(extract(response.text, output_format="json"))
                if json_response:
                    self.saver.add_result(json_response)
                    saved_pages += 1
            except Exception as e:
                self.console.log(
                    f"[red]Error parsing response: {response.url} - {str(e)}"
                )
                continue

        self.console.log(f"[green]Saved pages: {saved_pages}")


class CrawlerArgs(BaseModel):
    """Arguments for the crawl command"""

    start_urls: List[str] = Field(..., title="Start URLs")
    max_depth: int = Field(2, title="Maximum depth")
    output_path: str = Field("output", title="Output path")
    delay: float = Field(0.3, title="Delay between requests")
    push_to_hub: bool = Field(False, title="Push to HuggingFace Hub")
    username: Optional[str] = Field(None, title="HuggingFace username")
    folder: Optional[str] = Field(None, title="HuggingFace folder")
    robot_txt: bool = Field(False, title="Respect robots.txt")

    @validator("start_urls", pre=True)
    def validate_start_urls(cls, v):
        """Validate start urls"""
        if isinstance(v, str):
            return [v]
        return v

    class Config:
        """Pydantic config"""

        title = "Crawl arguments"
        schema_extra = {
            "example": {
                "start_urls": ["https://www.example.com"],
                "max_depth": 2,
                "output_path": "output",
                "delay": 0.3,
                "push_to_hub": False,
                "username": None,
                "folder": None,
                "robot_txt": False,
            }
        }


class Crawler(BaseModel):
    """Crawl command"""

    args: CrawlerArgs = Field(..., title="Crawl arguments")

    async def run_async(
        self, url_filter: UrlFilter, callbacks: Optional[Dict[str, Callable]] = None
    ):
        """Run the crawl command asynchronously"""
        url_filter = url_filter or UrlFilter()
        async with HttpxCrawler(
            url_filter=url_filter,
            callbacks=callbacks,
            delay=self.args.delay,
            output_path=self.args.output_path,
            robot_txt=self.args.robot_txt,
        ) as crawler:
            await crawler.run(self.args.start_urls, max_depth=self.args.max_depth)

    def run(
        self, url_filter: UrlFilter, callbacks: Optional[Dict[str, Callable]] = None
    ):
        """Run the crawl command synchronously"""
        asyncio.run(self.run_async(url_filter, callbacks))

    def push_to_hub(self):
        """Push the results to the HuggingFace Hub"""
        if not self.args.push_to_hub:
            return

        if not self.args.username:
            raise ValueError("Please specify a username to push to the Hub")

        if not self.args.folder:
            raise ValueError("Please specify a folder to push to the Hub")

        dataset = Dataset.from_json(self.args.output_path)
        dataset = dataset.filter(
            lambda x: x["text"] and x["text"] != "", keep_in_memory=True
        )
        dataset = dataset.unique("text")
        dataset.push_to_hub(
            username=self.args.username,
            repo_name=self.args.folder,
        )
