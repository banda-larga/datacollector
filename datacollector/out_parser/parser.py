from abc import ABC, abstractmethod
from typing import Dict
from enum import Enum


class OutputParser(ABC):
    @abstractmethod
    def parse(self, output: Dict) -> Dict:
        pass
