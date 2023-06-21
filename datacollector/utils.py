import openai
import backoff
from typing import List, Optional, Dict
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


@backoff.on_exception(backoff.expo, openai.error.TooManyRequestsError, max_time=60)
def generate(
    messages: List[Dict[str, str]],
    functions=None,
    function_call: str = "auto",
    model: Optional[str] = "gpt-3.5-turbo-0613",
) -> Dict:
    """Generate a single response from the OpenAI API."""

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            functions=functions,
            function_call=function_call,
        )

        response_message = response["choices"][0]["message"]
        if response_message.get("function_call"):
            function_args = json.loads(response_message["function_call"]["arguments"])
            return function_args

        elif response_message.get("role") == "assistant":
            return response_message["content"]

    except Exception as e:
        print(f"Error occurred: {e}")
        return {}

    return {}


def generate_batch(
    messages: List[Dict[str, str]],
    functions=None,
    batch_size: int = 10,
) -> List[Dict]:
    results = []

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [
            executor.submit(generate, message, functions) for message in messages
        ]

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error occurred in generate_batch: {e}")
                results.append({})

    return results
