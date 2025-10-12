from openai import APIError, APIConnectionError, RateLimitError, OpenAI
import os
import time


### OpenAI API Setup ####

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def api_request_with_retry(model, max_tokens, temperature, messages,
                           max_retries=5, base_delay=1.0, backoff_factor=2.0):
    """
    Send a request to the OpenAI API with exponential backoff retry logic.
    Uses the global `openai_client`.

    Args:
        model (str): Model name to use for completion.
        max_tokens (int): Maximum tokens in the response.
        temperature (float): Sampling temperature for randomness.
        messages (list): Chat completion messages.
        max_retries (int, optional): Max number of retry attempts. Default 5.
        base_delay (float, optional): Initial delay between retries in seconds. Default 1.0.
        backoff_factor (float, optional): Exponential backoff multiplier. Default 2.0.

    Returns:
        Response object from OpenAI API.

    Raises:
        Exception: If all retry attempts fail.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return openai_client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
        except (APIError, APIConnectionError, RateLimitError) as e:
            print(f"Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                delay = base_delay * (backoff_factor ** (attempt - 1))
                print(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)

    raise Exception("Failed to complete the request after multiple retries.")


def prompt_gpt(text: str, model: str, max_tokens: int, temperature: float) -> str:
    """
    Sends a text prompt to the GPT model and returns the response.
    """
    messages=[{"role": "user", "content": text}]
    response = api_request_with_retry(
        model = model,
        max_tokens = max_tokens,
        temperature = temperature,
        messages = messages
    )
    return response.choices[0].message.content
