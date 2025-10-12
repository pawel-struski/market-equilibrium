from openai import APIError, APIConnectionError, RateLimitError, Client, OpenAI
import os
import time


### OpenAI API Setup ####


openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def api_request_with_retry(model, max_tokens, temperature, messages, 
                           max_retries=5, base_delay=1.0, backoff_factor=2.0):
    retries = 0

    while retries <= max_retries:
        try:
            response = openai_client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            return response

        except APIError as e:
            print(f"OpenAI API returned an API Error: {e}")
            retries += 1

        except APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}")
            retries += 1

        except RateLimitError as e:
            print(f"OpenAI API request exceeded rate limit: {e}")
            retries += 1

        if retries <= max_retries:
            delay = base_delay * (backoff_factor ** (retries - 1))
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)

    raise Exception("Failed to complete request after multiple retries")


def api_request_with_retry_2(prompt, model, max_tokens, temperature, 
                             max_retries=5, base_delay=1.0, backoff_factor=2.0):
    retries = 0

    while retries <= max_retries:
        try:
            response = openai_client.completions.create(prompt=prompt, model=model, max_tokens=max_tokens, temperature=temperature)
            return response

        except APIError as e:
            print(f"OpenAI API returned an API Error: {e}")
            retries += 1

        except APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}")
            retries += 1

        except RateLimitError as e:
            print(f"OpenAI API request exceeded rate limit: {e}")
            retries += 1

        if retries <= max_retries:
            delay = base_delay * (backoff_factor ** (retries - 1))
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)

    raise Exception("Failed to complete request after multiple retries")


def act_gpt(text: str, model: str, max_tokens: int, temperature: float) -> str:
    """
    Sends a text prompt to the GPT model specified in exp_config and 
    returns the response.
    """
    messages=[{"role": "user", "content": text}]
    response = api_request_with_retry(
        model = model,
        max_tokens = max_tokens,
        temperature = temperature,
        messages = messages
    )
    return response.choices[0].message.content
