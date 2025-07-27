# from huggingface_hub import InferenceClient
from openai import APIError, APIConnectionError, RateLimitError, Client, OpenAI
import os
import time

######################################################################
### Llama 3 Setup
######################################################################

# client = InferenceClient(
#     model="meta-llama/Llama-3.1-70B",
#     token=os.environ.get("HF_TOKEN"),
#     provider="featherless-ai"
# )


######################################################################
### GPT Setup
######################################################################

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def api_request_with_retry(model, max_tokens, temperature, messages, max_retries=5, base_delay=1.0, backoff_factor=2.0):
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

def api_request_with_retry_2(prompt, model, max_tokens, temperature, max_retries=5, base_delay=1.0, backoff_factor=2.0):
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


######################################################################
### All models
######################################################################

# def act_llama3(text, prev):
#     generated = client.text_generation(
#         text, 
#         max_new_tokens=1,     
#         temperature=1e-6 
#     )
#     return generated.strip()

def act_gpt4(text):
    messages=[{"role": "user", "content": text}]
    response = api_request_with_retry(
        model = "gpt-4-0613",
        max_tokens = 1,
        temperature = 0.0,
        messages = messages
    )
    return response.choices[0].message.content

def act_gpt35(text):
    response = api_request_with_retry_2(
        model = "gpt-3.5-turbo-instruct",
        prompt = text,
        max_tokens = 1,
        temperature = 0.0,
    )
    return response.choices[0].text.strip()


def act_gpt4_test(text):
    messages=[{"role": "user", "content": text}]
    response = api_request_with_retry(
        model = "gpt-4.1-mini-2025-04-14",
        max_tokens = 5,
        temperature = 0.0,
        messages = messages
    )
    return response.choices[0].message.content
