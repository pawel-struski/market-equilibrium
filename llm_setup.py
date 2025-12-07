from openai import APIError, APIConnectionError, RateLimitError
import time
import logging
from contextlib import contextmanager

from langchain_openai import ChatOpenAI
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_community.callbacks.manager import openai_callback_var
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

logger = logging.getLogger(__name__)


class CostCallbackHandler(OpenAICallbackHandler):
    def __init__(self, logger=None):
        super().__init__()
        self.model_name = None
        self.logger = logger or logging.getLogger(__name__)


@contextmanager
def get_llm_callback(logger=None):
    cb = CostCallbackHandler(logger=logger)
    yield cb


def _to_langchain_messages(messages):
    """
    Convert OpenAI-style messages:
        {"role": "...", "content": "..."}
    into LangChain message objects.
    """
    lc_messages = []

    for m in messages:
        role = m.get("role")
        content = m.get("content", "")

        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role in ("assistant", "ai"):
            lc_messages.append(AIMessage(content=content))
        else:
            raise ValueError(f"Unsupported role: {role!r}")

    return lc_messages


def api_request_with_retry(
    model,
    max_tokens,
    temperature,
    messages,
    max_retries: int = 5,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    callbacks=None,
):
    """
    Send a request to the OpenAI API via LangChain's ChatOpenAI
    with exponential backoff retry logic.

    Args:
        model (str): Model name to use for completion (e.g. "gpt-5.1").
        max_tokens (int): Maximum tokens in the response.
        temperature (float): Sampling temperature for randomness.
        messages (list[dict]): List of {"role": ..., "content": ...}.
        max_retries (int, optional): Max number of retry attempts. Default 5.
        base_delay (float, optional): Initial delay between retries in seconds. Default 1.0.
        backoff_factor (float, optional): Exponential backoff multiplier. Default 2.0.
        callbacks (list, optional): List of callbacks to pass to the LLM.

    Returns:
        LangChain AIMessage with the model's response.

    Raises:
        Exception: If all retry attempts fail.
    """

    if "gemini" in model.lower():
        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
        )
    else:
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            # max_retries=0, # TODO
        )

    lc_messages = _to_langchain_messages(messages)

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            return llm.invoke(
                lc_messages, config={"callbacks": callbacks}, max_tokens=max_tokens
            )
        except (APIError, APIConnectionError, RateLimitError) as e:
            last_error = e
            print(f"Attempt {attempt}/{max_retries} failed: {e}")

            if attempt < max_retries:
                delay = base_delay * (backoff_factor ** (attempt - 1))
                print(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
        except Exception as e:
            # Catch other errors (e.g. from Gemini)
            last_error = e
            print(f"Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                delay = base_delay * (backoff_factor ** (attempt - 1))
                print(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)

    raise Exception(
        f"Failed to complete the request after multiple retries. Last error: {last_error}"
    )


def prompt_gpt(
    text: str, model: str, max_tokens: int, temperature: float, callbacks=None
) -> str:
    """
    Sends a text prompt to the GPT model and returns the response text
    using LangChain's ChatOpenAI under the hood.
    """
    messages = [{"role": "user", "content": text}]

    response = api_request_with_retry(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages,
        callbacks=callbacks,
    )

    return response.content
