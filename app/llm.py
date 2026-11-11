import logging
import time

import openai
from openai import OpenAI

from . import config

log = logging.getLogger("policy_qa")

if not config.LLM_API_KEY:
    log.warning("LLM_API_KEY is not set; /ask will return 500 until it is configured")

_client = OpenAI(
    base_url=config.LLM_BASE_URL,
    api_key=config.LLM_API_KEY or "missing-key",
    max_retries=0,   # retries are handled by the caller so the request deadline is respected
)
_use_json_mode = True


def chat(messages: list[dict], deadline: float) -> str:
    """One chat completion that never runs past `deadline` (a time.monotonic() value)."""
    global _use_json_mode
    while True:
        remaining = deadline - time.monotonic()
        if remaining < 2:
            raise TimeoutError("request time budget exhausted")
        kwargs = dict(model=config.LLM_MODEL, messages=messages, temperature=0,
                      timeout=min(config.LLM_TIMEOUT, remaining))
        if _use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if config.LLM_REASONING_EFFORT:
            kwargs["reasoning_effort"] = config.LLM_REASONING_EFFORT
        try:
            resp = _client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except openai.BadRequestError as e:
            # Some providers or models reject JSON mode; switch it off once and retry.
            # Any other 400 (for example an invalid API key) is passed on as an error.
            if not _use_json_mode or not any(w in str(e).lower() for w in ("response_format", "json")):
                raise
            _use_json_mode = False
