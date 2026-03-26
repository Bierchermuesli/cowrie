# ABOUTME: LLM client for communicating with OpenAI-compatible APIs.
# ABOUTME: Sends shell commands to an LLM and returns simulated responses.

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import requests

from twisted.internet import defer, reactor
from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.internet.threads import deferToThread
from twisted.python import log

from cowrie.core.config import CowrieConfig

if TYPE_CHECKING:
    from collections.abc import Generator


class LLMClient:
    """
    Client for communicating with OpenAI-compatible LLM APIs.
    Uses requests via deferToThread so proxy env vars (HTTPS_PROXY etc.) work correctly.
    """

    def __init__(self) -> None:
        self.api_key = CowrieConfig.get("llm", "api_key", fallback="")
        self.model = CowrieConfig.get("llm", "model", fallback="gpt-4o-mini")
        self.host = CowrieConfig.get("llm", "host", fallback="https://api.openai.com")
        self.path = CowrieConfig.get("llm", "path", fallback="/v1/chat/completions")
        self.max_tokens = CowrieConfig.getint("llm", "max_tokens", fallback=500)
        self.temperature = CowrieConfig.getfloat("llm", "temperature", fallback=0.7)
        self.debug = CowrieConfig.getboolean("llm", "debug", fallback=False)

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
        )

        if not self.api_key:
            log.msg("WARNING: No LLM API key configured in [llm] section")

    def _format_request_body(self, prompt: list[str]) -> dict:
        """Structure the request body for OpenAI chat completions API."""
        messages = []
        for i, message in enumerate(prompt):
            if i == 0:
                messages.append({"role": "system", "content": message})
            elif message.startswith("User:"):
                content = message[5:].strip()
                messages.append({"role": "user", "content": content})
            elif message.startswith("System:"):
                content = message[7:].strip()
                messages.append({"role": "assistant", "content": content})
            else:
                messages.append({"role": "user", "content": message})

        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def _do_request(self, prompt: list[str]) -> tuple[int, bytes]:
        """Blocking HTTP request — runs in a thread via deferToThread."""
        request_body = self._format_request_body(prompt)

        if self.debug:
            log.msg(f"LLM request: {json.dumps(request_body, indent=2)}")

        url = f"{self.host}{self.path}"
        try:
            response = self._session.post(url, json=request_body, timeout=30)
            return response.status_code, response.content
        except Exception as e:
            return 500, str(e).encode("utf-8")

    def _send_request(self, prompt: list[str]) -> Deferred[tuple[int, bytes]]:
        """Send request to the LLM API asynchronously."""
        return deferToThread(self._do_request, prompt)

    @inlineCallbacks
    def get_response(
        self, prompt: list[str]
    ) -> Generator[Deferred[Any], Any, str]:
        """
        Get a response from the LLM for the given prompt.

        Args:
            prompt: List of messages. First is system prompt, rest are
                    conversation history with "User:" and "System:" prefixes.

        Returns:
            The LLM's response text, or empty string on error.
        """
        status_code, response = yield self._send_request(prompt)

        if status_code != 200:
            log.err(f"LLM API error (status {status_code}): {response.decode('utf-8')}")
            return ""

        try:
            response_json = json.loads(response)
        except json.JSONDecodeError as e:
            log.err(f"Failed to parse LLM response: {e}")
            return ""

        if self.debug:
            log.msg(f"LLM response: {json.dumps(response_json, indent=2)}")

        if "choices" in response_json and len(response_json["choices"]) > 0:
            content: str = response_json["choices"][0]["message"]["content"]
            return content

        log.err(f"Unexpected LLM response format: {response}")
        return ""
