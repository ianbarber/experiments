"""Minimal chat-completions client for vLLM with token-id and logprob capture (docs/03 §5)."""
from __future__ import annotations
import time
import httpx

class ChatResult:
    def __init__(self, data: dict):
        ch = data["choices"][0]
        self.raw = data
        self.content: str = ch["message"].get("content") or ""
        self.reasoning: str = ch["message"].get("reasoning") or ch["message"].get("reasoning_content") or ""
        self.finish_reason: str = ch.get("finish_reason") or ""
        self.prompt_token_ids: list[int] = data.get("prompt_token_ids") or []
        self.token_ids: list[int] = ch.get("token_ids") or []
        lp = ch.get("logprobs") or {}
        self.logprobs: list[float] = [t["logprob"] for t in (lp.get("content") or [])]
        u = data.get("usage") or {}
        self.prompt_tokens: int = u.get("prompt_tokens") or len(self.prompt_token_ids)
        self.completion_tokens: int = u.get("completion_tokens") or len(self.token_ids)

class VLLMChat:
    def __init__(self, base_url: str, model: str, timeout_s: float = 900.0, retries: int = 5):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.retries = retries
        self._client = httpx.Client(timeout=httpx.Timeout(timeout_s, connect=30.0))

    def complete(self, messages: list[dict], *, max_tokens: int, temperature: float, top_p: float,
                 enable_thinking: bool, seed: int | None = None, stop: list[str] | None = None, logprobs: bool = True) -> ChatResult:
        body = {
            "model": self.model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": top_p, "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if logprobs:   # the training path; servers without logprob support (LAN sglang with speculative decoding) run with it off
            body.update({"logprobs": True, "top_logprobs": 0, "return_token_ids": True})
        if seed is not None: body["seed"] = seed
        if stop: body["stop"] = stop
        last = None
        for attempt in range(self.retries):
            try:
                r = self._client.post(f"{self.base_url}/chat/completions", json=body)
                if r.status_code >= 500 or r.status_code == 429:
                    raise httpx.HTTPStatusError(f"{r.status_code}: {r.text[:300]}", request=r.request, response=r)
                r.raise_for_status()
                return ChatResult(r.json())
            except (httpx.HTTPError, ValueError, KeyError) as ex:
                last = ex
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"vLLM request failed after {self.retries} attempts: {last}")

    def healthy(self) -> bool:
        try:
            return self._client.get(self.base_url.rsplit("/v1", 1)[0] + "/health", timeout=5).status_code == 200
        except httpx.HTTPError:
            return False
