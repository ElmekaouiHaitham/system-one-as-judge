import json
import math
import os
import time

import requests

from .accounting import measurement


class ProviderError(RuntimeError):
    def __init__(self, message, record, terminal=False):
        super().__init__(message)
        self.record = {**record, "terminal_error": terminal}


def probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a numeric probability")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Probability outside [0, 1]")
    return float(value)


def parse_clef(body):
    if body.get("success") is False:
        raise ValueError(f"Cloudflare returned errors: {body.get('errors')}")
    result = body.get("result", body)
    if not isinstance(result, dict):
        raise ValueError("Invalid Clef result object")
    answer = result["answers"]["category_positive"]
    # Workers AI wraps noul answers; the standalone SystemOne server may return
    # the numeric probability directly.
    if isinstance(answer, dict):
        if answer.get("type") != "noul":
            raise ValueError("Unexpected Clef answer type")
        answer = answer["noul"]
    return probability(answer), result.get("usage", {})


def parse_gemini(body):
    candidates = body.get("candidates", [])
    if not candidates or candidates[0].get("finishReason") != "STOP":
        raise ValueError("Gemini blocked, truncated, or returned no completed verdict")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought", False))
    verdict = json.loads(text)
    if not isinstance(verdict, dict):
        raise ValueError("Invalid Gemini verdict object")
    if verdict.get("label") not in ("positive", "negative"):
        raise ValueError("Invalid Gemini label")
    return int(verdict["label"] == "positive"), body.get("usageMetadata", {})


class Judges:
    def __init__(self, config, policy):
        self.config, self.policy = config, policy
        self.session = requests.Session()
        self.last_gemini = 0.0

    def post(self, url, headers, payload, provider, self_hosted=False):
        started = time.perf_counter()
        attempts, retry_wait = [], 0.0
        rates = self.config["prices_usd_per_million"]

        def snapshot(body):
            return measurement(provider, body, {
                "latency_s": time.perf_counter() - started,
                "request_latency_s": sum(a["duration_s"] for a in attempts),
                "retry_wait_s": retry_wait, "retry_count": max(0, len(attempts) - 1),
                "attempts": attempts,
            }, rates, self_hosted)

        for attempt in range(self.config["max_attempts"]):
            attempt_started = time.perf_counter()
            try:
                response = self.session.post(url, headers=headers, json=payload,
                                             timeout=self.config["timeout_seconds"])
            except requests.RequestException as exc:
                attempts.append({"http_status": None, "duration_s": time.perf_counter() - attempt_started,
                                 "usage": {}, "error": type(exc).__name__})
                # A timeout can be billable: retain unknown usage instead of assuming zero.
                raise ProviderError(type(exc).__name__, snapshot({})) from None
            duration = time.perf_counter() - attempt_started
            try:
                body = response.json()
                if not isinstance(body, dict):
                    body = {}
            except ValueError:
                body = {}
            result = body.get("result", body)
            usage = (result.get("usage", {}) if isinstance(result, dict) else {}) if provider == "clef" else body.get("usageMetadata", {})
            attempts.append({"http_status": response.status_code, "duration_s": duration, "usage": usage})
            if response.status_code == 429 or response.status_code >= 500:
                if attempt + 1 < self.config["max_attempts"]:
                    retry = response.headers.get("Retry-After", "")
                    delay = float(retry) if retry.isdigit() else 2 ** (attempt + 1)
                    wait_started = time.perf_counter()
                    time.sleep(min(delay, 60))
                    retry_wait += time.perf_counter() - wait_started
                    continue
            # Avoid including credential-bearing URLs in exceptions/logs.
            if not response.ok:
                raise ProviderError(f"Provider HTTP {response.status_code}; check credentials, quota and model access", snapshot(body))
            return body, snapshot(body)
        raise RuntimeError("Retry budget exhausted")

    def judge(self, provider, row):
        state = json.dumps({"prompt": row["prompt"], "response": row["response"]}, ensure_ascii=False)
        c = self.config
        if provider == "clef":
            endpoint = os.getenv("CLEF_ENDPOINT")
            if endpoint:
                key = os.getenv("CLEF_API_KEY", "")
            else:
                account = os.environ["CLOUDFLARE_ACCOUNT_ID"]
                endpoint = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{c['clef_model']}"
                key = os.environ["CLOUDFLARE_API_TOKEN"]
            body, record = self.post(endpoint, {"Authorization": f"Bearer {key}"}, {
                "model": c["clef_request_model"], "state": state,
                "questions": {"category_positive": {"type": "noul", "instructions": self.policy}},
            }, "clef", self_hosted=bool(os.getenv("CLEF_ENDPOINT")))
            try:
                p, usage = parse_clef(body)
            except (ValueError, KeyError, TypeError) as exc:
                raise ProviderError(f"Invalid Clef verdict: {exc}", record, terminal=True) from None
            label = int(p >= 0.5)
            extra = {"probability": p, "confidence": max(p, 1 - p)}
        else:
            wait = 60 / c["gemini_requests_per_minute"] - (time.monotonic() - self.last_gemini)
            if wait > 0:
                time.sleep(wait)
            self.last_gemini = time.monotonic()
            generation = {
                "temperature": 0, "maxOutputTokens": c["gemini_max_output_tokens"],
                "responseMimeType": "application/json",
                "responseSchema": {"type": "OBJECT", "properties": {
                    "label": {"type": "STRING", "enum": ["positive", "negative"]}},
                    "required": ["label"]},
                "thinkingConfig": {"thinkingLevel": c["gemini_thinking_level"]},
            }
            body, record = self.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{c['gemini_model']}:generateContent",
                {"x-goog-api-key": os.environ["GEMINI_API_KEY"]}, {
                    "systemInstruction": {"parts": [{"text": self.policy}]},
                    "contents": [{"role": "user", "parts": [{"text": state}]}],
                    "generationConfig": generation,
                }, "gemini")
            try:
                label, usage = parse_gemini(body)
            except (ValueError, KeyError, TypeError) as exc:
                raise ProviderError(f"Invalid Gemini verdict: {exc}", record, terminal=True) from None
            extra = {}
        return {"label": label, **extra, **record}
