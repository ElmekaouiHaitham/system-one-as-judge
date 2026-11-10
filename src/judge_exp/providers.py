import json
import math
import os
import time

import requests


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
    return probability(result["answers"]["category_positive"]), result.get("usage", {})


def parse_gemini(body):
    candidates = body.get("candidates", [])
    if not candidates or candidates[0].get("finishReason") != "STOP":
        raise ValueError("Gemini blocked, truncated, or returned no completed verdict")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought", False))
    verdict = json.loads(text)
    if verdict.get("label") not in ("positive", "negative"):
        raise ValueError("Invalid Gemini label")
    return int(verdict["label"] == "positive"), body.get("usageMetadata", {})


class Judges:
    def __init__(self, config, policy):
        self.config, self.policy = config, policy
        self.session = requests.Session()
        self.last_gemini = 0.0

    def post(self, url, headers, payload):
        started = time.perf_counter()
        for attempt in range(self.config["max_attempts"]):
            response = self.session.post(url, headers=headers, json=payload,
                                         timeout=self.config["timeout_seconds"])
            if response.status_code == 429 or response.status_code >= 500:
                if attempt + 1 < self.config["max_attempts"]:
                    retry = response.headers.get("Retry-After", "")
                    delay = float(retry) if retry.isdigit() else 2 ** (attempt + 1)
                    time.sleep(min(delay, 60))
                    continue
            # Avoid including credential-bearing URLs in exceptions/logs.
            if not response.ok:
                raise RuntimeError(f"Provider HTTP {response.status_code}; check credentials, quota and model access")
            return response.json(), time.perf_counter() - started
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
            body, latency = self.post(endpoint, {"Authorization": f"Bearer {key}"}, {
                "model": c["clef_request_model"], "state": state,
                "questions": {"category_positive": {"type": "noul", "instructions": self.policy}},
            })
            p, usage = parse_clef(body)
            label = int(p >= 0.5)
            inputs = usage.get("input_tokens", usage.get("prompt_tokens"))
            outputs = 0
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
            body, latency = self.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{c['gemini_model']}:generateContent",
                {"x-goog-api-key": os.environ["GEMINI_API_KEY"]}, {
                    "systemInstruction": {"parts": [{"text": self.policy}]},
                    "contents": [{"role": "user", "parts": [{"text": state}]}],
                    "generationConfig": generation,
                })
            label, usage = parse_gemini(body)
            inputs = usage.get("promptTokenCount")
            outputs = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
            extra = {}
        rates = c["prices_usd_per_million"]
        cost = None if inputs is None else (
            inputs * rates[f"{provider}_input"] + outputs * rates.get(f"{provider}_output", 0)) / 1e6
        return {"label": label, **extra, "latency_s": latency, "input_tokens": inputs,
                "output_tokens": outputs, "list_cost_usd": cost, "usage": usage, "raw": body}
