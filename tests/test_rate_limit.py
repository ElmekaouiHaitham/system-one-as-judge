import json

import pytest

from judge_exp.rate_limit import RateLimiter
from judge_exp.providers import Judges, retry_after_seconds
from judge_exp.common import ROOT


def test_no_bursts_and_cooldown_persist_across_resume(tmp_path):
    clock = [1000.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    path = tmp_path / "limiter.json"
    limiter = RateLimiter(2, path, clock=lambda: clock[0], sleep=sleep)
    assert limiter.wait() == 0
    assert limiter.wait() == 30
    limiter.throttled(75)
    resumed = RateLimiter(2, path, clock=lambda: clock[0], sleep=sleep)
    assert resumed.wait() == 75
    assert resumed.interval == 45
    assert max(sleeps) <= 30
    for _ in range(100):
        resumed.succeeded()
    assert resumed.interval == 30


def test_invalid_rate_and_retry_after_parsing(tmp_path):
    for rate in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            RateLimiter(rate, tmp_path / "limiter.json")
    assert retry_after_seconds("75.5") == 75.5
    assert retry_after_seconds("nan") == 0
    assert retry_after_seconds(None) == 0
    assert retry_after_seconds("bad") == 0


def test_every_retry_is_paced_and_initial_queue_wait_is_outside_latency(tmp_path, monkeypatch):
    import judge_exp.providers as providers
    clock = [1000.0]
    sleep = lambda n: clock.__setitem__(0, clock[0] + n)
    monkeypatch.setattr(providers.time, "perf_counter", lambda: clock[0])
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "offline-only")
    c = json.loads((ROOT / "configs" / "bedrock-kimi-k3.json").read_text())
    judges = Judges(c, "policy")
    limiter = RateLimiter(2, tmp_path / "limiter.json", clock=lambda: clock[0], sleep=sleep)
    limiter.wait()  # A previous collector reserved a slot.
    judges.bedrock_limiter = limiter
    starts = []

    class Response:
        def __init__(self, code):
            self.status_code, self.ok = code, code == 200
            self.headers = {"Retry-After": "75"} if code == 429 else {}

        def json(self):
            if not self.ok:
                return {"message": "throttled"}
            return {"stopReason": "end_turn", "metrics": {"latencyMs": 500},
                "output": {"message": {"content": [{"text": '{"label":"negative"}'}]}},
                "usage": {"inputTokens": 100, "outputTokens": 10}}

    responses = iter([429, 200])

    def post(*a, **kw):
        starts.append(clock[0])
        sleep(1)
        return Response(next(responses))

    monkeypatch.setattr(judges.session, "post", post)
    r = judges.judge("gemini", {"prompt": "context", "response": "answer"})
    assert starts == [1030, 1106]
    assert r["pre_request_pacing_s"] == 30
    assert r["latency_s"] == 77  # 1s rejected + 75s cooldown + 1s success.
    assert r["retry_wait_s"] == 75
    assert r["request_latency_s"] == 2
    assert r["provider_latency_s"] == .5
    assert r["rate_limit"]["effective_interval_s"] == 45
