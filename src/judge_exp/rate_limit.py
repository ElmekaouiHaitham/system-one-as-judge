"""Single-request pacing with persistent cooldowns; no bursts on resume."""
import json
import math
import time
from pathlib import Path

from .common import write_json


class RateLimiter:
    STRATEGY = "persistent_paced_cooldown_v2"

    def __init__(self, requests_per_minute, path, clock=None, sleep=None):
        if not math.isfinite(requests_per_minute) or requests_per_minute <= 0:
            raise ValueError("requests-per-minute must be a finite positive number")
        self.path = Path(path)
        self.clock, self.sleep = clock or time.time, sleep or time.sleep
        self.minimum_interval = 60 / requests_per_minute
        state = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.interval = self.minimum_interval
        self.next_allowed = state.get("next_allowed_unix_s", 0)
        self.consecutive_throttles = state.get("consecutive_throttles", 0)

    def save(self):
        temporary = self.path.with_suffix(".tmp")
        write_json(temporary, {"minimum_interval_s": self.minimum_interval,
            "interval_s": self.interval, "next_allowed_unix_s": self.next_allowed,
            "consecutive_throttles": self.consecutive_throttles, "strategy": self.STRATEGY})
        temporary.replace(self.path)

    def wait(self):
        started = self.clock()
        while self.clock() < self.next_allowed:
            self.sleep(min(30, self.next_allowed - self.clock()))
        self.next_allowed = self.clock() + self.interval
        self.save()
        return self.clock() - started

    def throttled(self, retry_after=0):
        self.consecutive_throttles += 1
        cooldown = min(120, 60 * 2 ** min(self.consecutive_throttles - 1, 1))
        self.next_allowed = max(self.next_allowed, self.clock() + max(cooldown, self.interval, retry_after))
        self.save()

    def succeeded(self):
        self.consecutive_throttles = 0
        self.save()

    def metadata(self):
        return {"minimum_interval_s": self.minimum_interval, "effective_interval_s": self.interval,
                "consecutive_throttles": self.consecutive_throttles, "strategy": self.STRATEGY}
