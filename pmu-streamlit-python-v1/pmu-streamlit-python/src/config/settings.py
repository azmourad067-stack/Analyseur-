from dataclasses import dataclass

PERIODS = {"week": 7, "month": 30, "quarter": 90, "semester": 180, "year": 365}
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2
TIMEOUT = 30
