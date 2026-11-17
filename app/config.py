"""Runtime configuration, read from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


@dataclass(frozen=True)
class Settings:
    """Paths and timing shared by the API and the workers.

    ``db_path`` is one SQLite file on a shared Docker volume, mounted into the
    API and both worker containers.  Jobs, checkpoints, leases and heartbeats
    live in it, so a checkpoint commit and the lease check that authorises it
    are a single atomic transaction.
    """

    db_path: Path
    lease_seconds: float
    heartbeat_interval: float
    poll_interval: float
    batch_delay_seconds: float
    notification_poll_interval: float = 0.5
    notification_timeout: float = 5.0
    notification_retry_min: float = 1.0
    notification_retry_max: float = 30.0
    notification_claim_seconds: float = 15.0

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(os.environ.get("TRAINING_DATA_DIR", str(ROOT / "data")))
        db_path = Path(os.environ.get("TRAINING_DB_PATH", str(data_dir / "training.db")))
        return cls(
            db_path=db_path.resolve(),
            lease_seconds=_float("TRAINING_LEASE_SECONDS", 6.0),
            heartbeat_interval=_float("TRAINING_HEARTBEAT_INTERVAL", 1.0),
            poll_interval=_float("TRAINING_POLL_INTERVAL", 0.3),
            batch_delay_seconds=_float("TRAINING_BATCH_DELAY_SECONDS", 0.0),
            notification_poll_interval=_float("TRAINING_NOTIFY_POLL_INTERVAL", 0.5),
            notification_timeout=_float("TRAINING_NOTIFY_TIMEOUT", 5.0),
            notification_retry_min=_float("TRAINING_NOTIFY_RETRY_MIN", 1.0),
            notification_retry_max=_float("TRAINING_NOTIFY_RETRY_MAX", 30.0),
            notification_claim_seconds=_float("TRAINING_NOTIFY_CLAIM_SECONDS", 15.0),
        )

    def validate(self) -> None:
        if self.lease_seconds <= 0:
            raise ValueError("TRAINING_LEASE_SECONDS must be positive")
        if not (0 < self.heartbeat_interval < self.lease_seconds):
            raise ValueError("heartbeat interval must be positive and shorter than the lease")
        if self.poll_interval <= 0:
            raise ValueError("TRAINING_POLL_INTERVAL must be positive")
        if self.batch_delay_seconds < 0:
            raise ValueError("TRAINING_BATCH_DELAY_SECONDS must not be negative")
        if self.notification_poll_interval <= 0:
            raise ValueError("TRAINING_NOTIFY_POLL_INTERVAL must be positive")
        if self.notification_timeout <= 0:
            raise ValueError("TRAINING_NOTIFY_TIMEOUT must be positive")
        if not (0 < self.notification_retry_min <= self.notification_retry_max):
            raise ValueError("notification retry bounds must be positive and ordered")
        if self.notification_claim_seconds <= 0:
            raise ValueError("TRAINING_NOTIFY_CLAIM_SECONDS must be positive")
