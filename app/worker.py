"""Background training worker.

Both worker containers run this process with a different ``WORKER_ID``.  Each
worker registers itself, publishes a heartbeat, polls for an unfinished job
whose lease is free or expired, and then trains one batch at a time.  Every
batch commit writes the full checkpoint (weights, momentum, batch loss, dataset
loss, step, timestamp, worker and lease epoch) plus the job's progress in one
transaction.

If the process dies, its heartbeat stops; the other worker waits for the lease
to expire and resumes from the newest committed step.  If a worker is merely
suspended and wakes up later, its committed fence check fails and it cannot
write, so it can neither roll the breakpoint back nor report a false completion.
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import threading
import time
import uuid
from pathlib import Path

from . import lease as lease_mod
from . import store
from .config import Settings
from .db import init_db
from .lease import Lease, LeaseLost
from .notifier import Notifier


class Worker:
    def __init__(self, settings: Settings, worker_id: str) -> None:
        self.settings = settings
        self.worker_id = worker_id
        self.stop_event = threading.Event()
        self.lease: Lease | None = None
        self.notifier = Notifier(settings, worker_id, self.stop_event)
        self._last_heartbeat = 0.0
        self._last_status = ""

    def request_stop(self, *_: object) -> None:
        self.stop_event.set()

    def beat(self, status: str) -> bool:
        """Publish liveness and renew the lease.

        Returns ``False`` when the lease has been taken over, which the training
        loop treats as a hard stop.
        """
        now = time.monotonic()
        heartbeat_due = (
            status != self._last_status
            or now - self._last_heartbeat >= self.settings.heartbeat_interval
        )
        if heartbeat_due:
            lease_mod.heartbeat(self.settings.db_path, self.worker_id, status)
            self._last_heartbeat = now
            self._last_status = status
        if self.lease is None:
            return True
        return lease_mod.renew(self.settings.db_path, self.lease, self.settings.lease_seconds)

    def run(self) -> None:
        self.settings.validate()
        init_db(self.settings.db_path)
        lease_mod.register_worker(self.settings.db_path, self.worker_id)
        self.beat("idle")
        # Delivery runs beside training, not after it: a completed job's message
        # is retried even while this worker trains the next job, and a message
        # left pending by a dead worker is picked up by any live one.
        self.notifier.start()
        print(f"[{self.worker_id}] started", flush=True)
        try:
            while not self.stop_event.is_set():
                if not self._run_one_job():
                    self.beat("idle")
                    self.stop_event.wait(self.settings.poll_interval)
                    continue
                try:
                    self._train()
                finally:
                    if self.lease is not None:
                        lease_mod.release(self.settings.db_path, self.lease)
                        self.lease = None
        finally:
            self.stop_event.set()
            self.notifier.join()
            if self.lease is not None:
                lease_mod.release(self.settings.db_path, self.lease)
            lease_mod.heartbeat(self.settings.db_path, self.worker_id, "stopped")
            print(f"[{self.worker_id}] stopped", flush=True)

    def _run_one_job(self) -> bool:
        """Claim one job's lease. Returns ``False`` when nothing is claimable."""
        for job_id in store.claimable_jobs(self.settings.db_path):
            lease = lease_mod.acquire(
                self.settings.db_path,
                job_id,
                self.worker_id,
                self.settings.lease_seconds,
            )
            if lease is not None:
                self.lease = lease
                return True
        return False

    def _train(self) -> None:
        assert self.lease is not None
        job_id = self.lease.job_id
        job = store.get_job(self.settings.db_path, job_id)
        if job is None:
            return
        store.record_event(
            self.settings.db_path,
            job_id,
            "training_started",
            {
                "worker_id": self.worker_id,
                "lease_epoch": self.lease.epoch,
                "resumed_from_step": job["completed_steps"],
            },
        )
        print(
            f"[{self.worker_id}] training {job_id} from step "
            f"{job['completed_steps']}/{job['total_steps']} (lease epoch {self.lease.epoch})",
            flush=True,
        )
        while not self.stop_event.is_set():
            try:
                # Renew first: if another worker took over while we were
                # suspended, leave without touching the job.  The commit below
                # is independently fenced by the lease epoch.
                if not self.beat("training"):
                    raise LeaseLost("lease was taken over")
                result = store.advance_job(
                    self.settings.db_path,
                    self.lease,
                    self.settings.batch_delay_seconds,
                )
            except LeaseLost:
                print(f"[{self.worker_id}] lost lease on {job_id}; stopping", flush=True)
                store.record_event(
                    self.settings.db_path,
                    job_id,
                    "lease_lost",
                    {"worker_id": self.worker_id, "lease_epoch": self.lease.epoch},
                )
                self.lease = None
                return
            print(
                f"[{self.worker_id}] {job_id} step {result['step']}/{result['total_steps']} "
                f"loss={result['loss']:.6f}",
                flush=True,
            )
            if result["status"] == "completed":
                print(f"[{self.worker_id}] completed {job_id}", flush=True)
                return
            self._pause_after_batch(job_id)

    def _pause_after_batch(self, job_id: str) -> None:
        """Simulated freezes for the failover demo; disabled by default.

        ``TRAINING_DEMO_PAUSE_FILE`` (a path shared with the demo) pauses this
        worker between batches, outside any SQLite transaction, without
        heartbeating - so its lease expires and the other worker takes over.

        With ``TRAINING_DEMO_CRASH_AFTER`` set to a step number, the process
        exits hard right after that step commits, modelling a container crash.
        Both are no-ops unless the environment variables are set.
        """
        crash_after = os.environ.get("TRAINING_DEMO_CRASH_AFTER")
        if crash_after and int(crash_after) == int(
            store.get_job(self.settings.db_path, job_id)["completed_steps"]
        ):
            print(f"[{self.worker_id}] simulated crash after step {crash_after}", flush=True)
            os._exit(70)

        pause_file = os.environ.get("TRAINING_DEMO_PAUSE_FILE")
        if not pause_file:
            return
        path = Path(pause_file)
        if path.exists():
            print(f"[{self.worker_id}] paused by {path}", flush=True)
            # No heartbeat while paused: this is what lets the lease expire.
            while path.exists() and not self.stop_event.is_set():
                self.stop_event.wait(self.settings.poll_interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Training worker")
    parser.add_argument("--worker-id", default=None)
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    worker_id = args.worker_id or f"worker-{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
    worker = Worker(settings, worker_id)
    signal.signal(signal.SIGTERM, worker.request_stop)
    signal.signal(signal.SIGINT, worker.request_stop)
    worker.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
