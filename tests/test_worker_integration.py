"""Process-level integration test: real API server, two real worker processes.

Flow:
1. start ``uvicorn`` against a fresh database and wait for readiness;
2. start worker A and let it commit a few batches;
3. SIGKILL worker A (an abrupt container-style death, no cleanup);
4. start worker B, which must take over after the lease expires;
5. compare the final weights with an uninterrupted run of the same input.

Also covers a "late" worker: A is suspended between batches long enough for B
to take over, then resumed.  A must not regress the checkpoint or claim
completion.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
LEASE_SECONDS = 2.0
HEARTBEAT_INTERVAL = 0.3
BATCH_DELAY = 0.25


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def env_for(db: Path, **extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "TRAINING_DB_PATH": str(db),
            "TRAINING_LEASE_SECONDS": str(LEASE_SECONDS),
            "TRAINING_HEARTBEAT_INTERVAL": str(HEARTBEAT_INTERVAL),
            "TRAINING_POLL_INTERVAL": "0.05",
            "TRAINING_BATCH_DELAY_SECONDS": str(BATCH_DELAY),
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.update(extra)
    return env


class Server:
    def __init__(self, db: Path, port: int) -> None:
        self.db = db
        self.port = port
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> "Server":
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--log-level",
                "warning",
            ],
            cwd=ROOT,
            env=env_for(self.db),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                self.get("/health")
                return self
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("API did not start")

    def __exit__(self, *exc) -> None:
        if self.proc is not None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)

    def get(self, path: str):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5) as response:
            return json.loads(response.read())

    def post(self, path: str, body: dict):
        data = json.dumps(body).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read())


def start_worker(db: Path, worker_id: str, **env: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "app.worker", "--worker-id", worker_id],
        cwd=ROOT,
        env=env_for(db, **env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def wait_for(predicate, timeout: float, interval: float = 0.05, message: str = "condition"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError(f"timed out waiting for {message}")


def wait_completed(server: Server, job_id: str, timeout: float = 90):
    return wait_for(
        lambda: (state := server.get(f"/jobs/{job_id}"))["status"] == "completed" and state,
        timeout=timeout,
        message="job completion",
    )


def stop_worker(proc: subprocess.Popen, hard: bool = False) -> None:
    if proc.poll() is not None:
        return
    if hard:
        proc.kill()
    else:
        proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def dataset(records: int = 40, seed: int = 21):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(records, 4)).tolist()
    labels = [1 if row[0] + row[1] > 0 else 0 for row in features]
    return {"features": features, "labels": labels, "epochs": 3}


@pytest.fixture(scope="module")
def uninterrupted(tmp_path_factory):
    """Reference run with no interruption at all."""
    db = tmp_path_factory.mktemp("reference") / "training.db"
    with Server(db, free_port()) as server:
        job = server.post("/jobs", dataset())
        worker = start_worker(db, "solo")
        try:
            final = wait_completed(server, job["job_id"])
        finally:
            stop_worker(worker)
    return final


class NotificationReceiver:
    """Real HTTP server standing in for the operations receiver."""

    def __init__(self, status: int = 201) -> None:
        self.status = status
        self.bodies: list[bytes] = []
        self.event_ids: list[str | None] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                outer.bodies.append(body)
                outer.event_ids.append(self.headers.get("X-Event-Id"))
                self.send_response(outer.status)
                self.end_headers()
                self.wfile.write(b'{"accepted":true}')

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}/notify"

    def __enter__(self) -> "NotificationReceiver":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def test_two_workers_deliver_completion_message_exactly_once(tmp_path):
    """Both workers have a notifier; only one may send the completed job."""
    db = tmp_path / "notify.db"
    with NotificationReceiver() as receiver_server:
        with Server(db, free_port()) as server:
            payload = {**dataset(records=24, seed=13),
                       "notification_url": receiver_server.url}
            job = server.post("/jobs", payload)
            workers = [
                start_worker(db, f"worker-{index}",
                             TRAINING_NOTIFY_POLL_INTERVAL="0.05")
                for index in (1, 2)
            ]
            try:
                final = wait_completed(server, job["job_id"])
                delivered = wait_for(
                    lambda: (items := server.get(f"/jobs/{job['job_id']}/notifications"))
                    and items[0]["status"] == "delivered"
                    and items[0],
                    timeout=30,
                    message="notification delivery",
                )
                # Give the other worker's notifier a chance to (wrongly) resend.
                time.sleep(0.5)
                notifications = server.get(f"/jobs/{job['job_id']}/notifications")
                latest = server.get(f"/jobs/{job['job_id']}")
            finally:
                for worker in workers:
                    stop_worker(worker)

        assert len(receiver_server.bodies) == 1
        message = json.loads(receiver_server.bodies[0])
        assert message["event_id"] == delivered["event_id"]
        assert receiver_server.event_ids == [delivered["event_id"]]
        assert message["final_weights"] == final["final_weights"]
        assert message["loss"] == final["loss"]
        assert message["job_id"] == job["job_id"]
        assert notifications[0]["status"] == "delivered"
        assert notifications[0]["attempts"] == 1
        assert len(notifications) == 1
        assert latest["notification"]["status"] == "delivered"
        assert latest["notification"]["payload"] == message
        assert latest["final_weights"] == final["final_weights"]


def test_crash_and_takeover_matches_uninterrupted(tmp_path, uninterrupted):
    db = tmp_path / "training.db"
    with Server(db, free_port()) as server:
        job = server.post("/jobs", dataset())
        worker_a = start_worker(db, "worker-a")
        try:
            wait_for(
                lambda: server.get(f"/jobs/{job['job_id']}")["completed_steps"] >= 3,
                timeout=60,
                message="first batches",
            )
        finally:
            # Abrupt death: no cleanup handler runs, heartbeat simply stops.
            stop_worker(worker_a, hard=True)
        at_crash = server.get(f"/jobs/{job['job_id']}")
        assert at_crash["status"] == "running"
        assert 3 <= at_crash["completed_steps"] < at_crash["total_steps"]

        worker_b = start_worker(db, "worker-b")
        try:
            final = wait_completed(server, job["job_id"])
        finally:
            stop_worker(worker_b)

        assert final["completed_steps"] == final["total_steps"]
        assert final["total_steps"] == uninterrupted["total_steps"]
        assert final["final_weights"] == uninterrupted["final_weights"]
        assert final["loss"] == uninterrupted["loss"]
        assert final["momentum_velocity"] == uninterrupted["momentum_velocity"]

        checkpoints = server.get(f"/jobs/{job['job_id']}/checkpoints")
        assert [c["step"] for c in checkpoints] == list(range(1, final["total_steps"] + 1))
        workers = {c["worker_id"] for c in checkpoints}
        assert "worker-a" in workers and "worker-b" in workers
        epochs = [c["lease_epoch"] for c in checkpoints]
        assert epochs == sorted(epochs) and epochs[-1] > epochs[0]

        events = server.get(f"/jobs/{job['job_id']}/events")
        starts = [e for e in events if e["event"] == "training_started"]
        assert [e["detail"]["worker_id"] for e in starts] == ["worker-a", "worker-b"]
        assert starts[1]["detail"]["resumed_from_step"] == at_crash["completed_steps"]


def test_late_worker_cannot_regress_or_false_complete(tmp_path):
    """A is suspended between batches until its lease expires; B takes over.

    A is then resumed.  It must not write, and B's completion must stand
    unchanged.

    The pause is implemented with ``TRAINING_DEMO_PAUSE_FILE`` rather than
    SIGSTOP so it is deterministic: a process frozen by the OS while inside a
    SQLite write transaction keeps the write lock, and no takeover worker could
    progress until it was killed.
    """
    db = tmp_path / "training.db"
    pause_file = tmp_path / "pause-a"
    with Server(db, free_port()) as server:
        job = server.post("/jobs", dataset(records=24, seed=5))
        worker_a = start_worker(db, "worker-a", TRAINING_DEMO_PAUSE_FILE=str(pause_file))
        wait_for(
            lambda: server.get(f"/jobs/{job['job_id']}")["completed_steps"] >= 2,
            timeout=60,
            message="A progress",
        )
        pause_file.write_text("paused")
        # Wait until A has stopped heartbeating so its lease can expire; by
        # then it cannot be mid-batch, so this snapshot is stable.
        wait_for(
            lambda: server.get("/workers")["leases"][0]["expired"],
            timeout=30,
            message="A lease to expire",
        )
        frozen_at = server.get(f"/jobs/{job['job_id']}")

        worker_b = start_worker(db, "worker-b")
        try:
            completed = wait_completed(server, job["job_id"])
            before_resume = server.get(f"/jobs/{job['job_id']}")
            checkpoints_before = server.get(f"/jobs/{job['job_id']}/checkpoints")

            pause_file.unlink()
            # A wakes up, finds its lease taken over, and must not write.
            time.sleep(BATCH_DELAY * 4)
            stop_worker(worker_a)

            after = server.get(f"/jobs/{job['job_id']}")
            checkpoints_after = server.get(f"/jobs/{job['job_id']}/checkpoints")
            assert after["completed_steps"] == completed["completed_steps"]
            assert after["weights"] == completed["weights"]
            assert after["momentum_velocity"] == completed["momentum_velocity"]
            assert after["status"] == "completed"
            assert after["finished_at"] == completed["finished_at"]
            assert before_resume["weights"] == completed["weights"]
            assert checkpoints_after == checkpoints_before

            steps = [c["step"] for c in checkpoints_after]
            assert steps == list(range(1, frozen_at["total_steps"] + 1))
            # No duplicate or regressed checkpoints were written by A.
            assert len(steps) == len(set(steps))
            late = [c for c in checkpoints_after if c["worker_id"] == "worker-a"]
            assert max(c["step"] for c in late) <= frozen_at["completed_steps"]
        finally:
            stop_worker(worker_a)
            stop_worker(worker_b)
