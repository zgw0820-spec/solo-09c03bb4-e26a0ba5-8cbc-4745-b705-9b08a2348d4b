#!/usr/bin/env python3
"""End-to-end demo: submit a job, watch it train, kill a worker, watch failover.

Run against Docker Compose (the delivered setup)::

    docker compose up -d --build
    python3 scripts/demo.py

Run against local processes (no Docker available)::

    python3 scripts/demo.py --local

The script only uses the public HTTP API plus, for the disruptive part,
``docker compose kill`` or a process signal.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")


def notify_url(local: bool) -> str:
    """Address the *workers* use for the demo receiver.

    In Compose the workers reach the API by service name; for local processes
    the API is on the loopback address. ``NOTIFY_URL`` overrides both.
    """
    override = os.environ.get("NOTIFY_URL")
    if override:
        return override
    return "http://127.0.0.1:8000/demo/receiver" if local else "http://api:8000/demo/receiver"


def call(method: str, path: str, body: dict | None = None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def wait_for(predicate, timeout: float, interval: float = 0.2, message: str = ""):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise SystemExit(f"timed out waiting for {message}")


def _lease_if_holder_expired(job_id: str, dead: str) -> dict | None:
    lease = _lease_for(job_id)
    if lease is None or lease["worker_id"] == dead:
        return None
    return lease


def _lease_for(job_id: str) -> dict | None:
    for lease in call("GET", "/workers")["leases"]:
        if lease["job_id"] == job_id:
            return lease
    return None


def _state_at_least(job_id: str, step: int) -> dict | None:
    state = call("GET", f"/jobs/{job_id}")
    return state if state["completed_steps"] >= step else None


def _state_if_completed(job_id: str) -> dict | None:
    state = call("GET", f"/jobs/{job_id}")
    return state if state["status"] == "completed" else None


def sample_records(count: int = 40, seed: int = 3, notification_url: str | None = None) -> dict:
    """Synthetic fan vibration/current records with shutdown labels."""
    rng = random.Random(seed)
    features, labels = [], []
    for index in range(count):
        vibration = rng.gauss(0.2, 1.0)
        current = rng.gauss(1.0, 1.0)
        temperature = rng.gauss(0.5, 1.0)
        label = 1 if vibration + current + 0.3 * temperature > 2.1 else 0
        if index % 11 == 0:
            label = 1
        features.append(
            [
                round(vibration, 4),
                round(current, 4),
                round(temperature, 4),
                round(abs(vibration - current), 4),
            ]
        )
        labels.append(label)
    payload = {"features": features, "labels": labels, "epochs": 3}
    if notification_url:
        payload["notification_url"] = notification_url
    return payload


def show(job: dict) -> None:
    print(
        f"  status={job['status']:<9} steps={job['completed_steps']}/{job['total_steps']}"
        f" progress={job['progress']:.0%} loss={job['loss']:.6f}"
        f" weights={[round(w, 6) for w in job['weights']]}"
    )


def kill_worker(mode: str, worker_id: str) -> None:
    """Hard-kill the worker that currently owns the job."""
    if mode == "docker":
        service = worker_id if worker_id.startswith("worker-") else f"worker-{worker_id}"
        subprocess.run(
            ["docker", "compose", "kill", service], cwd=ROOT, check=True, capture_output=True
        )
        print(f"killed {service} via docker compose")
    else:
        pid_file = ROOT / f".demo-{worker_id}.pid"
        if not pid_file.exists():
            raise SystemExit(f"no local pid file for {worker_id}")
        pid = int(pid_file.read_text())
        os.kill(pid, signal.SIGKILL)
        print(f"killed local {worker_id} (pid {pid})")


def step_1_submit(url: str | None) -> dict:
    print("\n[1] submitting 40 records, 3 epochs (8 per batch -> 5 batches/epoch, 15 steps)")
    if url:
        print(f"    notification_url={url}")
    job = call("POST", "/jobs", sample_records(notification_url=url))
    print(f"    job_id={job['job_id']}")
    show(job)
    return job


def step_2_watch(job_id: str, stop_after: int) -> dict:
    print(f"\n[2] watching progress until step >= {stop_after}")
    state = wait_for(
        lambda: _state_at_least(job_id, stop_after),
        timeout=120,
        message="training progress",
    )
    show(state)
    checkpoints = call("GET", f"/jobs/{job_id}/checkpoints")
    for checkpoint in checkpoints[-3:]:
        print(
            f"    step {checkpoint['step']:>2} training_epoch {checkpoint['training_epoch']}"
            f" batch {checkpoint['batch_in_epoch']} progress={checkpoint['progress']:.0%}"
            f" worker={checkpoint['worker_id']}"
            f" lease_epoch={checkpoint['lease_epoch']}"
            f" batch_loss={checkpoint['batch_loss']:.6f}"
        )
    return state


def step_3_kill(mode: str, job_id: str) -> dict:
    before = call("GET", f"/jobs/{job_id}")
    lease = _lease_for(job_id)
    if lease is None:
        raise SystemExit("job already finished; submit a larger job or lower --stop-after")
    print(
        f"\n[3] killing {lease['worker_id']} (lease epoch {lease['lease_epoch']})"
        f" while the job is at confirmed step {before['completed_steps']}"
    )
    print("    confirmed state before kill:")
    show(before)
    kill_worker(mode, lease["worker_id"])
    dead = lease["worker_id"]
    wait_for(
        lambda: _lease_if_holder_expired(job_id, dead),
        timeout=30,
        message="lease expiry / takeover",
    )
    after = call("GET", f"/jobs/{job_id}")
    print(
        f"    after takeover: holder={_lease_for(job_id)['worker_id']}"
        f" (epoch {_lease_for(job_id)['lease_epoch']}),"
        f" confirmed step still {after['completed_steps']}"
    )
    assert after["completed_steps"] >= before["completed_steps"]
    return before


def step_4_failover(job_id: str) -> dict:
    print("\n[4] waiting for the second worker to take over and finish")
    final = wait_for(
        lambda: _state_if_completed(job_id),
        timeout=180,
        message="completion after failover",
    )
    show(final)
    events = call("GET", f"/jobs/{job_id}/events")
    for event in events:
        if event["event"] in {"training_started", "job_completed"}:
            print(f"    {event['event']}: {event['detail']}")
    checkpoints = call("GET", f"/jobs/{job_id}/checkpoints")
    steps = [c["step"] for c in checkpoints]
    assert steps == list(range(1, final["total_steps"] + 1)), steps
    workers = sorted({c["worker_id"] for c in checkpoints})
    print(f"    checkpoint steps strictly 1..{final['total_steps']}, written by {workers}")
    epochs = [c["lease_epoch"] for c in checkpoints]
    assert epochs == sorted(epochs), epochs
    print(f"    lease epochs are monotonic: {sorted(set(epochs))}")
    return final


def step_5_final(job_id: str) -> dict:
    print("\n[5] final weights")
    final = call("GET", f"/jobs/{job_id}")
    print(f"    weights = {final['final_weights']}")
    print(f"    loss    = {final['loss']:.12f}")
    print(f"    saved to data/ (SQLite). Re-read from the API at /jobs/{job_id}")
    return final


def step_6_reference(interrupted: dict, url: str | None) -> None:
    """Train the same input again with no interruption and compare exactly."""
    print("\n[6] equivalence check: same input, no interruption")
    job = call("POST", "/jobs", sample_records(notification_url=url))
    reference = wait_for(
        lambda: _state_if_completed(job["job_id"]),
        timeout=180,
        message="uninterrupted reference completion",
    )
    same_weights = reference["final_weights"] == interrupted["final_weights"]
    same_loss = reference["loss"] == interrupted["loss"]
    print(f"    uninterrupted weights = {reference['final_weights']}")
    print(f"    weights identical      : {same_weights}")
    print(f"    loss identical         : {same_loss} ({reference['loss']!r})")
    if not (same_weights and same_loss):
        raise SystemExit("FAIL: resumed run differs from the uninterrupted run")
    print("    PASS: resumed run is bit-identical to an uninterrupted run")


def local_up() -> subprocess.Popen:
    """Start API + two workers locally for environments without Docker."""
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.update(
        TRAINING_DATA_DIR=str(data),
        TRAINING_LEASE_SECONDS="3",
        TRAINING_HEARTBEAT_INTERVAL="0.8",
        TRAINING_BATCH_DELAY_SECONDS="0.5",
        PYTHONPATH=str(ROOT),
        PYTHONUNBUFFERED="1",
    )
    logs = ROOT / "data" / "logs"
    logs.mkdir(exist_ok=True)
    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8000"],
        cwd=ROOT,
        env=env,
        stdout=open(logs / "api.log", "w"),
        stderr=subprocess.STDOUT,
    )
    (ROOT / ".demo-api.pid").write_text(str(api.pid))
    for worker in ("worker-1", "worker-2"):
        proc = subprocess.Popen(
            [sys.executable, "-m", "app.worker", "--worker-id", worker],
            cwd=ROOT,
            env=env,
            stdout=open(logs / f"{worker}.log", "w"),
            stderr=subprocess.STDOUT,
        )
        (ROOT / f".demo-{worker}.pid").write_text(str(proc.pid))
    wait_for(_healthy, timeout=30, message="local API")
    return api


def _healthy() -> bool:
    try:
        return call("GET", "/health")["status"] == "ok"
    except (urllib.error.URLError, KeyError, TimeoutError):
        return False


def local_down() -> None:
    for name in ("worker-1", "worker-2"):
        pid_file = ROOT / f".demo-{name}.pid"
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
            pid_file.unlink()
    api_pid = ROOT / ".demo-api.pid"
    if api_pid.exists():
        try:
            os.kill(int(api_pid.read_text()), signal.SIGKILL)
        except ProcessLookupError:
            pass
        api_pid.unlink()
    (ROOT / ".demo-worker-a.pid").unlink(missing_ok=True)


def step_7_notification(job_id: str) -> dict:
    """Show that the completion message reached the demo receiver exactly."""
    print("\n[7] completion notification")
    notification = wait_for(
        lambda: (
            items := call("GET", f"/jobs/{job_id}/notifications")
        )
        and items[0]["status"] == "delivered"
        and items[0],
        timeout=120,
        message="notification delivery",
    )
    payload = notification["payload"]
    print(f"    event_id = {notification['event_id']}")
    print(f"    status   = {notification['status']} (attempts {notification['attempts']})")
    print(f"    payload  = job_id={payload['job_id']} loss={payload['loss']:.12f}")
    print(f"               final_weights={payload['final_weights']}")

    inbox = wait_for(
        lambda: [
            entry
            for entry in call("GET", "/demo/receiver/inbox")
            if entry["event_id"] == notification["event_id"]
        ],
        timeout=30,
        message="demo receiver inbox entry",
    )
    entry = inbox[0]
    print(
        f"    receiver received event {entry['event_id']} "
        f"({entry['receive_count']} delivery/deliveries, de-duplicated by event_id)"
    )
    assert entry["payload"] == payload
    assert notification["status"] == "delivered"
    return notification


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="start API and workers locally")
    parser.add_argument("--stop-after", type=int, default=4)
    parser.add_argument(
        "--no-notify", action="store_true", help="submit without a notification_url"
    )
    args = parser.parse_args()

    if args.local:
        local_up()
        mode = "local"
    else:
        wait_for(_healthy, timeout=30, message="API (is docker compose up?)")
        mode = "docker"

    url = None if args.no_notify else notify_url(args.local)
    try:
        job = step_1_submit(url)
        step_2_watch(job["job_id"], args.stop_after)
        step_3_kill(mode, job["job_id"])
        step_4_failover(job["job_id"])
        final = step_5_final(job["job_id"])
        step_6_reference(final, url)
        if url:
            step_7_notification(job["job_id"])
    finally:
        if args.local:
            local_down()
    return 0


if __name__ == "__main__":
    sys.exit(main())
