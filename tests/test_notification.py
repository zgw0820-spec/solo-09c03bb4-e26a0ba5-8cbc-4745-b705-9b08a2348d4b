"""Completion notifications: atomic enqueue, durable retry, stable event id.

The receiver here is a real local HTTP server, so the tests exercise the same
``urllib`` code path a deployed worker uses.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from app import lease as lm
from app import outbox, receiver, store
from app.config import Settings
from app.db import init_db
from app.notifier import Notifier
from app.worker import Worker


def make_settings(tmp_path, **overrides) -> Settings:
    base = dict(
        db_path=tmp_path / "training.db",
        lease_seconds=0.6,
        heartbeat_interval=0.1,
        poll_interval=0.02,
        batch_delay_seconds=0.0,
        notification_poll_interval=0.02,
        notification_timeout=1.0,
        notification_retry_min=0.05,
        notification_retry_max=0.2,
        notification_claim_seconds=0.5,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture()
def settings(tmp_path):
    s = make_settings(tmp_path)
    s.validate()
    init_db(s.db_path)
    return s


def sample(records: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(records, 3)).tolist()
    labels = [int(i) % 2 for i in range(records)]
    return features, labels


def complete_job(settings, url: str | None, seed: int = 0):
    job = store.create_job(settings.db_path, *sample(10, seed), 2, url)
    lm.register_worker(settings.db_path, "trainer")
    lease = lm.acquire(settings.db_path, job["job_id"], "trainer", settings.lease_seconds)
    while True:
        result = store.advance_job(settings.db_path, lease)
        if result["status"] == "completed":
            return job


class Receiver:
    """Minimal HTTP receiver that records every request body."""

    def __init__(self) -> None:
        self.bodies: list[bytes] = []
        self.event_ids: list[str | None] = []
        self.failures_before_success = 0
        self.status = 201
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                outer.bodies.append(body)
                outer.event_ids.append(self.headers.get("X-Event-Id"))
                if outer.failures_before_success > 0:
                    outer.failures_before_success -= 1
                    self.send_response(503)
                    self.end_headers()
                    return
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"accepted":true}')

            def log_message(self, *args):  # keep pytest output clean
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}/notify"

    def __enter__(self) -> "Receiver":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def closed_port_url() -> str:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}/notify"


def test_completion_atomically_queues_message_with_final_state(settings):
    with Receiver() as receiver_server:
        job = complete_job(settings, receiver_server.url)

    final = store.get_job(settings.db_path, job["job_id"])
    notifications = outbox.list_notifications(settings.db_path, job["job_id"])
    assert len(notifications) == 1
    message = notifications[0]
    assert message["status"] == "pending"
    payload = message["payload"]
    assert payload["job_id"] == job["job_id"]
    assert payload["event"] == "job.completed"
    assert payload["event_id"] == outbox.event_id_for(job["job_id"])
    assert payload["final_weights"] == final["final_weights"]
    assert payload["loss"] == final["loss"]
    assert payload["completed_steps"] == final["total_steps"]
    assert message["attempts"] == 0

    events = [e["event"] for e in store.list_events(settings.db_path, job["job_id"])]
    assert events.count("notification_queued") == 1

    # The training result is already fully committed and independent of sending.
    assert final["status"] == "completed"
    assert final["completed_steps"] == final["total_steps"]


def test_job_without_url_has_no_notification(settings):
    job = complete_job(settings, None)
    assert store.get_job(settings.db_path, job["job_id"])["notification"] is None
    assert outbox.list_notifications(settings.db_path, job["job_id"]) == []


def test_notifier_delivers_and_marks_delivered(settings):
    with Receiver() as receiver_server:
        job = complete_job(settings, receiver_server.url)
        notifier = Notifier(settings, "w1", threading.Event())
        assert notifier.run_once() is True

    assert len(receiver_server.bodies) == 1
    body = json.loads(receiver_server.bodies[0])
    assert body["job_id"] == job["job_id"]
    assert receiver_server.event_ids == [outbox.event_id_for(job["job_id"])]

    message = outbox.list_notifications(settings.db_path, job["job_id"])[0]
    assert message["status"] == "delivered"
    assert message["attempts"] == 1
    assert message["delivered_status"] == 201
    assert notifier.run_once() is False  # nothing left to send

    final = store.get_job(settings.db_path, job["job_id"])
    assert final["status"] == "completed"
    assert final["notification"]["status"] == "delivered"


def test_unavailable_receiver_retries_with_identical_payload(settings):
    job = complete_job(settings, closed_port_url())
    notifier = Notifier(settings, "w1", threading.Event())

    assert notifier.run_once() is False
    pending = outbox.list_notifications(settings.db_path, job["job_id"])[0]
    assert pending["status"] == "pending"
    assert pending["attempts"] == 1
    assert "connect failed" in pending["last_error"]
    first_bytes = json.dumps(pending["payload"], sort_keys=True, separators=(",", ":"))

    # Training result is untouched by the delivery failure.
    failed_state = store.get_job(settings.db_path, job["job_id"])
    assert failed_state["status"] == "completed"
    assert failed_state["final_weights"] == pending["payload"]["final_weights"]

    with Receiver() as receiver_server:
        # Point the queued message at the now-available receiver without
        # changing its payload: the URL is delivery metadata, not content.
        with store.transaction(settings.db_path) as conn:
            conn.execute(
                "UPDATE outbox SET url=?, next_attempt_at=0 WHERE job_id=?",
                (receiver_server.url, job["job_id"]),
            )
        notifier.settings = settings
        assert notifier.run_once() is True

    assert len(receiver_server.bodies) == 1
    assert json.loads(receiver_server.bodies[0]) == json.loads(first_bytes)

    delivered = outbox.list_notifications(settings.db_path, job["job_id"])[0]
    assert delivered["status"] == "delivered"
    assert delivered["attempts"] == 2
    assert delivered["payload"] == pending["payload"]
    final = store.get_job(settings.db_path, job["job_id"])
    assert final["final_weights"] == failed_state["final_weights"]
    assert final["loss"] == failed_state["loss"]
    assert final["finished_at"] == failed_state["finished_at"]


def test_receiver_rejection_then_retry_keeps_event_id_and_content(settings):
    with Receiver() as receiver_server:
        receiver_server.failures_before_success = 2
        job = complete_job(settings, receiver_server.url)
        notifier = Notifier(settings, "w1", threading.Event())
        for _ in range(3):
            notifier.run_once()
            # Backoff is short in tests; ignore the due time.
            with store.transaction(settings.db_path) as conn:
                conn.execute("UPDATE outbox SET next_attempt_at=0 WHERE job_id=?", (job["job_id"],))
            if outbox.list_notifications(settings.db_path, job["job_id"])[0]["status"] == "delivered":
                break

    assert len(receiver_server.bodies) == 3
    assert len(set(receiver_server.bodies)) == 1  # byte-identical on every retry
    assert receiver_server.event_ids == [outbox.event_id_for(job["job_id"])] * 3
    message = outbox.list_notifications(settings.db_path, job["job_id"])[0]
    assert message["status"] == "delivered"
    assert message["attempts"] == 3
    assert message["last_error"] is None


def test_two_workers_deliver_a_job_only_once(settings):
    with Receiver() as receiver_server:
        job = complete_job(settings, receiver_server.url)
        events = [threading.Event() for _ in range(2)]
        notifiers = [Notifier(settings, f"w{i}", events[i - 1]) for i in (1, 2)]
        threads = [
            threading.Thread(target=lambda n=n: n.run_once()) for n in notifiers
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

    assert len(receiver_server.bodies) == 1
    message = outbox.list_notifications(settings.db_path, job["job_id"])[0]
    assert message["status"] == "delivered"
    assert message["attempts"] == 1


def test_restarted_worker_sends_message_left_by_dead_worker(settings):
    """No worker is running when the job completes; a later worker sends it."""
    with Receiver() as receiver_server:
        job = complete_job(settings, receiver_server.url)
        assert outbox.list_notifications(settings.db_path, job["job_id"])[0]["status"] == "pending"

        worker = Worker(settings, "restarted")
        thread = threading.Thread(target=worker.run, daemon=True)
        thread.start()
        try:
            deadline = time.time() + 10
            while time.time() < deadline:
                if outbox.list_notifications(settings.db_path, job["job_id"])[0]["status"] == "delivered":
                    break
                time.sleep(0.02)
        finally:
            worker.request_stop()
            thread.join(timeout=10)

    assert not thread.is_alive()
    message = outbox.list_notifications(settings.db_path, job["job_id"])[0]
    assert message["status"] == "delivered"
    assert json.loads(receiver_server.bodies[0])["event_id"] == message["event_id"]


def test_completion_rolls_back_entirely_if_message_cannot_be_queued(settings, monkeypatch):
    """Final weights, completed status and the message are one commit.

    If the outbox insert fails, the batch must roll back completely: the job
    stays unfinished at the previous checkpoint and nothing is sent.
    """
    job = store.create_job(settings.db_path, *sample(10, seed=3), 2, "http://127.0.0.1:9/notify")
    lm.register_worker(settings.db_path, "trainer")
    lease = lm.acquire(settings.db_path, job["job_id"], "trainer", settings.lease_seconds)
    result = store.advance_job(settings.db_path, lease)
    while result["step"] < job["total_steps"] - 1:
        result = store.advance_job(settings.db_path, lease)
    before = store.get_job(settings.db_path, job["job_id"])
    assert before["completed_steps"] == before["total_steps"] - 1

    def fail_enqueue(*args, **kwargs):
        raise RuntimeError("outbox write failed")

    monkeypatch.setattr(outbox, "enqueue", fail_enqueue)
    with pytest.raises(RuntimeError):
        store.advance_job(settings.db_path, lease)
    monkeypatch.undo()

    after = store.get_job(settings.db_path, job["job_id"])
    assert after["status"] == "running"
    assert after["completed_steps"] == before["completed_steps"]
    assert after["weights"] == before["weights"]
    assert after["loss"] == before["loss"]
    assert after["finished_at"] is None
    assert outbox.list_notifications(settings.db_path, job["job_id"]) == []
    assert len(store.list_checkpoints(settings.db_path, job["job_id"])) == before["completed_steps"]
    events = [e["event"] for e in store.list_events(settings.db_path, job["job_id"])]
    assert "job_completed" not in events

    # A later attempt still completes and queues exactly one message.
    final = store.advance_job(settings.db_path, lease)
    assert final["status"] == "completed"
    assert len(outbox.list_notifications(settings.db_path, job["job_id"])) == 1


def test_duplicate_completion_never_queues_a_second_message(settings):
    with Receiver() as receiver_server:
        job = complete_job(settings, receiver_server.url)
        # A late worker that still believes it owns the job calls advance_job
        # after completion; the write is fenced, so it cannot re-queue or
        # rewrite anything.
        with store.transaction(settings.db_path) as conn:
            conn.execute(
                "UPDATE leases SET worker_id='late', epoch=99, heartbeat_at=? "
                "WHERE job_id=?",
                (time.time(), job["job_id"]),
            )
        late = lm.Lease(job["job_id"], "late", 99)
        result = store.advance_job(settings.db_path, late)
        assert result["status"] == "completed"

    notifications = outbox.list_notifications(settings.db_path, job["job_id"])
    assert len(notifications) == 1
    assert notifications[0]["event_id"] == outbox.event_id_for(job["job_id"])
    events = [e["event"] for e in store.list_events(settings.db_path, job["job_id"])]
    assert events.count("job_completed") == 1
    assert events.count("notification_queued") == 1


def test_lost_acknowledgement_resends_identical_event(settings, monkeypatch):
    """The receiver saves the message but its 201 never reaches us.

    A crashed sender leaves the row pending after its claim fence expires; the
    next sender must resend byte-identical content under the same event id,
    which is what lets the receiver de-duplicate.
    """
    with Receiver() as receiver_server:
        job = complete_job(settings, receiver_server.url)
        notifier = Notifier(settings, "w1", threading.Event())

        def crash_after_send(*args, **kwargs):
            raise RuntimeError("sender died before recording the acknowledgement")

        monkeypatch.setattr("app.notifier.outbox.mark_delivered", crash_after_send)
        with pytest.raises(RuntimeError):
            notifier.run_once()

        pending = outbox.list_notifications(settings.db_path, job["job_id"])[0]
        assert pending["status"] == "pending"
        assert pending["attempts"] == 1

        monkeypatch.undo()
        # Wait for the dead sender's claim fence; the message is still pending.
        time.sleep(settings.notification_claim_seconds + 0.05)
        assert notifier.run_once() is True

    assert len(receiver_server.bodies) == 2
    assert receiver_server.bodies[0] == receiver_server.bodies[1]
    assert receiver_server.event_ids == [outbox.event_id_for(job["job_id"])] * 2
    delivered = outbox.list_notifications(settings.db_path, job["job_id"])[0]
    assert delivered["status"] == "delivered"
    assert delivered["attempts"] == 2


def test_existing_database_is_upgraded_in_place(tmp_path):
    """A database created by the previous release gains the new schema."""
    import sqlite3

    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY, status TEXT NOT NULL, feature_dim INTEGER NOT NULL,
            epochs INTEGER NOT NULL, batch_size INTEGER NOT NULL, record_count INTEGER NOT NULL,
            total_batches INTEGER NOT NULL, learning_rate REAL NOT NULL, momentum REAL NOT NULL,
            features_json TEXT NOT NULL, labels_json TEXT NOT NULL, weights_json TEXT NOT NULL,
            velocity_json TEXT NOT NULL, loss REAL NOT NULL,
            completed_steps INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL, updated_at REAL NOT NULL, finished_at REAL
        );
        """
    )
    conn.execute(
        "INSERT INTO jobs VALUES('old', 'completed', 2, 2, 8, 8, 1, 0.1, 0.9, '[]', '[]',"
        " '[0.0]', '[0.0]', 0.5, 1, 1.0, 1.0, 2.0)"
    )
    conn.commit()
    conn.close()

    init_db(db)
    init_db(db)  # idempotent when a second container starts

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
    tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master")}
    old = conn.execute("SELECT * FROM jobs WHERE job_id='old'").fetchone()
    conn.close()
    assert "notification_url" in columns
    assert {"outbox", "demo_inbox"} <= tables
    assert old["notification_url"] is None
    assert old["completed_steps"] == 1


def test_notification_url_validation(settings):
    from app.features import InputError

    features, labels = sample(10)
    for url in ("ftp://receiver/notify", "receiver/notify", "http://"):
        with pytest.raises(InputError):
            store.create_job(settings.db_path, features, labels, 2, url)
    assert store.create_job(settings.db_path, features, labels, 2, "")["notification_url"] is None


def test_demo_receiver_deduplicates_by_event_id(settings):
    payload = json.loads(
        outbox.build_payload("job-1", [0.25, -1.0], 0.5, 6, 6, time.time())
    )
    raw = json.dumps(payload).encode()

    first = receiver.receive(settings.db_path, raw)
    second = receiver.receive(settings.db_path, raw)
    assert first["duplicate"] is False and first["receive_count"] == 1
    assert second["duplicate"] is True and second["receive_count"] == 2
    inbox = receiver.list_inbox(settings.db_path)
    assert len(inbox) == 1 and inbox[0]["event_id"] == payload["event_id"]

    changed = dict(payload, loss=0.6)
    with pytest.raises(receiver.InvalidNotification):
        receiver.receive(settings.db_path, json.dumps(changed).encode())

    with pytest.raises(receiver.InvalidNotification):
        receiver.receive(settings.db_path, b"not json")
    with pytest.raises(receiver.InvalidNotification):
        receiver.receive(settings.db_path, b'{"loss": 1.0}')
    assert receiver.clear_inbox(settings.db_path) == 1


def test_demo_receiver_accepts_worker_delivery(settings):
    """The API endpoint path used by Compose: deliver through the demo inbox."""
    from fastapi.testclient import TestClient

    import app.main as main_module

    main_module.settings = settings
    with TestClient(main_module.app) as client:
        job = complete_job(settings, "http://api:8000/demo/receiver")
        message = outbox.list_notifications(settings.db_path, job["job_id"])[0]
        raw = json.dumps(message["payload"]).encode()

        response = client.post(
            "/demo/receiver",
            content=raw,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 201
        assert response.json()["duplicate"] is False
        repeat = client.post("/demo/receiver", content=raw)
        assert repeat.json()["duplicate"] is True and repeat.json()["receive_count"] == 2

        inbox = client.get("/demo/receiver/inbox").json()
        assert [entry["event_id"] for entry in inbox] == [message["event_id"]]
        notifications = client.get(f"/jobs/{job['job_id']}/notifications").json()
        assert notifications[0]["payload"] == message["payload"]
