"""Local demonstration receiver for completion notifications.

This is the "operations receiver" used by the Compose demo and Swagger.  It
records each event once by ``event_id``: a retried delivery of the same event
increments ``receive_count`` instead of adding a second entry, which is exactly
the de-duplication a real receiver can rely on because the sender never changes
the event id or the payload bytes.

The inbox is deliberately tiny and demo-only; it lives in the same SQLite file
so ``docker compose down -v`` resets it with everything else.
"""
from __future__ import annotations

import json
import time
from typing import Any

from .db import reader, transaction

MAX_PAYLOAD_BYTES = 256 * 1024


class InvalidNotification(ValueError):
    pass


def receive(db_path, raw_body: bytes, now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    if len(raw_body) > MAX_PAYLOAD_BYTES:
        raise InvalidNotification("notification payload is too large")
    try:
        payload = json.loads(raw_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidNotification("notification payload must be JSON") from exc
    if not isinstance(payload, dict):
        raise InvalidNotification("notification payload must be a JSON object")
    event_id = payload.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        raise InvalidNotification("notification payload needs a non-empty event_id")

    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with transaction(db_path) as conn:
        existing = conn.execute(
            "SELECT payload_json, receive_count FROM demo_inbox WHERE event_id=?",
            (event_id,),
        ).fetchone()
        if existing is not None:
            if existing["payload_json"] != body:
                raise InvalidNotification(
                    "the same event_id was resent with different content"
                )
            conn.execute(
                "UPDATE demo_inbox SET receive_count=receive_count + 1, last_received_at=? "
                "WHERE event_id=?",
                (now, event_id),
            )
            duplicate = True
        else:
            conn.execute(
                """
                INSERT INTO demo_inbox(
                    event_id, payload_json, first_received_at, last_received_at,
                    receive_count
                ) VALUES(?, ?, ?, ?, 1)
                """,
                (event_id, body, now, now),
            )
            duplicate = False
        row = conn.execute(
            "SELECT * FROM demo_inbox WHERE event_id=?", (event_id,)
        ).fetchone()
    return {
        "accepted": True,
        "event_id": event_id,
        "duplicate": duplicate,
        "receive_count": int(row["receive_count"]),
        "first_received_at": float(row["first_received_at"]),
        "last_received_at": float(row["last_received_at"]),
        "payload": json.loads(row["payload_json"]),
    }


def list_inbox(db_path) -> list[dict[str, Any]]:
    with reader(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM demo_inbox ORDER BY first_received_at, event_id"
        ).fetchall()
    return [
        {
            "event_id": row["event_id"],
            "payload": json.loads(row["payload_json"]),
            "receive_count": int(row["receive_count"]),
            "first_received_at": float(row["first_received_at"]),
            "last_received_at": float(row["last_received_at"]),
        }
        for row in rows
    ]


def clear_inbox(db_path) -> int:
    with transaction(db_path) as conn:
        deleted = conn.execute("DELETE FROM demo_inbox").rowcount
    return int(deleted)
