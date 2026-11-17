"""Transactional outbox for completion notifications.

The final batch commit writes the last weights, the ``completed`` status and
the pending notification row in one ``BEGIN IMMEDIATE`` transaction, so a job
can never be completed without its message being durably queued, and a message
can never be queued for a job whose final weights were rolled back.

A row is immutable once written: ``payload_json`` is rendered once from the
committed final weights, and every retry sends exactly those bytes under the
same ``event_id``.  Receivers can therefore de-duplicate by ``event_id`` even
when an acknowledgement is lost and the message is sent again.
"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Any
from urllib.parse import urlsplit

from .db import reader, transaction

EVENT_NAME = "job.completed"
MAX_URL_LENGTH = 2048


class InvalidNotificationURL(ValueError):
    """The notification receiver URL is not an absolute http(s) URL."""


def validate_url(raw: str | None) -> str | None:
    if raw is None:
        return None
    url = raw.strip()
    if not url:
        return None
    if len(url) > MAX_URL_LENGTH:
        raise InvalidNotificationURL(f"notification_url must be at most {MAX_URL_LENGTH} characters")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise InvalidNotificationURL("notification_url must be an absolute http(s) URL")
    return url


def event_id_for(job_id: str) -> str:
    """A deterministic, never-changing event id for a job's completion."""
    return f"{EVENT_NAME}:{job_id}"


def build_payload(
    job_id: str,
    final_weights: list[float],
    loss: float,
    completed_steps: int,
    total_steps: int,
    finished_at: float,
) -> str:
    """Serialise the completion message once, with a stable key order."""
    payload = {
        "event_id": event_id_for(job_id),
        "event": EVENT_NAME,
        "job_id": job_id,
        "final_weights": [float(value) for value in final_weights],
        "loss": float(loss),
        "completed_steps": int(completed_steps),
        "total_steps": int(total_steps),
        "finished_at": float(finished_at),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def enqueue(
    conn: sqlite3.Connection,
    job_id: str,
    url: str,
    payload_json: str,
    now: float,
) -> str:
    """Insert the pending message inside the caller's completion transaction."""
    event_id = event_id_for(job_id)
    conn.execute(
        """
        INSERT INTO outbox(
            event_id, job_id, url, payload_json, status, attempts,
            next_attempt_at, created_at, updated_at
        ) VALUES(?, ?, ?, ?, 'pending', 0, ?, ?, ?)
        """,
        (event_id, job_id, url, payload_json, now, now, now),
    )
    return event_id


def _summary(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "event_id": row["event_id"],
        "event": EVENT_NAME,
        "url": row["url"],
        "payload": json.loads(row["payload_json"]),
        "status": row["status"],
        "attempts": int(row["attempts"]),
        "last_error": row["last_error"],
        "next_attempt_at": float(row["next_attempt_at"]),
        "created_at": float(row["created_at"]),
        "updated_at": float(row["updated_at"]),
        "sent_at": row["sent_at"],
        "delivered_status": row["delivered_status"],
        "delivered_by": row["delivered_by"],
    }


def notification_for_job(conn: sqlite3.Connection, job_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM outbox WHERE job_id=? ORDER BY created_at, event_id LIMIT 1",
        (job_id,),
    ).fetchone()
    return _summary(row)


def list_notifications(db_path, job_id: str) -> list[dict[str, Any]]:
    with reader(db_path) as conn:
        if conn.execute("SELECT 1 FROM jobs WHERE job_id=?", (job_id,)).fetchone() is None:
            raise LookupError(job_id)
        rows = conn.execute(
            "SELECT * FROM outbox WHERE job_id=? ORDER BY created_at, event_id",
            (job_id,),
        ).fetchall()
    return [_summary(row) for row in rows]


def claim_next(
    db_path, worker_id: str, claim_seconds: float, now: float | None = None
) -> dict[str, Any] | None:
    """Claim one due pending message, fencing other senders for a while.

    The short claim is not a delivery guarantee: a sender that dies after the
    receiver accepted the message leaves the row pending, and the next claim
    re-sends it.  That is safe because the payload and event id never change.
    """
    now = time.time() if now is None else now
    with transaction(db_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM outbox
             WHERE status='pending'
               AND next_attempt_at <= ?
               AND (claim_expires_at IS NULL OR claim_expires_at <= ?)
             ORDER BY created_at, event_id
             LIMIT 1
            """,
            (now, now),
        ).fetchone()
        if row is None:
            return None
        updated = conn.execute(
            """
            UPDATE outbox
               SET claimed_by=?, claim_expires_at=?, attempts=attempts + 1, updated_at=?
             WHERE event_id=? AND status='pending'
               AND (claim_expires_at IS NULL OR claim_expires_at <= ?)
            """,
            (worker_id, now + claim_seconds, now, row["event_id"], now),
        )
        if updated.rowcount != 1:
            return None
        claimed = conn.execute(
            "SELECT * FROM outbox WHERE event_id=?", (row["event_id"],)
        ).fetchone()
    return {
        "event_id": claimed["event_id"],
        "job_id": claimed["job_id"],
        "url": claimed["url"],
        "payload_json": claimed["payload_json"],
        "attempts": int(claimed["attempts"]),
    }


def mark_delivered(
    db_path, event_id: str, worker_id: str, status_code: int, now: float | None = None
) -> None:
    now = time.time() if now is None else now
    with transaction(db_path) as conn:
        conn.execute(
            """
            UPDATE outbox
               SET status='delivered', sent_at=?, delivered_status=?, delivered_by=?,
                   last_error=NULL, updated_at=?, claimed_by=NULL, claim_expires_at=NULL
             WHERE event_id=? AND status='pending'
            """,
            (now, status_code, worker_id, now, event_id),
        )


def mark_failed(
    db_path,
    event_id: str,
    worker_id: str,
    error: str,
    retry_at: float,
    now: float | None = None,
) -> None:
    """Record a failed attempt and schedule the retry.

    Only touches the row if this sender still holds the claim (or it already
    expired), so a sender that stalled past its claim cannot overwrite a newer
    attempt's backoff.
    """
    now = time.time() if now is None else now
    with transaction(db_path) as conn:
        conn.execute(
            """
            UPDATE outbox
               SET last_error=?, next_attempt_at=?, updated_at=?,
                   claimed_by=NULL, claim_expires_at=NULL
             WHERE event_id=? AND status='pending'
               AND (claimed_by=? OR claimed_by IS NULL OR claim_expires_at <= ?)
            """,
            (error[:2000], retry_at, now, event_id, worker_id, now),
        )
