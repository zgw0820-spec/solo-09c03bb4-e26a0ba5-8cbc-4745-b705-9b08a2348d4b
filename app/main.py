"""Training API service."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from . import outbox, receiver, store
from .config import Settings
from .db import init_db, reader
from .features import InputError

settings = Settings.from_env()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.validate()
    init_db(settings.db_path)
    yield


app = FastAPI(
    title="Fan Shutdown Early-Warning Training API",
    description=(
        "Submit 8-64 fan vibration/current records and train a logistic "
        "regression with batch size 8, learning rate 0.1 and momentum 0.9.\n\n"
        "All training is executed by workers. Every batch commits weights, "
        "momentum, loss and progress in one atomic checkpoint; after a failure "
        "the next worker resumes from the newest committed checkpoint.\n\n"
        "If `notification_url` is given, the final weights, completed status and "
        "a pending completion message are committed together. Workers deliver "
        "that message durably, retrying the identical payload (same `event_id`) "
        "after any failure or restart; the receiver can de-duplicate by that id. "
        "For the demo, set `notification_url` to "
        "`http://api:8000/demo/receiver` (Compose) or "
        "`http://127.0.0.1:8000/demo/receiver` (local) and watch "
        "`GET /jobs/{job_id}/notifications` and `GET /demo/receiver/inbox`."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class TrainingRequest(BaseModel):
    features: list[list[float]] = Field(
        ...,
        description=(
            "8-64 records; each record is a fan vibration/current feature vector. "
            "Columns are standardised automatically."
        ),
        examples=[
            [
                [0.12, 3.4, 41.0],
                [0.15, 3.5, 41.4],
                [0.18, 3.3, 42.1],
                [0.21, 3.6, 41.8],
                [0.62, 5.1, 55.2],
                [0.65, 5.0, 56.0],
                [0.71, 5.4, 57.3],
                [0.74, 5.2, 56.6],
            ]
        ],
    )
    labels: list[int] = Field(
        ...,
        description="One shutdown label per record: 0 = healthy, 1 = shutdown warning.",
        examples=[[0, 0, 0, 0, 1, 1, 1, 1]],
    )
    epochs: int = Field(2, ge=2, le=5, description="Training epochs (2-5).")
    notification_url: str | None = Field(
        None,
        description=(
            "Optional absolute http(s) URL of the operations receiver. When set, "
            "a completion message with the job id, final weights, loss and a fixed "
            "event id is delivered durably after the final weights are committed."
        ),
        examples=["http://receiver:9000/notify"],
    )


@app.get("/health", tags=["system"])
def health() -> dict[str, Any]:
    with reader(settings.db_path) as conn:
        conn.execute("SELECT 1")
    return {"status": "ok", "service": "api", "database": str(settings.db_path)}


@app.post("/jobs", status_code=201, tags=["training"])
def submit_job(request: TrainingRequest) -> dict[str, Any]:
    """Validate and enqueue a training job; it is picked up by a worker."""
    try:
        return store.create_job(
            settings.db_path,
            request.features,
            request.labels,
            request.epochs,
            request.notification_url,
        )
    except InputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/jobs", tags=["training"])
def get_jobs(limit: int = Query(50, ge=1, le=200)) -> list[dict[str, Any]]:
    return store.list_jobs(settings.db_path, limit)


@app.get("/jobs/{job_id}", tags=["training"])
def get_job(job_id: str) -> dict[str, Any]:
    """Confirmed progress, loss, and the latest weights/momentum."""
    try:
        job = store.get_job(settings.db_path, job_id)
    except store.StateConflict as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/jobs/{job_id}/checkpoints", tags=["training"])
def get_checkpoints(
    job_id: str,
    after_step: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=1000),
) -> list[dict[str, Any]]:
    """Batch checkpoints in step order, each with weights, momentum and loss."""
    try:
        return store.list_checkpoints(settings.db_path, job_id, after_step, limit)
    except store.JobNotFound as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc


@app.get("/jobs/{job_id}/events", tags=["training"])
def get_events(job_id: str) -> list[dict[str, Any]]:
    """Audit trail: creation, takeover/resume points and completion."""
    try:
        return store.list_events(settings.db_path, job_id)
    except store.JobNotFound as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc


@app.get("/jobs/{job_id}/notifications", tags=["training"])
def get_notifications(job_id: str) -> list[dict[str, Any]]:
    """Completion messages for the job and their delivery state.

    ``status`` is ``pending`` until the receiver acknowledges, then
    ``delivered``. Delivery failures never change the training result; the
    same stored payload is retried under the same ``event_id``.
    """
    try:
        return outbox.list_notifications(settings.db_path, job_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc


@app.get("/workers", tags=["system"])
def get_workers() -> dict[str, Any]:
    """Worker liveness and current lease state, for the demo."""
    now = time.time()
    with reader(settings.db_path) as conn:
        workers = conn.execute(
            "SELECT worker_id, status, started_at, heartbeat_at FROM workers ORDER BY worker_id"
        ).fetchall()
        leases = conn.execute(
            "SELECT job_id, worker_id, epoch, heartbeat_at FROM leases ORDER BY job_id"
        ).fetchall()
    return {
        "lease_seconds": settings.lease_seconds,
        "workers": [
            {
                "worker_id": row["worker_id"],
                "status": row["status"],
                "started_at": row["started_at"],
                "heartbeat_at": row["heartbeat_at"],
                "seconds_since_heartbeat": round(max(0.0, now - row["heartbeat_at"]), 3),
                "alive": now - row["heartbeat_at"] < settings.lease_seconds,
            }
            for row in workers
        ],
        "leases": [
            {
                "job_id": row["job_id"],
                "worker_id": row["worker_id"],
                "lease_epoch": int(row["epoch"]),
                "expired": now - float(row["heartbeat_at"]) >= settings.lease_seconds,
                "seconds_since_heartbeat": round(max(0.0, now - row["heartbeat_at"]), 3),
            }
            for row in leases
        ],
    }


DEMO_NOTIFICATION_EXAMPLE = {
    "event_id": "job.completed:2f1c9a4e8b7d4c6f9a0e1b2c3d4e5f60",
    "event": "job.completed",
    "job_id": "2f1c9a4e8b7d4c6f9a0e1b2c3d4e5f60",
    "final_weights": [1.2241765631413866, 0.4595338020726801, 0.10336845199201995],
    "loss": 0.3006965745459481,
    "completed_steps": 15,
    "total_steps": 15,
    "finished_at": 1763390000.0,
}


@app.post(
    "/demo/receiver",
    status_code=201,
    tags=["demo receiver"],
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "required": ["event_id", "job_id", "final_weights", "loss"],
                        "properties": {
                            "event_id": {"type": "string"},
                            "event": {"type": "string"},
                            "job_id": {"type": "string"},
                            "final_weights": {
                                "type": "array",
                                "items": {"type": "number"},
                            },
                            "loss": {"type": "number"},
                            "completed_steps": {"type": "integer"},
                            "total_steps": {"type": "integer"},
                            "finished_at": {"type": "number"},
                        },
                    },
                    "example": DEMO_NOTIFICATION_EXAMPLE,
                }
            },
        }
    },
)
async def demo_receive(request: Request) -> dict[str, Any]:
    """Built-in operations receiver entry point (for the Swagger demo).

    Accepts the completion message as-is, stores it once per ``event_id`` and
    returns 201 for both first delivery and duplicates (``duplicate: true``).
    A resent event with different content is rejected with 422 - the real
    sender never does that, because the payload is immutable once queued.
    """
    raw = await request.body()
    try:
        return receiver.receive(settings.db_path, raw)
    except receiver.InvalidNotification as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/demo/receiver/inbox", tags=["demo receiver"])
def demo_inbox() -> list[dict[str, Any]]:
    """Everything the demo receiver accepted, with duplicate counts."""
    return receiver.list_inbox(settings.db_path)


@app.delete("/demo/receiver/inbox", tags=["demo receiver"])
def demo_inbox_clear() -> dict[str, int]:
    """Empty the demo inbox (does not touch training or outbox state)."""
    return {"deleted": receiver.clear_inbox(settings.db_path)}
