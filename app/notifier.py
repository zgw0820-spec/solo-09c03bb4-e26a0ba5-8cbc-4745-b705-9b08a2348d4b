"""Durable delivery of completion messages to the operations receiver.

Both workers run a :class:`Notifier` thread alongside their training loop.
Every due pending outbox row is claimed atomically (so the two workers do not
both send at the same instant), POSTed as JSON, and marked ``delivered`` only
after a 2xx acknowledgement.  Anything else - connection refused, timeout,
5xx - is a retry: the row stays pending with exponential backoff, and the
exact stored bytes are re-sent under the same ``event_id``.

Delivery is independent of training: it never touches the job, checkpoint or
lease rows.  If every worker is down, the message stays in the shared database
and the next worker to start sends it; a restart does not lose it and does not
change its content.
"""
from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request

from . import outbox
from .config import Settings


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Notifier:
    def __init__(self, settings: Settings, worker_id: str, stop_event: threading.Event) -> None:
        self.settings = settings
        self.worker_id = worker_id
        self.stop_event = stop_event
        self._thread: threading.Thread | None = None
        self._opener = urllib.request.build_opener(_NoRedirect)

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name=f"notifier-{self.worker_id}", daemon=True
        )
        self._thread.start()

    def join(self, timeout: float = 10.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                delivered = self.run_once()
            except Exception as exc:  # never let delivery kill the worker
                print(f"[{self.worker_id}] notifier error: {exc}", flush=True)
                delivered = False
            # Drain a backlog without an artificial pause; back off only when
            # there was nothing due or the attempt failed.
            if not delivered:
                self.stop_event.wait(self.settings.notification_poll_interval)

    def run_once(self) -> bool:
        """Attempt one due message. Returns True if it was delivered."""
        claimed = outbox.claim_next(
            self.settings.db_path,
            self.worker_id,
            self.settings.notification_claim_seconds,
        )
        if claimed is None:
            return False
        event_id = claimed["event_id"]
        try:
            status_code = self._post(
                claimed["url"], claimed["payload_json"], claimed["job_id"]
            )
        except Exception as exc:
            delay = self._backoff(claimed["attempts"])
            outbox.mark_failed(
                self.settings.db_path,
                event_id,
                self.worker_id,
                _describe_error(exc),
                time.time() + delay,
            )
            print(
                f"[{self.worker_id}] notify {event_id} failed "
                f"(attempt {claimed['attempts']}); retry in {delay:.1f}s: {_describe_error(exc)}",
                flush=True,
            )
            return False
        outbox.mark_delivered(self.settings.db_path, event_id, self.worker_id, status_code)
        print(
            f"[{self.worker_id}] notify {event_id} delivered (HTTP {status_code},"
            f" attempt {claimed['attempts']})",
            flush=True,
        )
        return True

    def _backoff(self, attempts: int) -> float:
        exponent = min(max(attempts - 1, 0), 16)
        return min(
            self.settings.notification_retry_max,
            self.settings.notification_retry_min * (2**exponent),
        )

    def _post(self, url: str, payload_json: str, job_id: str) -> int:
        request = urllib.request.Request(
            url,
            data=payload_json.encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Event-Id": outbox.event_id_for(job_id),
            },
            method="POST",
        )
        try:
            # Redirects are not followed: urllib would turn a redirected POST
            # into a body-less GET, which is not a delivery. A 3xx is a failure
            # and is retried (or fixed by the operator).
            with self._opener.open(request, timeout=self.settings.notification_timeout) as response:
                response.read()
                if 200 <= response.status < 300:
                    return response.status
                raise RuntimeError(f"receiver returned HTTP {response.status}")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"receiver returned HTTP {exc.code}") from exc


def _describe_error(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.URLError) and not isinstance(exc, urllib.error.HTTPError):
        return f"connect failed: {exc.reason}"
    return str(exc) or exc.__class__.__name__
