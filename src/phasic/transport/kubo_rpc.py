"""Minimal kubo HTTP RPC client.

Replaces the dead ``ipfshttpclient`` dependency with a direct wrapper
around the kubo ``/api/v0/*`` HTTP RPC. Tested against kubo >= 0.9; the
endpoints used here have been stable since kubo 0.5.

The kubo RPC uses ``POST`` for every endpoint (yes, even reads). Each
response is a JSON object (``add``, ``version``, ``pin/ls``) or raw
bytes (``cat``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

from ..exceptions import PTDBackendError
from ..logging_config import get_logger
from ._retry import request_with_retry

logger = get_logger(__name__)


class KuboRPC:
    """Thin client for the kubo HTTP RPC.

    Parameters
    ----------
    host : str, default '127.0.0.1'
        Daemon API host.
    port : int, default 5001
        Daemon API port.
    timeout : float, default 30.0
        Per-request timeout in seconds. The :meth:`is_alive` probe
        uses a shorter, hard-coded 1.0 s timeout instead.
    session : requests.Session, optional
        Custom session for connection pooling. A new session is
        created if omitted.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5001,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._session = session or requests.Session()
        self._base = f"http://{host}:{port}/api/v0"

    # ------------------------------------------------------------------
    # Probing
    # ------------------------------------------------------------------

    def is_alive(self) -> bool:
        """Return ``True`` if a kubo daemon is reachable.

        Uses a hard-coded 1-second timeout so the probe never stalls
        startup. Any error — connection refused, timeout, non-2xx —
        returns ``False``.
        """
        try:
            r = self._session.post(f"{self._base}/version", timeout=1.0)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def version(self) -> dict[str, Any]:
        """Return the daemon's version info.

        Raises
        ------
        PTDBackendError
            If the daemon is unreachable or returns a non-2xx status.
        """
        try:
            r = request_with_retry(
                f"{self._base}/version",
                method="POST",
                timeout=self.timeout,
                session=self._session,
            )
            return r.json()
        except requests.RequestException as e:
            raise PTDBackendError(f"kubo version probe failed: {e}") from e

    # ------------------------------------------------------------------
    # Content
    # ------------------------------------------------------------------

    def cat(self, cid: str) -> bytes:
        """Return the raw bytes of *cid*.

        Parameters
        ----------
        cid : str
            Content identifier (CIDv0 or CIDv1).

        Returns
        -------
        bytes
            The full content. The kubo RPC streams chunks; this
            helper accumulates them into a single ``bytes`` object.
        """
        try:
            r = request_with_retry(
                f"{self._base}/cat",
                method="POST",
                params={"arg": cid},
                timeout=self.timeout,
                session=self._session,
                stream=True,
            )
            chunks: list[bytes] = []
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    chunks.append(chunk)
            return b"".join(chunks)
        except requests.RequestException as e:
            raise PTDBackendError(f"kubo cat {cid} failed: {e}") from e

    def add(self, path: Path, pin: bool = True) -> str:
        """Publish a file to the local daemon and return its CID.

        Parameters
        ----------
        path : Path
            Local file to upload.
        pin : bool, default True
            If ``True``, also pin the content so it is not
            garbage-collected.

        Returns
        -------
        str
            CID of the published file.
        """
        path = Path(path)
        if not path.is_file():
            raise PTDBackendError(
                f"kubo add: not a file: {path}")
        try:
            with path.open("rb") as fh:
                files = {"file": (path.name, fh, "application/octet-stream")}
                r = request_with_retry(
                    f"{self._base}/add",
                    method="POST",
                    params={"pin": "true" if pin else "false",
                            "quieter": "true"},
                    files=files,
                    timeout=self.timeout,
                    session=self._session,
                )
        except requests.RequestException as e:
            raise PTDBackendError(f"kubo add {path} failed: {e}") from e

        # kubo /add streams one JSON object per file; with --quieter
        # it streams only the root hash. Parse the last JSON line.
        last_line = ""
        for line in r.text.splitlines():
            if line.strip():
                last_line = line
        if not last_line:
            raise PTDBackendError(
                f"kubo add {path}: empty response body")
        try:
            doc = json.loads(last_line)
        except json.JSONDecodeError as e:
            raise PTDBackendError(
                f"kubo add {path}: malformed response {last_line!r}: {e}"
            ) from e
        cid = doc.get("Hash")
        if not cid:
            raise PTDBackendError(
                f"kubo add {path}: response missing Hash field: {doc!r}")
        return cid

    # ------------------------------------------------------------------
    # Pinning
    # ------------------------------------------------------------------

    def pin_add(self, cid: str) -> None:
        """Pin *cid* so the daemon does not garbage-collect it."""
        try:
            request_with_retry(
                f"{self._base}/pin/add",
                method="POST",
                params={"arg": cid},
                timeout=self.timeout,
                session=self._session,
            )
        except requests.RequestException as e:
            raise PTDBackendError(f"kubo pin add {cid} failed: {e}") from e

    def pin_rm(self, cid: str) -> None:
        """Unpin *cid*; eligible for GC after this call."""
        try:
            request_with_retry(
                f"{self._base}/pin/rm",
                method="POST",
                params={"arg": cid},
                timeout=self.timeout,
                session=self._session,
            )
        except requests.RequestException as e:
            raise PTDBackendError(f"kubo pin rm {cid} failed: {e}") from e

    def pin_ls(self, cid: str) -> bool:
        """Check whether *cid* is pinned. Returns ``False`` on error."""
        try:
            r = self._session.post(
                f"{self._base}/pin/ls",
                params={"arg": cid},
                timeout=self.timeout,
            )
            if r.status_code != 200:
                return False
            doc = r.json()
            return cid in doc.get("Keys", {})
        except requests.RequestException:
            return False


__all__ = ["KuboRPC"]
