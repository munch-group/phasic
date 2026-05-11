"""Composite IPFS transport backend.

Combines the kubo HTTP RPC client (:class:`KuboRPC`) with the read-only
HTTP gateway client (:class:`GatewayClient`) into a single
:class:`TransportBackend` implementation:

- :meth:`get` tries the local daemon first, then falls back to public
  gateways. Works without a daemon.
- :meth:`add`, :meth:`pin`, :meth:`unpin` all require a daemon and
  raise :class:`PTDBackendError` if one is not available.
- :meth:`status` reports both halves of the composite.

Replaces the legacy ``IPFSBackend`` from ``trace_repository.py`` which
depended on the abandoned ``ipfshttpclient`` package.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..exceptions import PTDBackendError
from ..logging_config import get_logger
from . import TransportBackend
from .gateway import DEFAULT_GATEWAYS, GatewayClient
from .kubo_rpc import KuboRPC

logger = get_logger(__name__)


class IPFSBackend(TransportBackend):
    """Composite kubo-daemon + HTTP-gateway transport backend.

    Parameters
    ----------
    daemon_host : str, default '127.0.0.1'
        Hostname for the kubo daemon's HTTP RPC.
    daemon_port : int, default 5001
        Port for the kubo daemon's HTTP RPC.
    gateways : tuple[str, ...], optional
        Public HTTP gateway base URLs. Defaults to
        :data:`phasic.transport.gateway.DEFAULT_GATEWAYS`.
    timeout : float, default 30.0
        Per-request timeout for both the RPC client and the gateway
        client.

    Attributes
    ----------
    kubo : KuboRPC
        The underlying RPC client. ``kubo.is_alive()`` reports
        whether a daemon is reachable.
    gateway : GatewayClient
        Read-only HTTP gateway fallback.
    """

    def __init__(
        self,
        daemon_host: str = "127.0.0.1",
        daemon_port: int = 5001,
        gateways: tuple[str, ...] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.kubo = KuboRPC(
            host=daemon_host, port=daemon_port, timeout=timeout)
        self.gateway = GatewayClient(
            gateways=gateways, timeout=timeout)

    @property
    def name(self) -> str:
        return "ipfs"

    @property
    def daemon_available(self) -> bool:
        """Whether a kubo daemon is currently reachable."""
        return self.kubo.is_alive()

    # ------------------------------------------------------------------
    # TransportBackend interface
    # ------------------------------------------------------------------

    def get(self, cid_or_url: str, output_path: Path) -> None:
        """Fetch *cid_or_url* and write to *output_path*.

        Tries the kubo daemon first, then HTTP gateways. The daemon
        is bypassed if it is not running.
        """
        output_path = Path(output_path)
        if not output_path.parent.exists():
            raise PTDBackendError(
                f"output directory does not exist: {output_path.parent}")

        if self.kubo.is_alive() and not cid_or_url.startswith(
                ("http://", "https://")):
            try:
                content = self.kubo.cat(cid_or_url)
                output_path.write_bytes(content)
                return
            except PTDBackendError as e:
                logger.debug(
                    "kubo cat failed for %s (%s); falling back to gateways",
                    cid_or_url, e,
                )

        self.gateway.fetch(cid_or_url, output_path)

    def add(self, path: Path) -> str:
        """Publish *path* via the local daemon. Daemon required."""
        if not self.kubo.is_alive():
            raise PTDBackendError(
                "Publishing requires a running kubo daemon.\n"
                "  Start with: ipfs daemon\n"
                "  Install:    https://docs.ipfs.tech/install/")
        return self.kubo.add(path, pin=True)

    def pin(self, cid: str) -> None:
        if not self.kubo.is_alive():
            raise PTDBackendError(
                "Pinning requires a running kubo daemon.")
        self.kubo.pin_add(cid)

    def unpin(self, cid: str) -> None:
        if not self.kubo.is_alive():
            raise PTDBackendError(
                "Unpinning requires a running kubo daemon.")
        self.kubo.pin_rm(cid)

    def is_pinned(self, cid: str) -> bool:
        if not self.kubo.is_alive():
            return False
        return self.kubo.pin_ls(cid)

    def status(self) -> dict[str, Any]:
        """Return a summary of the backend's connection state."""
        daemon = self.kubo.is_alive()
        info: dict[str, Any] = {
            "name": self.name,
            "daemon": daemon,
            "daemon_version": None,
            "gateways": list(self.gateway.gateways),
        }
        if daemon:
            try:
                info["daemon_version"] = self.kubo.version().get("Version")
            except PTDBackendError:
                pass
        return info


__all__ = ["IPFSBackend", "DEFAULT_GATEWAYS"]
