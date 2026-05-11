"""Read-only HTTP gateway client for IPFS content.

A consumer that has no kubo daemon installed can still fetch
content-addressed artifacts through public HTTP gateways. This module
implements the fallback-loop pattern: try each configured gateway in
sequence, remembering the last error, and raise only if every gateway
fails.

Publishing is not supported here — that requires a kubo daemon (see
:class:`phasic.transport.kubo_rpc.KuboRPC`).
"""
from __future__ import annotations

from pathlib import Path

import requests

from ..exceptions import PTDBackendError
from ..logging_config import get_logger
from ._retry import request_with_retry

logger = get_logger(__name__)


DEFAULT_GATEWAYS: tuple[str, ...] = (
    "https://ipfs.io",
    "https://cloudflare-ipfs.com",
    "https://dweb.link",
    "https://gateway.pinata.cloud",
)


class GatewayClient:
    """Fetch IPFS content via public HTTP gateways.

    Parameters
    ----------
    gateways : tuple[str, ...], optional
        Ordered list of gateway base URLs (no trailing ``/ipfs/``).
        Defaults to :data:`DEFAULT_GATEWAYS`.
    timeout : float, default 30.0
        Per-gateway request timeout in seconds.
    session : requests.Session, optional
        Custom session for connection pooling.
    """

    def __init__(
        self,
        gateways: tuple[str, ...] | None = None,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.gateways = tuple(gateways) if gateways else DEFAULT_GATEWAYS
        self.timeout = timeout
        self._session = session or requests.Session()

    def fetch(self, cid_or_url: str, output_path: Path) -> None:
        """Download *cid_or_url* and write it to *output_path*.

        Accepts three forms:

        - a bare CID (``"bafy..."`` or ``"Qm..."``) — tries every
          configured gateway with the path ``/ipfs/<cid>``;
        - a full HTTPS URL — fetched directly without gateway
          rotation;
        - a CID with a subpath (``"<cid>/file.bin"``) — tried at
          ``/ipfs/<cid>/file.bin`` on each gateway.

        Parameters
        ----------
        cid_or_url : str
            Content identifier or full URL.
        output_path : Path
            Destination file. Parent directory must already exist.

        Raises
        ------
        PTDBackendError
            If every gateway (or the single URL) fails.
        """
        if cid_or_url.startswith(("http://", "https://")):
            urls = [cid_or_url]
        else:
            urls = [f"{gw}/ipfs/{cid_or_url}" for gw in self.gateways]

        last_error: Exception | None = None
        for url in urls:
            try:
                r = request_with_retry(
                    url,
                    method="GET",
                    timeout=self.timeout,
                    session=self._session,
                    stream=True,
                )
            except requests.RequestException as e:
                logger.debug("Gateway %s failed: %s", url, e)
                last_error = e
                continue

            try:
                with output_path.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            fh.write(chunk)
            except OSError as e:
                last_error = e
                continue
            return

        raise PTDBackendError(
            f"Failed to fetch {cid_or_url} from any of "
            f"{len(urls)} gateway(s). Last error: {last_error}"
        )


__all__ = ["GatewayClient", "DEFAULT_GATEWAYS"]
