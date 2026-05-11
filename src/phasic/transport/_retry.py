"""HTTP retry helper shared by the transport backends.

Extracted from the legacy ``trace_repository.py`` module so the kubo
RPC client, the HTTP gateway client, and the composite IPFS backend
can all share a single implementation.
"""
from __future__ import annotations

import time

import requests

from ..logging_config import get_logger

logger = get_logger(__name__)


def request_with_retry(
    url: str,
    *,
    method: str = "GET",
    timeout: float = 30.0,
    max_retries: int = 3,
    backoff_base: float = 1.0,
    session: requests.Session | None = None,
    **kwargs,
) -> requests.Response:
    """GET/POST *url* with exponential-backoff retry on transient failures.

    Retries on connection errors and 5xx status codes.

    Parameters
    ----------
    url : str
        URL to fetch.
    method : str, default 'GET'
        HTTP method. ``'GET'`` and ``'POST'`` are common.
    timeout : float, default 30
        Per-request timeout in seconds.
    max_retries : int, default 3
        Maximum number of attempts.
    backoff_base : float, default 1.0
        Base delay in seconds (doubles each retry).
    session : requests.Session, optional
        Session to reuse for connection pooling. If omitted, the
        module-level ``requests.request`` is used.
    **kwargs
        Forwarded to ``requests.request``.

    Returns
    -------
    requests.Response
        Successful response (``raise_for_status`` has been called).

    Raises
    ------
    requests.RequestException
        After all retries are exhausted.
    """
    last_exc: Exception | None = None
    request = session.request if session is not None else requests.request

    for attempt in range(max_retries):
        try:
            response = request(method, url, timeout=timeout, **kwargs)
            if response.status_code >= 500 and attempt < max_retries - 1:
                delay = backoff_base * (2 ** attempt)
                logger.debug(
                    "Server error %d from %s, retrying in %.1fs",
                    response.status_code, url, delay,
                )
                time.sleep(delay)
                continue
            response.raise_for_status()
            return response
        except requests.ConnectionError as e:
            last_exc = e
            if attempt < max_retries - 1:
                delay = backoff_base * (2 ** attempt)
                logger.debug(
                    "Connection error for %s, retrying in %.1fs", url, delay,
                )
                time.sleep(delay)
            continue
        except requests.RequestException:
            raise

    assert last_exc is not None
    raise last_exc
