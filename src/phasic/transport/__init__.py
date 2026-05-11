"""Transport backends for the compute-graph sharing system.

This package provides pluggable transport backends for fetching and
publishing compute-graph artifacts (``.bin`` files produced by
``ptd_save_parameterized_reward_compute_graph``).

The default backend is :class:`IPFSBackend`, which uses a kubo daemon
when available and falls back to public HTTP gateways otherwise. The
:class:`TransportBackend` abstract base class lets third-party code
plug in alternative stores (S3, GCS, Azure Blob, etc.).

See Also
--------
phasic.compute_repository : the registry-level API that consumes
    these backends.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class TransportBackend(ABC):
    """Abstract base class for compute-artifact transport backends.

    Implementations provide content-addressed storage for compute
    artifact files. The default implementation is
    :class:`phasic.transport.ipfs.IPFSBackend`; future backends may
    target S3, GCS, or Azure Blob Storage.

    Subclasses must implement :meth:`get`, :meth:`add`, and the
    :attr:`name` property. Optionally override :meth:`pin`,
    :meth:`unpin`, :meth:`is_pinned`, and :meth:`status`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name (e.g. ``'ipfs'``, ``'s3'``)."""

    @abstractmethod
    def get(self, cid_or_url: str, output_path: Path) -> None:
        """Retrieve content by identifier and write to *output_path*.

        Parameters
        ----------
        cid_or_url : str
            Content identifier — interpretation is backend-specific.
            For IPFS this is a CID; for HTTPS it is a URL; for S3 it
            is an ``s3://bucket/key`` URI.
        output_path : Path
            Destination file. The backend writes content here on
            success and does not create the parent directory.
        """

    @abstractmethod
    def add(self, path: Path) -> str:
        """Publish a file and return its identifier.

        Parameters
        ----------
        path : Path
            Local file to publish.

        Returns
        -------
        str
            Content identifier the backend assigned to this content.
        """

    def pin(self, cid: str) -> None:
        """Pin *cid* on the backing store. Default raises."""
        raise NotImplementedError(
            f"{self.name} backend does not support pinning")

    def unpin(self, cid: str) -> None:
        """Unpin *cid* from the backing store. Default raises."""
        raise NotImplementedError(
            f"{self.name} backend does not support unpinning")

    def is_pinned(self, cid: str) -> bool:
        """Check whether *cid* is pinned. Default returns ``False``."""
        return False

    def status(self) -> dict[str, Any]:
        """Return a backend-specific status summary."""
        return {"name": self.name}


def _ipfs_backend():
    from .ipfs import IPFSBackend
    return IPFSBackend


__all__ = ["TransportBackend", "_ipfs_backend"]
