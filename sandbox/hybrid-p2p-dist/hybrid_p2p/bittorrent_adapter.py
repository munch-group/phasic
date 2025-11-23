"""BitTorrent adapter for .torrent creation and seeding."""

import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import libtorrent as lt

logger = logging.getLogger(__name__)


class BitTorrentError(Exception):
    """Raised when BitTorrent operations fail."""
    pass


class TorrentCreator:
    """Creates .torrent files from content."""
    
    def __init__(self, creator_name: str = "hybrid-p2p-dist"):
        """
        Initialize torrent creator.
        
        Args:
            creator_name: Name to embed in torrent metadata
        """
        self.creator_name = creator_name
    
    def create_torrent(
        self,
        path: Path,
        trackers: Optional[List[str]] = None,
        comment: Optional[str] = None,
        private: bool = False,
        piece_size: int = 16384,  # 16 KB default
    ) -> bytes:
        """
        Create a .torrent file from a file or directory.
        
        Args:
            path: Path to file or directory
            trackers: List of tracker URLs
            comment: Optional comment
            private: Whether torrent is private
            piece_size: Piece size in bytes (must be power of 2)
            
        Returns:
            Bencoded torrent data
            
        Raises:
            BitTorrentError: If creation fails
        """
        if not path.exists():
            raise BitTorrentError(f"Path not found: {path}")
        
        try:
            # Create file storage
            fs = lt.file_storage()
            
            if path.is_file():
                lt.add_files(fs, str(path))
            elif path.is_dir():
                lt.add_files(fs, str(path))
            else:
                raise BitTorrentError(f"Invalid path type: {path}")
            
            # Create torrent
            t = lt.create_torrent(fs, piece_size=piece_size)
            
            # Set creator
            t.set_creator(self.creator_name)
            
            # Set comment if provided
            if comment:
                t.set_comment(comment)
            
            # Set private flag
            if private:
                t.set_priv(True)
            
            # Add trackers
            if trackers:
                for tier, tracker_url in enumerate(trackers):
                    t.add_tracker(tracker_url, tier=tier)
            
            # Generate pieces
            if path.is_file():
                lt.set_piece_hashes(t, str(path.parent))
            else:
                lt.set_piece_hashes(t, str(path))
            
            # Generate torrent data
            torrent_data = lt.bencode(t.generate())
            
            logger.info(f"Created torrent for {path.name}")
            return torrent_data
            
        except Exception as e:
            raise BitTorrentError(f"Failed to create torrent: {e}") from e
    
    def save_torrent(
        self,
        path: Path,
        output_path: Path,
        trackers: Optional[List[str]] = None,
        comment: Optional[str] = None,
    ) -> Path:
        """
        Create and save a .torrent file.
        
        Args:
            path: Path to content
            output_path: Where to save .torrent file
            trackers: Tracker URLs
            comment: Optional comment
            
        Returns:
            Path to saved .torrent file
        """
        torrent_data = self.create_torrent(path, trackers=trackers, comment=comment)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(torrent_data)
        
        logger.info(f"Saved torrent to {output_path}")
        return output_path
    
    @staticmethod
    def get_info_hash(torrent_data: bytes) -> str:
        """
        Extract info hash from torrent data.
        
        Args:
            torrent_data: Bencoded torrent
            
        Returns:
            Hex-encoded info hash
        """
        try:
            info = lt.torrent_info(torrent_data)
            return str(info.info_hash())
        except Exception as e:
            raise BitTorrentError(f"Failed to extract info hash: {e}") from e
    
    @staticmethod
    def create_magnet_link(
        torrent_data: bytes,
        display_name: Optional[str] = None,
    ) -> str:
        """
        Create magnet link from torrent data.
        
        Args:
            torrent_data: Bencoded torrent
            display_name: Optional display name
            
        Returns:
            Magnet URI
        """
        try:
            info = lt.torrent_info(torrent_data)
            magnet = lt.make_magnet_uri(info)
            
            if display_name:
                # Add display name parameter
                magnet += f"&dn={display_name}"
            
            return magnet
            
        except Exception as e:
            raise BitTorrentError(f"Failed to create magnet link: {e}") from e


class TorrentSeeder:
    """Seeds torrents using libtorrent."""
    
    def __init__(
        self,
        listen_port: int = 6881,
        download_dir: Optional[Path] = None,
    ):
        """
        Initialize seeder.
        
        Args:
            listen_port: Port to listen on
            download_dir: Directory for downloads (temp if None)
        """
        self.listen_port = listen_port
        self.download_dir = download_dir or Path("/tmp/bt_downloads")
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session
        self.session = lt.session()
        self.session.listen_on(listen_port, listen_port + 10)
        
        # Active torrents
        self.handles: Dict[str, lt.torrent_handle] = {}
        
        logger.info(f"BitTorrent seeder initialized on port {listen_port}")
    
    def add_torrent(
        self,
        torrent_data: bytes,
        seed_path: Optional[Path] = None,
    ) -> str:
        """
        Add a torrent for seeding.
        
        Args:
            torrent_data: Bencoded torrent file
            seed_path: Path to existing data (for seeding)
            
        Returns:
            Info hash (hex string)
        """
        try:
            # Parse torrent
            info = lt.torrent_info(torrent_data)
            info_hash = str(info.info_hash())
            
            if info_hash in self.handles:
                logger.info(f"Torrent {info_hash} already added")
                return info_hash
            
            # Create add parameters
            params = {
                "ti": info,
                "save_path": str(seed_path.parent if seed_path else self.download_dir),
            }
            
            # If we have the data, seed it; otherwise download
            if seed_path and seed_path.exists():
                # Seeding mode
                params["flags"] = lt.torrent_flags.seed_mode
                logger.info(f"Adding torrent {info_hash} for seeding")
            else:
                logger.info(f"Adding torrent {info_hash} for download")
            
            # Add to session
            handle = self.session.add_torrent(params)
            self.handles[info_hash] = handle
            
            return info_hash
            
        except Exception as e:
            raise BitTorrentError(f"Failed to add torrent: {e}") from e
    
    def add_magnet(self, magnet_uri: str) -> str:
        """
        Add a magnet link for downloading.
        
        Args:
            magnet_uri: Magnet URI
            
        Returns:
            Info hash
        """
        try:
            params = lt.parse_magnet_uri(magnet_uri)
            params["save_path"] = str(self.download_dir)
            
            handle = self.session.add_torrent(params)
            info_hash = str(handle.info_hash())
            
            self.handles[info_hash] = handle
            
            logger.info(f"Added magnet link: {info_hash}")
            return info_hash
            
        except Exception as e:
            raise BitTorrentError(f"Failed to add magnet: {e}") from e
    
    def get_status(self, info_hash: str) -> Optional[Dict]:
        """
        Get status of a torrent.
        
        Args:
            info_hash: Info hash
            
        Returns:
            Status dict or None
        """
        handle = self.handles.get(info_hash)
        if not handle:
            return None
        
        status = handle.status()
        
        return {
            "state": str(status.state),
            "progress": status.progress,
            "download_rate": status.download_rate,
            "upload_rate": status.upload_rate,
            "num_peers": status.num_peers,
            "num_seeds": status.num_seeds,
            "is_seeding": status.is_seeding,
            "is_finished": status.is_finished,
        }
    
    def wait_for_completion(
        self,
        info_hash: str,
        timeout: int = 3600,
        check_interval: int = 5,
    ) -> bool:
        """
        Wait for a torrent to complete downloading.
        
        Args:
            info_hash: Info hash
            timeout: Max wait time in seconds
            check_interval: How often to check status
            
        Returns:
            True if completed, False if timeout
        """
        handle = self.handles.get(info_hash)
        if not handle:
            raise BitTorrentError(f"Torrent {info_hash} not found")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = handle.status()
            
            if status.is_finished:
                logger.info(f"Torrent {info_hash} completed")
                return True
            
            time.sleep(check_interval)
        
        logger.warning(f"Torrent {info_hash} timed out after {timeout}s")
        return False
    
    def remove_torrent(self, info_hash: str) -> None:
        """
        Remove a torrent from the session.
        
        Args:
            info_hash: Info hash
        """
        handle = self.handles.get(info_hash)
        if handle:
            self.session.remove_torrent(handle)
            del self.handles[info_hash]
            logger.info(f"Removed torrent {info_hash}")
    
    def shutdown(self) -> None:
        """Shutdown the seeder."""
        logger.info("Shutting down BitTorrent seeder")
        for info_hash in list(self.handles.keys()):
            self.remove_torrent(info_hash)
    
    def __enter__(self) -> "TorrentSeeder":
        """Context manager entry."""
        return self
    
    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.shutdown()
