"""Main client interface for hybrid P2P distribution."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from .bittorrent_adapter import BitTorrentError, TorrentCreator, TorrentSeeder
from .ipfs_adapter import IPFSAdapter, IPFSError
from .signing import KeyManager, Signer, Verifier, VerificationError
from .validation import (
    ContentManifest,
    ContentValidator,
    DistributionMetadata,
    create_manifest_from_files,
)

logger = logging.getLogger(__name__)


class ClientConfig:
    """Configuration for hybrid P2P client."""
    
    def __init__(
        self,
        # IPFS settings
        ipfs_api_url: str = "http://127.0.0.1:5001",
        ipfs_gateway_url: str = "http://127.0.0.1:8080",
        
        # BitTorrent settings
        bt_trackers: Optional[List[str]] = None,
        bt_listen_port: int = 6881,
        
        # Keys
        signing_key_path: Optional[Path] = None,
        signing_key_password: Optional[bytes] = None,
        
        # Validation
        strict_mime_types: bool = False,
        
        # Storage
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize client configuration.
        
        Args:
            ipfs_api_url: IPFS API endpoint
            ipfs_gateway_url: IPFS gateway for retrieval
            bt_trackers: BitTorrent tracker URLs
            bt_listen_port: Port for BitTorrent
            signing_key_path: Path to Ed25519 private key
            signing_key_password: Password for private key
            strict_mime_types: Enforce strict MIME type validation
            cache_dir: Local cache directory
        """
        self.ipfs_api_url = ipfs_api_url
        self.ipfs_gateway_url = ipfs_gateway_url
        self.bt_trackers = bt_trackers or [
            "udp://tracker.opentrackr.org:1337/announce",
            "udp://open.tracker.cl:1337/announce",
        ]
        self.bt_listen_port = bt_listen_port
        self.signing_key_path = signing_key_path
        self.signing_key_password = signing_key_password
        self.strict_mime_types = strict_mime_types
        self.cache_dir = cache_dir or Path.home() / ".hybrid_p2p" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)


class Client:
    """
    High-level client for hybrid P2P content distribution.
    
    Provides publish and fetch operations that handle validation,
    signing, and distribution across IPFS and BitTorrent.
    """
    
    def __init__(self, config: ClientConfig):
        """
        Initialize client.
        
        Args:
            config: Client configuration
        """
        self.config = config
        
        # Initialize components
        self.validator = ContentValidator(strict_mime=config.strict_mime_types)
        self.ipfs = IPFSAdapter(
            api_url=config.ipfs_api_url,
            gateway_url=config.ipfs_gateway_url,
        )
        
        # Initialize key manager if key provided
        self.key_manager: Optional[KeyManager] = None
        self.signer: Optional[Signer] = None
        
        if config.signing_key_path:
            self.key_manager = KeyManager.from_private_key_file(
                config.signing_key_path,
                password=config.signing_key_password,
            )
            self.signer = Signer(self.key_manager)
        
        # Torrent creator
        self.torrent_creator = TorrentCreator()
        
        logger.info("Hybrid P2P client initialized")
    
    def publish(
        self,
        files: List[Path],
        name: str,
        version: str,
        uploader_id: str,
        description: Optional[str] = None,
        create_torrent: bool = True,
        pin_to_ipfs: bool = True,
    ) -> ContentManifest:
        """
        Publish content to P2P networks.
        
        This validates files, creates a manifest, signs it, adds content
        to IPFS, optionally creates a torrent, and returns the manifest.
        
        Args:
            files: List of file paths to publish
            name: Package name
            version: Semantic version
            uploader_id: Uploader identifier
            description: Optional description
            create_torrent: Whether to create BitTorrent metadata
            pin_to_ipfs: Whether to pin to IPFS
            
        Returns:
            Signed manifest with distribution metadata
            
        Raises:
            ValueError: If validation fails
            IPFSError: If IPFS operations fail
            BitTorrentError: If torrent creation fails
        """
        logger.info(f"Publishing {name} v{version} with {len(files)} files")
        
        # Validate all files first
        for file_path in files:
            if not file_path.exists():
                raise ValueError(f"File not found: {file_path}")
        
        # Create initial manifest
        manifest = create_manifest_from_files(
            files=files,
            name=name,
            version=version,
            uploader_id=uploader_id,
            description=description,
        )
        
        # Add to IPFS
        ipfs_cids = {}
        
        if len(files) == 1:
            # Single file
            result = self.ipfs.add_file(files[0], pin=pin_to_ipfs)
            ipfs_cids[files[0].name] = result["Hash"]
            root_cid = result["Hash"]
        else:
            # Multiple files - create temp directory structure
            import tempfile
            import shutil
            
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir) / name
                tmp_path.mkdir()
                
                for file_path in files:
                    shutil.copy(file_path, tmp_path / file_path.name)
                
                result = self.ipfs.add_directory(tmp_path, pin=pin_to_ipfs)
                root_cid = result["Hash"]
        
        logger.info(f"Added to IPFS: {root_cid}")
        
        # Update file entries with CIDs
        for file_entry in manifest.files:
            if file_entry.path in ipfs_cids:
                file_entry.cid = ipfs_cids[file_entry.path]
        
        # Create distribution metadata
        distribution = DistributionMetadata(
            ipfs_cid=root_cid,
            ipfs_gateway=self.config.ipfs_gateway_url,
        )
        
        # Create torrent if requested
        if create_torrent:
            try:
                # Create torrent from first file (or directory)
                torrent_path = files[0] if len(files) == 1 else files[0].parent
                torrent_data = self.torrent_creator.create_torrent(
                    torrent_path,
                    trackers=self.config.bt_trackers,
                    comment=f"{name} v{version}",
                )
                
                # Extract info hash
                info_hash = TorrentCreator.get_info_hash(torrent_data)
                magnet = TorrentCreator.create_magnet_link(torrent_data, name)
                
                distribution.torrent_infohash = info_hash
                distribution.torrent_magnet = magnet
                
                # Save torrent file
                torrent_file = self.config.cache_dir / f"{name}-{version}.torrent"
                torrent_file.write_bytes(torrent_data)
                
                logger.info(f"Created torrent: {info_hash}")
                
            except Exception as e:
                logger.warning(f"Failed to create torrent: {e}")
        
        # Add distribution metadata to manifest
        manifest.distribution = distribution
        
        # Sign manifest
        if self.signer:
            manifest = self.signer.sign_manifest(manifest)
            logger.info("Manifest signed")
        else:
            logger.warning("No signer configured - manifest not signed")
        
        # Save manifest locally
        manifest_path = self.config.cache_dir / f"{name}-{version}-manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2))
        logger.info(f"Saved manifest to {manifest_path}")
        
        return manifest
    
    def fetch(
        self,
        manifest: Union[ContentManifest, Path, str],
        output_dir: Optional[Path] = None,
        verify_signature: bool = True,
        prefer_ipfs: bool = True,
    ) -> Path:
        """
        Fetch content from P2P networks.
        
        This retrieves content using the manifest, verifies signatures
        and hashes, and saves files to output directory.
        
        Args:
            manifest: Manifest object, path to manifest file, or JSON string
            output_dir: Where to save files (cache if None)
            verify_signature: Whether to verify manifest signature
            prefer_ipfs: Try IPFS before BitTorrent
            
        Returns:
            Path to output directory
            
        Raises:
            VerificationError: If signature verification fails
            ValueError: If content validation fails
        """
        # Parse manifest if needed
        if isinstance(manifest, (Path, str)):
            if isinstance(manifest, Path):
                manifest_json = manifest.read_text()
            else:
                manifest_json = manifest
            
            manifest = ContentManifest.model_validate_json(manifest_json)
        
        logger.info(f"Fetching {manifest.name} v{manifest.version}")
        
        # Verify signature
        if verify_signature:
            try:
                Verifier.verify_manifest(manifest)
                logger.info("Manifest signature verified")
            except VerificationError as e:
                raise VerificationError(f"Manifest signature invalid: {e}") from e
        
        # Determine output directory
        if output_dir is None:
            output_dir = self.config.cache_dir / f"{manifest.name}-{manifest.version}"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Fetch via IPFS
        if prefer_ipfs and manifest.distribution and manifest.distribution.ipfs_cid:
            try:
                self._fetch_from_ipfs(manifest, output_dir)
                logger.info("Successfully fetched from IPFS")
                return output_dir
            except IPFSError as e:
                logger.warning(f"IPFS fetch failed: {e}")
        
        # Fetch via BitTorrent
        if manifest.distribution and manifest.distribution.torrent_magnet:
            try:
                self._fetch_from_bittorrent(manifest, output_dir)
                logger.info("Successfully fetched from BitTorrent")
                return output_dir
            except BitTorrentError as e:
                logger.warning(f"BitTorrent fetch failed: {e}")
        
        raise ValueError("Failed to fetch content from any source")
    
    def _fetch_from_ipfs(
        self,
        manifest: ContentManifest,
        output_dir: Path,
    ) -> None:
        """Fetch content from IPFS and verify."""
        if not manifest.distribution or not manifest.distribution.ipfs_cid:
            raise IPFSError("No IPFS CID in manifest")
        
        cid = manifest.distribution.ipfs_cid
        
        # Fetch each file
        for file_entry in manifest.files:
            output_path = output_dir / file_entry.path
            
            if file_entry.cid:
                # Fetch by file CID
                self.ipfs.get_file(file_entry.cid, output_path)
            else:
                # Fetch from root CID
                self.ipfs.get_file(f"{cid}/{file_entry.path}", output_path)
            
            # Verify hash
            self.validator.validate_file_against_entry(output_path, file_entry)
    
    def _fetch_from_bittorrent(
        self,
        manifest: ContentManifest,
        output_dir: Path,
    ) -> None:
        """Fetch content from BitTorrent and verify."""
        if not manifest.distribution or not manifest.distribution.torrent_magnet:
            raise BitTorrentError("No magnet link in manifest")
        
        with TorrentSeeder(
            listen_port=self.config.bt_listen_port,
            download_dir=output_dir,
        ) as seeder:
            info_hash = seeder.add_magnet(manifest.distribution.torrent_magnet)
            
            # Wait for completion
            if not seeder.wait_for_completion(info_hash, timeout=3600):
                raise BitTorrentError("Download timeout")
            
            # Verify files
            for file_entry in manifest.files:
                file_path = output_dir / file_entry.path
                self.validator.validate_file_against_entry(file_path, file_entry)
    
    def verify_local_content(
        self,
        manifest: Union[ContentManifest, Path],
        content_dir: Path,
    ) -> bool:
        """
        Verify local content against manifest.
        
        Args:
            manifest: Manifest object or path
            content_dir: Directory containing files
            
        Returns:
            True if all files match manifest
            
        Raises:
            ValueError: If verification fails
        """
        if isinstance(manifest, Path):
            manifest = ContentManifest.model_validate_json(manifest.read_text())
        
        for file_entry in manifest.files:
            file_path = content_dir / file_entry.path
            self.validator.validate_file_against_entry(file_path, file_entry)
        
        return True
    
    def close(self) -> None:
        """Clean up resources."""
        self.ipfs.close()
    
    def __enter__(self) -> "Client":
        """Context manager entry."""
        return self
    
    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.close()
