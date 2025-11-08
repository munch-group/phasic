"""IPFS adapter for content-addressed storage and retrieval."""

import io
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class IPFSError(Exception):
    """Raised when IPFS operations fail."""
    pass


class IPFSAdapter:
    """
    Adapter for IPFS HTTP API.
    
    Supports both local go-ipfs daemon and remote gateways.
    """
    
    def __init__(
        self,
        api_url: str = "http://127.0.0.1:5001",
        gateway_url: str = "http://127.0.0.1:8080",
        timeout: int = 300,
    ):
        """
        Initialize IPFS adapter.
        
        Args:
            api_url: IPFS API endpoint (for adding/pinning)
            gateway_url: IPFS gateway endpoint (for retrieval)
            timeout: Request timeout in seconds
        """
        self.api_url = api_url.rstrip("/")
        self.gateway_url = gateway_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
    
    def check_connection(self) -> bool:
        """
        Check if IPFS daemon is reachable.
        
        Returns:
            True if connected
        """
        try:
            response = self._session.post(
                f"{self.api_url}/api/v0/version",
                timeout=5,
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"IPFS connection check failed: {e}")
            return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def add_file(
        self,
        path: Path,
        pin: bool = True,
        wrap_with_directory: bool = False,
    ) -> Dict[str, str]:
        """
        Add a file to IPFS.
        
        Args:
            path: Path to file
            pin: Whether to pin the file
            wrap_with_directory: Wrap file in a directory
            
        Returns:
            Dict with 'Hash' (CID), 'Name', and 'Size'
            
        Raises:
            IPFSError: If operation fails
        """
        if not path.exists():
            raise IPFSError(f"File not found: {path}")
        
        params = {
            "pin": "true" if pin else "false",
            "wrap-with-directory": "true" if wrap_with_directory else "false",
        }
        
        try:
            with open(path, "rb") as f:
                files = {"file": (path.name, f)}
                response = self._session.post(
                    f"{self.api_url}/api/v0/add",
                    params=params,
                    files=files,
                    timeout=self.timeout,
                )
            
            response.raise_for_status()
            
            # Response is newline-delimited JSON
            lines = response.text.strip().split("\n")
            results = [json.loads(line) for line in lines]
            
            # Return the last result (root if wrapped, file otherwise)
            result = results[-1]
            
            logger.info(f"Added {path.name} to IPFS: {result['Hash']}")
            return result
            
        except Exception as e:
            raise IPFSError(f"Failed to add file to IPFS: {e}") from e
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def add_directory(
        self,
        path: Path,
        pin: bool = True,
        recursive: bool = True,
    ) -> Dict[str, str]:
        """
        Add a directory to IPFS.
        
        Args:
            path: Path to directory
            pin: Whether to pin the content
            recursive: Recursively add subdirectories
            
        Returns:
            Dict with root directory CID
            
        Raises:
            IPFSError: If operation fails
        """
        if not path.is_dir():
            raise IPFSError(f"Not a directory: {path}")
        
        params = {
            "pin": "true" if pin else "false",
            "recursive": "true" if recursive else "false",
        }
        
        try:
            # Collect all files
            files_to_add = []
            for file_path in path.rglob("*") if recursive else path.glob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(path)
                    files_to_add.append((str(relative_path), open(file_path, "rb")))
            
            if not files_to_add:
                raise IPFSError(f"No files found in directory: {path}")
            
            files = [("file", (name, fh)) for name, fh in files_to_add]
            
            response = self._session.post(
                f"{self.api_url}/api/v0/add",
                params=params,
                files=files,
                timeout=self.timeout,
            )
            
            # Close file handles
            for _, fh in files_to_add:
                fh.close()
            
            response.raise_for_status()
            
            # Parse response (last line is root directory)
            lines = response.text.strip().split("\n")
            results = [json.loads(line) for line in lines]
            root_result = results[-1]
            
            logger.info(f"Added directory {path.name} to IPFS: {root_result['Hash']}")
            return root_result
            
        except Exception as e:
            raise IPFSError(f"Failed to add directory to IPFS: {e}") from e
    
    def add_bytes(
        self,
        data: bytes,
        filename: str = "data",
        pin: bool = True,
    ) -> Dict[str, str]:
        """
        Add raw bytes to IPFS.
        
        Args:
            data: Bytes to add
            filename: Name for the data
            pin: Whether to pin
            
        Returns:
            Dict with CID
            
        Raises:
            IPFSError: If operation fails
        """
        params = {"pin": "true" if pin else "false"}
        
        try:
            files = {"file": (filename, io.BytesIO(data))}
            response = self._session.post(
                f"{self.api_url}/api/v0/add",
                params=params,
                files=files,
                timeout=self.timeout,
            )
            
            response.raise_for_status()
            result = json.loads(response.text)
            
            logger.info(f"Added {len(data)} bytes to IPFS: {result['Hash']}")
            return result
            
        except Exception as e:
            raise IPFSError(f"Failed to add bytes to IPFS: {e}") from e
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def get_file(
        self,
        cid: str,
        output_path: Optional[Path] = None,
    ) -> Union[bytes, Path]:
        """
        Retrieve a file from IPFS.
        
        Args:
            cid: Content identifier (CID)
            output_path: Optional path to save file
            
        Returns:
            File bytes if no output_path, else Path to saved file
            
        Raises:
            IPFSError: If retrieval fails
        """
        try:
            # Try API first, fall back to gateway
            try:
                response = self._session.post(
                    f"{self.api_url}/api/v0/cat",
                    params={"arg": cid},
                    timeout=self.timeout,
                    stream=True,
                )
                response.raise_for_status()
            except Exception:
                logger.debug("API cat failed, trying gateway")
                response = self._session.get(
                    f"{self.gateway_url}/ipfs/{cid}",
                    timeout=self.timeout,
                    stream=True,
                )
                response.raise_for_status()
            
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Retrieved {cid} to {output_path}")
                return output_path
            else:
                data = response.content
                logger.info(f"Retrieved {cid} ({len(data)} bytes)")
                return data
                
        except Exception as e:
            raise IPFSError(f"Failed to retrieve {cid} from IPFS: {e}") from e
    
    def pin_add(self, cid: str) -> bool:
        """
        Pin content by CID.
        
        Args:
            cid: Content identifier
            
        Returns:
            True if pinned successfully
            
        Raises:
            IPFSError: If pinning fails
        """
        try:
            response = self._session.post(
                f"{self.api_url}/api/v0/pin/add",
                params={"arg": cid},
                timeout=self.timeout,
            )
            response.raise_for_status()
            
            logger.info(f"Pinned {cid}")
            return True
            
        except Exception as e:
            raise IPFSError(f"Failed to pin {cid}: {e}") from e
    
    def pin_rm(self, cid: str) -> bool:
        """
        Unpin content by CID.
        
        Args:
            cid: Content identifier
            
        Returns:
            True if unpinned successfully
        """
        try:
            response = self._session.post(
                f"{self.api_url}/api/v0/pin/rm",
                params={"arg": cid},
                timeout=self.timeout,
            )
            response.raise_for_status()
            
            logger.info(f"Unpinned {cid}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to unpin {cid}: {e}")
            return False
    
    def pin_ls(self, cid: Optional[str] = None) -> List[str]:
        """
        List pinned content.
        
        Args:
            cid: Optional specific CID to check
            
        Returns:
            List of pinned CIDs
        """
        try:
            params = {"arg": cid} if cid else {}
            response = self._session.post(
                f"{self.api_url}/api/v0/pin/ls",
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            
            data = response.json()
            pins = list(data.get("Keys", {}).keys())
            return pins
            
        except Exception as e:
            logger.warning(f"Failed to list pins: {e}")
            return []
    
    def get_stats(self, cid: str) -> Dict[str, any]:
        """
        Get object statistics.
        
        Args:
            cid: Content identifier
            
        Returns:
            Dict with stats (size, etc.)
        """
        try:
            response = self._session.post(
                f"{self.api_url}/api/v0/object/stat",
                params={"arg": cid},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.warning(f"Failed to get stats for {cid}: {e}")
            return {}
    
    def close(self) -> None:
        """Close the session."""
        self._session.close()
    
    def __enter__(self) -> "IPFSAdapter":
        """Context manager entry."""
        return self
    
    def __exit__(self, *args: any) -> None:
        """Context manager exit."""
        self.close()
