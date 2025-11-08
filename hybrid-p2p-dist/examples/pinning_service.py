"""
Example: Automated pinning service integration.

This demonstrates how to set up automated pinning to a remote
IPFS pinning service (like Pinata, Web3.Storage, or self-hosted).
"""

import os
from pathlib import Path
from typing import Optional

import requests
from hybrid_p2p import Client, ClientConfig, ContentManifest


class PinningService:
    """Integration with remote IPFS pinning services."""
    
    def __init__(self, api_key: str, service_url: str):
        """
        Initialize pinning service client.
        
        Args:
            api_key: API key for authentication
            service_url: Base URL of pinning service
        """
        self.api_key = api_key
        self.service_url = service_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
        })
    
    def pin_by_cid(self, cid: str, name: Optional[str] = None) -> dict:
        """
        Pin content by CID.
        
        Args:
            cid: IPFS CID to pin
            name: Optional name for the pin
            
        Returns:
            Pin status information
        """
        data = {
            "cid": cid,
            "name": name or cid,
        }
        
        response = self.session.post(
            f"{self.service_url}/api/v1/pins",
            json=data,
        )
        response.raise_for_status()
        
        return response.json()
    
    def get_pin_status(self, request_id: str) -> dict:
        """
        Get status of a pin request.
        
        Args:
            request_id: Pin request ID
            
        Returns:
            Status information
        """
        response = self.session.get(
            f"{self.service_url}/api/v1/pins/{request_id}",
        )
        response.raise_for_status()
        
        return response.json()
    
    def list_pins(self) -> list:
        """
        List all pins.
        
        Returns:
            List of pin objects
        """
        response = self.session.get(
            f"{self.service_url}/api/v1/pins",
        )
        response.raise_for_status()
        
        return response.json()["results"]


def publish_and_pin(
    files: list[Path],
    name: str,
    version: str,
    pinning_service: PinningService,
    uploader_id: str = "automated-publisher",
) -> ContentManifest:
    """
    Publish content and pin to remote service.
    
    Args:
        files: Files to publish
        name: Package name
        version: Version
        pinning_service: Pinning service instance
        uploader_id: Uploader identifier
        
    Returns:
        Published manifest
    """
    # Initialize client
    config = ClientConfig(
        signing_key_path=Path(os.getenv("SIGNING_KEY_PATH", "signing_key.pem")),
    )
    
    with Client(config) as client:
        # Publish to local IPFS
        print(f"Publishing {name} v{version}...")
        manifest = client.publish(
            files=files,
            name=name,
            version=version,
            uploader_id=uploader_id,
            create_torrent=True,
            pin_to_ipfs=True,  # Pin locally first
        )
        
        # Pin to remote service
        if manifest.distribution and manifest.distribution.ipfs_cid:
            cid = manifest.distribution.ipfs_cid
            
            print(f"Pinning {cid} to remote service...")
            pin_result = pinning_service.pin_by_cid(
                cid,
                name=f"{name}-{version}",
            )
            
            print(f"Pin request created: {pin_result}")
            
            # Update manifest with pinning info
            if not manifest.distribution.pinning_nodes:
                manifest.distribution.pinning_nodes = []
            
            manifest.distribution.pinning_nodes.append(
                pinning_service.service_url
            )
        
        return manifest


def setup_automated_pinning():
    """
    Set up automated pinning for CI/CD pipeline.
    
    This can be called from GitHub Actions or other CI systems.
    """
    # Get credentials from environment
    pinning_api_key = os.getenv("PINNING_API_KEY")
    pinning_service_url = os.getenv("PINNING_SERVICE_URL", "https://api.pinata.cloud")
    
    if not pinning_api_key:
        raise ValueError("PINNING_API_KEY environment variable required")
    
    # Initialize pinning service
    pinning = PinningService(
        api_key=pinning_api_key,
        service_url=pinning_service_url,
    )
    
    # Example: Pin content from a manifest file
    manifest_path = Path("manifest.json")
    
    if manifest_path.exists():
        manifest = ContentManifest.model_validate_json(manifest_path.read_text())
        
        if manifest.distribution and manifest.distribution.ipfs_cid:
            print(f"Pinning {manifest.name} v{manifest.version}")
            
            result = pinning.pin_by_cid(
                manifest.distribution.ipfs_cid,
                name=f"{manifest.name}-{manifest.version}",
            )
            
            print(f"✓ Pinned: {result}")
        else:
            print("No IPFS CID in manifest")
    else:
        print("No manifest found")


if __name__ == "__main__":
    # Example usage
    
    # For local testing:
    # 1. Set environment variables:
    #    export PINNING_API_KEY="your-api-key"
    #    export SIGNING_KEY_PATH="path/to/key.pem"
    
    # 2. Create test files
    test_files = [
        Path("test1.txt"),
        Path("test2.json"),
    ]
    
    for f in test_files:
        if not f.exists():
            f.write_text(f"Test content for {f.name}")
    
    # 3. Pin using service (requires valid API key)
    try:
        pinning_api_key = os.getenv("PINNING_API_KEY")
        
        if pinning_api_key:
            service = PinningService(
                api_key=pinning_api_key,
                service_url=os.getenv(
                    "PINNING_SERVICE_URL",
                    "https://api.pinata.cloud"
                ),
            )
            
            manifest = publish_and_pin(
                files=test_files,
                name="test-package",
                version="1.0.0",
                pinning_service=service,
            )
            
            print(f"\n✓ Published and pinned: {manifest.content_id}")
        else:
            print("Set PINNING_API_KEY to test remote pinning")
            
    except Exception as e:
        print(f"Error: {e}")
