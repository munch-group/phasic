"""
Comprehensive example: Publishing and fetching content.

This example demonstrates:
1. Key generation
2. Creating and validating content
3. Publishing to IPFS + BitTorrent
4. Fetching and verifying content
"""

import tempfile
from pathlib import Path

from hybrid_p2p import (
    Client,
    ClientConfig,
    generate_keypair,
    ContentManifest,
)


def main():
    """Run the complete workflow."""
    
    # Setup temporary directories
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        keys_dir = base / "keys"
        content_dir = base / "content"
        download_dir = base / "downloads"
        
        keys_dir.mkdir()
        content_dir.mkdir()
        download_dir.mkdir()
        
        print("=" * 60)
        print("Hybrid P2P Distribution Example")
        print("=" * 60)
        
        # Step 1: Generate keys
        print("\n1. Generating Ed25519 key pair...")
        private_key, public_key = generate_keypair(
            output_dir=keys_dir,
            key_name="example_key",
        )
        print(f"   Private key: {private_key}")
        print(f"   Public key: {public_key}")
        
        # Step 2: Create sample content
        print("\n2. Creating sample content...")
        files = []
        
        # Text file
        text_file = content_dir / "readme.txt"
        text_file.write_text("Hello from hybrid P2P distribution!\n" * 100)
        files.append(text_file)
        
        # JSON file
        json_file = content_dir / "metadata.json"
        json_file.write_text('{"version": "1.0", "author": "example"}\n')
        files.append(json_file)
        
        # Binary file
        binary_file = content_dir / "data.bin"
        binary_file.write_bytes(b'\x00\x01\x02\x03' * 1000)
        files.append(binary_file)
        
        print(f"   Created {len(files)} files:")
        for f in files:
            print(f"     - {f.name} ({f.stat().st_size} bytes)")
        
        # Step 3: Initialize client
        print("\n3. Initializing client...")
        config = ClientConfig(
            signing_key_path=private_key,
            ipfs_api_url="http://127.0.0.1:5001",
            ipfs_gateway_url="http://127.0.0.1:8080",
        )
        
        client = Client(config)
        
        # Check IPFS connection
        if client.ipfs.check_connection():
            print("   ✓ IPFS daemon connected")
        else:
            print("   ✗ IPFS daemon not available (will fail at publish)")
            print("     Start IPFS: ipfs daemon")
            return
        
        # Step 4: Publish content
        print("\n4. Publishing content...")
        print("   - Validating files")
        print("   - Creating manifest")
        print("   - Signing manifest")
        print("   - Adding to IPFS")
        print("   - Creating BitTorrent metadata")
        
        manifest = client.publish(
            files=files,
            name="example-package",
            version="1.0.0",
            uploader_id="alice@example.com",
            description="Example package for hybrid P2P distribution",
            create_torrent=True,
            pin_to_ipfs=True,
        )
        
        print(f"\n   ✓ Published successfully!")
        print(f"     Content ID: {manifest.content_id}")
        print(f"     IPFS CID: {manifest.distribution.ipfs_cid}")
        print(f"     Torrent Hash: {manifest.distribution.torrent_infohash}")
        
        # Save manifest
        manifest_path = base / "manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2))
        print(f"     Manifest: {manifest_path}")
        
        # Step 5: Fetch content (simulating different client)
        print("\n5. Fetching content...")
        print("   - Loading manifest")
        print("   - Verifying signature")
        print("   - Retrieving from IPFS")
        print("   - Validating checksums")
        
        fetched_dir = client.fetch(
            manifest=manifest_path,
            output_dir=download_dir,
            verify_signature=True,
            prefer_ipfs=True,
        )
        
        print(f"\n   ✓ Fetched successfully!")
        print(f"     Downloaded to: {fetched_dir}")
        
        # Step 6: Verify content
        print("\n6. Verifying downloaded content...")
        
        try:
            client.verify_local_content(manifest, fetched_dir)
            print("   ✓ All files verified successfully!")
        except Exception as e:
            print(f"   ✗ Verification failed: {e}")
        
        # Step 7: Compare files
        print("\n7. Comparing original and downloaded files...")
        all_match = True
        
        for orig_file in files:
            downloaded = fetched_dir / orig_file.name
            
            if not downloaded.exists():
                print(f"   ✗ {orig_file.name}: Not found")
                all_match = False
                continue
            
            orig_data = orig_file.read_bytes()
            down_data = downloaded.read_bytes()
            
            if orig_data == down_data:
                print(f"   ✓ {orig_file.name}: Match")
            else:
                print(f"   ✗ {orig_file.name}: Mismatch!")
                all_match = False
        
        if all_match:
            print("\n" + "=" * 60)
            print("SUCCESS: All operations completed successfully!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("FAILURE: Some operations failed")
            print("=" * 60)
        
        # Cleanup
        client.close()


if __name__ == "__main__":
    # Note: This requires a running IPFS daemon
    # Start with: ipfs daemon
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
