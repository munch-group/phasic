# Hybrid P2P Distribution System

A Python library for hybrid content distribution combining GitHub (canonical storage), IPFS (content-addressed P2P), and BitTorrent (high-throughput distribution).

## Features

- **Hybrid Architecture**: Central GitHub repository + decentralized P2P networks
- **Content Addressing**: IPFS CIDs and SHA-256 hashes for integrity
- **Cryptographic Signing**: Ed25519 signatures for manifests and files
- **Format Validation**: Pydantic schemas with strict type checking
- **BitTorrent Compatibility**: .torrent generation and magnet links
- **Pinning Support**: Dedicated pinning nodes for high availability
- **CLI Interface**: Rich terminal interface for all operations

## Architecture

```
┌─────────────┐
│   GitHub    │ ← Canonical manifests & metadata
│  Repository │
└──────┬──────┘
       │
       ├─────────────┐
       │             │
┌──────▼──────┐ ┌───▼────────┐
│    IPFS     │ │ BitTorrent │ ← P2P distribution
│   Network   │ │   Network  │
└──────┬──────┘ └───┬────────┘
       │             │
       └──────┬──────┘
              │
       ┌──────▼──────┐
       │   Clients   │ ← Validate, fetch, verify
       └─────────────┘
```

## Installation

### From source

```bash
git clone https://github.com/your-org/hybrid-p2p-dist.git
cd hybrid-p2p-dist
pip install -e ".[dev]"
```

### Prerequisites

1. **IPFS daemon** (required for IPFS operations):
   ```bash
   # Install go-ipfs
   wget https://dist.ipfs.io/go-ipfs/v0.24.0/go-ipfs_v0.24.0_linux-amd64.tar.gz
   tar -xvzf go-ipfs_v0.24.0_linux-amd64.tar.gz
   cd go-ipfs
   sudo bash install.sh
   
   # Initialize and start
   ipfs init
   ipfs daemon
   ```

2. **libtorrent** (required for BitTorrent):
   ```bash
   # Ubuntu/Debian
   sudo apt-get install python3-libtorrent
   
   # macOS
   brew install libtorrent-rasterbar
   ```

## Quick Start

### 1. Generate signing keys

```bash
hybrid-p2p keygen -o ~/.hybrid_p2p/keys -n mykey
```

### 2. Publish content

```bash
hybrid-p2p publish file1.txt file2.json \
  --name mypackage \
  --version 1.0.0 \
  --uploader-id alice@example.com \
  --key ~/.hybrid_p2p/keys/mykey.pem \
  --output manifest.json
```

### 3. Fetch content

```bash
hybrid-p2p fetch manifest.json -o ./downloads
```

### 4. Verify content

```bash
hybrid-p2p verify manifest.json ./downloads
```

## Python API

### Basic Usage

```python
from pathlib import Path
from hybrid_p2p import Client, ClientConfig

# Initialize client
config = ClientConfig(
    signing_key_path=Path("~/.hybrid_p2p/keys/mykey.pem"),
)

with Client(config) as client:
    # Publish
    manifest = client.publish(
        files=[Path("data.txt"), Path("metadata.json")],
        name="my-dataset",
        version="1.0.0",
        uploader_id="alice@example.com",
        description="Example dataset",
    )
    
    print(f"Published: {manifest.distribution.ipfs_cid}")
    
    # Fetch
    output_dir = client.fetch(
        manifest=manifest,
        output_dir=Path("./downloads"),
        verify_signature=True,
    )
    
    print(f"Downloaded to: {output_dir}")
```

### Advanced: Key Management

```python
from hybrid_p2p import KeyManager, Signer, Verifier

# Generate keys
km = KeyManager()
km.save_keys(
    private_path=Path("private.pem"),
    public_path=Path("public.pem"),
    password=b"secure-password",  # Optional
)

# Sign manifest
signer = Signer(km)
signed_manifest = signer.sign_manifest(manifest)

# Verify manifest
is_valid = Verifier.verify_manifest(signed_manifest)
```

### Advanced: IPFS Operations

```python
from hybrid_p2p import IPFSAdapter

with IPFSAdapter() as ipfs:
    # Add file
    result = ipfs.add_file(Path("data.txt"))
    cid = result["Hash"]
    
    # Retrieve file
    data = ipfs.get_file(cid)
    
    # Pin content
    ipfs.pin_add(cid)
```

### Advanced: BitTorrent

```python
from hybrid_p2p import TorrentCreator, TorrentSeeder

# Create torrent
creator = TorrentCreator()
torrent_data = creator.create_torrent(
    path=Path("data.txt"),
    trackers=["udp://tracker.example.com:6969/announce"],
)

# Get magnet link
magnet = creator.create_magnet_link(torrent_data, "My Data")

# Seed torrent
with TorrentSeeder() as seeder:
    info_hash = seeder.add_torrent(
        torrent_data,
        seed_path=Path("data.txt"),
    )
    # Keep seeding...
```

## Deployment

### Using Docker Compose

Start IPFS, IPFS Cluster, and monitoring:

```bash
docker-compose up -d
```

Services:
- IPFS API: http://localhost:5001
- IPFS Gateway: http://localhost:8080
- IPFS Cluster: http://localhost:9094
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

### Production Setup

1. **Set up pinning nodes**:
   ```bash
   # On dedicated server
   ipfs init --profile server
   ipfs daemon
   ```

2. **Configure IPFS Cluster** for distributed pinning:
   ```bash
   ipfs-cluster-service init
   ipfs-cluster-service daemon
   ```

3. **Set up monitoring**:
   - Prometheus scrapes IPFS and cluster metrics
   - Grafana dashboards for visualization
   - Alerts for pin failures

4. **GitHub Actions integration**:
   - Automatic manifest validation on PR
   - Pin content to cluster on merge
   - Update release assets

## Manifest Schema

```json
{
  "manifest_version": "1.0",
  "content_id": "abc123...",
  "version": "1.0.0",
  "name": "my-package",
  "description": "Package description",
  "files": [
    {
      "path": "file.txt",
      "size": 1024,
      "sha256": "abc123...",
      "cid": "Qm...",
      "mime_type": "text/plain"
    }
  ],
  "total_size": 1024,
  "signature": {
    "algorithm": "ed25519",
    "public_key": "base64...",
    "signature": "base64...",
    "signed_at": "2025-01-01T00:00:00Z"
  },
  "provenance": {
    "uploader_id": "alice@example.com",
    "timestamp": "2025-01-01T00:00:00Z",
    "git_commit": "abc123...",
    "git_ref": "main"
  },
  "distribution": {
    "ipfs_cid": "Qm...",
    "ipfs_gateway": "https://ipfs.io",
    "torrent_infohash": "abc123...",
    "torrent_magnet": "magnet:?xt=...",
    "pinning_nodes": ["node1", "node2"]
  }
}
```

## Security Model

1. **Content Integrity**: SHA-256 hashes + IPFS CIDs (content-addressed)
2. **Authenticity**: Ed25519 signatures on manifests
3. **Availability**: Dedicated pinning nodes + BitTorrent seeders
4. **Privacy**: Optional end-to-end encryption for sensitive content
5. **Validation**: Strict schema enforcement before publish/fetch

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=hybrid_p2p --cov-report=html

# Run specific test file
pytest tests/test_validation.py -v
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Format code
black hybrid_p2p tests

# Lint
ruff check hybrid_p2p tests

# Type check
mypy hybrid_p2p
```

## References

1. **Benet, J. (2014)**. IPFS - Content Addressed, Versioned, P2P File System. arXiv.
2. **Legout, A., et al. (2005)**. Understanding BitTorrent: An Experimental Perspective.
3. **Merkle, R. (1979)**. A Certified Digital Signature.
4. **libp2p Project**. Protocol and routing foundations.

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

- Issues: https://github.com/your-org/hybrid-p2p-dist/issues
- Documentation: https://hybrid-p2p-dist.readthedocs.io
- Discussions: https://github.com/your-org/hybrid-p2p-dist/discussions
