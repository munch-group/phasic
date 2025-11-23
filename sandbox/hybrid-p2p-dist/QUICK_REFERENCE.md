# Quick Reference Card

## Installation

```bash
./quickstart.sh              # Automated setup
# OR
pip install -e ".[dev]"      # Manual install
ipfs daemon &                 # Start IPFS
```

## CLI Commands

### Generate Keys
```bash
hybrid-p2p keygen -o ~/.hybrid_p2p/keys -n mykey
hybrid-p2p keygen -o keys -n mykey --password  # Encrypted
```

### Publish Content
```bash
hybrid-p2p publish file1.txt file2.json \
  --name package-name \
  --version 1.0.0 \
  --uploader-id user@example.com \
  --key ~/.hybrid_p2p/keys/mykey.pem \
  --output manifest.json
```

### Fetch Content
```bash
hybrid-p2p fetch manifest.json -o ./downloads
hybrid-p2p fetch manifest.json --no-verify      # Skip signature check
hybrid-p2p fetch manifest.json --prefer-bt      # Use BitTorrent first
```

### Verify Content
```bash
hybrid-p2p verify manifest.json ./downloads
```

### Inspect Manifest
```bash
hybrid-p2p inspect manifest.json
```

## Python API

### Basic Publish/Fetch
```python
from pathlib import Path
from hybrid_p2p import Client, ClientConfig

config = ClientConfig(signing_key_path=Path("key.pem"))
with Client(config) as client:
    # Publish
    manifest = client.publish(
        files=[Path("data.txt")],
        name="dataset",
        version="1.0.0",
        uploader_id="alice",
    )
    
    # Fetch
    client.fetch(manifest, output_dir=Path("downloads"))
```

### Key Management
```python
from hybrid_p2p import generate_keypair, KeyManager

# Generate
private, public = generate_keypair(Path("keys"), "mykey")

# Load
km = KeyManager.from_private_key_file(private, password=b"pass")
```

### IPFS Operations
```python
from hybrid_p2p import IPFSAdapter

with IPFSAdapter() as ipfs:
    # Add file
    result = ipfs.add_file(Path("data.txt"))
    cid = result["Hash"]
    
    # Retrieve
    data = ipfs.get_file(cid)
    
    # Pin
    ipfs.pin_add(cid)
```

### Signing
```python
from hybrid_p2p import Signer, Verifier, KeyManager

km = KeyManager()
signer = Signer(km)

# Sign manifest
signed = signer.sign_manifest(manifest)

# Verify
Verifier.verify_manifest(signed)
```

## Docker

```bash
docker-compose up -d                    # Start all services
docker-compose ps                       # Check status
docker-compose logs -f ipfs             # View logs
docker-compose down                     # Stop services
```

**Services:**
- IPFS: http://localhost:5001 (API), http://localhost:8080 (Gateway)
- IPFS Cluster: http://localhost:9094
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## IPFS Commands

```bash
ipfs daemon                             # Start daemon
ipfs id                                 # Show node info
ipfs add file.txt                       # Add file
ipfs cat QmX...                         # Retrieve content
ipfs pin ls                             # List pins
ipfs repo gc                            # Garbage collect
```

## Manifest Structure

```json
{
  "content_id": "sha256-hash",
  "version": "1.0.0",
  "name": "package-name",
  "files": [
    {
      "path": "file.txt",
      "size": 1024,
      "sha256": "abc...",
      "cid": "Qm...",
      "mime_type": "text/plain"
    }
  ],
  "signature": {
    "algorithm": "ed25519",
    "public_key": "base64...",
    "signature": "base64..."
  },
  "distribution": {
    "ipfs_cid": "Qm...",
    "torrent_infohash": "abc...",
    "torrent_magnet": "magnet:?xt=..."
  }
}
```

## Testing

```bash
pytest                                   # Run all tests
pytest -v                                # Verbose
pytest --cov=hybrid_p2p                  # With coverage
pytest tests/test_validation.py          # Specific file
pytest -k "test_sign"                    # By name pattern
```

## Troubleshooting

### IPFS not running
```bash
ipfs id                                  # Check if running
ipfs daemon &                            # Start if needed
tail -f ~/.ipfs/logs/latest-log.txt     # View logs
```

### Permission denied on key file
```bash
chmod 600 ~/.hybrid_p2p/keys/mykey.pem  # Fix permissions
```

### Import errors
```bash
pip install -e ".[dev]" --force-reinstall
python -c "import hybrid_p2p; print(hybrid_p2p.__version__)"
```

### Signature verification failed
```bash
# Check manifest signature manually
python3 -c "
from hybrid_p2p import ContentManifest, Verifier
from pathlib import Path
m = ContentManifest.model_validate_json(Path('manifest.json').read_text())
Verifier.verify_manifest(m)
"
```

## Environment Variables

```bash
export SIGNING_KEY_PATH=~/.hybrid_p2p/keys/mykey.pem
export IPFS_API_URL=http://127.0.0.1:5001
export IPFS_GATEWAY_URL=http://127.0.0.1:8080
export PINNING_API_KEY=your-pinata-key
```

## File Locations

```
~/.hybrid_p2p/
├── cache/              # Downloaded content
├── keys/               # Signing keys
└── manifests/          # Saved manifests

~/.ipfs/                # IPFS repository
├── config              # IPFS config
└── datastore/          # Content storage
```

## Common Patterns

### Publish with Description
```bash
hybrid-p2p publish *.csv \
  --name dataset \
  --version 2.0.0 \
  --uploader-id researcher@university.edu \
  --description "Research dataset for paper XYZ" \
  --key keys/research.pem
```

### Batch Processing
```python
for version in ["1.0.0", "1.1.0", "1.2.0"]:
    manifest = client.publish(
        files=get_files_for_version(version),
        name="package",
        version=version,
        uploader_id="automated",
    )
    save_manifest(manifest, f"manifest-{version}.json")
```

### Verify Before Publishing
```python
validator = ContentValidator()
for file_path in files:
    validator.validate_file(file_path)  # Raises if invalid
manifest = client.publish(...)          # Safe to publish
```

## Monitoring

### Prometheus Queries
```promql
ipfs_peers_total                        # Number of peers
rate(ipfs_bandwidth_in_bytes_total[5m]) # Inbound bandwidth
cluster_pinqueue_size                   # Pin queue size
```

### Health Check
```bash
curl http://localhost:5001/api/v0/id    # IPFS health
curl http://localhost:9094/id           # Cluster health
```

## URLs

- IPFS WebUI: http://127.0.0.1:5001/webui
- IPFS Gateway: http://127.0.0.1:8080
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## Documentation

- `README.md` - Main documentation
- `GETTING_STARTED.md` - Tutorial
- `DEPLOYMENT.md` - Production guide
- `PROJECT_SUMMARY.md` - Architecture
- `examples/` - Code examples
