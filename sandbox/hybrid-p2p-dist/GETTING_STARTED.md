# Getting Started Guide

## Project Structure

```
hybrid-p2p-dist/
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI/CD
├── examples/
│   ├── basic_usage.py                # Complete usage example
│   └── pinning_service.py            # Pinning service integration
├── hybrid_p2p/                       # Main package
│   ├── __init__.py                   # Public API exports
│   ├── bittorrent_adapter.py         # BitTorrent integration
│   ├── cli.py                        # Command-line interface
│   ├── client.py                     # High-level client API
│   ├── ipfs_adapter.py               # IPFS integration
│   ├── signing.py                    # Cryptographic signing
│   └── validation.py                 # Schema validation
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── test_signing.py               # Signing tests
│   └── test_validation.py            # Validation tests
├── .gitignore                        # Git ignore patterns
├── DEPLOYMENT.md                     # Production deployment guide
├── PROJECT_SUMMARY.md                # Architecture overview
├── README.md                         # Main documentation
├── docker-compose.yml                # Docker orchestration
├── prometheus.yml                    # Monitoring configuration
├── pyproject.toml                    # Package metadata & dependencies
└── quickstart.sh                     # Automated setup script
```

## Installation Methods

### Method 1: Quick Start (Recommended for Testing)

```bash
# Clone and run automated setup
git clone https://github.com/your-org/hybrid-p2p-dist.git
cd hybrid-p2p-dist
chmod +x quickstart.sh
./quickstart.sh
```

This script will:
1. Check Python version (3.10+)
2. Install system dependencies (libmagic, libtorrent)
3. Install and initialize IPFS
4. Install the Python package
5. Generate test signing keys
6. Start IPFS daemon
7. Create sample files

### Method 2: Manual Installation

#### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3.10 python3-pip libmagic1 python3-libtorrent

# macOS
brew install python@3.10 libmagic libtorrent-rasterbar
```

#### Install IPFS

```bash
# Download IPFS
wget https://dist.ipfs.io/go-ipfs/v0.24.0/go-ipfs_v0.24.0_linux-amd64.tar.gz
tar -xzf go-ipfs_v0.24.0_linux-amd64.tar.gz
cd go-ipfs
sudo bash install.sh

# Initialize IPFS
ipfs init
ipfs config Addresses.API /ip4/127.0.0.1/tcp/5001
ipfs config Addresses.Gateway /ip4/127.0.0.1/tcp/8080

# Start daemon
ipfs daemon &
```

#### Install Package

```bash
cd hybrid-p2p-dist
pip install -e ".[dev]"
```

### Method 3: Docker (for Development)

```bash
cd hybrid-p2p-dist
docker-compose up -d
```

This starts:
- IPFS node (API: 5001, Gateway: 8080)
- IPFS Cluster (API: 9094)
- Prometheus (UI: 9090)
- Grafana (UI: 3000)

## First Steps

### 1. Generate Signing Keys

```bash
hybrid-p2p keygen -o ~/.hybrid_p2p/keys -n mykey
```

Output:
```
✓ Key pair generated!
Private key: /home/user/.hybrid_p2p/keys/mykey.pem
Public key: /home/user/.hybrid_p2p/keys/mykey.pub
Keep the private key secure!
```

### 2. Create Sample Content

```bash
mkdir ~/sample-data
echo "Hello, P2P World!" > ~/sample-data/readme.txt
echo '{"version": "1.0"}' > ~/sample-data/metadata.json
```

### 3. Publish Content

```bash
cd ~/sample-data

hybrid-p2p publish readme.txt metadata.json \
  --name my-first-package \
  --version 1.0.0 \
  --uploader-id alice@example.com \
  --key ~/.hybrid_p2p/keys/mykey.pem \
  --output manifest.json
```

Output:
```
✓ Published successfully!

┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Field       ┃ Value                            ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Name        │ my-first-package                 │
│ Version     │ 1.0.0                            │
│ Content ID  │ abc123...                        │
│ Total Size  │ 45 bytes                         │
│ Files       │ 2                                │
│ IPFS CID    │ QmX...                           │
│ Torrent Hash│ def456...                        │
└─────────────┴──────────────────────────────────┘

Manifest saved to: manifest.json
```

### 4. Inspect Manifest

```bash
hybrid-p2p inspect manifest.json
```

### 5. Fetch Content (Simulate Another User)

```bash
cd ~
mkdir downloads
hybrid-p2p fetch ~/sample-data/manifest.json -o downloads
```

Output:
```
✓ Fetched successfully!
Files saved to: /home/user/downloads

┏━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┓
┃ File         ┃ Size    ┃ Type       ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━┩
│ readme.txt   │ 18      │ text/plain │
│ metadata.json│ 19      │ application│
│              │         │ /json      │
└──────────────┴─────────┴────────────┘
```

### 6. Verify Content

```bash
hybrid-p2p verify ~/sample-data/manifest.json downloads
```

Output:
```
✓ All files verified successfully!
```

## Python API Usage

### Basic Publishing

```python
from pathlib import Path
from hybrid_p2p import Client, ClientConfig

# Configure client
config = ClientConfig(
    signing_key_path=Path("~/.hybrid_p2p/keys/mykey.pem").expanduser(),
    ipfs_api_url="http://127.0.0.1:5001",
    ipfs_gateway_url="http://127.0.0.1:8080",
)

# Publish content
with Client(config) as client:
    manifest = client.publish(
        files=[
            Path("data.txt"),
            Path("metadata.json"),
        ],
        name="my-dataset",
        version="1.0.0",
        uploader_id="alice@example.com",
        description="Example dataset",
        create_torrent=True,
        pin_to_ipfs=True,
    )
    
    print(f"Published: {manifest.distribution.ipfs_cid}")
    
    # Save manifest
    manifest_path = Path("manifest.json")
    manifest_path.write_text(manifest.model_dump_json(indent=2))
```

### Basic Fetching

```python
# Fetch content
output_dir = client.fetch(
    manifest=Path("manifest.json"),
    output_dir=Path("./downloads"),
    verify_signature=True,
    prefer_ipfs=True,
)

print(f"Downloaded to: {output_dir}")

# Verify
is_valid = client.verify_local_content(
    manifest=Path("manifest.json"),
    content_dir=output_dir,
)
print(f"Content valid: {is_valid}")
```

### Working with Keys

```python
from hybrid_p2p import generate_keypair, KeyManager, Signer

# Generate new keys
private_path, public_path = generate_keypair(
    output_dir=Path("~/.hybrid_p2p/keys").expanduser(),
    key_name="production_key",
    password=b"secure-password",  # Optional
)

# Load existing keys
km = KeyManager.from_private_key_file(
    private_path,
    password=b"secure-password",
)

# Sign manifest
signer = Signer(km)
signed_manifest = signer.sign_manifest(manifest)
```

## Common Workflows

### Workflow 1: Research Data Publishing

```bash
# 1. Organize data
mkdir research-dataset-v1
cp data/*.csv research-dataset-v1/
cp metadata.json research-dataset-v1/

# 2. Generate keys (once)
hybrid-p2p keygen -o keys -n research

# 3. Publish
cd research-dataset-v1
hybrid-p2p publish *.csv metadata.json \
  --name research-dataset \
  --version 1.0.0 \
  --uploader-id researcher@university.edu \
  --description "Research dataset for paper XYZ" \
  --key ../keys/research.pem \
  --output manifest.json

# 4. Commit manifest to GitHub
git add manifest.json
git commit -m "Release v1.0.0"
git push

# 5. Collaborators fetch
hybrid-p2p fetch manifest.json -o data/
```

### Workflow 2: Software Distribution

```python
from hybrid_p2p import Client, ClientConfig
from pathlib import Path
import subprocess

# Build software
subprocess.run(["make", "build"])

# Publish release
config = ClientConfig(signing_key_path=Path("release-key.pem"))
with Client(config) as client:
    manifest = client.publish(
        files=[Path("build/myapp.tar.gz")],
        name="myapp",
        version="2.1.0",
        uploader_id="releases@mycompany.com",
        description="MyApp v2.1.0 - Bug fixes and improvements",
    )
    
    # Save to releases/
    releases_dir = Path("releases")
    releases_dir.mkdir(exist_ok=True)
    
    manifest_path = releases_dir / f"myapp-v2.1.0-manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    
    print(f"IPFS: ipfs://{manifest.distribution.ipfs_cid}")
    print(f"Magnet: {manifest.distribution.torrent_magnet}")
```

### Workflow 3: Content Mirroring

```python
# Mirror content from one network to another
from hybrid_p2p import Client, ClientConfig

config = ClientConfig()
with Client(config) as client:
    # Fetch from IPFS
    local_dir = client.fetch(
        manifest="upstream-manifest.json",
        prefer_ipfs=True,
    )
    
    # Re-publish with your signature
    new_manifest = client.publish(
        files=list(local_dir.glob("*")),
        name="mirrored-content",
        version="1.0.0",
        uploader_id="mirror@example.com",
    )
    
    print(f"Mirrored as: {new_manifest.distribution.ipfs_cid}")
```

## Troubleshooting

### IPFS daemon not running

```bash
# Check if running
ipfs id

# If not, start it
ipfs daemon &

# Check logs
tail -f ~/.ipfs/logs/latest-log.txt
```

### Cannot connect to IPFS API

```bash
# Check configuration
ipfs config Addresses.API

# Should be: /ip4/127.0.0.1/tcp/5001

# Reconfigure if needed
ipfs config Addresses.API /ip4/127.0.0.1/tcp/5001
ipfs daemon restart
```

### Signature verification failed

```bash
# Check public key
cat ~/.hybrid_p2p/keys/mykey.pub

# Verify manifest manually
python3 << EOF
from hybrid_p2p import ContentManifest, Verifier
from pathlib import Path

manifest = ContentManifest.model_validate_json(
    Path("manifest.json").read_text()
)

try:
    Verifier.verify_manifest(manifest)
    print("✓ Signature valid")
except Exception as e:
    print(f"✗ Signature invalid: {e}")
EOF
```

### Import errors

```bash
# Reinstall with dependencies
pip install -e ".[dev]" --force-reinstall

# Check installation
python3 -c "import hybrid_p2p; print(hybrid_p2p.__version__)"
```

## Next Steps

1. **Read the full documentation**: See `README.md` for complete API reference
2. **Review examples**: Check `examples/` directory for real-world scenarios
3. **Set up production**: Follow `DEPLOYMENT.md` for production deployment
4. **Join the community**: GitHub Discussions for questions and ideas
5. **Contribute**: See `CONTRIBUTING.md` for guidelines

## Resources

- **IPFS Documentation**: https://docs.ipfs.io
- **IPFS Web UI**: http://127.0.0.1:5001/webui
- **IPFS Gateway**: http://127.0.0.1:8080
- **Project Repository**: https://github.com/your-org/hybrid-p2p-dist
- **Issue Tracker**: https://github.com/your-org/hybrid-p2p-dist/issues

## Support

- **Questions**: GitHub Discussions
- **Bugs**: GitHub Issues
- **Security**: security@example.com
- **Documentation**: https://hybrid-p2p-dist.readthedocs.io

Happy distributing! 🚀
