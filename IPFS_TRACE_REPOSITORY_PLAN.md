# IPFS-Based Trace Repository - Detailed Implementation Plan

**Version:** 1.1
**Date:** 2025-10-21
**Author:** Kasper Munch
**Status:** Design Complete - Awaiting Implementation
**Update:** Added Hybrid Auto-Start Daemon Management (Option 4)

---

## Executive Summary

This document outlines a comprehensive plan for implementing a decentralized, IPFS-based repository for PtDAlgorithms model traces. The system combines IPFS for content distribution with GitHub for human-readable indexing, providing zero-cost hosting, automatic redundancy, and offline-first functionality.

**Key Benefits:**
- **Zero hosting costs** for maintainers
- **Automatic redundancy** via IPFS network
- **Content integrity** via cryptographic hashing
- **Offline usage** after initial download
- **Community-driven** infrastructure

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Data Model](#data-model)
3. [Python Implementation](#python-implementation)
4. [IPFS Daemon Auto-Start](#ipfs-daemon-auto-start)
5. [Deployment & Operations](#deployment--operations)
6. [Migration Path](#migration-path)
7. [Cost Analysis](#cost-analysis)

---

## Architecture Overview

### Core Components

1. **IPFS Network** - Decentralized content storage
2. **Central Registry** - GitHub-hosted metadata/index (human-readable names → IPFS CIDs)
3. **Pinning Services** - Reliable mirrors (Pinata, Web3.Storage, institutional nodes)
4. **Python Client** - Unified API abstracting IPFS complexity
5. **Gateway Fallbacks** - HTTP gateways for users without IPFS daemon

### Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                  User's Machine                     │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │  phasic Python Library                │  │
│  │                                              │  │
│  │  trace = registry.get("coalescent_n5")      │  │
│  │         ↓                                    │  │
│  │  1. Check local cache                       │  │
│  │  2. Fetch CID from GitHub registry          │  │
│  │  3. Retrieve from IPFS (prioritize local)   │  │
│  │  4. Fallback to HTTP gateway if needed      │  │
│  └──────────────────────────────────────────────┘  │
│                     ↓                               │
│           ~/.phasic_traces/                  │
│           └── ipfs_cache/{CID}/trace.json.gz        │
└─────────────────────────────────────────────────────┘
                      ↓
        ┌─────────────┴─────────────┐
        ↓                           ↓
┌───────────────┐          ┌────────────────┐
│  IPFS Network │          │ HTTP Gateways  │
│               │          │                │
│ Local daemon  │   OR     │ ipfs.io        │
│ University    │          │ cloudflare-    │
│ peers         │          │   ipfs.com     │
└───────────────┘          └────────────────┘
        ↑
   (pinned by)
        ↓
┌─────────────────────────────────┐
│  Pinning Services / Mirrors     │
│                                 │
│  • Pinata (commercial)          │
│  • Web3.Storage (free tier)     │
│  • AU institutional IPFS node   │
│  • Community contributors       │
└─────────────────────────────────┘
```

---

## Data Model

### 1. Central Registry (GitHub Repository)

**Repository:** `munch-group/phasic-traces`
**File:** `registry.json`

```json
{
  "version": "1.0.0",
  "updated": "2025-10-21T12:00:00Z",
  "traces": {
    "coalescent_n5_theta1": {
      "cid": "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi",
      "description": "Kingman coalescent, n=5 samples, 1 parameter (theta)",
      "metadata": {
        "model_type": "coalescent",
        "domain": "population-genetics",
        "param_length": 1,
        "vertices": 5,
        "edges": 10,
        "trace_version": "1.0",
        "created": "2025-10-15",
        "author": "Kasper Munch <kaspermunch@birc.au.dk>",
        "citation": "Røikjer, Hobolth & Munch (2022)",
        "tags": ["coalescent", "kingman", "population-genetics"]
      },
      "files": {
        "trace.json.gz": {
          "cid": "bafkreih4...",
          "size_bytes": 15432
        },
        "metadata.json": {
          "cid": "bafkreie3...",
          "size_bytes": 842
        },
        "example.py": {
          "cid": "bafkreig7...",
          "size_bytes": 1204
        }
      },
      "pins": [
        {"service": "pinata", "status": "pinned", "pin_id": "..."},
        {"service": "web3storage", "status": "pinned"},
        {"service": "au-institutional", "url": "https://ipfs.birc.au.dk"}
      ],
      "checksum": "sha256:a3f2e9c8b1d4...",
      "license": "MIT"
    },

    "coalescent_n10_theta2": {
      "cid": "bafybeiabc123...",
      "description": "Kingman coalescent, n=10 samples, 2 parameters (theta, rho)",
      "metadata": {
        "model_type": "coalescent",
        "domain": "population-genetics",
        "param_length": 2,
        "vertices": 45,
        "edges": 90,
        "trace_version": "1.0",
        "based_on": "coalescent_n5_theta1",
        "created": "2025-10-16"
      },
      "files": {
        "trace.json.gz": {"cid": "bafkreih5...", "size_bytes": 84321}
      },
      "pins": [...]
    },

    "structured_coalescent_2pop": {
      "cid": "bafybeidef456...",
      "description": "Structured coalescent, 2 populations with migration",
      "metadata": {
        "model_type": "structured-coalescent",
        "domain": "population-genetics",
        "param_length": 3,
        "vertices": 120,
        "edges": 340,
        "trace_version": "1.0"
      },
      "files": {
        "trace.json.gz": {"cid": "bafkreih6...", "size_bytes": 256000},
        "construction_code.py": {"cid": "bafkreih7..."}
      },
      "pins": [...]
    }
  },

  "collections": {
    "coalescent_basic": {
      "description": "Basic Kingman coalescent models (n=3 to n=20)",
      "cid": "bafybeicollection123...",
      "traces": [
        "coalescent_n5_theta1",
        "coalescent_n10_theta2",
        "coalescent_n15_theta1"
      ]
    }
  },

  "pinning_services": {
    "pinata": {
      "api_url": "https://api.pinata.cloud",
      "gateway": "https://gateway.pinata.cloud"
    },
    "web3storage": {
      "api_url": "https://api.web3.storage",
      "gateway": "https://w3s.link"
    },
    "au-institutional": {
      "gateway": "https://ipfs.birc.au.dk",
      "contact": "kaspermunch@birc.au.dk",
      "status": "active"
    }
  }
}
```

### 2. Trace Package Structure (IPFS Directory)

Each trace is stored as an IPFS directory (UnixFS DAG):

```
ipfs://bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi/
├── trace.json.gz           # Main trace data (gzipped)
├── metadata.json           # Detailed metadata
├── checksum.sha256         # Integrity verification
├── example.py              # Usage example
├── construction_code.py    # Optional: code that built this trace
└── README.md               # Human-readable documentation
```

**Benefits of IPFS directories:**
- Files share common sub-graphs → automatic deduplication
- Can add files later while preserving main CID references
- Partial downloads (fetch just `trace.json.gz` if needed)

### 3. Trace Metadata File

**File:** `metadata.json` (stored in IPFS with trace)

```json
{
  "trace_id": "coalescent_n5_theta1",
  "format_version": "1.0",
  "created": "2025-10-15T14:30:00Z",
  "author": {
    "name": "Kasper Munch",
    "email": "kaspermunch@birc.au.dk",
    "orcid": "0000-0001-2345-6789"
  },
  "citation": {
    "text": "Røikjer, Hobolth & Munch (2022)",
    "doi": "10.1007/s11222-022-10155-6",
    "url": "https://doi.org/10.1007/s11222-022-10155-6"
  },
  "model": {
    "type": "coalescent",
    "variant": "kingman",
    "description": "Standard Kingman coalescent for n=5 haploid samples",
    "parameters": [
      {"name": "theta", "description": "Scaled mutation rate (4*N_e*mu)", "domain": "[0, ∞)"}
    ],
    "state_space": {
      "vertices": 5,
      "edges": 10,
      "parameterized_edges": 10,
      "max_state_length": 1
    }
  },
  "construction": {
    "method": "callback",
    "code_file": "construction_code.py",
    "dependencies": {
      "phasic": ">=0.21.3"
    }
  },
  "validation": {
    "test_cases": [
      {
        "theta": 1.0,
        "time": 2.0,
        "expected_pdf": 0.18394,
        "tolerance": 1e-5
      }
    ]
  },
  "performance": {
    "trace_recording_time_ms": 45,
    "trace_size_bytes": 15432,
    "evaluation_time_ms": 0.8,
    "vertices": 5
  },
  "license": "MIT",
  "keywords": ["coalescent", "population-genetics", "kingman", "theta"]
}
```

---

## Python Implementation

### New Module: `src/phasic/trace_repository.py`

**Key Classes:**

1. **`IPFSBackend`** - IPFS client with HTTP gateway fallback and auto-start capability
2. **`TraceRegistry`** - Main API for browsing/downloading traces
3. **Helper functions** - `get_trace()`, `install_trace_library()`

**Core Features:**
- **Progressive Enhancement**: Works without IPFS → faster with daemon → optimal with service
- **Auto-start daemon**: Automatically launches IPFS daemon when installed
- **Automatic fallback**: IPFS daemon → HTTP gateways
- Local caching in `~/.phasic_traces/`
- Checksum verification
- Metadata search/filtering
- Publishing workflow

### Usage Examples

#### End User (Download & Use)

```python
from phasic.trace_repository import TraceRegistry, get_trace
from phasic.trace_elimination import trace_to_log_likelihood
from phasic import SVGD
import numpy as np

# Simple one-liner
trace = get_trace("coalescent_n5_theta1")

# Or with registry
registry = TraceRegistry()

# Browse available traces
traces = registry.list_traces(domain="population-genetics")
for t in traces:
    print(f"{t['trace_id']}: {t['description']}")

# Download and use trace
trace = registry.get_trace("coalescent_n5_theta1")

# Use for inference
observed_times = np.array([1.2, 2.3, 0.8, 1.5, 3.2])
log_lik = trace_to_log_likelihood(trace, observed_times, granularity=100)

svgd = SVGD(
    log_prob=log_lik,
    theta_dim=1,
    n_particles=100,
    n_iterations=1000
)

results = svgd.fit()
print(f"Posterior mean theta: {results['theta_mean']}")
```

#### Contributor (Publish Trace)

```python
from phasic import Graph
from phasic.trace_elimination import record_elimination_trace
from phasic.trace_repository import TraceRegistry
import numpy as np

# Build model
def coalescent_callback(state):
    n = state[0]
    if n <= 1:
        return []
    rate = n * (n - 1) / 2
    return [(np.array([n - 1]), 0.0, [rate])]

graph = Graph(
    state_length=1,
    callback=coalescent_callback,
    parameterized=True,
    nr_samples=5
)

# Record trace
trace = record_elimination_trace(graph, param_length=1)

# Create metadata
metadata = {
    "model_type": "coalescent",
    "domain": "population-genetics",
    "param_length": 1,
    "vertices": 5,
    "edges": 10,
    "description": "Kingman coalescent for n=5 samples with theta parameter",
    "created": "2025-10-21",
    "author": "Kasper Munch <kaspermunch@birc.au.dk>",
    "citation": {
        "text": "Røikjer, Hobolth & Munch (2022)",
        "doi": "10.1007/s11222-022-10155-6"
    },
    "tags": ["coalescent", "kingman", "population-genetics"],
    "license": "MIT"
}

# Publish to IPFS
registry = TraceRegistry()
cid = registry.publish_trace(
    trace=trace,
    trace_id="coalescent_n5_theta1",
    metadata=metadata,
    submit_pr=True  # Prints PR instructions
)

print(f"✓ Published to IPFS: {cid}")
print(f"✓ To make publicly available, submit PR to registry")
```

---

## IPFS Daemon Auto-Start

### Progressive Enhancement Strategy

The system uses a **three-tier approach** to provide optimal user experience:

**Tier 1 (Zero Config):** HTTP gateways work out of the box
- No IPFS installation required
- Downloads via public IPFS gateways (ipfs.io, cloudflare-ipfs.com)
- Slower but always works

**Tier 2 (Auto-Start):** Python auto-starts daemon when IPFS installed
- User installs IPFS once: `brew install ipfs` or equivalent
- Python automatically starts daemon when needed
- Faster downloads via local IPFS node
- No manual intervention required

**Tier 3 (Optimal):** IPFS installed as system service
- Daemon always running, starts on login
- Lowest latency, best performance
- Shares content with IPFS network
- Optional setup via one-line installer

### Implementation: Auto-Start in IPFSBackend

```python
class IPFSBackend:
    """
    IPFS backend with automatic daemon management.

    Tries to connect to existing daemon, auto-starts if needed,
    falls back to HTTP gateways if IPFS not installed.
    """

    def __init__(
        self,
        daemon_addr: str = "/ip4/127.0.0.1/tcp/5001",
        gateways: Optional[List[str]] = None,
        auto_start: bool = True,
        timeout: int = 30
    ):
        self.daemon_addr = daemon_addr
        self.timeout = timeout
        self.auto_start = auto_start
        self.client = None
        self.daemon_process = None

        # Configure HTTP gateway fallbacks
        if gateways is None:
            self.gateways = [
                "https://ipfs.io",
                "https://cloudflare-ipfs.com",
                "https://dweb.link",
                "https://gateway.pinata.cloud"
            ]
        else:
            self.gateways = gateways

        # Try connecting to IPFS daemon
        if HAS_IPFS_CLIENT:
            try:
                # Try existing daemon first
                self.client = ipfshttpclient.connect(daemon_addr, timeout=5)
                version = self.client.version()
                print(f"✓ Connected to IPFS daemon (version {version['Version']})")
            except Exception as e:
                # Try auto-starting daemon
                if auto_start and self._start_daemon():
                    time.sleep(2)  # Give daemon time to initialize
                    try:
                        self.client = ipfshttpclient.connect(daemon_addr, timeout=5)
                        print(f"✓ Started IPFS daemon automatically")
                    except:
                        warnings.warn("IPFS daemon started but connection failed. Using HTTP gateways.")
                else:
                    if not auto_start:
                        print("IPFS daemon not running. Using HTTP gateways.")
                    else:
                        warnings.warn(f"IPFS not available. Using HTTP gateways.")

    def _start_daemon(self) -> bool:
        """
        Attempt to start IPFS daemon in background.

        Returns
        -------
        bool
            True if daemon started successfully, False otherwise
        """
        import subprocess
        import shutil

        # Check if ipfs is installed
        ipfs_path = shutil.which('ipfs')
        if not ipfs_path:
            return False

        try:
            # Check if daemon already running
            result = subprocess.run(
                ['pgrep', '-x', 'ipfs'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return True  # Already running

            # Check if IPFS is initialized
            ipfs_dir = Path.home() / ".ipfs"
            if not ipfs_dir.exists():
                # Initialize IPFS
                subprocess.run(
                    ['ipfs', 'init'],
                    capture_output=True,
                    check=True
                )
                print("✓ Initialized IPFS repository")

            # Start daemon in background (detached from parent)
            self.daemon_process = subprocess.Popen(
                ['ipfs', 'daemon'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True  # Detach from parent process
            )

            return True
        except Exception as e:
            warnings.warn(f"Failed to start IPFS daemon: {e}")
            return False

    def __del__(self):
        """
        Cleanup on object destruction.

        Note: We intentionally do NOT kill the daemon here, as it should
        persist for other processes and future use.
        """
        pass
```

### Optional: System Service Installation

For users who want optimal performance, provide a post-install script that sets up IPFS as a system service.

#### Post-Install Hook (setup.py)

```python
def post_install():
    """
    Post-installation hook for optional IPFS setup.

    Called after: pip install phasic[ipfs]
    """
    import shutil
    import subprocess
    from pathlib import Path

    # Check if IPFS is installed
    if not shutil.which('ipfs'):
        print("\n" + "="*60)
        print("OPTIONAL: Install IPFS for faster downloads")
        print("="*60)
        print("IPFS is not required but provides:")
        print("  • Faster downloads via local daemon")
        print("  • Offline sharing with collaborators")
        print("  • Contributing traces to the network")
        print("\nInstall: https://docs.ipfs.tech/install/")
        print("  macOS:  brew install ipfs")
        print("  Linux:  See https://docs.ipfs.tech/install/")
        print("="*60 + "\n")
        return

    # Initialize IPFS if not already done
    ipfs_dir = Path.home() / ".ipfs"
    if not ipfs_dir.exists():
        subprocess.run(['ipfs', 'init'], check=True)
        print("✓ IPFS initialized")

    # Ask user about auto-start preference
    print("\n" + "="*60)
    print("IPFS Daemon Auto-Start Options")
    print("="*60)
    print("1. Manual (run 'ipfs daemon' when needed)")
    print("2. Install as system service (recommended)")
    print("3. Auto-start from Python (default)")
    print("="*60)

    choice = input("\nChoose option [1-3] (press Enter for 3): ").strip() or "3"

    if choice == "2":
        install_system_service()
    elif choice == "3":
        print("\n✓ IPFS will auto-start when phasic needs it")
        print("  No further action required!")
    else:
        print("\n✓ Manual mode selected")
        print("  Run 'ipfs daemon' before using IPFS features")

    print("="*60 + "\n")


def install_system_service():
    """Install IPFS as system service (OS-specific)."""
    import platform
    import subprocess
    import shutil
    from pathlib import Path

    system = platform.system()
    ipfs_path = shutil.which('ipfs')

    if system == 'Darwin':  # macOS
        install_macos_service(ipfs_path)
    elif system == 'Linux':
        install_linux_service(ipfs_path)
    elif system == 'Windows':
        install_windows_service(ipfs_path)
    else:
        print(f"⚠ System service installation not supported on {system}")
        print("  IPFS will auto-start from Python when needed")


def install_macos_service(ipfs_path: str):
    """Install IPFS as macOS launchd service."""
    from pathlib import Path
    import subprocess

    plist_path = Path.home() / "Library/LaunchAgents/io.ipfs.daemon.plist"

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>io.ipfs.daemon</string>
    <key>ProgramArguments</key>
    <array>
        <string>{ipfs_path}</string>
        <string>daemon</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{Path.home() / '.ipfs/daemon.log'}</string>
    <key>StandardErrorPath</key>
    <string>{Path.home() / '.ipfs/daemon.error.log'}</string>
</dict>
</plist>
"""

    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(plist_content)

    # Load service
    subprocess.run(['launchctl', 'load', str(plist_path)], check=True)

    print(f"✓ IPFS installed as macOS service")
    print(f"  Daemon will auto-start on login")
    print(f"  Logs: {Path.home() / '.ipfs/daemon.log'}")


def install_linux_service(ipfs_path: str):
    """Install IPFS as systemd user service."""
    from pathlib import Path
    import subprocess

    service_path = Path.home() / ".config/systemd/user/ipfs.service"

    service_content = f"""[Unit]
Description=IPFS Daemon
After=network.target

[Service]
ExecStart={ipfs_path} daemon
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
"""

    service_path.parent.mkdir(parents=True, exist_ok=True)
    service_path.write_text(service_content)

    # Enable and start
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', '--user', 'enable', 'ipfs'], check=True)
    subprocess.run(['systemctl', '--user', 'start', 'ipfs'], check=True)

    print(f"✓ IPFS installed as systemd user service")
    print(f"  Daemon will auto-start on login")
    print(f"  Status: systemctl --user status ipfs")


def install_windows_service(ipfs_path: str):
    """Install IPFS as Windows scheduled task."""
    import subprocess

    # Create scheduled task to run on logon
    subprocess.run([
        'schtasks', '/create',
        '/tn', 'IPFS Daemon',
        '/tr', f'"{ipfs_path}" daemon',
        '/sc', 'onlogon',
        '/f'  # Force overwrite if exists
    ], check=True)

    # Start task immediately
    subprocess.run(['schtasks', '/run', '/tn', 'IPFS Daemon'], check=True)

    print(f"✓ IPFS installed as Windows scheduled task")
    print(f"  Daemon will auto-start on login")
```

### One-Line Installer Script

Provide a convenience script for advanced users:

**File:** `scripts/install_ipfs_service.sh`

```bash
#!/bin/bash
# Install IPFS and configure as system service for PtDAlgorithms

set -e

echo "Installing IPFS for PtDAlgorithms..."

# Detect OS
OS=$(uname -s)

# Install IPFS if not present
if ! command -v ipfs &> /dev/null; then
    if [ "$OS" == "Darwin" ]; then
        if ! command -v brew &> /dev/null; then
            echo "Error: Homebrew required. Install from: https://brew.sh"
            exit 1
        fi
        brew install ipfs
    elif [ "$OS" == "Linux" ]; then
        wget https://dist.ipfs.tech/kubo/v0.24.0/kubo_v0.24.0_linux-amd64.tar.gz
        tar -xvzf kubo_v0.24.0_linux-amd64.tar.gz
        cd kubo
        sudo bash install.sh
        cd ..
        rm -rf kubo kubo_v0.24.0_linux-amd64.tar.gz
    else
        echo "Error: Unsupported OS: $OS"
        exit 1
    fi
fi

# Initialize IPFS
if [ ! -d "$HOME/.ipfs" ]; then
    ipfs init
    echo "✓ IPFS initialized"
fi

# Install as service
if [ "$OS" == "Darwin" ]; then
    # macOS launchd
    PLIST_PATH="$HOME/Library/LaunchAgents/io.ipfs.daemon.plist"
    mkdir -p "$HOME/Library/LaunchAgents"

    cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>io.ipfs.daemon</string>
    <key>ProgramArguments</key>
    <array>
        <string>$(which ipfs)</string>
        <string>daemon</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
EOF

    launchctl load "$PLIST_PATH"
    echo "✓ IPFS installed as macOS service (auto-starts on login)"

elif [ "$OS" == "Linux" ]; then
    # systemd user service
    SERVICE_PATH="$HOME/.config/systemd/user/ipfs.service"
    mkdir -p "$HOME/.config/systemd/user"

    cat > "$SERVICE_PATH" <<EOF
[Unit]
Description=IPFS Daemon
After=network.target

[Service]
ExecStart=$(which ipfs) daemon
Restart=always

[Install]
WantedBy=default.target
EOF

    systemctl --user daemon-reload
    systemctl --user enable ipfs
    systemctl --user start ipfs
    echo "✓ IPFS installed as systemd service (auto-starts on login)"
fi

echo ""
echo "✓ Setup complete!"
echo "  IPFS daemon is now running and will auto-start on login"
echo "  PtDAlgorithms will automatically use local daemon for faster downloads"
```

**Usage:**
```bash
curl -fsSL https://raw.githubusercontent.com/munch-group/phasic/master/scripts/install_ipfs_service.sh | bash
```

### User Experience Flow

**First-time user (no IPFS):**
```python
>>> from phasic.trace_repository import get_trace
>>> trace = get_trace("coalescent_n5_theta1")
Updating registry from munch-group/phasic-traces...
✓ Registry updated
Downloading trace 'coalescent_n5_theta1' from IPFS...
IPFS not available. Using HTTP gateways.
  Downloading from https://ipfs.io/ipfs/bafybeig...
  Progress: 100.0% (15.1 KB)
✓ Downloaded to /Users/kasper/.phasic_traces/traces/coalescent_n5_theta1/trace.json.gz
```

**User with IPFS installed:**
```python
>>> from phasic.trace_repository import get_trace
>>> trace = get_trace("coalescent_n5_theta1")
✓ Started IPFS daemon automatically
Downloading trace 'coalescent_n5_theta1' from IPFS...
✓ Downloaded via local IPFS node (2.3s)
```

**User with IPFS service installed:**
```python
>>> from phasic.trace_repository import get_trace
>>> trace = get_trace("coalescent_n5_theta1")
✓ Connected to IPFS daemon (version 0.24.0)
Downloading trace 'coalescent_n5_theta1' from IPFS...
✓ Downloaded via local IPFS node (1.1s)
```

---

## Deployment & Operations

### 1. Initial Setup (Repository Maintainer)

**Install IPFS:**
```bash
# macOS
brew install ipfs

# Linux
wget https://dist.ipfs.tech/kubo/v0.24.0/kubo_v0.24.0_linux-amd64.tar.gz
tar -xvzf kubo_v0.24.0_linux-amd64.tar.gz
cd kubo
sudo bash install.sh

# Initialize and start daemon
ipfs init
ipfs daemon &
```

**Create GitHub Registry Repository:**
```bash
# Create munch-group/phasic-traces
git clone https://github.com/munch-group/phasic-traces.git
cd phasic-traces

# Create initial registry
cat > registry.json <<'EOF'
{
  "version": "1.0.0",
  "updated": "2025-10-21T12:00:00Z",
  "traces": {},
  "collections": {},
  "pinning_services": {
    "pinata": {
      "api_url": "https://api.pinata.cloud",
      "gateway": "https://gateway.pinata.cloud"
    }
  }
}
EOF

git add registry.json
git commit -m "Initialize trace registry"
git push
```

### 2. Publishing Workflow

**Step 1:** Build and record trace (see Python example above)

**Step 2:** Pin to pinning services
```bash
# Using Pinata (commercial service with free tier)
export PINATA_JWT="your_jwt_token"

curl -X POST "https://api.pinata.cloud/pinning/pinByHash" \
  -H "Authorization: Bearer $PINATA_JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "hashToPin": "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi",
    "pinataMetadata": {
      "name": "coalescent_n5_theta1",
      "keyvalues": {
        "model_type": "coalescent",
        "library": "phasic"
      }
    }
  }'

# Or using Web3.Storage (free)
w3 up --name coalescent_n5_theta1 bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi
```

**Step 3:** Submit PR to registry
- Fork `munch-group/phasic-traces`
- Add trace entry to `registry.json`
- Submit PR
- Maintainer reviews and merges

### 3. Institutional Mirror (Aarhus University)

**Setup dedicated IPFS node:**

```bash
# Server: ipfs.birc.au.dk

# Install IPFS
apt-get install ipfs

# Configure for server use
ipfs init --profile server
ipfs config Addresses.Gateway /ip4/0.0.0.0/tcp/8080
ipfs config --json Datastore.StorageMax '"500GB"'

# Pin all phasic traces
ipfs pin add bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi
# ... (or script to pin all from registry)

# Run as systemd service
cat > /etc/systemd/system/ipfs.service <<'EOF'
[Unit]
Description=IPFS Daemon
After=network.target

[Service]
User=ipfs
ExecStart=/usr/local/bin/ipfs daemon
Restart=always

[Install]
WantedBy=multi-user.target
EOF

systemctl enable ipfs
systemctl start ipfs

# Configure nginx reverse proxy for HTTPS
server {
    server_name ipfs.birc.au.dk;
    location / {
        proxy_pass http://localhost:8080;
    }
}
```

### 4. Monitoring & Maintenance

**Check pin status:**
```python
# scripts/check_pins.py
import requests
from pathlib import Path
import json

registry_path = Path("registry.json")
registry = json.loads(registry_path.read_text())

for trace_id, info in registry['traces'].items():
    cid = info['cid']

    # Check IPFS gateways
    for gateway in ["https://ipfs.io", "https://cloudflare-ipfs.com"]:
        url = f"{gateway}/ipfs/{cid}/trace.json.gz"
        try:
            response = requests.head(url, timeout=10)
            status = "✓" if response.ok else "✗"
        except:
            status = "✗"

        print(f"{trace_id}: {gateway} {status}")
```

**Auto-repin script:**
```bash
#!/bin/bash
# scripts/repin_traces.sh

# Fetch latest registry
curl -s https://raw.githubusercontent.com/munch-group/phasic-traces/main/registry.json > registry.json

# Extract all CIDs
jq -r '.traces[].cid' registry.json | while read cid; do
    echo "Pinning $cid..."
    ipfs pin add --progress $cid
done
```

---

## Migration Path

The IPFS system can coexist with existing `cloud_cache.py`:

### Phase 1: Add IPFS Backend

Extend existing `CloudBackend` hierarchy:

```python
# src/phasic/cloud_cache.py (extend existing)

class IPFSCloudBackend(CloudBackend):
    """IPFS backend compatible with existing CloudBackend API"""

    def __init__(self, daemon_addr="/ip4/127.0.0.1/tcp/5001"):
        self.ipfs = IPFSBackend(daemon_addr)

    def upload_cache(self, local_cache_dir, remote_prefix='', **kwargs):
        # Add directory to IPFS, return CID
        cid = self.ipfs.add(Path(local_cache_dir))
        print(f"Uploaded to IPFS: {cid}")
        return cid

    def download_cache(self, local_cache_dir, remote_cid, **kwargs):
        # Retrieve from IPFS by CID
        self.ipfs.get_directory(remote_cid, Path(local_cache_dir))
```

### Phase 2: Dual-Mode Operation

Users choose backend:

```python
from phasic.cloud_cache import S3Backend, IPFSCloudBackend

# Traditional cloud
backend = S3Backend('my-bucket')
backend.upload_cache('~/.phasic_cache/symbolic')

# Or IPFS
backend = IPFSCloudBackend()
cid = backend.upload_cache('~/.phasic_cache/symbolic')
# Share CID with collaborators!
```

### Phase 3: Gradual Transition

- Keep S3/GCS for large institutional deployments
- Use IPFS for community-contributed traces
- Registry tracks both: `"storage": {"type": "ipfs", "cid": "..."}`

---

## Cost Analysis

### Free Tier Options

- **IPFS Public Network:** $0 (distributed hosting)
- **Web3.Storage:** 5GB free, $0.03/GB thereafter
- **Pinata:** 1GB free, then $20/month for 100GB
- **GitHub:** Free for public repos (registry.json only)

### Institutional Hosting

- **Server:** $50-100/month (VPS with 500GB storage)
- **Bandwidth:** Typically free (IPFS distributes load)
- **Maintenance:** ~2 hours/month

### Estimated Costs for 1000 Traces

- Average trace size: 50KB
- Total: 50MB (fits in all free tiers)
- Large models (100+ vertices): Up to 1MB each
- **Total cost:** $0 using free tiers + institutional mirror

---

## Comparison: IPFS vs. Alternatives

| Feature | IPFS | S3/GCS | GitHub Releases |
|---------|------|--------|-----------------|
| **Storage Cost** | $0 | $0.02/GB/month | $0 (limit: 2GB) |
| **Bandwidth Cost** | $0 | $0.09/GB | $0 (soft limit) |
| **Decentralization** | ✓ | ✗ | ✗ |
| **Censorship Resistance** | ✓ | ✗ | ✗ |
| **Offline-First** | ✓ | ✗ | ✗ |
| **Content Addressing** | ✓ | ✗ | ✗ |
| **Auto Deduplication** | ✓ | ✗ | ✗ |
| **Community Mirrors** | ✓ | ✗ | ✗ |
| **Setup Complexity** | Medium | Low | Low |

---

## Key Design Principles

1. **Offline-first:** Cached traces work without internet
2. **Content-addressed:** SHA-256 checksums for integrity
3. **Lazy loading:** Download on first use, not at install
4. **Metadata-driven:** JSON index enables search without downloading
5. **Versioning:** Track trace format versions for compatibility

---

## Implementation Checklist

### Core Implementation
- [ ] Create `munch-group/phasic-traces` GitHub repository
- [ ] Implement `IPFSBackend` class in `trace_repository.py`
  - [ ] Add auto-start daemon functionality
  - [ ] Implement HTTP gateway fallback
  - [ ] Add initialization check and auto-init
- [ ] Implement `TraceRegistry` class
- [ ] Add tests for IPFS integration
  - [ ] Test auto-start functionality
  - [ ] Test gateway fallback
  - [ ] Test with/without IPFS installed

### IPFS Daemon Management
- [ ] Implement post-install hook for system service setup
  - [ ] macOS launchd support
  - [ ] Linux systemd support
  - [ ] Windows Task Scheduler support
- [ ] Create `scripts/install_ipfs_service.sh` installer script
- [ ] Add auto-start logic to `IPFSBackend.__init__()`
- [ ] Test daemon management on all platforms

### Infrastructure
- [ ] Set up Pinata/Web3.Storage accounts
- [ ] Configure AU institutional IPFS node
- [ ] Document publishing workflow
- [ ] Create initial trace library (coalescent models)

### Package & Documentation
- [ ] Update package dependencies (`ipfshttpclient`, `requests`)
- [ ] Add `[ipfs]` optional dependency group
- [ ] Write user documentation
  - [ ] Three-tier usage guide (no IPFS / auto-start / service)
  - [ ] Installation instructions
  - [ ] Publishing guide for contributors

---

## Timeline Estimate

- **Week 1:** Core Python implementation
  - IPFSBackend with auto-start capability
  - TraceRegistry class
  - HTTP gateway fallback
- **Week 2:** Daemon management
  - Post-install hooks for system services
  - One-line installer script
  - Cross-platform testing (macOS, Linux, Windows)
- **Week 3:** Testing and documentation
  - Unit tests for all auto-start scenarios
  - User documentation for three-tier approach
  - GitHub repository setup
- **Week 4:** Initial trace library creation
  - 5-10 basic coalescent models
  - Pinning service integration
- **Week 5:** Institutional mirror setup
  - AU IPFS node configuration
  - Monitoring scripts
- **Week 6:** Beta testing and public release
  - Collaborator testing
  - Performance benchmarking
  - Public announcement

---

## References

- **IPFS Documentation:** https://docs.ipfs.tech/
- **Pinata API:** https://docs.pinata.cloud/
- **Web3.Storage:** https://web3.storage/docs/
- **PtDAlgorithms Paper:** [Røikjer, Hobolth & Munch (2022)](https://doi.org/10.1007/s11222-022-10155-6)

---

*Last updated: 2025-10-21*
