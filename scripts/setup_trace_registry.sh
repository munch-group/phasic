#!/bin/bash
# Setup script for creating phasic-traces GitHub repository
#
# Usage: ./scripts/setup_trace_registry.sh

set -e

REPO_NAME="phasic-traces"
ORG_NAME="munch-group"

echo "Setting up $ORG_NAME/$REPO_NAME repository..."

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is required but not installed."
    echo "Install: brew install gh (macOS) or see https://cli.github.com/"
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo "Error: Not authenticated with GitHub CLI."
    echo "Run: gh auth login"
    exit 1
fi

# Create temporary directory for repository
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

echo "Creating repository structure in $TEMP_DIR..."

# Initialize git repository
git init

# Create initial registry.json
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
    },
    "web3storage": {
      "api_url": "https://api.web3.storage",
      "gateway": "https://w3s.link"
    }
  }
}
EOF

# Create README
cat > README.md <<'EOF'
# PtDAlgorithms Trace Repository

This repository contains the central registry for pre-computed elimination traces for the PtDAlgorithms library (phasic).

## Overview

- **Traces** are stored on IPFS (decentralized, content-addressed storage)
- **Registry** (this repo) maps human-readable names to IPFS CIDs
- **Users** download traces automatically via `phasic.trace_repository`

## Usage

### Download a Trace

```python
from phasic.trace_repository import get_trace

# Download and load trace
trace = get_trace("coalescent_n5_theta1")

# Use with SVGD
from phasic.trace_elimination import trace_to_log_likelihood
from phasic import SVGD

log_lik = trace_to_log_likelihood(trace, observed_times)
svgd = SVGD(log_lik, theta_dim=1, n_particles=100)
results = svgd.fit()
```

### Browse Available Traces

```python
from phasic.trace_repository import TraceRegistry

registry = TraceRegistry()
traces = registry.list_traces(domain="population-genetics")

for t in traces:
    print(f"{t['trace_id']}: {t['description']}")
```

## Contributing a Trace

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed instructions.

Quick overview:

1. Build and record your trace:
   ```python
   from phasic import Graph
   from phasic.trace_elimination import record_elimination_trace

   # Build model
   graph = Graph(...)
   trace = record_elimination_trace(graph, param_length=1)
   ```

2. Publish to IPFS:
   ```python
   from phasic.trace_repository import TraceRegistry

   registry = TraceRegistry()
   cid = registry.publish_trace(
       trace=trace,
       trace_id="my_model",
       metadata={...},
       submit_pr=True
   )
   ```

3. Submit pull request to add your trace to `registry.json`

## Registry Structure

See `registry.json` for the complete schema. Each trace entry contains:

- **CID**: IPFS content identifier
- **Description**: Human-readable description
- **Metadata**: Model type, domain, parameters, etc.
- **Files**: Individual file CIDs (trace.json.gz, metadata.json, etc.)
- **Pins**: Pinning service status

## Pinning Services

To ensure traces remain available, we use multiple pinning services:

- **Pinata**: Commercial pinning service
- **Web3.Storage**: Free tier available
- **Institutional nodes**: University IPFS nodes

## License

All traces are licensed under the MIT License unless otherwise specified in their metadata.

## References

- **PtDAlgorithms Paper**: [Røikjer, Hobolth & Munch (2022)](https://doi.org/10.1007/s11222-022-10155-6)
- **IPFS Documentation**: https://docs.ipfs.tech/
- **Repository**: https://github.com/munch-group/phasic
EOF

# Create CONTRIBUTING guide
cat > CONTRIBUTING.md <<'EOF'
# Contributing Traces to the Repository

Thank you for contributing to the PtDAlgorithms trace repository!

## Prerequisites

1. Install IPFS:
   ```bash
   # macOS
   brew install ipfs

   # Linux
   wget https://dist.ipfs.tech/kubo/v0.24.0/kubo_v0.24.0_linux-amd64.tar.gz
   tar -xvzf kubo_v0.24.0_linux-amd64.tar.gz
   cd kubo
   sudo bash install.sh
   ```

2. Initialize IPFS:
   ```bash
   ipfs init
   ipfs daemon &
   ```

3. Install phasic with IPFS support:
   ```bash
   pip install phasic[ipfs]
   ```

## Publishing a Trace

### 1. Build Your Model

```python
from phasic import Graph
import numpy as np

def your_callback(state):
    # Your model logic
    ...
    return [(next_state, base_weight, [coefficients])]

graph = Graph(
    state_length=...,
    callback=your_callback,
    parameterized=True,
    nr_samples=...
)
```

### 2. Record Trace

```python
from phasic.trace_elimination import record_elimination_trace

trace = record_elimination_trace(graph, param_length=...)
```

### 3. Prepare Metadata

```python
metadata = {
    "model_type": "coalescent",  # or "queuing", "survival", etc.
    "domain": "population-genetics",
    "param_length": 1,
    "vertices": 5,
    "edges": 10,
    "description": "Clear, concise description of your model",
    "created": "2025-10-21",
    "author": "Your Name <email@domain.com>",
    "citation": {
        "text": "Author et al. (2025)",
        "doi": "10.xxxx/xxxxx",
        "url": "https://doi.org/..."
    },
    "tags": ["coalescent", "kingman", "population-genetics"],
    "license": "MIT"
}
```

### 4. Publish to IPFS

```python
from phasic.trace_repository import TraceRegistry

registry = TraceRegistry()

cid = registry.publish_trace(
    trace=trace,
    trace_id="unique_descriptive_name",
    metadata=metadata,
    construction_code=open("build_model.py").read(),  # Optional
    example_code=open("example_usage.py").read(),     # Optional
    submit_pr=True  # Prints PR instructions
)
```

### 5. Submit Pull Request

1. Fork this repository
2. Add your trace entry to `registry.json` (follow the printed format)
3. Submit pull request
4. Maintainer will:
   - Review metadata and description
   - Pin to institutional/commercial services
   - Merge PR

## Trace Entry Format

```json
{
  "your_trace_id": {
    "cid": "bafybeig...",
    "description": "Short description (one sentence)",
    "metadata": {
      "model_type": "coalescent",
      "domain": "population-genetics",
      "param_length": 1,
      "vertices": 5,
      "edges": 10,
      "trace_version": "1.0",
      "created": "2025-10-21",
      "author": "Your Name <email@domain.com>",
      "citation": {
        "text": "Author et al. (2025)",
        "doi": "10.xxxx/xxxxx"
      },
      "tags": ["tag1", "tag2"]
    },
    "files": {
      "trace.json.gz": {
        "cid": "bafybeig.../trace.json.gz",
        "size_bytes": 15432
      }
    },
    "checksum": "sha256:...",
    "license": "MIT"
  }
}
```

## Naming Conventions

- Use lowercase with underscores: `model_type_n_params`
- Include key parameters: `coalescent_n10_theta2`
- Be descriptive but concise: `structured_coalescent_2pop_mig`

## Quality Guidelines

1. **Documentation**: Include clear description and tags
2. **Metadata**: Fill all relevant fields
3. **Testing**: Verify trace works before publishing
4. **Reproducibility**: Include construction code when possible
5. **Citation**: Cite papers/methods used

## Questions?

Open an issue or contact: kaspermunch@birc.au.dk
EOF

# Create .gitignore
cat > .gitignore <<'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
.Python
env/
venv/
*.egg-info/
dist/
build/

# OS
.DS_Store
Thumbs.db

# Editor
.vscode/
.idea/
*.swp
*.swo
*~
EOF

# Initial commit
git add .
git commit -m "Initialize phasic-traces registry

- Create registry.json schema
- Add README and CONTRIBUTING guides
- Setup repository structure"

echo ""
echo "Repository structure created in: $TEMP_DIR"
echo ""
echo "Next steps:"
echo "1. Review the files in $TEMP_DIR"
echo "2. Create repository on GitHub:"
echo "   cd $TEMP_DIR"
echo "   gh repo create $ORG_NAME/$REPO_NAME --public --source=. --remote=origin --push"
echo ""
echo "Or manually:"
echo "   cd $TEMP_DIR"
echo "   git remote add origin git@github.com:$ORG_NAME/$REPO_NAME.git"
echo "   git push -u origin master"
echo ""
