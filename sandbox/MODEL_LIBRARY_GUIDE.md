# Model Library Repository Guide

**For:** Creating and distributing shared symbolic DAG caches

This guide explains how to create a public or organization-wide repository of pre-computed symbolic DAGs for common phase-type distribution models.

---

## Quick Start

### Install a Pre-Computed Library

```python
from ptdalgorithms.cloud_cache import install_model_library

# Install official coalescent models
install_model_library('coalescent')

# Models are now cached and ready to use instantly!
```

Or via CLI:
```bash
ptd-cache install coalescent
```

### Use Cached Models

```python
from ptdalgorithms import Graph

# Build model (uses cache automatically)
g = Graph(callback=coalescent_callback, parameterized=True)
model = Graph.pmf_from_graph(g)  # Instant from cache!

# Evaluate
pdf = model(theta, times)  # No symbolic elimination needed
```

---

## Repository Structure

For a model library repository (e.g., `ptdalgorithms-models`):

```
ptdalgorithms-models/
├── README.md
├── coalescent/
│   ├── n_samples_3.json
│   ├── n_samples_3.meta
│   ├── n_samples_5.json
│   ├── n_samples_5.meta
│   ├── n_samples_10.json
│   ├── n_samples_10.meta
│   └── manifest.json
├── queuing/
│   ├── mm1_queue.json
│   ├── mm1_queue.meta
│   ├── mmk_queue.json
│   ├── mmk_queue.meta
│   └── manifest.json
├── reliability/
│   ├── series_system.json
│   ├── series_system.meta
│   ├── parallel_system.json
│   ├── parallel_system.meta
│   └── manifest.json
└── .github/
    └── workflows/
        └── release.yml
```

### File Formats

#### Symbolic DAG JSON (`*.json`)
```json
{
  "vertices_length": 1024,
  "state_length": 2,
  "param_length": 1,
  "is_acyclic": true,
  "is_discrete": false,
  "vertices": [...],
  "expressions": [...]
}
```

#### Metadata (`*.meta`)
```json
{
  "model_name": "coalescent_n10",
  "description": "Coalescent model for 10 samples",
  "ptdalgorithms_version": "0.21.3",
  "created": "2025-10-13T10:30:00Z",
  "graph_hash": "a3f2e9c8b1d4...",
  "stats": {
    "vertices": 1024,
    "edges": 2048,
    "elimination_time_ms": 234.5
  },
  "parameters": {
    "n_samples": 10,
    "mutation_rate_param": true
  },
  "compatible_versions": ["0.21.x", "0.22.x"]
}
```

#### Manifest (`manifest.json`)
```json
{
  "category": "coalescent",
  "description": "Population genetics coalescent models",
  "version": "1.0.0",
  "models": [
    {
      "name": "n_samples_3",
      "hash": "a3f2e9c8b1d4...",
      "description": "3 sample coalescent",
      "vertices": 256,
      "parameters": ["mutation_rate"]
    },
    {
      "name": "n_samples_10",
      "hash": "f8d7c6a5b4e3...",
      "description": "10 sample coalescent",
      "vertices": 1024,
      "parameters": ["mutation_rate"]
    }
  ],
  "checksums": {
    "n_samples_3.json": "sha256:1234...",
    "n_samples_10.json": "sha256:5678..."
  }
}
```

---

## Creating a Model Library

### Step 1: Build and Cache Models

```python
from ptdalgorithms import Graph
from ptdalgorithms.symbolic_cache import SymbolicCache
import numpy as np

cache = SymbolicCache()

# Build models
models = []

# Coalescent with 3 samples
def coalescent_n3(state, n=3):
    if len(state) == 0:
        return [(np.array([n]), 1.0, [1.0])]
    if state[0] > 1:
        k = state[0]
        return [(np.array([k-1]), 0.0, [k*(k-1)/2])]
    return []

g3 = Graph(callback=lambda s: coalescent_n3(s, n=3), parameterized=True)
_ = Graph.pmf_from_graph(g3)  # Triggers symbolic elimination
hash3 = cache._compute_graph_hash(g3)
models.append(('coalescent_n3', hash3))

# Repeat for n=5, n=10, etc.
# ...

print("Models cached:")
for name, hash_key in models:
    print(f"  {name}: {hash_key}")
```

### Step 2: Export Cache

```python
# Export to organized directory
cache.export_library(
    output_dir='coalescent_models',
    hash_keys=[h for _, h in models]
)

# Directory structure:
# coalescent_models/
# ├── a3f2e9c8....json
# ├── a3f2e9c8....meta
# ├── f8d7c6a5....json
# ├── f8d7c6a5....meta
# └── manifest.json
```

### Step 3: Add Metadata

```python
import json

# Create detailed metadata for each model
for model_name, hash_key in models:
    meta = {
        'model_name': model_name,
        'description': f'Coalescent model: {model_name}',
        'ptdalgorithms_version': '0.21.3',
        'created': '2025-10-13T10:30:00Z',
        'graph_hash': hash_key,
        # ... add more details
    }

    meta_file = f'coalescent_models/{hash_key}.meta'
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)
```

### Step 4: Create Package

```bash
# Create tarball
cd coalescent_models
tar -czf ../coalescent_models_v1.0.0.tar.gz .

# Compute checksum
sha256sum coalescent_models_v1.0.0.tar.gz > coalescent_models_v1.0.0.tar.gz.sha256
```

---

## Publishing to GitHub

### Option 1: GitHub Releases

```bash
# Create repository
gh repo create ptdalgorithms-models --public

# Add files
git add coalescent/ queuing/ reliability/
git commit -m "Add model libraries v1.0.0"
git push

# Create release
gh release create v1.0.0 \
  --title "Model Library v1.0.0" \
  --notes "Initial release with coalescent, queuing, and reliability models" \
  coalescent_models_v1.0.0.tar.gz \
  queuing_models_v1.0.0.tar.gz \
  reliability_models_v1.0.0.tar.gz
```

### Option 2: Git LFS (for large libraries)

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.json"
git lfs track "*.tar.gz"

# Add and commit
git add .gitattributes coalescent/
git commit -m "Add coalescent models"
git push
```

---

## Publishing to Cloud Storage

### S3

```python
from ptdalgorithms.cloud_cache import S3Backend

backend = S3Backend(
    bucket_name='ptdalgorithms-models',
    prefix='v1.0.0/'
)

backend.upload_cache(
    local_cache_dir='coalescent_models',
    remote_prefix='coalescent/'
)
```

### GCS

```python
from ptdalgorithms.cloud_cache import GCSBackend

backend = GCSBackend(
    bucket_name='ptdalgorithms-models',
    prefix='v1.0.0/'
)

backend.upload_cache(
    local_cache_dir='coalescent_models',
    remote_prefix='coalescent/'
)
```

---

## Using Published Libraries

### From GitHub Releases

```python
from ptdalgorithms.cloud_cache import download_from_github_release

download_from_github_release(
    repo='munch-group/ptdalgorithms-models',
    tag='v1.0.0',
    asset_name='coalescent_models_v1.0.0.tar.gz',
    extract_to='~/.ptdalgorithms_cache/symbolic',
    verify_checksum='a3f2e9c8...'  # Optional but recommended
)
```

### From S3

```python
from ptdalgorithms.cloud_cache import S3Backend

backend = S3Backend(
    bucket_name='ptdalgorithms-models',
    prefix='v1.0.0/coalescent/'
)

backend.download_cache(
    local_cache_dir='~/.ptdalgorithms_cache/symbolic'
)
```

### Via CLI

```bash
# Install official library
ptd-cache install coalescent

# Import from URL
ptd-cache import https://example.com/models/coalescent_v1.tar.gz --symbolic

# Import from file
ptd-cache import downloaded_models.tar.gz --symbolic
```

---

## Example: Creating Organization Library

For internal use within an organization:

```python
from ptdalgorithms.symbolic_cache import SymbolicCache
from ptdalgorithms.cloud_cache import S3Backend

# Build and cache organization-specific models
cache = SymbolicCache()

# Export to S3
backend = S3Backend(
    bucket_name='my-org-models',
    prefix='ptdalgorithms/',
    profile='my-aws-profile'
)

backend.upload_cache(
    local_cache_dir=cache.cache_dir,
    remote_prefix='symbolic/'
)

# Team members download:
backend.download_cache(
    local_cache_dir='~/.ptdalgorithms_cache/symbolic'
)
```

---

## Versioning and Compatibility

### Semantic Versioning

Use semantic versioning for model libraries:
- **Major**: Breaking changes (incompatible format)
- **Minor**: New models added
- **Patch**: Bug fixes, metadata updates

### Compatibility Matrix

Document compatible PtDAlgorithms versions in metadata:

```json
{
  "ptdalgorithms_min_version": "0.21.0",
  "ptdalgorithms_max_version": "0.22.x",
  "format_version": "1.0"
}
```

### Migration Guide

When format changes, provide migration script:

```python
# migrate_v1_to_v2.py
def migrate_cache_entry(old_json):
    """Convert v1 format to v2"""
    # ... conversion logic ...
    return new_json
```

---

## Best Practices

### 1. Documentation

Include README.md with:
- Model descriptions
- Parameter explanations
- Usage examples
- Citation information

### 2. Testing

Test models before publishing:

```python
def test_cached_model(hash_key):
    """Verify cached model works"""
    cache = SymbolicCache()
    symbolic_json = cache.load(hash_key)
    assert symbolic_json is not None
    # ... test instantiation ...
```

### 3. Checksums

Always provide SHA-256 checksums:

```bash
sha256sum *.tar.gz > SHA256SUMS
```

### 4. License

Include LICENSE file (e.g., MIT, Apache 2.0)

### 5. Reproducibility

Include model generation scripts:

```python
# build_models.py
"""
Script to rebuild all models in this library.
Ensures reproducibility.
"""
```

---

## Example Repository: ptdalgorithms-models

See reference implementation at:
https://github.com/munch-group/ptdalgorithms-models

Features:
- ✅ Pre-computed models for common use cases
- ✅ Comprehensive metadata
- ✅ Checksums for verification
- ✅ GitHub Actions for automated releases
- ✅ S3 mirror for fast downloads
- ✅ Usage examples and documentation

---

## Automated Release Workflow

GitHub Actions example (`.github/workflows/release.yml`):

```yaml
name: Release Model Library

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Create packages
        run: |
          tar -czf coalescent_models.tar.gz coalescent/
          tar -czf queuing_models.tar.gz queuing/
          tar -czf reliability_models.tar.gz reliability/
          sha256sum *.tar.gz > SHA256SUMS

      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          files: |
            *.tar.gz
            SHA256SUMS
          body_path: RELEASE_NOTES.md
```

---

## Support and Contributions

For questions about creating model libraries:
- Open an issue at: https://github.com/munch-group/ptdalgorithms/issues
- Email: kaspermunch@birc.au.dk

To contribute models to official library:
- Fork: https://github.com/munch-group/ptdalgorithms-models
- Add your models following structure above
- Submit pull request with documentation

---

**Happy model sharing!** 🚀
