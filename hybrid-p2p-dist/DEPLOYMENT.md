# Deployment and Operations Guide

## Table of Contents

1. [Infrastructure Setup](#infrastructure-setup)
2. [IPFS Configuration](#ipfs-configuration)
3. [Pinning Strategy](#pinning-strategy)
4. [GitHub Integration](#github-integration)
5. [Monitoring and Alerts](#monitoring-and-alerts)
6. [Security Best Practices](#security-best-practices)
7. [Backup and Recovery](#backup-and-recovery)
8. [Scaling Considerations](#scaling-considerations)

## Infrastructure Setup

### Minimum Requirements

**Development Environment:**
- 1 IPFS node (local or remote)
- Python 3.10+
- 10GB storage

**Production Environment:**
- 3+ IPFS nodes (for redundancy)
- IPFS Cluster (3+ nodes)
- Load balancer for IPFS gateway
- Monitoring stack (Prometheus + Grafana)
- 100GB+ storage per node

### Server Specifications

**IPFS Node:**
```
CPU: 4 cores
RAM: 8GB minimum, 16GB recommended
Storage: 500GB SSD (grows with pinned content)
Network: 1Gbps uplink
```

**IPFS Cluster Node:**
```
CPU: 2 cores
RAM: 4GB
Storage: 50GB SSD
Network: 1Gbps
```

## IPFS Configuration

### 1. Install IPFS

```bash
# Download and install go-ipfs
wget https://dist.ipfs.io/go-ipfs/v0.24.0/go-ipfs_v0.24.0_linux-amd64.tar.gz
tar -xvzf go-ipfs_v0.24.0_linux-amd64.tar.gz
cd go-ipfs
sudo bash install.sh
```

### 2. Initialize IPFS

```bash
# Production profile
ipfs init --profile server

# Configuration
ipfs config Addresses.Gateway /ip4/0.0.0.0/tcp/8080
ipfs config Addresses.API /ip4/127.0.0.1/tcp/5001
ipfs config --json API.HTTPHeaders.Access-Control-Allow-Origin '["*"]'
ipfs config --json API.HTTPHeaders.Access-Control-Allow-Methods '["GET", "POST"]'
```

### 3. Systemd Service

Create `/etc/systemd/system/ipfs.service`:

```ini
[Unit]
Description=IPFS Daemon
After=network.target

[Service]
Type=simple
User=ipfs
Environment="IPFS_PATH=/data/ipfs"
ExecStart=/usr/local/bin/ipfs daemon --enable-gc
Restart=always
RestartSec=10
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable ipfs
sudo systemctl start ipfs
```

### 4. IPFS Cluster Setup

```bash
# Install IPFS Cluster
wget https://dist.ipfs.io/ipfs-cluster-service/v1.0.5/ipfs-cluster-service_v1.0.5_linux-amd64.tar.gz
tar -xvzf ipfs-cluster-service_v1.0.5_linux-amd64.tar.gz
sudo cp ipfs-cluster-service/ipfs-cluster-service /usr/local/bin/

# Initialize
ipfs-cluster-service init

# Configure
export CLUSTER_SECRET=$(od -vN 32 -An -tx1 /dev/urandom | tr -d ' \n')
echo $CLUSTER_SECRET > /etc/ipfs-cluster/secret
```

Cluster service (`/etc/systemd/system/ipfs-cluster.service`):

```ini
[Unit]
Description=IPFS Cluster Service
Requires=ipfs.service
After=ipfs.service

[Service]
Type=simple
User=ipfs
Environment="IPFS_CLUSTER_PATH=/data/ipfs-cluster"
ExecStart=/usr/local/bin/ipfs-cluster-service daemon
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Pinning Strategy

### Replication Levels

1. **Critical Content**: 5+ pins across geographically distributed nodes
2. **Important Content**: 3 pins
3. **Standard Content**: 2 pins
4. **Ephemeral Content**: 1 pin (local only)

### Pinning Workflow

```python
# Example pinning automation
from hybrid_p2p import IPFSAdapter

def ensure_redundancy(cid: str, target_pins: int = 3):
    """Ensure content is pinned on multiple nodes."""
    nodes = [
        "http://node1.example.com:5001",
        "http://node2.example.com:5001",
        "http://node3.example.com:5001",
    ]
    
    pinned_count = 0
    
    for node_url in nodes:
        try:
            ipfs = IPFSAdapter(api_url=node_url)
            ipfs.pin_add(cid)
            pinned_count += 1
            
            if pinned_count >= target_pins:
                break
        except Exception as e:
            print(f"Failed to pin on {node_url}: {e}")
    
    return pinned_count
```

## GitHub Integration

### 1. Repository Structure

```
repository/
├── .github/
│   └── workflows/
│       ├── validate-manifests.yml
│       └── pin-on-merge.yml
├── manifests/
│   ├── dataset-v1.0.0.json
│   └── dataset-v1.0.1.json
├── public-keys/
│   ├── alice.pub
│   └── bob.pub
└── README.md
```

### 2. Validation Workflow

`.github/workflows/validate-manifests.yml`:

```yaml
name: Validate Manifests

on:
  pull_request:
    paths:
      - 'manifests/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install hybrid-p2p-dist
      
      - name: Validate manifests
        run: |
          python -c "
from pathlib import Path
from hybrid_p2p import ContentManifest, Verifier

for manifest_path in Path('manifests').glob('*.json'):
    manifest = ContentManifest.model_validate_json(
        manifest_path.read_text()
    )
    Verifier.verify_manifest(manifest)
    print(f'✓ {manifest_path.name} is valid')
          "
```

### 3. Auto-Pin Workflow

`.github/workflows/pin-on-merge.yml`:

```yaml
name: Pin to Cluster

on:
  push:
    branches: [main]
    paths:
      - 'manifests/**'

jobs:
  pin:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Pin new content
        env:
          CLUSTER_URL: ${{ secrets.IPFS_CLUSTER_URL }}
          CLUSTER_AUTH: ${{ secrets.IPFS_CLUSTER_AUTH }}
        run: |
          # Extract CIDs from changed manifests
          # Pin to cluster
          curl -X POST "$CLUSTER_URL/pins/$CID" \
            -H "Authorization: Bearer $CLUSTER_AUTH"
```

## Monitoring and Alerts

### Prometheus Metrics

Key metrics to monitor:

```yaml
# IPFS metrics
- ipfs_peers_total
- ipfs_pinned_objects_total
- ipfs_repo_size_bytes
- ipfs_bandwidth_in_bytes_total
- ipfs_bandwidth_out_bytes_total

# Cluster metrics
- cluster_peers_total
- cluster_pinqueue_size
- cluster_pins_total
- cluster_pins_pin_error_total
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Hybrid P2P Distribution",
    "panels": [
      {
        "title": "IPFS Peers",
        "targets": ["ipfs_peers_total"]
      },
      {
        "title": "Pinned Objects",
        "targets": ["ipfs_pinned_objects_total"]
      },
      {
        "title": "Bandwidth",
        "targets": [
          "rate(ipfs_bandwidth_in_bytes_total[5m])",
          "rate(ipfs_bandwidth_out_bytes_total[5m])"
        ]
      }
    ]
  }
}
```

### Alert Rules

`alert_rules.yml`:

```yaml
groups:
  - name: ipfs_alerts
    rules:
      - alert: IPFSNodeDown
        expr: up{job="ipfs"} == 0
        for: 5m
        annotations:
          summary: "IPFS node is down"
      
      - alert: PinQueueBacklog
        expr: cluster_pinqueue_size > 100
        for: 15m
        annotations:
          summary: "Pin queue has {{ $value }} items"
      
      - alert: PinErrors
        expr: rate(cluster_pins_pin_error_total[5m]) > 0.1
        annotations:
          summary: "High pin error rate"
```

## Security Best Practices

### 1. Key Management

```bash
# Generate production keys with strong password
hybrid-p2p keygen -o /etc/hybrid-p2p/keys -n production --password

# Set restrictive permissions
chmod 600 /etc/hybrid-p2p/keys/production.pem
chmod 644 /etc/hybrid-p2p/keys/production.pub
chown hybrid-p2p:hybrid-p2p /etc/hybrid-p2p/keys/*
```

### 2. Network Security

```bash
# Firewall rules (UFW)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 4001/tcp  # IPFS swarm
sudo ufw allow 4001/udp  # IPFS swarm
sudo ufw deny 5001/tcp   # Block external API access
sudo ufw allow 8080/tcp  # IPFS gateway (if public)
sudo ufw enable
```

### 3. API Authentication

Configure reverse proxy (nginx):

```nginx
location /api/v0 {
    auth_basic "IPFS API";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass http://127.0.0.1:5001;
}
```

### 4. Content Scanning

```python
# Integrate virus scanning before pinning
import subprocess

def scan_content(cid: str) -> bool:
    """Scan content before pinning."""
    # Retrieve content
    ipfs = IPFSAdapter()
    data = ipfs.get_file(cid)
    
    # Scan with ClamAV
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(data)
        tmp.flush()
        
        result = subprocess.run(
            ["clamdscan", tmp.name],
            capture_output=True,
        )
        
        return result.returncode == 0
```

## Backup and Recovery

### 1. IPFS Repository Backup

```bash
#!/bin/bash
# backup-ipfs.sh

BACKUP_DIR="/backups/ipfs"
DATE=$(date +%Y%m%d)

# Stop IPFS
systemctl stop ipfs

# Backup repository
tar -czf "$BACKUP_DIR/ipfs-$DATE.tar.gz" /data/ipfs

# Start IPFS
systemctl start ipfs

# Cleanup old backups (keep 30 days)
find "$BACKUP_DIR" -name "ipfs-*.tar.gz" -mtime +30 -delete
```

### 2. Manifest Recovery

Store manifests in:
1. GitHub repository (primary)
2. IPFS (redundant)
3. S3/blob storage (backup)

### 3. Disaster Recovery Plan

1. **Data Loss**: Re-pin from cluster nodes
2. **Cluster Failure**: Bootstrap new cluster from backups
3. **Key Compromise**: Generate new keys, re-sign manifests

## Scaling Considerations

### Horizontal Scaling

```
                    ┌─────────────┐
                    │ Load Balancer│
                    └──────┬───────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
       ┌────▼───┐     ┌────▼───┐    ┌────▼───┐
       │ IPFS 1 │     │ IPFS 2 │    │ IPFS 3 │
       └────┬───┘     └────┬───┘    └────┬───┘
            │              │              │
            └──────────────┼──────────────┘
                           │
                    ┌──────▼───────┐
                    │ IPFS Cluster │
                    └──────────────┘
```

### Performance Optimization

1. **Pin Subset Strategy**: Don't pin everything on every node
2. **CDN Integration**: Use CloudFlare or similar for gateway
3. **Regional Clusters**: Deploy geographically distributed clusters
4. **Caching Layer**: Redis for metadata, IPFS for content

### Cost Optimization

```python
# Implement tiered storage
TIER_1_NODES = 5  # Critical content
TIER_2_NODES = 3  # Important content
TIER_3_NODES = 1  # Standard content

def pin_with_tier(cid: str, tier: int):
    node_count = {
        1: TIER_1_NODES,
        2: TIER_2_NODES,
        3: TIER_3_NODES,
    }[tier]
    
    ensure_redundancy(cid, target_pins=node_count)
```

## Maintenance Tasks

### Daily
- Check node health
- Monitor pin queue
- Review error logs

### Weekly
- Garbage collection
- Bandwidth analysis
- Security updates

### Monthly
- Repository backup
- Capacity planning
- Performance review

## Troubleshooting

### Common Issues

**1. Pin Queue Stuck:**
```bash
ipfs-cluster-ctl pin ls --status queued
ipfs-cluster-ctl pin rm <cid>
ipfs-cluster-ctl pin add <cid>
```

**2. Node Out of Sync:**
```bash
ipfs-cluster-ctl peers ls
ipfs-cluster-ctl state export > state.json
ipfs-cluster-ctl state import state.json
```

**3. High Memory Usage:**
```bash
# Adjust repository size
ipfs config Datastore.StorageMax 100GB
ipfs repo gc
```

## Support and Resources

- Documentation: https://docs.ipfs.io
- IPFS Cluster: https://cluster.ipfs.io
- Community: https://discuss.ipfs.io
- GitHub: https://github.com/ipfs/go-ipfs
