#!/bin/bash
# Quick start script for hybrid P2P distribution system
# This script sets up a complete development environment

set -e

echo "======================================"
echo "Hybrid P2P Distribution - Quick Setup"
echo "======================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    echo -e "${RED}Unsupported OS: $OSTYPE${NC}"
    exit 1
fi

echo -e "${GREEN}Detected OS: $OS${NC}"

# Step 1: Check Python version
echo -e "\n${YELLOW}[1/7] Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo -e "${GREEN}✓ Python $PYTHON_VERSION (OK)${NC}"
else
    echo -e "${RED}✗ Python 3.10+ required (found $PYTHON_VERSION)${NC}"
    exit 1
fi

# Step 2: Install system dependencies
echo -e "\n${YELLOW}[2/7] Installing system dependencies...${NC}"
if [ "$OS" = "linux" ]; then
    sudo apt-get update
    sudo apt-get install -y libmagic1 python3-libtorrent wget curl
    echo -e "${GREEN}✓ Installed libmagic and libtorrent${NC}"
elif [ "$OS" = "macos" ]; then
    brew install libmagic libtorrent-rasterbar wget curl
    echo -e "${GREEN}✓ Installed libmagic and libtorrent${NC}"
fi

# Step 3: Install IPFS
echo -e "\n${YELLOW}[3/7] Installing IPFS...${NC}"
if command -v ipfs &> /dev/null; then
    echo -e "${GREEN}✓ IPFS already installed ($(ipfs --version))${NC}"
else
    IPFS_VERSION="v0.24.0"
    IPFS_DIST="go-ipfs_${IPFS_VERSION}_${OS}-amd64.tar.gz"
    
    wget -q "https://dist.ipfs.io/go-ipfs/${IPFS_VERSION}/${IPFS_DIST}"
    tar -xzf "$IPFS_DIST"
    cd go-ipfs
    sudo bash install.sh
    cd ..
    rm -rf go-ipfs "$IPFS_DIST"
    
    echo -e "${GREEN}✓ IPFS installed${NC}"
fi

# Step 4: Initialize IPFS
echo -e "\n${YELLOW}[4/7] Initializing IPFS...${NC}"
if [ -d "$HOME/.ipfs" ]; then
    echo -e "${GREEN}✓ IPFS already initialized${NC}"
else
    ipfs init
    ipfs config Addresses.API /ip4/127.0.0.1/tcp/5001
    ipfs config Addresses.Gateway /ip4/127.0.0.1/tcp/8080
    echo -e "${GREEN}✓ IPFS initialized${NC}"
fi

# Step 5: Install Python package
echo -e "\n${YELLOW}[5/7] Installing hybrid-p2p-dist...${NC}"
pip install -e ".[dev]"
echo -e "${GREEN}✓ Package installed${NC}"

# Step 6: Generate test keys
echo -e "\n${YELLOW}[6/7] Generating test keys...${NC}"
KEYS_DIR="$HOME/.hybrid_p2p/keys"
mkdir -p "$KEYS_DIR"

if [ -f "$KEYS_DIR/test_key.pem" ]; then
    echo -e "${GREEN}✓ Keys already exist${NC}"
else
    hybrid-p2p keygen -o "$KEYS_DIR" -n test_key
    echo -e "${GREEN}✓ Keys generated at $KEYS_DIR${NC}"
fi

# Step 7: Start IPFS daemon
echo -e "\n${YELLOW}[7/7] Starting IPFS daemon...${NC}"
if pgrep -x "ipfs" > /dev/null; then
    echo -e "${GREEN}✓ IPFS daemon already running${NC}"
else
    echo -e "${YELLOW}Starting IPFS daemon in background...${NC}"
    ipfs daemon > /dev/null 2>&1 &
    sleep 3
    
    if pgrep -x "ipfs" > /dev/null; then
        echo -e "${GREEN}✓ IPFS daemon started${NC}"
    else
        echo -e "${RED}✗ Failed to start IPFS daemon${NC}"
        exit 1
    fi
fi

# Create sample content
echo -e "\n${YELLOW}Creating sample content...${NC}"
SAMPLE_DIR="$HOME/.hybrid_p2p/samples"
mkdir -p "$SAMPLE_DIR"

cat > "$SAMPLE_DIR/readme.txt" << EOF
Sample content for hybrid P2P distribution
==========================================

This file was created by the quick setup script.
You can publish it using:

    hybrid-p2p publish readme.txt \\
      --name sample-package \\
      --version 1.0.0 \\
      --uploader-id $(whoami)@example.com \\
      --key ~/.hybrid_p2p/keys/test_key.pem
EOF

cat > "$SAMPLE_DIR/data.json" << EOF
{
  "name": "Sample Dataset",
  "version": "1.0.0",
  "items": [
    {"id": 1, "value": "alpha"},
    {"id": 2, "value": "beta"},
    {"id": 3, "value": "gamma"}
  ]
}
EOF

echo -e "${GREEN}✓ Sample files created in $SAMPLE_DIR${NC}"

# Print success message
echo -e "\n======================================"
echo -e "${GREEN}✓ Setup complete!${NC}"
echo -e "======================================\n"

echo -e "${YELLOW}Quick Start Commands:${NC}"
echo -e ""
echo -e "1. Publish sample content:"
echo -e "   ${GREEN}cd $SAMPLE_DIR${NC}"
echo -e "   ${GREEN}hybrid-p2p publish readme.txt data.json \\${NC}"
echo -e "   ${GREEN}     --name sample-package \\${NC}"
echo -e "   ${GREEN}     --version 1.0.0 \\${NC}"
echo -e "   ${GREEN}     --uploader-id $(whoami)@example.com \\${NC}"
echo -e "   ${GREEN}     --key ~/.hybrid_p2p/keys/test_key.pem \\${NC}"
echo -e "   ${GREEN}     --output manifest.json${NC}"
echo -e ""
echo -e "2. Inspect manifest:"
echo -e "   ${GREEN}hybrid-p2p inspect manifest.json${NC}"
echo -e ""
echo -e "3. Fetch content (in another directory):"
echo -e "   ${GREEN}hybrid-p2p fetch manifest.json -o ./downloads${NC}"
echo -e ""
echo -e "4. Verify content:"
echo -e "   ${GREEN}hybrid-p2p verify manifest.json ./downloads${NC}"
echo -e ""

echo -e "${YELLOW}Docker Setup (optional):${NC}"
echo -e "   ${GREEN}docker-compose up -d${NC}"
echo -e "   This starts IPFS, IPFS Cluster, Prometheus, and Grafana"
echo -e ""

echo -e "${YELLOW}Run Tests:${NC}"
echo -e "   ${GREEN}pytest${NC}"
echo -e ""

echo -e "${YELLOW}Documentation:${NC}"
echo -e "   README.md         - General usage"
echo -e "   DEPLOYMENT.md     - Production deployment"
echo -e "   PROJECT_SUMMARY.md - Architecture overview"
echo -e ""

echo -e "${YELLOW}Resources:${NC}"
echo -e "   IPFS WebUI:    http://127.0.0.1:5001/webui"
echo -e "   IPFS Gateway:  http://127.0.0.1:8080"
echo -e ""

echo -e "${GREEN}Happy distributing! 🚀${NC}\n"
