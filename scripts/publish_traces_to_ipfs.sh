#!/bin/bash
# Publish traces to IPFS and update registry
#
# Prerequisites:
#   brew install ipfs
#   ipfs init
#   ipfs daemon &

set -e

TRACES_DIR="/tmp/phasic_traces"
REGISTRY_DIR="/tmp/phasic-traces"

echo "Publishing traces to IPFS..."
echo "="*70

# Check if IPFS daemon is running
if ! ipfs swarm peers &> /dev/null; then
    echo "Error: IPFS daemon not running"
    echo "Start it with: ipfs daemon &"
    exit 1
fi

# Function to publish a trace directory
publish_trace() {
    local trace_dir=$1
    local trace_id=$(basename "$trace_dir")

    echo ""
    echo "Publishing $trace_id..."

    # Add directory to IPFS
    local cid=$(ipfs add -r -Q "$trace_dir")

    echo "  Directory CID: $cid"

    # Get individual file CIDs
    local trace_file_cid=$(ipfs add -Q "$trace_dir/trace.json.gz")
    local metadata_cid=$(ipfs add -Q "$trace_dir/metadata.json")

    echo "  trace.json.gz CID: $trace_file_cid"
    echo "  metadata.json CID: $metadata_cid"

    # Store CIDs for registry update
    echo "$trace_id|$cid|$trace_file_cid|$metadata_cid" >> /tmp/ipfs_cids.txt

    echo "  ✓ Published $trace_id"
}

# Clear previous CID file
rm -f /tmp/ipfs_cids.txt

# Publish each trace
for trace_dir in "$TRACES_DIR"/coalescent_*; do
    if [ -d "$trace_dir" ]; then
        publish_trace "$trace_dir"
    fi
done

echo ""
echo "="*70
echo "All traces published to IPFS!"
echo "="*70
echo ""
echo "CIDs saved to: /tmp/ipfs_cids.txt"
echo ""
echo "Next steps:"
echo "  1. Run: python scripts/update_registry_with_cids.py"
echo "  2. cd /tmp/phasic-traces && git add registry.json"
echo "  3. git commit -m 'Update with real IPFS CIDs'"
echo "  4. git push"
echo "="*70
