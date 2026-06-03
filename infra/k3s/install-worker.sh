#!/usr/bin/env bash
# Run on node-02 and node-03 (worker + additional etcd members)
# Usage: K3S_TOKEN=<token> bash install-worker.sh
set -euo pipefail

CONTROL_IP="192.168.10.10"    # node-01 IP
K3S_TOKEN="${K3S_TOKEN:?Set K3S_TOKEN env var to the token from install-control.sh}"

curl -sfL https://get.k3s.io | sh -s - server \
  --server="https://${CONTROL_IP}:6443" \
  --token="${K3S_TOKEN}" \
  --disable=local-storage \
  --node-label="ts/role=worker"

echo "=== k3s worker node ready and joined cluster ==="
