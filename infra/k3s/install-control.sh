#!/usr/bin/env bash
# Run on node-01 (control plane + first etcd member)
set -euo pipefail

NODE_IP="192.168.10.10"   # this node's IP — update if different

curl -sfL https://get.k3s.io | sh -s - server \
  --cluster-init \
  --disable=local-storage \
  --tls-san="${NODE_IP}" \
  --node-label="ts/role=infra"

echo ""
echo "=== k3s control node ready ==="
echo "Join token for worker nodes:"
cat /var/lib/rancher/k3s/server/node-token
echo ""
echo "Copy to admin workstation:"
echo "  scp root@${NODE_IP}:/etc/rancher/k3s/k3s.yaml ~/.kube/config"
echo "  sed -i 's/127.0.0.1/${NODE_IP}/' ~/.kube/config"
