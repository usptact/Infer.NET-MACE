#!/usr/bin/env bash
# Run on admin workstation to fetch kubeconfig from node-01
set -euo pipefail

NODE_IP="192.168.10.10"
KUBECONFIG_PATH="${HOME}/.kube/config"

mkdir -p "$(dirname "${KUBECONFIG_PATH}")"
scp "root@${NODE_IP}:/etc/rancher/k3s/k3s.yaml" "${KUBECONFIG_PATH}"
sed -i "s/127.0.0.1/${NODE_IP}/" "${KUBECONFIG_PATH}"
chmod 600 "${KUBECONFIG_PATH}"

echo "kubeconfig written to ${KUBECONFIG_PATH}"
kubectl get nodes
