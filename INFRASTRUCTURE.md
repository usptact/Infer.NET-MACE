# ThreatSense: On-Premises Kubernetes Deployment Guide

This document describes how to deploy ThreatSense on a physical server rack using k3s Kubernetes. All services run in containers; all persistent data lives on a NAS via NFS. No internet connectivity is required after initial setup.

For the system architecture and API design, see [THREATSENSE_DESIGN.md](THREATSENSE_DESIGN.md).

---

## Hardware Requirements

### Minimum (small facility, ≤100 sensors)

| Component | Spec | Notes |
|---|---|---|
| node-01 | 8 cores, 32 GB RAM, 200 GB SSD | k3s control+worker |
| node-02 | 8 cores, 32 GB RAM, 200 GB SSD | k3s worker |
| node-03 | 8 cores, 32 GB RAM, 200 GB SSD | k3s worker |
| NAS | 4 TB usable, NFS capable | All persistent data |
| Switch | Managed, VLAN-capable | Sensor / operator isolation |

### Network Layout

```
Sensor VLAN 10  (192.168.10.0/24)  — cameras, mics, access control
Operator VLAN 20 (192.168.20.0/24) — workstations, admin laptops
Cluster LAN      (same as VLAN 10) — inter-node k3s traffic
```

### Static IP Assignments

| Host | IP |
|---|---|
| node-01 | 192.168.10.10 |
| node-02 | 192.168.10.11 |
| node-03 | 192.168.10.12 |
| NAS | 192.168.10.50 |
| MQTT broker (MetalLB) | 192.168.10.200 |
| Ingress / API / Console (MetalLB) | 192.168.10.201 |

---

## Software Stack

| Layer | Technology | Why |
|---|---|---|
| Kubernetes | k3s | Single binary, embedded etcd HA, built-in Traefik + CoreDNS |
| Storage | NFS subdir provisioner | Auto-provision PVs from NAS exports |
| Load balancer | MetalLB (L2 mode) | Bare-metal LoadBalancer, no BGP required |
| Ingress | Traefik (k3s built-in) | HTTP + WebSocket + TLS termination |
| TLS | cert-manager (self-signed) | Internal CA, no internet dependency |
| Monitoring | kube-prometheus-stack | Prometheus + Grafana, optional |
| Registry | registry:2 | Private container image store |
| Message broker | Eclipse Mosquitto | MQTT 3.1.1 + 5.0 |
| Database | PostgreSQL 16 | Incidents, beliefs, audit log |
| Cache/bus | Redis 7 | AOF persistence, Streams for events |

---

## Service Map

```
External (sensor VLAN, MetalLB 192.168.10.200):
  mqtt-broker:1883      — MQTT for cameras, sensors, access control

External (operator network, MetalLB 192.168.10.201 via Traefik):
  /                     → operator-console   (web UI)
  /api/v1/*             → sensor-gateway     (REST API)
  /ws/v1/*              → threat-score-svc   (WebSocket)
  /grafana              → Grafana            (monitoring, optional)

Internal (ClusterIP only):
  redis:6379            postgres:5432
  sensor-gateway:8080/8081
  incident-manager:8080
  mace-inference:8080
  threat-score-svc:8080
  feedback-processor:8080
```

---

## Day-1 Bootstrap

### Prerequisites
- All nodes running Ubuntu 22.04 LTS (or equivalent)
- SSH access from admin workstation to all nodes
- NAS NFS exports created: `/exports/postgres`, `/exports/redis`, `/exports/mqtt`, `/exports/registry`, `/exports/auditlog`
- `helm` and `kubectl` installed on admin workstation

### Step-by-step

```bash
# 1. Clone this repo on your admin workstation
git clone <repo-url> && cd Infer.NET-MACE

# 2. Bootstrap k3s cluster
scp infra/k3s/install-control.sh node-01:~/ && ssh node-01 bash install-control.sh
scp infra/k3s/install-worker.sh  node-02:~/ && ssh node-02 bash install-worker.sh
scp infra/k3s/install-worker.sh  node-03:~/ && ssh node-03 bash install-worker.sh
bash infra/k3s/kubeconfig-export.sh

# 3. Install cluster infrastructure via Helm
helm repo add metallb   https://metallb.github.io/metallb
helm repo add nfs-subdir https://kubernetes-sigs.github.io/nfs-subdir-external-provisioner/
helm repo add jetstack  https://charts.jetstack.io
helm repo update

helm install metallb   metallb/metallb        -n metallb-system --create-namespace
kubectl apply -f infra/helm-values/metallb-ippool.yaml

helm install nfs-provisioner nfs-subdir/nfs-subdir-external-provisioner \
  -f infra/helm-values/nfs-provisioner-values.yaml -n kube-system

helm install cert-manager jetstack/cert-manager \
  --set installCRDs=true -n cert-manager --create-namespace
kubectl apply -f infra/helm-values/cert-manager-issuer.yaml

# 4. Apply namespace
kubectl apply -f infra/k8s/00-namespace.yaml

# 5. Deploy infrastructure (Postgres, Redis, MQTT)
kubectl apply -f infra/k8s/01-infrastructure/
kubectl wait --for=condition=ready pod -l app=postgres -n threatsense --timeout=120s
kubectl wait --for=condition=ready pod -l app=redis    -n threatsense --timeout=60s

# 6. Deploy private registry + configure k3s to trust it
kubectl apply -f infra/k8s/02-registry/
kubectl wait --for=condition=available deployment/registry -n threatsense --timeout=60s
# Copy k3s-registries.yaml to each node and restart k3s agent
for node in node-01 node-02 node-03; do
  scp infra/k8s/02-registry/k3s-registries.yaml $node:/etc/rancher/k3s/registries.yaml
  ssh $node systemctl restart k3s || ssh $node systemctl restart k3s-agent
done

# 7. Build and push service images
make -C infra build push

# 8. Deploy services
kubectl apply -f infra/k8s/04-services/

# 9. Apply network policies and ingress (last — avoids blocking infra traffic)
kubectl apply -f infra/k8s/03-network/

# 10. Verify
kubectl get pods -n threatsense
open https://192.168.10.201
```

### Optional: Monitoring

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install monitoring prometheus-community/kube-prometheus-stack \
  -f infra/helm-values/monitoring-values.yaml -n monitoring --create-namespace
kubectl apply -f infra/k8s/05-monitoring/
```

---

## Day-2 Operations

### Deploy a new image version
```bash
make -C infra push TAG=<git-sha>
make -C infra deploy TAG=<git-sha>
```

### Scale the inference service
```bash
kubectl scale deployment mace-inference --replicas=4 -n threatsense
```

### View logs
```bash
kubectl logs -l app=mace-inference -n threatsense --tail=100 -f
```

### Check sensor reliability (live model state)
```bash
kubectl exec -n threatsense deploy/feedback-processor -- \
  curl -s http://mace-inference:8080/api/v1/model/sensor-reliability | jq .
```

### Backup NAS data
The NAS `/exports` directory contains all state. Back it up with your NAS vendor's snapshot/replication tools. For PostgreSQL consistency, run before backup:
```bash
kubectl exec -n threatsense sts/postgres -- pg_dump -U threatsense threatsense > backup.sql
```

---

## TLS / Browser Setup

The cluster uses a self-signed internal CA. Each operator workstation needs to trust the CA once:
1. Download the CA cert: `kubectl get secret threatsense-ca -n cert-manager -o jsonpath='{.data.tls\.crt}' | base64 -d > threatsense-ca.crt`
2. Install in OS trust store:
   - **macOS**: `sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain threatsense-ca.crt`
   - **Windows**: Import into "Trusted Root Certification Authorities" via `certmgr.msc`
   - **Linux/Chrome**: `certutil -d sql:$HOME/.pki/nssdb -A -t "CT,," -n ThreatSense -i threatsense-ca.crt`

---

## Failure Recovery

| Failure | Recovery |
|---|---|
| Single node down | k3s reschedules pods on remaining nodes; MetalLB re-advertises IPs |
| All nodes down | Restart nodes; k3s auto-restores from etcd; data intact on NAS |
| NAS unavailable | Services run but cannot persist; Redis/Postgres pods restart-loop — restore NAS then pods self-heal |
| Image push failure | Previous `:latest` image continues running; retry `make push` |
| Postgres data corruption | Restore from NAS snapshot; `kubectl delete pod -l app=postgres -n threatsense` to trigger re-mount |
