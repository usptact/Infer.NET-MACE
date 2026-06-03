# ThreatSense Infrastructure

Kubernetes manifests, Dockerfiles, and setup scripts for deploying ThreatSense on a physical server rack.

See [../INFRASTRUCTURE.md](../INFRASTRUCTURE.md) for the full deployment guide.

## Quick Reference

```
infra/
├── k3s/                  # cluster bootstrap scripts
├── helm-values/          # Helm chart configuration
├── k8s/
│   ├── 00-namespace.yaml
│   ├── 01-infrastructure/  mqtt / redis / postgres
│   ├── 02-registry/        private container registry
│   ├── 03-network/         ingress, TLS, network policies
│   ├── 04-services/        all ThreatSense application services
│   └── 05-monitoring/      Prometheus ServiceMonitors + Grafana dashboards
├── docker-compose.yml    # local development
└── Makefile              # build / push / deploy

dockerfiles/
├── sensor-gateway/
├── incident-manager/
├── mace-inference/       # NOTE: debian base required (Infer.NET MKL)
├── threat-score-svc/
├── feedback-processor/
└── operator-console/     # node:20 build → nginx:alpine serve
```

## Common Tasks

| Task | Command |
|---|---|
| Build all images | `make -C infra build` |
| Push to registry | `make -C infra push` |
| Deploy/update | `make -C infra deploy TAG=<sha>` |
| View pod status | `make -C infra status` |
| Tail service logs | `make -C infra logs SVC=mace-inference` |
| Local dev stack | `docker compose -f infra/docker-compose.yml up -d` |

## Environment Variables to Change Before Deploying

| File | Variable | Action |
|---|---|---|
| `k8s/01-infrastructure/postgres/secret.yaml` | `POSTGRES_PASSWORD` | Generate new, base64-encode |
| `helm-values/metallb-ippool.yaml` | IP range | Match your sensor VLAN |
| `helm-values/nfs-provisioner-values.yaml` | `nfs.server` | Your NAS IP |
| `k8s/03-network/certificate.yaml` | `ipAddresses` | Your ingress IP |
| `k8s/02-registry/k3s-registries.yaml` | Registry IP | Your ingress IP |
| All `deployment.yaml` files | `image:` | Your registry IP |
