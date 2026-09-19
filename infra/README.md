# Deploying the MACE inference service

Manifests for the service built from this repository. Everything here refers to
code that is present and buildable.

```
infra/k8s/
├── 00-namespace.yaml
├── 04-services/mace-inference/   deployment, service, configmap, pvc
└── 05-monitoring/                ServiceMonitor for the metrics port
```

Build and deploy:

```bash
docker build -f dockerfiles/mace-inference/Dockerfile -t mace-inference .
kubectl apply -f infra/k8s/00-namespace.yaml
kubectl apply -f infra/k8s/04-services/mace-inference/
kubectl apply -f infra/k8s/05-monitoring/
```

For local work, `docker-compose.yml` at the repository root runs the service
together with the REST gateway, Prometheus and Grafana, and needs no cluster.

## Two things worth knowing before changing these

**The deployment is a single replica on purpose.** Worker reliability lives in
the process and is written to `Mace__BeliefStorePath` at shutdown. A second
replica would learn from whichever feedback reached it, drift away from the
first, and overwrite its file on exit — so scaling out would lose evidence
rather than share it. Serve more load by raising `Mace__PoolSize`, which raises
concurrency inside one process. Sharing reliability across replicas means moving
the belief store behind shared storage, which is a change to `BeliefStore`, not
to a manifest.

**Probes and scrapes must target the metrics port.** gRPC over plaintext
requires HTTP/2, which a kubelet probe and a Prometheus scrape cannot speak, so
the container exposes gRPC on 8080 and everything else on 9090.

## The wider deployment

These manifests were extracted from the ThreatSense stack, where this service
was one of six. The rest of that deployment — sensor ingest, incident
correlation, threat scoring, the operator console, the feedback processor, and
the MQTT, Postgres, Redis, registry and network layers underneath them — is kept
under [`docs/threatsense/`](../docs/threatsense/) as reference. Those manifests
build from source that is not in this repository and cannot be deployed from
here.
