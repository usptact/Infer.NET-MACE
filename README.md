# ThreatSense — MACE Inference Service

[![.NET](https://img.shields.io/badge/.NET-10.0-blue.svg)](https://dotnet.microsoft.com/download/dotnet/10.0)
[![Infer.NET](https://img.shields.io/badge/Infer.NET-0.4.2504.701-purple.svg)](https://dotnet.github.io/infer/)
[![gRPC](https://img.shields.io/badge/transport-gRPC-cyan.svg)](https://grpc.io)
[![Python](https://img.shields.io/badge/test--client-FastAPI-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A real-time multi-modal threat assessment system whose inference core is built on the **MACE** (Multi-Annotator Competence Estimation) Bayesian model. Sensors of any modality — cameras, microphones, access-control readers, door sensors — are treated as annotators whose reliability is learned over time. Operator feedback propagates belief updates back into the model, enabling continuous online learning.

The main deliverable in this repository is the **MACE Inference Service**: an ASP.NET Core gRPC pod that can be deployed behind a load balancer, receives per-incident annotation vectors with Bayesian priors, runs Infer.NET VMP inference, and returns full posterior distributions over threat levels. A FastAPI test gateway wraps it with HTTP/JSON endpoints for development and manual testing.

---

## Contents

- [MACE Model](#mace-model)
- [ThreatSense Domain Mapping](#threatsense-domain-mapping)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [gRPC API](#grpc-api)
- [Test Client (FastAPI Gateway)](#test-client-fastapi-gateway)
- [Configuration](#configuration)
- [Unit Tests](#unit-tests)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Design Documents](#design-documents)
- [References](#references)

---

## MACE Model

MACE (Hovy et al., NAACL 2013) is a hierarchical Bayesian model that aggregates noisy annotations from workers with unknown reliability to infer the true label of each item.

### Model Variables

| Variable | Type | Description |
|---|---|---|
| `T[i]` | Discrete | True threat level for incident `i` |
| `S[i,j]` | Bernoulli | Fault indicator — whether sensor `j` produced a faulty reading on incident `i` |
| `θ[j]` | Beta | Sensor `j`'s base fault rate (unreliability) |
| `φ[j]` | Dirichlet | Sensor `j`'s bias when unreliable |

### Generative Process

```
For each incident i:
    T[i] ~ DiscreteUniform(numThreatLevels)

    For each sensor j:
        S[i,j] ~ Bernoulli(θ[j])

        if S[i,j] = 0:   A[i,j] = T[i]           // reliable: reports true threat level
        if S[i,j] = 1:   A[i,j] ~ Discrete(φ[j]) // faulty: reports biased reading
```

Inference uses **Variational Message Passing (VMP)** via Microsoft Infer.NET. VMP is approximate Bayesian inference that converges quickly and handles missing sensor readings (absent sensors) natively through the sparse matrix structure.

### Online Learning

In the batch formulation T, S, θ, φ are inferred jointly from a fixed matrix. In the online setting used here:

1. Each active **incident** is a single item (`numIncidents = 1`).
2. **Priors** `θ[j]` and `φ[j]` are loaded from the Belief Store per request and reflect everything learned from past incidents.
3. After the operator closes an incident with a verdict, the **Feedback Processor** applies a Bayesian Beta update to `θ[j]`, reinforcing reliable sensors and penalising false-alarm contributors.
4. Updated priors are written back to the Belief Store and injected into the next request — completing the online learning loop.

---

## ThreatSense Domain Mapping

| MACE Concept | Physical Security Equivalent |
|---|---|
| Worker `j` | Sensor / modality (camera, mic, door sensor, badge reader…) |
| Item `i` | Threat incident (spatiotemporal event cluster) |
| Annotation `A[i,j]` | Sensor reading `A[i,j]` — sensor `j`'s discretised threat assessment for incident `i` (0–4) |
| True label `T[i]` | True threat level `T[i]`: 0=CLEAR, 1=LOW, 2=MEDIUM, 3=HIGH, 4=CRITICAL |
| Spammer probability `θ[j]` | Sensor `j`'s fault rate (base probability of producing an unreliable reading) |
| Spammer preference `φ[j]` | Sensor `j`'s fault bias — the reading distribution when unreliable |
| Missing annotation | Absent sensor — not covering the incident zone or offline |
| Operator feedback | Gold-standard verdict used to update priors |

---

## Quick Start

### Prerequisites

- [.NET 10.0 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)
- Python 3.11+ (for the test client)
- Docker (optional, for local infra)

### 1 — Build and run the gRPC service

```bash
dotnet build MACE/MACE.csproj
dotnet run --project MACE
# Service starts on http://localhost:8080 (gRPC / H2C)
# Metrics on http://localhost:9090/metrics
```

### 2 — Set up the FastAPI test gateway

```bash
cd test-client
make install     # pip install -r requirements.txt
make generate    # generate Python stubs from the .proto file
make run         # uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

### 3 — Send a test inference request

```bash
curl -s -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "incident_id": "lobby-001",
    "sensor_readings": [3, 3, -1, -1, 2, -1, 0],
    "theta_priors": [
      {"alpha":1,"beta":9}, {"alpha":1,"beta":9}, {"alpha":5,"beta":5},
      {"alpha":5,"beta":5}, {"alpha":2,"beta":8}, {"alpha":5,"beta":5},
      {"alpha":5,"beta":5}
    ],
    "phi_priors": [
      {"pseudocounts":[1,1,1,1,1]}, {"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]}, {"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]}, {"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]}
    ]
  }' | python3 -m json.tool
```

Expected response: `threat_level: 3` (HIGH), `confidence ≥ 0.70`.

---

## Repository Structure

```
.
├── MACE/                          # The gRPC inference service (.NET 10 / Infer.NET)
│   ├── Protos/
│   │   └── mace_inference.proto   # gRPC service contract (Infer, UpdatePriors, Health)
│   ├── Core/
│   │   ├── IInferencePool.cs      # Pool abstraction (enables unit testing without Infer.NET)
│   │   ├── InferenceOptions.cs    # Config POCO (bound from appsettings / env vars)
│   │   └── InferencePool.cs       # Thread-safe pool of MACETrain instances
│   ├── Services/
│   │   ├── MaceInferenceGrpcService.cs  # gRPC service implementation
│   │   └── PriorUpdateService.cs        # Bayesian Beta prior update (pure arithmetic)
│   ├── MACEBase.cs                # Abstract Infer.NET model graph
│   ├── MACETrain.cs               # Batch + online inference; InferOnline() method
│   ├── ModelData.cs               # ModelData + OnlineInferenceResult records
│   ├── Program.cs                 # ASP.NET Core bootstrap (Minimal API + gRPC)
│   └── appsettings.json           # Default configuration
│
├── MACE.Tests/                    # xUnit test suite (104 tests)
│   ├── Core/
│   │   └── InferencePoolTests.cs  # Pool acquire/release, concurrency, dispose semantics
│   ├── Inference/
│   │   └── MACETrainTests.cs      # MACETrain/MACEBase with real Infer.NET VMP
│   ├── Logging/
│   │   └── ShortClassNameEnricherTests.cs  # Serilog enricher
│   ├── Services/
│   │   ├── MaceInferenceGrpcServiceTests.cs  # gRPC service with mock pool
│   │   └── PriorUpdateServiceTests.cs        # Beta prior update arithmetic
│   └── MACE.Tests.csproj
│
├── test-client/                   # FastAPI HTTP→gRPC bridge (Python)
│   ├── main.py                    # FastAPI app; /infer, /update-priors, /health
│   ├── generate_stubs.sh          # Generates Python stubs from the .proto file
│   ├── Makefile                   # make install / generate / run / curl-*
│   └── README.md
│
├── infra/                         # Kubernetes manifests and tooling
│   ├── k3s/                       # Cluster bootstrap scripts
│   ├── helm-values/               # MetalLB, NFS provisioner, cert-manager, monitoring
│   ├── k8s/                       # Namespace, infrastructure, services, network policies
│   ├── docker-compose.yml         # Full local dev stack (no Kubernetes needed)
│   └── Makefile                   # make build / push / deploy
│
├── dockerfiles/                   # Multi-stage Dockerfiles for all six services
│
├── testdata/                      # Batch annotation data for offline MACE experiments
│   ├── sample_data.txt
│   └── adult_data.txt
│
├── global.json                    # Pins .NET SDK to 10.0.x
├── THREATSENSE_DESIGN.md          # Full system design: requirements, APIs, sub-systems
├── MACE_SERVICE_DESIGN.md         # Source-level rationale for every change in this service
└── INFRASTRUCTURE.md              # On-premises Kubernetes deployment guide
```

---

## gRPC API

The service contract is defined in [`MACE/Protos/mace_inference.proto`](MACE/Protos/mace_inference.proto). The three RPCs are:

### `Infer` — hot path

Called by the Incident Manager each time an annotation vector is updated (typically every 200 ms per active incident).

**Key request fields:**

| Field | Type | Description |
|---|---|---|
| `incident_id` | string | Caller-defined incident identifier |
| `sensor_readings` | int32[] | One entry per sensor type; -1 = absent. Length must equal `NumSensorTypes`. |
| `theta_priors` | BetaParams[] | Current `θ` prior per sensor type |
| `phi_priors` | DirichletParams[] | Current `φ` prior per sensor type |
| `warm_start` | double[] | Optional: `threat_dist` from a previous call on the same incident |

**Key response fields:**

| Field | Type | Description |
|---|---|---|
| `threat_dist` | double[] | Full posterior `P(CLEAR…CRITICAL \| evidence)` |
| `threat_level` | int32 | `argmax(threat_dist)` |
| `confidence` | double | `max(threat_dist)` |
| `entropy` | double | Shannon entropy (uncertainty measure) |
| `sensor_reliability` | SensorReliability[] | Per-sensor fault probability and reliability score |
| `inference_ms` | int64 | VMP wall-clock time |

**Warm-start:** On the second and subsequent calls for the same incident, pass the previous `threat_dist` as `warm_start`. VMP initialises from that distribution instead of random, typically halving the iteration count.

---

### `UpdatePriors` — cold path

Called by the Feedback Processor after an operator verdict. Pure arithmetic — does not use Infer.NET.

Implements the update rules from [THREATSENSE_DESIGN.md §7.5](THREATSENSE_DESIGN.md):

| Condition | Effect on Beta(α, β) |
|---|---|
| TRUE_ALARM + sensor flagged (annotation ≥ 2) | `β += lr × (1 − fault_prob)` — reinforce reliability |
| FALSE_ALARM + sensor flagged | `α += lr × fault_prob` — penalise false alarm |
| TRUE_ALARM + sensor missed (annotation < 2) | `α += lr × 0.3` — penalise miss |
| FALSE_ALARM + sensor quiet (annotation < 2) | `β += lr × 0.3` — reinforce correct silence |
| annotation = -1 | No change — sensor was absent |

---

### `Health` — liveness / readiness probe

Returns pool availability and uptime. Kubernetes `readinessProbe` targets this RPC.

---

### Calling with grpcurl

```bash
# Health check
grpcurl -plaintext localhost:8080 mace.MaceInference/Health

# Inference (provide a JSON file or inline)
grpcurl -plaintext -d '{
  "incident_id": "test-001",
  "sensor_readings": [3,3,-1,-1,2,-1,0],
  "theta_priors": [
    {"alpha":1,"beta":9},{"alpha":1,"beta":9},{"alpha":5,"beta":5},
    {"alpha":5,"beta":5},{"alpha":2,"beta":8},{"alpha":5,"beta":5},
    {"alpha":5,"beta":5}
  ],
  "phi_priors": [
    {"pseudocounts":[1,1,1,1,1]},{"pseudocounts":[1,1,1,1,1]},
    {"pseudocounts":[1,1,1,1,1]},{"pseudocounts":[1,1,1,1,1]},
    {"pseudocounts":[1,1,1,1,1]},{"pseudocounts":[1,1,1,1,1]},
    {"pseudocounts":[1,1,1,1,1]}
  ]
}' localhost:8080 mace.MaceInference/Infer
```

---

## Test Client (FastAPI Gateway)

The test client in `test-client/` is a FastAPI application that bridges HTTP/JSON → gRPC, so any HTTP client (curl, Postman, browser) can reach the inference pod without a gRPC-aware tool.

```
test-client/main.py
├── POST /infer           → MaceInference.Infer
├── POST /update-priors   → MaceInference.UpdatePriors
├── GET  /health          → MaceInference.Health
└── GET  /docs            Swagger UI (auto-generated by FastAPI)
```

```bash
cd test-client

make install    # pip install -r requirements.txt
make generate   # generate mace_inference_pb2*.py from the proto
make run        # start gateway on port 8000

# Smoke tests
make curl-health
make curl-infer
make curl-update
```

Point at a remote pod:
```bash
make run TARGET=192.168.10.100:8080
```

### Sensor type index table

The default 7 sensor types (configurable via `INFERENCE__NUMSENSORTYPES`):

| Index | Sensor type | Annotation scale |
|---|---|---|
| 0 | CAMERA_CV | 0=no person → 3=high-confidence unknown person |
| 1 | MICROPHONE | 0=ambient → 3=gunshot detected |
| 2 | ACCESS_CONTROL | 0=normal → 3=forced entry |
| 3 | DOOR_SENSOR | 0=closed → 3=forced open |
| 4 | GLASS_BREAK | 0=none, 3=detected |
| 5 | BADGE_READER | 0=valid → 3=invalid off-hours |
| 6 | TIME_CONTEXT | 0=business hours → 2=late night |

---

## Configuration

All values live in `MACE/appsettings.json` and can be overridden with environment variables using the standard .NET double-underscore convention.

| Setting | Default | Env var | Description |
|---|---|---|---|
| `NumSensorTypes` | 7 | `INFERENCE__NUMSENSORTYPES` | Must match the annotation vector length |
| `NumThreatLevels` | 5 | `INFERENCE__NUMTHREATLEVELS` | Threat levels: CLEAR=0 … CRITICAL=4 |
| `PoolSize` | 4 | `INFERENCE__POOLSIZE` | Pre-warmed MACETrain instances |
| `MinSensorsForInference` | 2 | `INFERENCE__MINSENSORSFORINFERENCE` | Below this, returns max-annotation fallback |
| `PoolAcquireTimeoutMs` | 5000 | `INFERENCE__POOLACQUIRETIMEOUTMS` | gRPC UNAVAILABLE after this many ms |

**Port:** gRPC on `8080` (H2C — cleartext HTTP/2). Prometheus metrics on `9090`.

**Pool sizing:** `PoolSize` should equal the expected request concurrency per pod. Each slot holds one pre-compiled Infer.NET factor graph (~100 MB resident). Requests queue when all slots are busy; none are rejected until `PoolAcquireTimeoutMs` expires.

---

## Unit Tests

The test suite lives in `MACE.Tests/` and uses **xUnit 2.9**, **FluentAssertions**, and **Moq**. It contains 104 tests organised into four parts.

### Run all tests

```bash
dotnet test MACE.Tests/MACE.Tests.csproj
```

### Run only the fast tests (skip Infer.NET inference)

Tests that invoke real VMP inference are tagged `[Trait("Category", "Integration")]`. Filter them out when you want a sub-second feedback loop:

```bash
dotnet test MACE.Tests/MACE.Tests.csproj --filter "Category!=Integration"
```

This runs Parts 1 and the non-Infer.NET subset of Part 4 in under one second.

### Run a specific part or class

```bash
# All pool tests
dotnet test MACE.Tests/MACE.Tests.csproj --filter "FullyQualifiedName~InferencePoolTests"

# All gRPC service tests
dotnet test MACE.Tests/MACE.Tests.csproj --filter "FullyQualifiedName~MaceInferenceGrpcServiceTests"
```

### Test breakdown

| Part | File | Tests | Speed | What's covered |
|---|---|---|---|---|
| 1 — Pure logic | `Services/PriorUpdateServiceTests.cs` | 33 | <1 ms each | `UpdateTheta` all four verdict×flagged combinations, boundary at `MediumThreshold`, learning-rate scaling; `ParseVerdict` case-insensitivity and error cases |
| 1 — Logging | `Logging/ShortClassNameEnricherTests.cs` | 5 | <1 ms each | Namespace stripping, no-namespace passthrough, missing `SourceContext`, property always named `ShortContext` |
| 2 — Inference core | `Inference/MACETrainTests.cs` | 32 | ~400 ms each | Constructor validation; `SetModelData`/`InitializeLabels` error paths; `InferOnline` output invariants (probs sum to 1, confidence = max, ThreatLevel = argmax, entropy ≥ 0); consensus vs disagreement entropy; warm-start; `InferModelData` shape |
| 3 — Pool | `Core/InferencePoolTests.cs` | 13 | <5 ms each* | Acquire/release `Available` counter; all-slots sequential and concurrent acquisition; pool-exhaustion cancellation; extra task unblocks when slot released; `ObjectDisposedException` on disposed pool; idempotent `Dispose` |
| 4 — gRPC service | `Services/MaceInferenceGrpcServiceTests.cs` | 26 | <10 ms each† | All seven validation branches (`InvalidArgument`); fallback path for `<MinSensorsForInference`; pool exhaustion → `Unavailable`; happy-path response invariants; `UpdatePriors` unknown verdict and default learning rate; `Health` pool state and uptime |

\* Pool creation pays the Infer.NET Roslyn JIT cost once per test class via `IClassFixture`; individual tests run in <5 ms.  
† Part 4 uses a mock `IInferencePool` — no Infer.NET involved. The four happy-path tests that call real VMP are tagged `Integration` and take ~400 ms each.

### Architecture note — `IInferencePool`

`MaceInferenceGrpcService` depends on `IInferencePool` (not the concrete `InferencePool` class) so the gRPC service layer can be tested entirely with a Moq mock, without spinning up Infer.NET. `InferencePool` implements `IInferencePool` and is registered in DI as `AddSingleton<IInferencePool, InferencePool>()`.

---

## Kubernetes Deployment

Full on-premises deployment on a 3-node k3s cluster with NAS-backed NFS storage is documented in [INFRASTRUCTURE.md](INFRASTRUCTURE.md).

The `infra/` directory contains everything needed:

```
infra/
├── k3s/               # Node bootstrap scripts
├── helm-values/       # MetalLB, NFS provisioner, cert-manager, monitoring
├── k8s/               # All Kubernetes manifests
├── docker-compose.yml # Local dev without Kubernetes
└── Makefile           # make build / push / deploy / rollback
```

**Local dev stack** (all services + dependencies, no Kubernetes):

```bash
docker compose -f infra/docker-compose.yml up -d
```

**Inference pod specifics** — two non-obvious requirements handled in the Deployment manifest:

```yaml
# /dev/shm for Infer.NET MKL shared memory
volumes:
- name: dshm
  emptyDir: { medium: Memory, sizeLimit: 256Mi }

# Debian base required — Infer.NET MKL uses glibc (not musl/Alpine)
image: mcr.microsoft.com/dotnet/aspnet:10.0
```

---

## Design Documents

| Document | Contents |
|---|---|
| [THREATSENSE_DESIGN.md](THREATSENSE_DESIGN.md) | Full system design: MACE domain mapping, functional and non-functional requirements, API specifications for all six services, sub-system details, data models, latency budget |
| [MACE_SERVICE_DESIGN.md](MACE_SERVICE_DESIGN.md) | Source-level rationale: exactly what changed in each file and why, thread-safety contract, model instance lifecycle, what the pod deliberately does not do |
| [INFRASTRUCTURE.md](INFRASTRUCTURE.md) | On-premises Kubernetes deployment guide: hardware topology, k3s bootstrap, NFS storage, MetalLB, cert-manager, day-1 runbook, day-2 operations, failure recovery |

---

## Development

### Build

```bash
dotnet build MACE/MACE.csproj
```

### Run the service with verbose logging

```bash
ASPNETCORE_ENVIRONMENT=Development dotnet run --project MACE
# PoolSize is automatically reduced to 1 in Development mode
```

### Regenerate Python stubs after proto changes

```bash
cd test-client && make generate
```

---

## References

1. **MACE Paper** — Dirk Hovy, Taylor Berg-Kirkpatrick, Ashish Vaswani, Eduard Hovy. "Learning Whom to Trust with MACE". *NAACL 2013*. [PDF](http://www.aclweb.org/anthology/N13-1132)
2. **Infer.NET** — Microsoft Research probabilistic programming framework. [dotnet.github.io/infer](https://dotnet.github.io/infer/)
3. **gRPC** — [grpc.io](https://grpc.io)
4. **FastAPI** — [fastapi.tiangolo.com](https://fastapi.tiangolo.com)

---

## License

MIT — see [LICENSE](LICENSE).
