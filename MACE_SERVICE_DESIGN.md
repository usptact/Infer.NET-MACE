# MACE Inference Service: Design Notes

This document records the rationale behind every source-level decision in the
MACE inference service. For system-level context (full six-service architecture,
requirements, API specs for the other services) see [THREATSENSE_DESIGN.md](THREATSENSE_DESIGN.md).

---

## 1. What This Service Does

The MACE inference pod is a **stateless gRPC service** that:

1. Accepts a flat annotation vector (`int[]`, one entry per sensor type, -1 = absent) together with Beta/Dirichlet priors loaded by the caller from the Belief Store.
2. Runs Infer.NET VMP inference to produce a posterior distribution over threat levels and per-sensor spammer probabilities.
3. Returns the result; it stores nothing.
4. Optionally computes updated Beta priors from an operator verdict (pure arithmetic, no Infer.NET).

**Why stateless?** The pod can be scaled horizontally, restarted without data loss, and tested with synthetic inputs without a database. All prior state lives in the Belief Store (Postgres + Redis), owned by the Feedback Processor.

**Why gRPC?** The pod is called service-to-service (Incident Manager → pod) at high frequency with numerical payloads (int arrays, double arrays). Protobuf is more efficient than JSON for this data type, and the generated C# client stubs remove HTTP boilerplate from callers. See the transport-choice discussion in the README.

---

## 2. Source File Audit

### Deleted

| File | Reason |
|---|---|
| `MACE/CsvReader.cs` | Batch CSV parser; replaced by structured JSON-over-gRPC |
| `MACE/Program.cs` (old) | CLI entry point `Main(string[] args)`; replaced by ASP.NET Core bootstrap |

Batch test data (`sample_data.txt`, `adult_data.txt`, `true_labels.txt`) moved to `testdata/` — kept for offline experiments, not bundled in the Docker image.

---

### `MACE/MACEBase.cs` — warm-start overload added

`InitializeLabels(int, int)` previously assigned a random point-mass label to each item for VMP symmetry-breaking. For the online pod, where the same incident is re-inferred as new sensor events arrive, we can warm-start from the previous posterior instead. VMP converges in fewer iterations when started near the true solution.

**Change:** Added `InitializeLabels(int numItems, int numCategories, Discrete[]? warmStart)`. When `warmStart` is non-null and matches `numItems`, those distributions are used directly. Null falls back to the original random behaviour. The original two-argument signature now delegates to this overload with `null`.

---

### `MACE/MACETrain.cs` — online constructor and `InferOnline`

**Problems with the batch API for online use:**

| Problem | Impact |
|---|---|
| Constructor requires `numItems` | Online pod always uses 1; the parameter is misleading |
| `InferModelData(int[][] data)` requires a jagged array | Every call must wrap a flat `int[]` in `new int[1][]` |
| `InferModelData` materialises `ThetaDist` and `PhiDist` | Two arrays allocated and immediately discarded in the online path |
| Return type `ModelData` uses batch arrays | Callers must remember to index `[0]`; risk of off-by-one |

**Changes:**

```csharp
// numItems=1 for the online pod
public MACETrain(int numSensorTypes, int numCategories)
    : this(numSensorTypes, numItems: 1, numCategories) { }
```

```csharp
public OnlineInferenceResult InferOnline(
    int[] annotations,           // length == numSensorTypes; -1 = absent
    ModelData priors,
    Discrete? warmStart = null)
```

`InferOnline` calls `InferenceEngine.Infer<>` only for `_trueLabels` and `_spammerIndicators`, skipping the `ThetaDist`/`PhiDist` extraction that `InferModelData` performs. VMP runs once regardless; the optimisation is allocation only.

The original batch constructor and `InferModelData(int[][] data)` are preserved — backward compatible for offline experiments using `testdata/`.

---

### `MACE/ModelData.cs` — `OnlineInferenceResult` added

```csharp
public record OnlineInferenceResult(
    Discrete    TDist,       // posterior over threat levels (length = NumCategories)
    Bernoulli[] SDist,       // spammer posteriors, one per sensor type
    int         ThreatLevel, // argmax(TDist.GetProbs())
    double      Confidence,  // max probability
    double      Entropy      // Shannon entropy
);
```

Named derived quantities (argmax, confidence, entropy) so the gRPC handler does not recompute them, and so callers never index `TDist[0]` by convention.

---

### `MACE/MACE.csproj` — SDK and packages

```xml
<Project Sdk="Microsoft.NET.Sdk.Web">   <!-- was Microsoft.NET.Sdk -->
```

Added packages:
```xml
<PackageReference Include="Grpc.AspNetCore" Version="2.65.0" />
<PackageReference Include="prometheus-net.AspNetCore" Version="8.2.1" />
```

Added proto build item:
```xml
<Protobuf Include="Protos/mace_inference.proto" GrpcServices="Server" />
```

Removed `<None Include="sample_data.txt" .../>` — test data no longer copied into the build output or Docker image.

---

## 3. New Files

### `MACE/Protos/mace_inference.proto` — service contract

Single source of truth for the gRPC interface. Three RPCs:

| RPC | Path | Notes |
|---|---|---|
| `Infer` | hot path | called by Incident Manager on each annotation update |
| `UpdatePriors` | cold path | called by Feedback Processor after operator verdict |
| `Health` | probe | Kubernetes liveness/readiness |

The C# server stubs are auto-generated at build time from this file. Python stubs for the test client are generated via `test-client/generate_stubs.sh`.

---

### `MACE/Core/InferenceOptions.cs`

Config POCO bound from `"Inference"` in `appsettings.json`. All fields overridable via `INFERENCE__*` environment variables (what Kubernetes ConfigMaps inject).

---

### `MACE/Core/InferencePool.cs` — thread-safety design

Infer.NET's `InferenceEngine` is not thread-safe: `Infer<T>()`, `ObservedValue` assignment, and `InitialiseTo()` all mutate internal state. Each concurrent RPC call needs exclusive ownership of a `MACETrain` instance.

Creating a new instance per request is too expensive because `CreateModel()` compiles the Infer.NET factor graph (JIT compilation of generated C# code, ~1–3 s per slot). The pool pays this cost once at startup.

**Design:**
- `SemaphoreSlim(_poolSize, _poolSize)` provides back-pressure: requests queue rather than being rejected, until `PoolAcquireTimeoutMs` expires and a gRPC `UNAVAILABLE` is returned.
- `PooledInference` is a `readonly struct` implementing `IDisposable`. The `using` pattern ensures the slot is returned even if `InferOnline` throws.
- Static Prometheus gauge `mace_pool_available` is updated after every `Infer` call and `Health` probe.

---

### `MACE/Services/PriorUpdateService.cs` — prior update math

The Bayesian Beta update triggered by operator feedback is pure arithmetic. Separating it from the gRPC service allows independent unit testing and keeps the math explicit and auditable.

Update rules (from THREATSENSE_DESIGN.md §7.5):

| Condition | Update |
|---|---|
| TRUE_ALARM + annotation ≥ MEDIUM (≥2) | `β += lr × (1 − spammerProbMean)` |
| FALSE_ALARM + annotation ≥ MEDIUM | `α += lr × spammerProbMean` |
| TRUE_ALARM + annotation < MEDIUM | `α += lr × 0.3` |
| FALSE_ALARM + annotation < MEDIUM | `β += lr × 0.3` |
| annotation == -1 | no change |

Returns a new immutable `BetaParameters(Alpha, Beta)` record.

---

### `MACE/Services/MaceInferenceGrpcService.cs`

Implements the three RPCs. Notable design points:

**`_uptime` is `static readonly`** — the service class is instantiated per-request by the gRPC framework, so instance fields reset on every call. Uptime must be static to survive across instances.

**Custom Prometheus metrics are static** — prometheus-net requires metrics be registered once with the global registry. Static fields guarantee this.

**Validation order in `Infer`:** lengths → Beta positivity → Dirichlet positivity → warm-start length. Invalid inputs return `INVALID_ARGUMENT` before acquiring a pool slot.

**Fallback path for insufficient sensors:** When fewer than `MinSensorsForInference` sensors contributed, MACE VMP has too little information to be meaningful. The pod returns the max observed annotation as `threat_level` with a uniform (maximum-entropy) `t_dist` and increments `mace_infer_requests_total{status="fallback"}`.

---

### `MACE/Program.cs` (replacement)

ASP.NET Core bootstrap:
- Kestrel configured for cleartext HTTP/2 (H2C) on port 8080 — TLS terminated at the load-balancer level in Kubernetes.
- `InferencePool` forced to construct before the first request via `GetRequiredService<InferencePool>()` — this pre-warms all slots synchronously at startup rather than on the first concurrent burst.
- Prometheus `UseMetricServer(port: 9090)` creates a separate HTTP/1.1 listener so Prometheus can scrape without H2C support.

---

## 4. Thread-Safety Contract

| Resource | Owner | Thread safety |
|---|---|---|
| `MACETrain` instance | `InferencePool` | One instance per concurrent request; `SemaphoreSlim` enforces exclusivity |
| `InferenceEngine` (inside each `MACETrain`) | Not thread-safe | Protected by pool |
| `PriorUpdateService` | Stateless | Safe to call concurrently without locking |
| Prometheus counters/histograms | `static` fields | prometheus-net is thread-safe |

---

## 5. What the Pod Deliberately Does Not Do

| Capability | Where it lives |
|---|---|
| Connect to Redis or PostgreSQL | Caller (Incident Manager / Feedback Processor) |
| Manage incident state | Incident Manager |
| Discretize raw sensor readings | Sensor Gateway |
| Trigger alerts | Threat Score Service |
| Persist priors between restarts | Caller passes priors per request; Belief Store persists them |

---

## 6. Verification

```bash
# Build and start
dotnet run --project MACE
# gRPC on :8080, Prometheus metrics on :9090

# Health via grpcurl
grpcurl -plaintext localhost:8080 mace.MaceInference/Health

# Inference: camera=HIGH(3), mic=HIGH(3), badge=MEDIUM(2), time=CLEAR(0)
grpcurl -plaintext -d '{
  "incident_id": "test-001",
  "annotations": [3,3,-1,-1,2,-1,0],
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
# Expect: threat_level=3, confidence ≥ 0.70

# Prior update after TRUE_ALARM — camera and mic both correctly flagged
grpcurl -plaintext -d '{
  "verdict": "TRUE_ALARM",
  "learning_rate": 0.5,
  "sensors": [
    {"sensor_type_index":0,"annotation":3,"spammer_prob_mean":0.09,
     "current_theta":{"alpha":1,"beta":9}},
    {"sensor_type_index":1,"annotation":3,"spammer_prob_mean":0.12,
     "current_theta":{"alpha":1,"beta":9}}
  ]
}' localhost:8080 mace.MaceInference/UpdatePriors
# Expect: beta increases for sensor 0 and 1

# Prometheus metrics
curl -s localhost:9090/metrics | grep mace_

# FastAPI test gateway (alternative to grpcurl)
cd test-client && make install generate run
open http://localhost:8000/docs
```
