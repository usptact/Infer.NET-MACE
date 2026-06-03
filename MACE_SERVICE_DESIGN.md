# MACE Inference Service: Transformation Design

This document describes what must change, and precisely why, to transform the current batch CLI into a self-contained ASP.NET Core HTTP service that operates as a stateless online inference pod within the ThreatSense architecture.

For system-level context see [THREATSENSE_DESIGN.md](THREATSENSE_DESIGN.md).

---

## 1. Goal

Transform the MACE codebase from a batch CLI tool into a **deployable inference pod** that:

1. Reads configuration on startup (sensor type count, category count, pool size)
2. Pre-warms a pool of Infer.NET model instances
3. Exposes REST endpoints for per-incident VMP inference and Bayesian prior updates
4. Is **fully stateless** — priors and annotations arrive in each request, results return in the response
5. Runs in Docker; scales horizontally behind a load balancer

---

## 2. Current State Audit

| File | Current Role | Fate |
|---|---|---|
| `MACE/Program.cs` | CLI entry: parse args → read CSV → batch inference → write CSV | **DELETE** |
| `MACE/CsvReader.cs` | Batch CSV parser, data-quality validation | **DELETE** |
| `MACE/MACEBase.cs` | Abstract Infer.NET model graph | **KEEP + MODIFY** |
| `MACE/MACETrain.cs` | Concrete batch inference | **KEEP + MODIFY** |
| `MACE/ModelData.cs` | Beta[], Dirichlet[], Discrete[], Bernoulli[][] container | **KEEP + ADD** |
| `MACE/MACE.csproj` | CLI project file, no ASP.NET references | **MODIFY** |
| `MACE/sample_data.txt`, `adult_data.txt`, `true_labels.txt` | Batch test fixtures | **MOVE to `testdata/`** (not in Docker image) |

---

## 3. What Is Deleted and Why

### `Program.cs` — DELETE

The entire entry point assumes batch operation:
- `args[0]` is a file path on disk
- `numItems` is discovered at runtime from the CSV row count
- Inference runs once and the process exits

None of this pattern exists in the service world. The ASP.NET Core bootstrap replaces it entirely.

### `CsvReader.cs` — DELETE

Its only consumer is the old `Program.cs`. The service receives annotations as a JSON array in the HTTP request body. The data-quality checks it contains (minimum 3 workers, duplicate detection) belong upstream in the Sensor Gateway / Incident Manager, not in the inference pod. No part of this file is reusable.

---

## 4. Files That Change

### `MACE/MACEBase.cs` — Modify: add warm-start overload

**What it does today:** Defines the Infer.NET model graph — `_trueLabels`, `_spammerIndicators`, `_theta`, `_phi`, the `item` and `worker` ranges, and prior variable arrays. `CreateModel()` wires priors to random variables. `SetModelData()` sets observed prior values. `InitializeLabels()` breaks symmetry with random point-mass initialization.

**The problem with `InitializeLabels()` for online use:**

It calls `Rand.Int(numCategories)` to assign a random starting label to each item. For a warm-start scenario — where a previous inference cycle already produced a `TDist[0]` posterior for the same incident — we want to initialize from that distribution, not from random. VMP converges in fewer iterations when started near the true solution.

**Change — add an overload:**

```csharp
// BEFORE (keep for backward compatibility):
public void InitializeLabels(int numItems, int numCategories)

// ADD:
public void InitializeLabels(int numItems, int numCategories, Discrete[]? warmStart)
```

When `warmStart` is non-null and its length matches `numItems`, use those distributions instead of `Discrete.PointMass(randomLabel, numCategories)`. When null, fall back to the existing random behavior.

No other changes to `MACEBase.cs`.

---

### `MACE/MACETrain.cs` — Modify: add online-optimized path

**What it does today:** Constructor takes `(int numWorkers, int numItems, int numCategories)`. For online use, `numItems` is always `1` — accepting it as a parameter is a source of misuse. `InferModelData(int[][] data)` runs inference and returns a `ModelData` with full batch arrays (`TDist[0..N-1]`, `SDist[0..N-1][0..M-1]`).

**Four problems for online use:**

| Problem | Impact |
|---|---|
| `numItems` passed at construction | Online pod always uses `1`; the parameter is misleading |
| `InferModelData` requires `int[][]` | Callers must wrap a flat `int[]` in `new int[1][]` every call |
| No warm-start before each inference | Initialization is caller's responsibility; pool reuse loses previous state |
| Return type `ModelData` exposes batch arrays | Callers must remember to index `[0]`; off-by-one risk |

**Changes:**

Add a convenience constructor for the online case:
```csharp
// numItems is always 1 for the online pod
public MACETrain(int numSensorTypes, int numCategories)
    : this(numSensorTypes, numItems: 1, numCategories) { }
```

Add `InferOnline` — the method the service calls on every HTTP request:
```csharp
public OnlineInferenceResult InferOnline(
    int[] annotations,           // length == numSensorTypes; -1 = not observed
    ModelData priors,            // theta + phi priors, indexed by sensor type
    Discrete? warmStart = null)  // optional: TDist[0] from previous cycle
```

Internally this method:
1. Calls `SetModelData(priors)` — sets observed values on `_thetaPriors` / `_phiPriors`
2. Calls `InitializeLabels(1, numCategories, warmStart != null ? new[] { warmStart } : null)` — warm or random start
3. Wraps `annotations` as `new int[1][] { annotations }` for the internal model
4. Delegates to the existing `InferModelData` logic
5. Unwraps `TDist[0]` and `SDist[0]` into an `OnlineInferenceResult`

The existing batch constructor and `InferModelData(int[][] data)` are **kept unchanged** — backward compatibility for the README examples and any future integration tests.

---

### `MACE/ModelData.cs` — Add `OnlineInferenceResult`

**What it does today:** Plain record/class with four public arrays. No behaviour.

**Addition:** A new record that represents a single-incident inference result:

```csharp
public record OnlineInferenceResult(
    Discrete   TDist,        // posterior distribution over threat levels
    Bernoulli[] SDist,       // per-sensor spammer posteriors (length = numSensorTypes)
    int        ThreatLevel,  // argmax(TDist.GetProbs())
    double     Confidence,   // max probability in TDist
    double     Entropy       // Shannon entropy of TDist
);
```

This is the return type of `MACETrain.InferOnline()`. It prevents callers from accidentally indexing `TDist[1]` instead of `TDist[0]`, and it names the derived quantities (argmax, confidence, entropy) so the HTTP handler doesn't have to recompute them.

---

### `MACE/MACE.csproj` — Modify

**SDK change** — brings in the ASP.NET Core framework reference:
```xml
<!-- BEFORE -->
<Project Sdk="Microsoft.NET.Sdk">

<!-- AFTER -->
<Project Sdk="Microsoft.NET.Sdk.Web">
```

**New NuGet packages:**
```xml
<PackageReference Include="Swashbuckle.AspNetCore" Version="6.*" />
<PackageReference Include="prometheus-net.AspNetCore" Version="8.*" />
```

**Remove** the `<None>` items that copy batch test data into the build output — these do not belong in the Docker image:
```xml
<!-- REMOVE both of these -->
<None Include="sample_data.txt">
  <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
</None>
<None Include="true_labels.txt">
  <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
</None>
```

**Keep** the existing Infer.NET references unchanged.

---

## 5. New Files

### `MACE/Program.cs` (full replacement)

ASP.NET Core Minimal API bootstrap. **Minimal API** (not MVC controllers) because:
- The service has ≤5 endpoints — no need for controller boilerplate
- The endpoint code stays inline and is readable in one pass
- This style is the .NET equivalent of FastAPI: terse, explicit, schema-first

What it does:
1. Reads config from `appsettings.json` and environment variables
2. Registers `InferencePool` as `Singleton`
3. Registers `PriorUpdateService` as `Singleton`
4. Adds Swagger/OpenAPI for discoverability in development
5. Adds Prometheus `/metrics` endpoint
6. Maps the four endpoints (`/infer`, `/update-priors`, `/health`, `/metrics`)

---

### `MACE/Core/InferencePool.cs`

**Why it exists:** Infer.NET's `InferenceEngine` is not thread-safe. `Infer<T>()`, `ObservedValue` assignment, and `InitialiseTo()` all mutate internal state. Each concurrent HTTP request needs exclusive ownership of a `MACETrain` instance. Creating a new instance per request is too expensive — `CreateModel()` compiles the entire Infer.NET factor graph, which is the dominant cost.

The pool pays the `CreateModel()` cost once per slot at startup, then leases slots to requests.

```csharp
public sealed class InferencePool : IDisposable
{
    private readonly SemaphoreSlim _semaphore;          // backpressure
    private readonly ConcurrentQueue<MACETrain> _available;
    private readonly int _total;

    // Startup: creates poolSize instances, calls CreateModel() on each
    public InferencePool(int poolSize, int numSensorTypes, int numCategories)

    // Returns a PooledInference (IDisposable) that gives exclusive access to one MACETrain
    public async Task<PooledInference> AcquireAsync(CancellationToken ct = default)

    // Called by PooledInference.Dispose() to return the slot
    internal void Return(MACETrain instance)

    public int Available => _available.Count;
    public int Total     => _total;
}

// Using-statement wrapper — releases the slot on Dispose()
public readonly struct PooledInference : IDisposable
{
    public MACETrain Inferencer { get; }
    private readonly InferencePool _pool;
    public void Dispose() => _pool.Return(Inferencer);
}
```

Usage in the `/infer` handler:
```csharp
using var lease = await pool.AcquireAsync(ctx.RequestAborted);
var result = lease.Inferencer.InferOnline(annotations, priors, warmStart);
```

`PoolSize` is configured via `Inference:PoolSize` (default: 4). Should match the expected concurrency of the pod. Requests queue on the `SemaphoreSlim` when all slots are busy — they are not rejected, they wait. The Kubernetes `deployment.yaml` already sets `cpu: "3000m"` per pod to handle burst inference load.

---

### `MACE/Services/PriorUpdateService.cs`

**Why it exists:** The Bayesian Beta update triggered by operator feedback is pure arithmetic — it does not touch Infer.NET at all. Separating it means it can be unit-tested with zero inference overhead, and the math is explicit and auditable.

```csharp
public sealed class PriorUpdateService
{
    public BetaParameters UpdateTheta(
        BetaParameters current,
        double spammerProbMean,   // S[0][j].GetMean() from a previous InferOnline call
        int annotation,           // what this sensor said; -1 = absent
        Verdict verdict,          // TRUE_ALARM or FALSE_ALARM
        double learningRate = 0.5)
    // Returns a new BetaParameters — immutable, no side effects
}
```

**Update rules** (from THREATSENSE_DESIGN.md §7.5):

| Condition | Update |
|---|---|
| TRUE_ALARM and annotation ≥ MEDIUM (≥2) | `beta  += lr × (1 − spammerProbMean)` — sensor was reliable |
| FALSE_ALARM and annotation ≥ MEDIUM | `alpha += lr × spammerProbMean` — sensor contributed to false alarm |
| TRUE_ALARM and annotation < MEDIUM (<2) | `alpha += lr × 0.3` — sensor missed the threat |
| FALSE_ALARM and annotation < MEDIUM | `beta  += lr × 0.3` — sensor correctly stayed quiet |
| annotation == -1 | no change — sensor was not present for this incident |

`BetaParameters` is a simple `record(double Alpha, double Beta)`.

---

### `MACE/appsettings.json`

```json
{
  "Inference": {
    "NumSensorTypes": 7,
    "NumCategories": 5,
    "PoolSize": 4,
    "MinSensorsForInference": 2,
    "PoolAcquireTimeoutMs": 5000
  },
  "Logging": {
    "LogLevel": {
      "Default": "Information",
      "Microsoft.AspNetCore": "Warning"
    }
  }
}
```

Every value under `"Inference"` maps to `INFERENCE__*` environment variables, which is what the Kubernetes `configmap.yaml` already sets (see `infra/k8s/04-services/mace-inference/configmap.yaml`).

### `MACE/appsettings.Development.json`

```json
{
  "Inference": { "PoolSize": 1 },
  "Logging": { "LogLevel": { "Default": "Debug" } }
}
```

Reduces pool size during local `dotnet run` to save memory.

---

## 6. REST API

### `POST /infer`

Hot path. Incident Manager calls this every time an annotation vector is updated.

**Request:**
```json
{
  "incident_id": "uuid",
  "annotations": [3, 3, -1, -1, 2, -1, 0],
  "theta_priors": [
    {"alpha": 1.0, "beta": 9.0},
    {"alpha": 1.0, "beta": 9.0},
    {"alpha": 5.0, "beta": 5.0},
    {"alpha": 5.0, "beta": 5.0},
    {"alpha": 2.0, "beta": 8.0},
    {"alpha": 5.0, "beta": 5.0},
    {"alpha": 5.0, "beta": 5.0}
  ],
  "phi_priors": [
    {"pseudocounts": [1.0, 1.0, 1.0, 1.0, 1.0]},
    ...
  ],
  "warm_start": null
}
```

- `annotations` length must equal `NumSensorTypes`; -1 = sensor absent
- `theta_priors` and `phi_priors` length must equal `NumSensorTypes`
- `warm_start`: null on first call; `t_dist` from a previous response on the same incident for subsequent calls

**Response 200:**
```json
{
  "incident_id": "uuid",
  "t_dist": [0.01, 0.04, 0.10, 0.847, 0.003],
  "threat_level": 3,
  "confidence": 0.847,
  "entropy": 0.612,
  "sensor_reliability": [
    {"sensor_type_index": 0, "annotation": 3, "spammer_prob": 0.09, "reliability": 0.91},
    {"sensor_type_index": 1, "annotation": 3, "spammer_prob": 0.12, "reliability": 0.88},
    {"sensor_type_index": 4, "annotation": 2, "spammer_prob": 0.21, "reliability": 0.79},
    {"sensor_type_index": 6, "annotation": 0, "spammer_prob": 0.47, "reliability": 0.53}
  ],
  "num_observations": 4,
  "inference_ms": 52
}
```

Only sensors with `annotation != -1` appear in `sensor_reliability`.

**Response 400:** `annotations` wrong length, or `theta_priors` wrong count.
**Response 503:** pool exhausted within `PoolAcquireTimeoutMs` milliseconds.

---

### `POST /update-priors`

Cold path. Feedback Processor calls this after an operator verdict. Does not use Infer.NET.

**Request:**
```json
{
  "verdict": "TRUE_ALARM",
  "learning_rate": 0.5,
  "sensors": [
    {"sensor_type_index": 0, "annotation": 3, "spammer_prob_mean": 0.09,
     "current_theta": {"alpha": 1.0, "beta": 9.0}},
    {"sensor_type_index": 1, "annotation": 3, "spammer_prob_mean": 0.12,
     "current_theta": {"alpha": 1.0, "beta": 9.0}},
    {"sensor_type_index": 2, "annotation": -1, "spammer_prob_mean": 0.0,
     "current_theta": {"alpha": 5.0, "beta": 5.0}}
  ]
}
```

**Response 200:**
```json
{
  "updated_thetas": [
    {"sensor_type_index": 0, "alpha": 1.0, "beta": 9.455},
    {"sensor_type_index": 1, "alpha": 1.0, "beta": 9.440},
    {"sensor_type_index": 2, "alpha": 5.0, "beta": 5.0}
  ]
}
```

Sensor 2 passes through unchanged because its annotation is -1.

---

### `GET /health`

Kubernetes liveness and readiness probe.

**Response 200:**
```json
{"status": "healthy", "pool_available": 3, "pool_total": 4, "uptime_seconds": 3612}
```

**Response 503:** if `pool_available == 0` for longer than the configured timeout (indicates stuck inference or resource exhaustion).

---

### `GET /metrics`

Prometheus text format, via `prometheus-net.AspNetCore`. Exposes:

| Metric | Type | Description |
|---|---|---|
| `mace_infer_duration_seconds` | Histogram | VMP inference wall-clock time |
| `mace_pool_available` | Gauge | Idle pool slots at this instant |
| `mace_infer_requests_total` | Counter | By label: `status={success,timeout,error}` |
| `mace_update_priors_requests_total` | Counter | By `verdict` label |

---

## 7. Thread Safety Contract

Infer.NET's `InferenceEngine` is not thread-safe. The following operations mutate internal state and must not run concurrently on the same instance:

- `InferenceEngine.Infer<T>(variable)` — runs the compiled inference algorithm
- `variable.ObservedValue = x` — sets observed data
- `variable.InitialiseTo(dist)` — sets VMP initialization

**Guarantee:** The `SemaphoreSlim` in `InferencePool` ensures each `MACETrain` instance is held by at most one concurrent request. The `PooledInference` struct uses `IDisposable` to release the lease deterministically.

`PriorUpdateService` has no mutable state and is safe to call concurrently.

---

## 8. Model Instance Lifecycle

```
Startup (once per pool slot):
  instance = new MACETrain(NumSensorTypes, NumCategories)
  instance.CreateModel()   ← compiles Infer.NET factor graph; ~1-3s per slot
  pool.Enqueue(instance)

Per-request POST /infer:
  lease = pool.Acquire()   ← waits if all busy
  lease.Inferencer.SetModelData(priors)
  lease.Inferencer.InitializeLabels(1, NumCategories, warmStart)
  result = lease.Inferencer.InferOnline(annotations)
  lease.Dispose()          ← returns to pool
  return HTTP 200 with result

Per-request POST /update-priors:
  // No pool interaction — pure arithmetic
  updatedPriors = priorUpdateService.UpdateTheta(...)
  return HTTP 200 with updatedPriors
```

`CreateModel()` is called **once per pool slot at startup**. This is the expensive step (~1-3 seconds, JIT + factor-graph compilation). Per-request cost is only VMP iteration (~30-80ms for 7 sensor types, 5 categories).

---

## 9. What the Pod Deliberately Does Not Do

| Capability | Where it lives instead |
|---|---|
| Connect to Redis or PostgreSQL | Incident Manager / Feedback Processor |
| Store incident state | Incident Manager |
| Discretize raw sensor readings | Sensor Gateway |
| Trigger alerts | Threat Score Service |
| Maintain learned priors between restarts | Caller passes priors in each request; Belief Store persists them |

This keeps the pod stateless, horizontally scalable, and independently testable.

---

## 10. Verification Checklist

```bash
# Start the service
dotnet run --project MACE

# 1. Health check — expect 200 with pool_available=4
curl http://localhost:5000/health

# 2. First inference — camera(3) + mic(3) + badge(2) + time(0)
curl -s -X POST http://localhost:5000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "incident_id":"test-001",
    "annotations":[3,3,-1,-1,2,-1,0],
    "theta_priors":[
      {"alpha":1,"beta":9},{"alpha":1,"beta":9},{"alpha":5,"beta":5},
      {"alpha":5,"beta":5},{"alpha":2,"beta":8},{"alpha":5,"beta":5},
      {"alpha":5,"beta":5}],
    "phi_priors":[
      {"pseudocounts":[1,1,1,1,1]},{"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]},{"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]},{"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]}]
  }' | jq .
# Expect: threat_level=3, confidence≥0.70, num_observations=4

# 3. Second inference with warm_start from step 2
# Set warm_start to t_dist from step 2 response
# Expect: same threat_level, inference_ms likely lower (warm VMP init)

# 4. Prior update — TRUE_ALARM, camera and mic both correct
curl -s -X POST http://localhost:5000/update-priors \
  -H "Content-Type: application/json" \
  -d '{
    "verdict":"TRUE_ALARM","learning_rate":0.5,
    "sensors":[
      {"sensor_type_index":0,"annotation":3,"spammer_prob_mean":0.09,
       "current_theta":{"alpha":1,"beta":9}},
      {"sensor_type_index":1,"annotation":3,"spammer_prob_mean":0.12,
       "current_theta":{"alpha":1,"beta":9}}
    ]
  }' | jq .
# Expect: beta increases for indices 0 and 1 (reliability reinforced)

# 5. Swagger UI (development)
open http://localhost:5000/swagger

# 6. Prometheus metrics
curl -s http://localhost:5000/metrics | grep mace_
```

---

## 11. File Disposition Summary

| File | Action |
|---|---|
| `MACE/Program.cs` | **Replace** with ASP.NET Core Minimal API bootstrap |
| `MACE/CsvReader.cs` | **Delete** |
| `MACE/MACEBase.cs` | **Modify** — add `Discrete[]? warmStart` overload to `InitializeLabels` |
| `MACE/MACETrain.cs` | **Modify** — add `MACETrain(int, int)` constructor; add `InferOnline(...)` method |
| `MACE/ModelData.cs` | **Modify** — add `OnlineInferenceResult` record |
| `MACE/MACE.csproj` | **Modify** — SDK to `.Web`; add Swashbuckle + prometheus-net; remove sample data entries |
| `MACE/Core/InferencePool.cs` | **New** |
| `MACE/Services/PriorUpdateService.cs` | **New** |
| `MACE/appsettings.json` | **New** |
| `MACE/appsettings.Development.json` | **New** |
| `testdata/sample_data.txt` etc. | **Move** from `MACE/` — not in Docker image |
