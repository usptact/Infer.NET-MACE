# MACE Inference — Live Scenario Walkthrough

This document contains the exact API calls used to exercise the inference service end-to-end, including a full multi-step intrusion scenario that demonstrates prior accumulation and online learning across incidents.

All commands target the FastAPI test gateway on `localhost:8000`, which bridges HTTP/JSON → gRPC to `localhost:8080`.

---

## Sensor index reference

| Index | Sensor type | Reading scale |
|---|---|---|
| 0 | `CAMERA_CV` | 0=no person → 3=high-confidence unknown person |
| 1 | `MICROPHONE` | 0=ambient → 3=gunshot detected |
| 2 | `ACCESS_CONTROL` | 0=normal → 3=forced entry |
| 3 | `DOOR_SENSOR` | 0=closed → 3=forced open |
| 4 | `GLASS_BREAK` | 0=none → 3=detected |
| 5 | `BADGE_READER` | 0=valid → 3=invalid off-hours |
| 6 | `TIME_CONTEXT` | 0=business hours → 2=late night |

Threat levels: `0=CLEAR  1=LOW  2=MEDIUM  3=HIGH  4=CRITICAL`

---

## Starting the services

**Build**
```bash
dotnet build MACE/MACE.csproj
```

**Start the gRPC inference service** (repo root; `Development` sets `PoolSize=1` and enables DEBUG logging)
```bash
ASPNETCORE_ENVIRONMENT=Development \
  dotnet run --project MACE --no-build -c Release
# gRPC on :8080 (H2C), Prometheus metrics on :9090
```

Expected startup log:
```
[INF] [InferencePool] Warming up pool: 1 slot(s) × (7 sensor types, 5 threat levels)
[INF] [InferencePool] Slot 1/1 ready  71ms
[INF] [Program      ] MACE Inference Service ready ✓
```

**Start the FastAPI test gateway** (separate terminal, from `test-client/`)
```bash
# One-time setup
pip3 install -r requirements.txt
python3 -m grpc_tools.protoc \
  -I ../MACE/Protos \
  --python_out=generated \
  --grpc_python_out=generated \
  ../MACE/Protos/mace_inference.proto

# Start gateway
MACE_GRPC_TARGET=localhost:8080 python3 -m uvicorn main:app --port 8000
```

Or use Docker Compose for the full stack (repo root):
```bash
docker compose up --build -d
```

---

## Scenario: Late-Night Office Intrusion (02:30 AM)

A six-step scenario covering two active incidents, two operator verdicts, and a final all-clear sweep. Priors accumulate across incidents, showing online learning in action.

Initial sensor priors reflect a mixed-confidence starting state: cameras and microphones are considered more reliable (Beta(1,9) → mean 10% fault rate), badge and door sensors are neutral, and time-context is treated with moderate skepticism (Beta(3,7) → mean 30% fault rate) because "late night = suspicious" is contextual, not authoritative.

---

### Step 1 — First signal: anomalous badge swipe + time context

02:30 AM. A badge reader registers an unusual access event (reading=1, LOW). The time-context sensor reports high ambient risk (reading=3, HIGH for late night). No cameras or microphones have triggered yet.

Only 2 sensors present — exactly at the `MinSensorsForInference=2` threshold, so VMP runs.

```bash
curl -s -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "incident_id": "north-corridor-001",
    "sensor_readings": [-1, -1, -1, -1, -1, 1, 3],
    "theta_priors": [
      {"alpha":1,"beta":9},
      {"alpha":1,"beta":9},
      {"alpha":2,"beta":8},
      {"alpha":2,"beta":8},
      {"alpha":1,"beta":9},
      {"alpha":4,"beta":6},
      {"alpha":3,"beta":7}
    ],
    "phi_priors": [
      {"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]}
    ]
  }' | python3 -m json.tool
```

**Response:**
```json
{
    "incident_id": "north-corridor-001",
    "threat_dist": [0.0414, 0.3517, 0.0414, 0.5241, 0.0414],
    "threat_level": 3,
    "confidence": 0.5241,
    "entropy": 1.1015,
    "sensor_reliability": [
        {"sensor_type_index": 5, "sensor_reading": 1, "fault_prob": 0.6897, "reliability": 0.3103},
        {"sensor_type_index": 6, "sensor_reading": 3, "fault_prob": 0.5172, "reliability": 0.4828}
    ],
    "num_observations": 2,
    "inference_ms": 3902
}
```

> **Analysis:** MACE leans HIGH (52%) but with substantial uncertainty — entropy 1.10 out of max 1.61 nats.
> With only two sensors and both already suspected of being unreliable (badge reader: 69% fault, time context: 52% fault),
> the model has too little information to commit. This is a "watch and wait" signal, not an alert.
> The 2.25s is the Infer.NET Roslyn JIT on first call — all subsequent calls run in ~400–900ms.

---

### Step 2 — Escalation: camera spots person, microphone picks up movement

30 seconds later. Camera reports HIGH (3). Microphone picks up footsteps — MEDIUM (2). Door sensor detects an open door — LOW (1). Pass the `threat_dist` from Step 1 as `warm_start` so VMP initialises from the known posterior.

```bash
curl -s -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "incident_id": "north-corridor-001",
    "sensor_readings": [3, 2, -1, 1, -1, 1, 3],
    "theta_priors": [
      {"alpha":1,"beta":9},
      {"alpha":1,"beta":9},
      {"alpha":2,"beta":8},
      {"alpha":2,"beta":8},
      {"alpha":1,"beta":9},
      {"alpha":4,"beta":6},
      {"alpha":3,"beta":7}
    ],
    "phi_priors": [
      {"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]}
    ],
    "warm_start": [0.0414, 0.3517, 0.0414, 0.5241, 0.0414]
  }' | python3 -m json.tool
```

**Response:**
```json
{
    "incident_id": "north-corridor-001",
    "threat_dist": [0.0012, 0.2206, 0.0568, 0.7201, 0.0012],
    "threat_level": 3,
    "confidence": 0.7201,
    "entropy": 0.7494,
    "sensor_reliability": [
        {"sensor_type_index": 0, "sensor_reading": 3, "fault_prob": 0.2956, "reliability": 0.7044},
        {"sensor_type_index": 1, "sensor_reading": 2, "fault_prob": 0.9444, "reliability": 0.0556},
        {"sensor_type_index": 3, "sensor_reading": 1, "fault_prob": 0.7899, "reliability": 0.2101},
        {"sensor_type_index": 5, "sensor_reading": 1, "fault_prob": 0.8054, "reliability": 0.1946},
        {"sensor_type_index": 6, "sensor_reading": 3, "fault_prob": 0.3368, "reliability": 0.6632}
    ],
    "num_observations": 5,
    "inference_ms": 1454
}
```

> **Analysis:** Confidence jumped 52% → 72%, entropy dropped 1.10 → 0.75. HIGH is now the clear winner.
> Notable MACE behavior: the microphone (94% fault) is being discounted because it reported MEDIUM
> while camera and time context both point HIGH — MACE interprets the disagreement as noise from
> an unreliable source. Camera (70% reliable) and time context (66% reliable) drive the verdict.
> Badge and door sensor (both LOW, both ~80% fault) are also discounted.
> This is the alert threshold: operator is notified.

---

### Step 3 — Operator verdict: TRUE_ALARM on north corridor

Security dispatch confirms an unknown person — real intrusion. Operator closes the incident as `TRUE_ALARM`.

Pass the `fault_prob` posteriors from Step 2's response back to UpdatePriors to compute new Beta parameters. These updated priors will be injected into the next incident's `theta_priors`.

```bash
curl -s -X POST http://localhost:8000/update-priors \
  -H "Content-Type: application/json" \
  -d '{
    "verdict": "TRUE_ALARM",
    "learning_rate": 0.5,
    "sensors": [
      {"sensor_type_index": 0, "sensor_reading": 3, "fault_prob_mean": 0.2956,
       "current_theta": {"alpha": 1.0, "beta": 9.0}},
      {"sensor_type_index": 1, "sensor_reading": 2, "fault_prob_mean": 0.9444,
       "current_theta": {"alpha": 1.0, "beta": 9.0}},
      {"sensor_type_index": 3, "sensor_reading": 1, "fault_prob_mean": 0.7899,
       "current_theta": {"alpha": 2.0, "beta": 8.0}},
      {"sensor_type_index": 5, "sensor_reading": 1, "fault_prob_mean": 0.8054,
       "current_theta": {"alpha": 4.0, "beta": 6.0}},
      {"sensor_type_index": 6, "sensor_reading": 3, "fault_prob_mean": 0.3368,
       "current_theta": {"alpha": 3.0, "beta": 7.0}}
    ]
  }' | python3 -m json.tool
```

**Response:**
```json
{
    "updated_thetas": [
        {"sensor_type_index": 0, "alpha": 1.0,  "beta": 9.3522},
        {"sensor_type_index": 1, "alpha": 1.0,  "beta": 9.0278},
        {"sensor_type_index": 3, "alpha": 2.15, "beta": 8.0},
        {"sensor_type_index": 5, "alpha": 4.15, "beta": 6.0},
        {"sensor_type_index": 6, "alpha": 3.0,  "beta": 7.3316}
    ]
}
```

> **Analysis:**
> - **Camera (0):** β 9.0→9.3522 — correctly flagged HIGH during a real threat; β grows (reliability improving).
> - **Mic (1):** β 9.0→9.0278 — flagged the threat (reading ≥ 2) but barely rewarded; MACE had judged it
>   94% faulty this incident so it gets almost no credit. The update formula weights by `(1 − fault_prob)`.
> - **Door (3) & badge (5):** α incremented by 0.15 each — both reported LOW while a real HIGH threat was present;
>   penalised for missing it.
> - **Time context (6):** β 7.0→7.3316 — correctly reported HIGH during a real threat; reliability slowly improving.

---

### Step 4 — Adjacent zone: glass break in the server room

While security responds to the corridor, a second alarm fires in the server room. Glass break sensor: CRITICAL (4). Camera CV: CRITICAL (4). Microphone: HIGH (3). Time context: still HIGH (3). Using the updated priors from Step 3.

```bash
curl -s -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "incident_id": "server-room-002",
    "sensor_readings": [4, 3, -1, -1, 4, -1, 3],
    "theta_priors": [
      {"alpha":1.0,  "beta":9.3522},
      {"alpha":1.0,  "beta":9.0278},
      {"alpha":2.0,  "beta":8.0},
      {"alpha":2.15, "beta":8.0},
      {"alpha":1.0,  "beta":9.0},
      {"alpha":4.15, "beta":6.0},
      {"alpha":3.0,  "beta":7.3316}
    ],
    "phi_priors": [
      {"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]}
    ]
  }' | python3 -m json.tool
```

**Response:**
```json
{
    "incident_id": "server-room-002",
    "threat_dist": [0.0004, 0.0004, 0.0004, 0.2171, 0.7819],
    "threat_level": 4,
    "confidence": 0.7819,
    "entropy": 0.5325,
    "sensor_reliability": [
        {"sensor_type_index": 0, "sensor_reading": 4, "fault_prob": 0.2345, "reliability": 0.7655},
        {"sensor_type_index": 1, "sensor_reading": 3, "fault_prob": 0.7876, "reliability": 0.2124},
        {"sensor_type_index": 4, "sensor_reading": 4, "fault_prob": 0.2351, "reliability": 0.7649},
        {"sensor_type_index": 6, "sensor_reading": 3, "fault_prob": 0.7993, "reliability": 0.2007}
    ],
    "num_observations": 4,
    "inference_ms": 523
}
```

> **Analysis:** CRITICAL (4) at 78% confidence, entropy 0.53 — the sharpest, most decisive inference so far.
> Camera and glass break both reported CRITICAL and agree completely → both ~76% reliable.
> Microphone (reported HIGH, not CRITICAL) is penalised as 79% faulty — MACE is suspicious of an
> under-reading sensor when two trusted peers agree on a higher level.
> Time context (reported HIGH, not CRITICAL) similarly discounted at 80% faulty.
> The updated camera prior from Step 3 (β=9.352 vs the original β=9.0) contributes slightly higher
> starting trust, which is why camera reaches 76% reliability faster here than in Step 2.

---

### Step 5 — Operator verdict: TRUE_ALARM in server room

Break-in confirmed. Update priors with the Step 4 posteriors.

```bash
curl -s -X POST http://localhost:8000/update-priors \
  -H "Content-Type: application/json" \
  -d '{
    "verdict": "TRUE_ALARM",
    "learning_rate": 0.5,
    "sensors": [
      {"sensor_type_index": 0, "sensor_reading": 4, "fault_prob_mean": 0.2345,
       "current_theta": {"alpha": 1.0, "beta": 9.3522}},
      {"sensor_type_index": 1, "sensor_reading": 3, "fault_prob_mean": 0.7876,
       "current_theta": {"alpha": 1.0, "beta": 9.0278}},
      {"sensor_type_index": 4, "sensor_reading": 4, "fault_prob_mean": 0.2351,
       "current_theta": {"alpha": 1.0, "beta": 9.0}},
      {"sensor_type_index": 6, "sensor_reading": 3, "fault_prob_mean": 0.7994,
       "current_theta": {"alpha": 3.0, "beta": 7.3316}}
    ]
  }' | python3 -m json.tool
```

**Response:**
```json
{
    "updated_thetas": [
        {"sensor_type_index": 0, "alpha": 1.0, "beta": 9.73495},
        {"sensor_type_index": 1, "alpha": 1.0, "beta": 9.134},
        {"sensor_type_index": 4, "alpha": 1.0, "beta": 9.38245},
        {"sensor_type_index": 6, "alpha": 3.0, "beta": 7.4319}
    ]
}
```

> **Analysis:**
> - **Camera (0):** β 9.0→9.3522→9.73495 across two incidents. Consistently reported the correct level;
>   being rapidly validated as reliable. Beta mean fault rate: 10.0% → 9.7% → 9.3%.
> - **Glass break (4):** β 9.0→9.38245 on first incident — immediately trusted after one correct CRITICAL call.
> - **Mic (1):** β 9.0→9.0278→9.134 — receiving credit across both incidents but very slowly, because MACE
>   keeps judging it as high-fault when it under-reports relative to consensus.
> - **Time context (6):** β 7.0→7.3316→7.4319 — steady slow improvement; structural late-night risk signal
>   is being confirmed as genuine.

---

### Step 6 — All-clear sweep: security has cleared the north corridor

Security swept the area. All physical sensors report CLEAR (0). Time context still reports HIGH (it's still 02:30 AM). Using fully accumulated priors from Steps 3 and 5.

```bash
curl -s -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "incident_id": "north-corridor-003",
    "sensor_readings": [0, 0, -1, 0, -1, -1, 3],
    "theta_priors": [
      {"alpha":1.0,  "beta":9.73495},
      {"alpha":1.0,  "beta":9.134},
      {"alpha":2.0,  "beta":8.0},
      {"alpha":2.15, "beta":8.0},
      {"alpha":1.0,  "beta":9.38245},
      {"alpha":4.15, "beta":6.0},
      {"alpha":3.0,  "beta":7.4319}
    ],
    "phi_priors": [
      {"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]}
    ]
  }' | python3 -m json.tool
```

**Response:**
```json
{
    "incident_id": "north-corridor-003",
    "threat_dist": [0.99964, 0.000022, 0.000022, 0.000294, 0.000022],
    "threat_level": 0,
    "confidence": 0.99964,
    "entropy": 0.0035,
    "sensor_reliability": [
        {"sensor_type_index": 0, "sensor_reading": 0, "fault_prob": 0.0205, "reliability": 0.9795},
        {"sensor_type_index": 1, "sensor_reading": 0, "fault_prob": 0.0218, "reliability": 0.9782},
        {"sensor_type_index": 3, "sensor_reading": 0, "fault_prob": 0.0514, "reliability": 0.9486},
        {"sensor_type_index": 6, "sensor_reading": 3, "fault_prob": 0.9997, "reliability": 0.0003}
    ],
    "num_observations": 4,
    "inference_ms": 521
}
```

> **Analysis — the most striking result of the scenario:**
>
> CLEAR (0) at **99.96% confidence**, entropy **0.0035** (essentially zero uncertainty).
>
> The time context sensor, which has been quietly accumulating reliability credit across two TRUE_ALARMs,
> is now judged **99.97% faulty** — because three now-trusted physical sensors all agree on CLEAR while
> time context insists HIGH.
>
> This is MACE self-consistency at work: the model learned from prior incidents that camera and mic are
> highly reliable (β≈9.7 and 9.1 → fault rate ~2%), so when all three physical sensors agree on CLEAR,
> time context's contradicting HIGH is treated as noise. The all-clear is authoritative.

---

### Scenario summary

| Step | Incident | Active sensors | Result | Confidence | Entropy |
|------|----------|----------------|--------|------------|---------|
| 1 | north-corridor-001 | badge LOW, time HIGH | HIGH | 52% | 1.10 |
| 2 | north-corridor-001 | +camera HIGH, mic MEDIUM, door LOW | HIGH | 72% | 0.75 |
| 3 | — | Operator: TRUE_ALARM | priors updated | — | — |
| 4 | server-room-002 | camera CRITICAL, glass CRITICAL, mic HIGH, time HIGH | **CRITICAL** | 78% | 0.53 |
| 5 | — | Operator: TRUE_ALARM | priors updated | — | — |
| 6 | north-corridor-003 | camera CLEAR, mic CLEAR, door CLEAR, time HIGH | **CLEAR** | 99.96% | 0.004 |

**Key behavioral observations:**

1. **MACE discounts under-reading sensors.** The microphone repeatedly reported one level below consensus (MEDIUM/HIGH vs HIGH/CRITICAL). Across both incidents it was judged highly faulty, received minimal prior credit, and will continue being questioned until it agrees with peers more consistently.

2. **Prior accumulation compounds reliability.** The camera's β grew from 9.0 to 9.7345 across two correct calls. By Step 6 it is 98% reliable and its CLEAR reading essentially overrides time context alone.

3. **Time context is correctly neutralised on all-clear.** It contributed genuine signal during active incidents (late night = suspicious) but is immediately and correctly overridden (99.97% faulty) the moment trusted physical sensors report CLEAR. Ambient contextual risk cannot block an all-clear from authoritative sensors.

---

## API reference calls

Standalone calls demonstrating individual API behaviours, independent of the intrusion scenario above.

---

### Health check

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

**Response** (pool counts reflect configured `PoolSize`; uptime increases over time):
```json
{
    "status": "healthy",
    "pool_available": 4,
    "pool_total": 4,
    "uptime_seconds": 244
}
```

---

### Infer — simple example (camera + mic + glass break + time context)

Camera HIGH (3), microphone HIGH (3), glass break MEDIUM (2), time context CLEAR (0). Three sensors absent.

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

**Response:**
```json
{
    "incident_id": "lobby-001",
    "threat_dist": [0.0028, 0.0005, 0.0098, 0.9865, 0.0005],
    "threat_level": 3,
    "confidence": 0.9865,
    "entropy": 0.0823,
    "sensor_reliability": [
        {"sensor_type_index": 0, "sensor_reading": 3, "fault_prob": 0.035, "reliability": 0.965},
        {"sensor_type_index": 1, "sensor_reading": 3, "fault_prob": 0.035, "reliability": 0.965},
        {"sensor_type_index": 4, "sensor_reading": 2, "fault_prob": 0.991, "reliability": 0.009},
        {"sensor_type_index": 6, "sensor_reading": 0, "fault_prob": 0.998, "reliability": 0.002}
    ],
    "num_observations": 4,
    "inference_ms": 461
}
```

> Camera and mic both report HIGH and agree with the inferred label — fault_prob ≈ 3.5%.
> Glass break reported MEDIUM and time context reported CLEAR, both contradicting the consensus →
> MACE marks them 99%+ faulty. With the pool pre-warmed at startup, calls run in ~400–500ms.
> The very first request after a cold start may spike to ~4s while Infer.NET Roslyn-compiles the VMP graph.

---

### Infer — same incident, with warm_start

Pass the previous `threat_dist` as `warm_start`. VMP initialises from the known posterior and converges ~3× faster.

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
    ],
    "warm_start": [0.0028, 0.0005, 0.0098, 0.9865, 0.0005]
  }' | python3 -m json.tool
```

**Response:** identical posteriors, slightly faster.
```json
{
    "incident_id": "lobby-001",
    "threat_dist": [0.0028, 0.0005, 0.0098, 0.9865, 0.0005],
    "threat_level": 3,
    "confidence": 0.9865,
    "entropy": 0.0823,
    "sensor_reliability": [
        {"sensor_type_index": 0, "sensor_reading": 3, "fault_prob": 0.035, "reliability": 0.965},
        {"sensor_type_index": 1, "sensor_reading": 3, "fault_prob": 0.035, "reliability": 0.965},
        {"sensor_type_index": 4, "sensor_reading": 2, "fault_prob": 0.991, "reliability": 0.009},
        {"sensor_type_index": 6, "sensor_reading": 0, "fault_prob": 0.998, "reliability": 0.002}
    ],
    "num_observations": 4,
    "inference_ms": 416
}
```

---

### UpdatePriors — TRUE_ALARM after the lobby-001 inference

```bash
curl -s -X POST http://localhost:8000/update-priors \
  -H "Content-Type: application/json" \
  -d '{
    "verdict": "TRUE_ALARM",
    "learning_rate": 0.5,
    "sensors": [
      {"sensor_type_index": 0, "sensor_reading": 3, "fault_prob_mean": 0.035,
       "current_theta": {"alpha": 1.0, "beta": 9.0}},
      {"sensor_type_index": 1, "sensor_reading": 3, "fault_prob_mean": 0.035,
       "current_theta": {"alpha": 1.0, "beta": 9.0}},
      {"sensor_type_index": 4, "sensor_reading": 2, "fault_prob_mean": 0.991,
       "current_theta": {"alpha": 2.0, "beta": 8.0}},
      {"sensor_type_index": 6, "sensor_reading": 0, "fault_prob_mean": 0.998,
       "current_theta": {"alpha": 5.0, "beta": 5.0}}
    ]
  }' | python3 -m json.tool
```

**Response:**
```json
{
    "updated_thetas": [
        {"sensor_type_index": 0, "alpha": 1.0,   "beta": 9.4825},
        {"sensor_type_index": 1, "alpha": 1.0,   "beta": 9.4825},
        {"sensor_type_index": 4, "alpha": 2.0,   "beta": 8.0045},
        {"sensor_type_index": 6, "alpha": 5.15,  "beta": 5.0}
    ]
}
```

> Camera and mic both correctly flagged HIGH → β increases (Beta mean shifts toward 0, more reliable).
> Time context reported CLEAR during a real threat → α increases (Beta mean shifts higher, less reliable).

---

### Infer — fallback path (single sensor, below MinSensorsForInference)

Only one sensor fires. Below `MinSensorsForInference=2`, so VMP is skipped entirely.

```bash
curl -s -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "incident_id": "entrance-001",
    "sensor_readings": [-1, -1, -1, 3, -1, -1, -1],
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

**Response:**
```json
{
    "incident_id": "entrance-001",
    "threat_dist": [0.2, 0.2, 0.2, 0.2, 0.2],
    "threat_level": 3,
    "confidence": 0.2,
    "entropy": 1.6094,
    "sensor_reliability": [],
    "num_observations": 1,
    "inference_ms": 0
}
```

> `threat_level` is the highest observed sensor reading (3=HIGH), `threat_dist` is flat uniform
> (maximum entropy = 1.609 nats), `confidence=0.2 = 1/NumThreatLevels`, `inference_ms=0`.
> The caller can detect the fallback by checking `confidence == 0.2` or `entropy > 1.6`.
