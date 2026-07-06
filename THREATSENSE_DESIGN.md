# ThreatSense: Real-Time Multi-Modal Threat Assessment System
## Design Document

---

## 1. Context and Motivation

Physical security operators face a high-dimensional signal integration problem. A single premises may produce thousands of sensor events per hour from heterogeneous modalities: camera-based computer vision (CV), acoustic detectors, access control systems, intrusion sensors, and time-contextual signals. Each modality has different reliability, different latency, and different relevance depending on the event type.

The core challenge is threefold:
1. **Signal fusion**: Aggregate noisy, partial, asynchronous observations into a coherent threat hypothesis
2. **Reliability learning**: Learn over time which sensors are credible in which contexts
3. **Operator in the loop**: Incorporate slow, expert feedback to improve the model without blocking real-time inference

This document describes **ThreatSense**, a system that uses the MACE (Multi-Annotator Competence Estimation) Bayesian model as its inference core. MACE's crowdsourcing framing maps naturally onto multi-modal sensor fusion: sensors are "annotators," threat incidents are "items," and sensor reliability is the latent variable that gets learned over time.

---

## 2. MACE Domain Mapping

The original MACE model (Hovy et al., NAACL 2013) estimates true labels by aggregating noisy annotations from workers with unknown reliability. The mapping to threat assessment is:

| MACE Concept | Threat Domain Equivalent |
|---|---|
| Worker `j` | Sensor / modality (camera, mic, door sensor, badge reader…) |
| Item `i` | Threat incident (spatiotemporal event cluster) |
| Annotation `A[i,j]` | Discretized threat assessment from sensor `j` for incident `i` |
| True label `T[i]` | Latent threat level for incident `i` |
| Spammer probability `θ[j]` | Sensor `j`'s base false-alarm rate (Beta prior) |
| Spammer preference `φ[j]` | Sensor `j`'s bias direction when unreliable (Dirichlet prior) |
| Missing annotation | Sensor not covering the incident zone / offline |
| Operator feedback | Gold-standard label for closed incidents |

**Key model properties that make MACE suitable:**
- Handles partial observations naturally (sparse annotation matrix — not every sensor sees every incident)
- Provides full posterior `P(threat_level | all_observations)` — not just a point estimate
- Learns sensor reliability through prior propagation between incidents
- Handles conflicting sensor signals probabilistically without ad-hoc weighting rules
- Operator feedback updates `θ` and `φ` priors retroactively, improving future inferences

**Threat level discretization:**
The latent variable `T[i]` takes values in `{0=CLEAR, 1=LOW, 2=MEDIUM, 3=HIGH, 4=CRITICAL}`. Each sensor's raw output is normalized to this 5-level scale via configurable per-sensor thresholds (see Section 7.1).

**Online adaptation of the existing batch implementation:**
The current codebase (`MACETrain.InferModelData`) runs batch VMP inference on a fixed annotation matrix. ThreatSense adapts it to streaming via:
1. Per-incident annotation accumulation buffers (one row per active incident)
2. Prior-fed re-inference: `posteriors(t) → priors(t+1)` using `ThetaDist` and `PhiDist` from `ModelData`
3. Warm-start label initialization from `TDist` of the previous inference cycle to speed VMP convergence

---

## 3. Functional Requirements

### 3.1 Sensor Ingestion
- **FR-01**: Accept sensor events from heterogeneous sources: cameras (CV scores), microphones (acoustic event detections), access control (badge events), door/window sensors (open/forced/glass-break), environmental sensors, and time-of-day context
- **FR-02**: Support push protocols: MQTT, gRPC, REST webhook
- **FR-03**: Normalize raw sensor outputs to the canonical 5-level threat label space using per-sensor configurable thresholds
- **FR-04**: Validate sensor events against registered sensor schema; reject malformed events and emit a validation error

### 3.2 Incident Management
- **FR-05**: Cluster incoming sensor events into incidents by spatial zone and configurable time window (default: 60s)
- **FR-06**: Assign each sensor event to exactly one active incident; create a new incident when no matching one exists
- **FR-07**: Manage incident lifecycle: `OPEN → ASSESSED → ESCALATED → RESOLVED`
- **FR-08**: Support manual incident creation by operators

### 3.3 Threat Inference
- **FR-09**: Run MACE inference for each active incident whenever its annotation matrix is updated
- **FR-10**: Produce and publish a threat score for each incident within 500ms of receiving a new sensor event
- **FR-11**: Express threat as a full posterior distribution `P(level)` plus a scalar `confidence` (max probability)
- **FR-12**: Identify which sensors are credible (low `S[i,j]` posterior) and which are acting as false alarmers for each incident
- **FR-13**: Re-use sensor reliability priors (`θ[j]`, `φ[j]`) learned from past incidents

### 3.4 Alerting
- **FR-14**: Emit real-time alerts when a threat score crosses operator-configured thresholds
- **FR-15**: Support per-zone, per-sensor-type, and global alert threshold configurations
- **FR-16**: Deliver alerts to operator consoles via WebSocket and to external systems via webhook

### 3.5 Operator Feedback
- **FR-17**: Accept operator verdicts on closed incidents: `TRUE_ALARM` or `FALSE_ALARM`
- **FR-18**: Update sensor reliability priors based on operator verdicts (posterior belief propagation)
- **FR-19**: Support partial feedback: operator can label specific sensors as contributing or not contributing to a given incident

### 3.6 Observability
- **FR-20**: Expose per-sensor reliability dashboards showing current `θ[j]` Beta distribution (mean, variance, trend over time)
- **FR-21**: Provide incident history with full audit trail of sensor events, MACE posteriors at each inference step, and operator actions
- **FR-22**: Emit structured logs and metrics compatible with standard observability stacks (Prometheus, OpenTelemetry)

---

## 4. Non-Functional Requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-01 | Latency | Threat score update ≤ 500ms P99 from sensor event receipt |
| NFR-02 | Throughput | Handle ≥ 1,000 sensor events/second per deployment node |
| NFR-03 | Availability | 99.9% uptime; inference service must degrade gracefully if one sensor type becomes unavailable |
| NFR-04 | Durability | All sensor events and MACE posteriors persisted before acknowledgment; no data loss on restart |
| NFR-05 | Scalability | Horizontal scaling of Inference Service; Incident Manager partitioned by zone |
| NFR-06 | Security | mTLS between all internal services; sensor credentials rotated per-device; operator actions authenticated and RBAC-controlled |
| NFR-07 | Auditability | Tamper-evident append-only log of all operator actions and model state changes |
| NFR-08 | Testability | All inference logic unit-testable with synthetic sensor scenarios; replay capability for historical incidents |
| NFR-09 | Explainability | Every threat score must be traceable to contributing sensors and their reliability scores |
| NFR-10 | Configurability | Sensor thresholds, incident time windows, alert thresholds, and prior hyperparameters configurable without code changes |

---

## 5. High-Level System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Physical Sensor Layer                         │
│  [NVIDIA DeepStream / Camera CV]  [Acoustic Detectors]           │
│  [Access Control / Badge Readers] [Door / Glass Break Sensors]   │
│  [Environmental / Time-Context]                                   │
└──────────────────────────┬───────────────────────────────────────┘
                           │  MQTT / gRPC / REST
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Sensor Gateway                               │
│  ┌──────────────┐  ┌───────────────┐  ┌─────────────────────┐   │
│  │ Protocol     │  │ Normalization │  │ Sensor Registry      │   │
│  │ Adapters     │→ │ + Threshold   │→ │ (auth, schema, caps) │   │
│  │ MQTT/gRPC    │  │ Discretizer   │  │                      │   │
│  └──────────────┘  └───────────────┘  └─────────────────────┘   │
└──────────────────────────┬───────────────────────────────────────┘
                           │  Canonical SensorEvent stream
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                     Incident Manager                              │
│  ┌────────────────────┐  ┌──────────────────────────────────┐   │
│  │ Temporal-Spatial   │  │ Annotation Accumulation Buffer    │   │
│  │ Event Clusterer    │→ │ (incident × sensor matrix)        │   │
│  └────────────────────┘  └──────────────────────────────────┘   │
└─────────┬──────────────────────────────────┬─────────────────────┘
          │ Prior request                    │ Updated annotation matrix
          ▼                                  ▼
┌──────────────────┐         ┌───────────────────────────────────┐
│   Belief Store   │         │      MACE Inference Service        │
│                  │         │                                     │
│  θ[j] Beta       │────────►│  Online MACE (Infer.NET VMP)        │
│  φ[j] Dirichlet  │◄────────│  - Prior injection                 │
│  Incident history│  Update │  - Per-incident re-inference       │
│  Posteriors log  │  priors │  - Posterior propagation           │
└──────────────────┘         └─────────────────┬─────────────────┘
                                               │ ModelData posteriors
                                               ▼
                             ┌───────────────────────────────────┐
                             │      Threat Score Service          │
                             │                                     │
                             │  - P(level) → scalar score         │
                             │  - Threshold evaluation            │
                             │  - Alert generation                │
                             │  - WebSocket publisher             │
                             └──────────────┬────────────────────┘
                                            │ Scores + Alerts
                                            ▼
                             ┌───────────────────────────────────┐
                             │        Operator Console            │
                             │                                     │
                             │  - Incident dashboard              │
                             │  - Alert management                │
                             │  - Sensor reliability view         │
                             │  - Feedback submission             │
                             └──────────────┬────────────────────┘
                                            │ TRUE/FALSE alarm verdict
                                            ▼
                             ┌───────────────────────────────────┐
                             │      Feedback Processor            │
                             │  - θ/φ prior update (EM step)      │
                             │  - Posterior backpropagation       │
                             │  - Audit log write                 │
                             └───────────────────────────────────┘
```

### 5.1 Data Flow Summary

1. A sensor detects an event and pushes it to the **Sensor Gateway**
2. The Gateway normalizes, authenticates, and discretizes the raw reading to a threat label (0–4)
3. The **Incident Manager** assigns the event to an active incident (or creates one), updating the annotation matrix
4. The **MACE Inference Service** fetches current sensor priors from the **Belief Store**, runs VMP inference on the updated annotation matrix, and publishes updated posteriors
5. The **Threat Score Service** converts the `T[i]` posterior to a scalar score and confidence, evaluates alert thresholds, and pushes updates to the **Operator Console** via WebSocket
6. When an operator resolves an incident, the **Feedback Processor** updates `θ[j]` and `φ[j]` priors in the **Belief Store** for the relevant sensor types

---

## 6. API Specifications

### 6.1 Sensor Ingest API

**Register a sensor**
```
POST /api/v1/sensors
Body: {
  "sensor_id": "cam-north-01",
  "type": "CAMERA_CV",
  "zone_id": "lobby",
  "capabilities": ["PERSON_DETECTION", "WEAPON_DETECTION"],
  "discretization_thresholds": {
    "PERSON_DETECTION": [0.30, 0.60, 0.85],   // → labels 0,1,2,3
    "WEAPON_DETECTION":  [0.20, 0.50, 0.75]
  },
  "initial_reliability": { "alpha": 2.0, "beta": 8.0 }  // Beta(2,8) → reliable prior
}
Response 201: { "sensor_id": "...", "registered_at": "..." }
```

**Push a sensor event**
```
POST /api/v1/sensors/{sensorId}/events
Body: {
  "event_id": "uuid",
  "timestamp": "ISO8601",
  "detection_type": "PERSON_DETECTION",
  "raw_score": 0.87,
  "metadata": { "bounding_box": [...], "track_id": "..." }
}
Response 202: { "event_id": "...", "incident_id": "...", "queued_at": "..." }
```

**Batch event push (for high-frequency sensors)**
```
POST /api/v1/sensors/{sensorId}/events/batch
Body: { "events": [ ...array of event objects... ] }
Response 202: { "accepted": N, "rejected": M, "errors": [...] }
```

### 6.2 Threat Query API

**Get threat score for an incident**
```
GET /api/v1/incidents/{incidentId}/threat
Response 200: {
  "incident_id": "...",
  "zone_id": "lobby",
  "threat_level": "HIGH",           // argmax of T[i] posterior
  "threat_score": 3,                // integer 0–4
  "confidence": 0.847,              // max probability in T[i]
  "distribution": {                 // full P(level | observations)
    "CLEAR": 0.01, "LOW": 0.04, "MEDIUM": 0.10,
    "HIGH": 0.847, "CRITICAL": 0.003
  },
  "contributing_sensors": [
    { "sensor_id": "cam-north-01", "reliability": 0.91,
      "annotation": "HIGH", "spammer_prob": 0.09 },
    { "sensor_id": "mic-lobby-01", "reliability": 0.76,
      "annotation": "HIGH", "spammer_prob": 0.24 }
  ],
  "inference_timestamp": "...",
  "num_observations": 4
}
```

**Get aggregated zone threat**
```
GET /api/v1/zones/{zoneId}/threat
Response 200: {
  "zone_id": "lobby",
  "active_incidents": [...],
  "worst_case_threat": "HIGH",
  "composite_score": 3.2            // weighted sum across active incidents
}
```

**List active incidents**
```
GET /api/v1/incidents?status=OPEN&zone=lobby&since=ISO8601
Response 200: {
  "incidents": [ { "incident_id": "...", "status": "...",
                   "threat_level": "...", "opened_at": "..." } ],
  "total": N
}
```

### 6.3 Real-Time Streaming API

**WebSocket: threat score stream**
```
WS /ws/v1/threats?zones=lobby,corridor&min_level=LOW

Server push messages:
{
  "type": "THREAT_UPDATE",
  "incident_id": "...",
  "zone_id": "...",
  "threat_level": "HIGH",
  "confidence": 0.847,
  "delta": "+1",                    // change from previous level
  "triggered_by": "mic-lobby-01",
  "timestamp": "..."
}

{
  "type": "ALERT",
  "incident_id": "...",
  "alert_type": "THRESHOLD_CROSSED",
  "threshold": "HIGH",
  "timestamp": "..."
}
```

### 6.4 Operator Feedback API

**Submit incident verdict**
```
POST /api/v1/incidents/{incidentId}/feedback
Body: {
  "verdict": "TRUE_ALARM",          // or FALSE_ALARM
  "true_threat_level": "HIGH",      // operator-resolved gold level; required for
                                    // TRUE_ALARM. FALSE_ALARM is pinned to CLEAR.
  "operator_id": "op-007",
  "notes": "confirmed gunshot on NW camera",
  "sensor_assessments": [           // optional per-sensor override of responsibility
    { "sensor_id": "cam-north-01", "was_correct": true },
    { "sensor_id": "mic-lobby-01", "was_correct": true }
  ]
}
Response 200: {
  "feedback_id": "...",
  "priors_updated": ["cam-north-01", "mic-lobby-01"],
  "reliability_deltas": {
    "cam-north-01": {
      "theta_before": { "alpha": 2.0, "beta": 8.0 },
      "theta_after":  { "alpha": 2.06, "beta": 8.44 },
      "phi_before": [8, 4, 3, 2, 1],
      "phi_after":  [8, 4, 3, 2.06, 1]
    }
  }
}
```

### 6.5 Configuration API

**Update sensor discretization thresholds**
```
PUT /api/v1/sensors/{sensorId}/config
Body: {
  "discretization_thresholds": { "PERSON_DETECTION": [0.25, 0.55, 0.80] },
  "initial_reliability": { "alpha": 1.0, "beta": 9.0 }
}
```

**Update alert thresholds**
```
PUT /api/v1/zones/{zoneId}/alert-config
Body: {
  "alert_on_level": "MEDIUM",
  "confidence_min": 0.70,
  "incident_window_seconds": 90
}
```

### 6.6 Admin / Model API

**Get sensor reliability state**
```
GET /api/v1/model/sensor-reliability
Response 200: {
  "sensors": [
    { "sensor_id": "cam-north-01",
      "theta_alpha": 2.8, "theta_beta": 10.2,
      "reliability_mean": 0.78,
      "incidents_seen": 142,
      "false_alarm_rate": 0.22 }
  ]
}
```

**Replay an incident (for testing / debugging)**
```
POST /api/v1/admin/replay
Body: { "incident_id": "...", "use_live_priors": false }
Response 200: { "replay_id": "...", "result_url": "..." }
```

---

## 7. Sub-System Details

### 7.1 Sensor Gateway

**Responsibility:** Accept raw, heterogeneous sensor events; authenticate and validate; normalize to `CanonicalSensorEvent`; route to the event bus.

**Protocol Adapters:**
- MQTT adapter: subscribes to `sensors/{sensorId}/events` topics
- gRPC adapter: implements `SensorService.PushEvent` RPC
- REST adapter: handles `POST /api/v1/sensors/{sensorId}/events`
- ONVIF/RTSP bridge: for cameras that push analytics metadata via RTSP

**Sensor Registry:**
- Maintains sensor metadata: type, zone, capabilities, credential, discretization config
- Validates incoming events against registered sensor schema
- Tracks sensor health: last-seen timestamp, online/offline status

**Normalization and Discretization Pipeline:**

Each sensor type registers an `IDiscretizer` that maps raw scores to the 0–4 threat label. Examples:

| Sensor Type | Mapping Logic |
|---|---|
| Camera CV (person) | `P(unknown_person)` → thresholds `[0.30, 0.60, 0.85]` → labels `{0,1,2,3}` |
| Weapon detection | `P(weapon)` → thresholds `[0.20, 0.50, 0.75]` → labels `{0,1,2,3}` |
| Gunshot mic | `P(gunshot)` → labels `{0,1,2,3}`; if reverb_flag is set, cap at label 2 |
| Badge reader | `(badge_valid=false AND off_hours)` → label 3; `(invalid AND business_hours)` → label 2 |
| Glass break | confidence > 0.5 → label 3 (always HIGH) |
| Forced door | detected → label 3 |
| Time context | business_hours → 0; weekend → 1; early/late → 1; late_night → 2 |

Discretizer thresholds are stored in the Sensor Registry and updatable at runtime (FR-10).

**Rate limiting:** Per-sensor token bucket prevents burst floods from misconfigured sensors from saturating the inference queue.

**Output — `CanonicalSensorEvent`:**
```
{
  event_id:     UUID
  sensor_id:    string
  sensor_type:  CAMERA_CV | MICROPHONE | ACCESS_CONTROL | DOOR_SENSOR |
                GLASS_BREAK | BADGE_READER | TIME_CONTEXT
  zone_id:      string
  timestamp:    DateTimeOffset
  threat_label: int            // 0-4 discretized
  raw_score:    float
  confidence:   float          // sensor's own confidence in the reading
  metadata:     object         // type-specific payload (bounding boxes, track IDs, etc.)
}
```

---

### 7.2 Incident Manager

**Responsibility:** Cluster `CanonicalSensorEvent` into `Incident` objects; maintain the per-incident annotation matrix; trigger re-inference.

**Incident definition:** A set of sensor events in the same zone within a configurable time window `W` (default 60s). Two events belong to the same incident if their zone IDs match and the time since the last event in the incident is < `W`. When the gap exceeds `W` with no new events, the incident transitions to `ASSESSED`.

**Annotation matrix:**
```
AnnotationBuffer {
  incident_id:    UUID
  zone_id:        string
  opened_at:      DateTimeOffset
  last_event_at:  DateTimeOffset
  annotations:    int[numSensorTypes]   // -1 = missing; multiple events from same
                                        // sensor type → keep MAX label (conservative)
  dirty:          bool                  // needs re-inference
}
```

**Re-inference trigger policy:**
- High-priority events (GLASS_BREAK, GUNSHOT, FORCED_DOOR, WEAPON_DETECTION ≥ HIGH): trigger immediate synchronous inference
- Standard events: set `dirty=true`; a background timer fires inference at most every 200ms for all dirty incidents
- This debounce prevents VMP thrashing during rapid event bursts from a single sensor

**Incident lifecycle:**
```
OPEN        → ASSESSED    when: no new events for W seconds
ASSESSED    → ESCALATED   when: threat_level ≥ HIGH AND confidence ≥ threshold
ASSESSED    → RESOLVED    when: operator marks false alarm OR threat drops to CLEAR
ESCALATED   → RESOLVED    when: operator closes
```

---

### 7.3 MACE Inference Service (Online Adaptation)

**Responsibility:** Run MACE VMP inference on active incidents; manage per-sensor belief state; propagate posteriors.

**Core adaptation — batch to online:**

The existing `MACETrain.InferModelData(int[][] data)` operates on a fixed `numItems × numWorkers` matrix. The online adaptation treats each active incident as a single-row matrix (`numItems=1`):

```
annotations[0][j] = sensor j's current label for this incident (-1 if not seen)
```

**Inference cycle for a single incident:**

1. Load current `ThetaDist[j]` (Beta) and `PhiDist[j]` (Dirichlet) from Belief Store for all registered sensor types
2. Construct `ModelData` priors and call `trainer.SetModelData(priors)`
3. If a previous posterior exists for this incident, warm-start `InitializeLabels` from `TDist[0]` rather than random; this reduces VMP iterations for incremental updates
4. Call `trainer.InferModelData(annotations)` to get updated posteriors
5. Publish `ThreatPosterior` (TDist, SDist) to Threat Score Service
6. Store posterior snapshot to Belief Store for audit trail

**What does NOT happen at inference time:** Global `ThetaDist` and `PhiDist` are not updated during active incidents. They are updated only after an incident closes and the operator provides feedback (see Section 7.5). This separation ensures that mid-incident inference uses stable priors.

**Model instance management:**
- Infer.NET `InferenceEngine` instances are not thread-safe for concurrent `Infer<>` calls; each inference job gets its own `MACETrain` instance
- Maintain a pool of pre-warmed `MACETrain` instances (pool size = max concurrent incidents) to amortize object graph construction cost
- `numWorkers` = total registered sensor types (fixed at startup); `numItems` = 1; `numCategories` = 5

**Minimum observation guard:**
When fewer than `min_sensors` (default: 2) sensor types have contributed to an incident, MACE has insufficient information for meaningful inference. In this case:
- Skip VMP inference
- Set `threat_level = max(annotations)`, `confidence = LOW_CONFIDENCE_FLAG`
- Flag the response so the operator console can render it differently

**Latency profile (estimated for 10 sensor types, 5 categories):**
- VMP typically converges in 5–15 iterations
- Estimated wall time: 30–80ms per inference call on a modern server CPU
- Well within the 500ms P99 budget after accounting for queue time and network

---

### 7.4 Threat Score Service

**Responsibility:** Convert MACE posteriors to actionable scores; evaluate alert thresholds; fan out to operator consoles and external systems.

**Threat score computation from `TDist[0]` posterior:**
```
threat_level  = argmax(TDist[0].GetProbs())        // most probable label
threat_score  = threat_level                        // integer 0–4
confidence    = TDist[0].GetProbs()[threat_level]   // P(most probable)
entropy       = H(TDist[0])                         // uncertainty measure
```

**Alert evaluation logic:**
```
Alert triggered if ALL of:
  threat_level >= zone.alert_on_level
  confidence   >= zone.confidence_min
  incident age >= anti_flicker_min_duration (default: 3s)
  last alert for this incident was at a lower level (suppress duplicate alerts)
```

**WebSocket fan-out:**
- Clients subscribe with `zones` filter and `min_level` filter
- On each inference cycle, publish `THREAT_UPDATE` to all matching subscribers
- On alert trigger, publish `ALERT` message with priority flag
- Connection state managed with heartbeat/ping-pong; missed heartbeats queue messages for reconnect

**External alert webhooks:**
- Configurable per-integration endpoint for PSIM, SIEM, and VMS systems
- Payload signed with HMAC-SHA256 (per-endpoint shared secret)
- Delivery with exponential backoff retry; failed deliveries logged for audit

---

### 7.5 Feedback Processor

**Responsibility:** Incorporate operator verdicts to update sensor reliability priors; write immutable audit log.

**Bayesian prior update logic (single-incident EM step):**

Once the operator resolves an incident to a **gold true level** `T`, both reliability
priors update in closed form — this is the EM step of the MACE generative model
(`MACETrain.CreateModel`), not a hand-tuned heuristic. `θ[j] ~ Beta(α, β)` is the
spammer probability (α counts "faulty" evidence, β counts "reliable" evidence);
`φ[j] ~ Dirichlet(m)` is the label distribution a sensor emits *when* faulty.

For each sensor `j` with reading `a` (skip absent sensors, `a = -1`), compute the
**fault responsibility** using the current prior means `θ̄`, `φ̄`:

```
r = P(faulty | a, T)
  = 1                                          if a ≠ T   (a reliable sensor must report T; it didn't)
  = θ̄·φ̄[a] / (θ̄·φ̄[a] + (1 − θ̄))               if a = T   (could be reliable, or faulty-but-emitted-truth)
```

Then apply the matching conjugate updates, damped by the learning rate:

```
θ:  α += lr · r,   β += lr · (1 − r)
φ:  m[a] += lr · r          // only the faulty branch informs φ; only bucket a moves
```

The gold level comes from the verdict: **FALSE_ALARM** pins `T = CLEAR (0)`;
**TRUE_ALARM** uses the operator-resolved `true_threat_level`. Because `r` is
recomputed from `T` (rather than reusing the mid-incident `S[0][j]` posterior), a
trusted sensor that disagrees with the gold label is fully penalised (`r = 1`),
and a distrusted one that agrees is credited — the update no longer discounts
surprising evidence. There is no `MEDIUM` threshold or fixed penalty constant.

`lr` = `learning_rate` (configurable, default: 0.5) damps how aggressively a single
incident moves the global priors.

**Beta distribution reference points:**
- `Beta(1, 9)`: mean=0.10 → highly reliable sensor
- `Beta(5, 5)`: mean=0.50 → neutral / cold-start prior
- `Beta(9, 1)`: mean=0.90 → highly unreliable sensor (spammer)

**Operator can also submit per-sensor assessments** (FR-19) that override the
model-derived responsibility. If the operator explicitly marks sensor `j` as "was
correct" (force `r = 0`) or "was wrong" (force `r = 1`), that assessment replaces
the computed `r` for both the θ and φ updates.

> **Note — no forgetting factor:** both α+β and the φ pseudocounts accumulate
> monotonically, so priors become progressively harder to move. A decay factor
> applied to both before each update is planned future work (tracked in `ISSUES.md`).

**Audit log entry (append-only):**
```json
{
  "timestamp": "...",
  "operator_id": "op-007",
  "incident_id": "...",
  "verdict": "TRUE_ALARM",
  "true_threat_level": 3,
  "sensor_updates": [
    {
      "sensor_id": "cam-north-01",
      "responsibility": 0.12,
      "theta_before": { "alpha": 2.0, "beta": 8.0 },
      "theta_after":  { "alpha": 2.06, "beta": 8.44 },
      "phi_before": [8, 4, 3, 2, 1],
      "phi_after":  [8, 4, 3, 2.06, 1]
    }
  ]
}
```

---

### 7.6 Belief Store

**Responsibility:** Persist and serve the learned Bayesian state: sensor reliability priors, incident history, posterior snapshots.

**Sensor Beliefs** (read on every inference call, written on operator feedback):
```sql
sensor_beliefs (
  sensor_id        TEXT PRIMARY KEY,
  theta_alpha      FLOAT NOT NULL DEFAULT 1.0,
  theta_beta       FLOAT NOT NULL DEFAULT 9.0,
  phi_pseudocounts FLOAT[] NOT NULL,          -- length = numCategories
  last_updated     TIMESTAMPTZ,
  incidents_seen   INT DEFAULT 0
)
```

**Incident Records** (written at open; updated at close):
```sql
incidents (
  incident_id        UUID PRIMARY KEY,
  zone_id            TEXT NOT NULL,
  status             TEXT NOT NULL,           -- OPEN|ASSESSED|ESCALATED|RESOLVED
  opened_at          TIMESTAMPTZ NOT NULL,
  closed_at          TIMESTAMPTZ,
  annotation_matrix  JSONB,                   -- snapshot at close
  final_posteriors   JSONB,                   -- TDist, SDist
  operator_verdict   TEXT,                    -- TRUE_ALARM|FALSE_ALARM
  operator_id        TEXT
)
```

**Posterior Snapshots** (append-only; one row per inference cycle):
```sql
posterior_snapshots (
  snapshot_id    UUID PRIMARY KEY,
  incident_id    UUID REFERENCES incidents,
  snapshot_at    TIMESTAMPTZ NOT NULL,
  t_dist         FLOAT[5] NOT NULL,           -- threat level posterior
  s_dist         JSONB NOT NULL,              -- per-sensor spammer posteriors
  trigger_sensor TEXT                         -- which sensor event triggered inference
)
```

**Recommended storage stack:**
- `sensor_beliefs`: Redis with AOF persistence (fast reads on hot path) + async PostgreSQL sync for durability
- `incidents`, `posterior_snapshots`: PostgreSQL
- Audit log: append-only Kafka topic or immutable S3 object store

---

## 8. Key Data Models

### CanonicalSensorEvent
```
{
  event_id:     UUID
  sensor_id:    string
  sensor_type:  CAMERA_CV | MICROPHONE | ACCESS_CONTROL | DOOR_SENSOR |
                GLASS_BREAK | BADGE_READER | TIME_CONTEXT
  zone_id:      string
  timestamp:    DateTimeOffset
  threat_label: int (0–4)
  raw_score:    float
  confidence:   float
  metadata:     object        // type-specific payload
}
```

### Incident
```
{
  incident_id:        UUID
  zone_id:            string
  status:             OPEN | ASSESSED | ESCALATED | RESOLVED
  opened_at:          DateTimeOffset
  last_event_at:      DateTimeOffset
  annotation_matrix:  int[]            // one slot per sensor type; -1 = missing
  current_posteriors: ThreatPosterior  // latest MACE output
  event_count:        int
  sensor_count:       int              // distinct sensor types that contributed
}
```

### ThreatPosterior
```
{
  incident_id:    UUID
  inferred_at:    DateTimeOffset
  t_dist:         float[5]       // P(CLEAR, LOW, MEDIUM, HIGH, CRITICAL)
  threat_level:   int            // argmax
  confidence:     float          // max probability
  entropy:        float          // uncertainty
  sensor_reliability: [{
    sensor_id:    string
    annotation:   int            // discretized label this sensor provided
    spammer_prob: float          // mean of S[0][j] Bernoulli posterior
    reliability:  float          // 1 - spammer_prob
  }]
}
```

---

## 9. Deployment Considerations

### 9.1 Component Topology

**Single-premises (≤100 sensors, ≤10 concurrent active incidents):**
- All services co-deployed on a single hardened host or small 3-node Kubernetes cluster
- MQTT broker: Eclipse Mosquitto
- Internal event bus: Redis Streams
- Storage: PostgreSQL + Redis

**Multi-premises / enterprise:**
- Sensor Gateway and Incident Manager partitioned per premises or building zone
- MACE Inference Service scales horizontally; 1 instance handles ~50 concurrent active incidents
- Centralized Belief Store optionally aggregates cross-premises sensor type reliability (useful for fleet deployments of identical sensor hardware)

### 9.2 Latency Budget

| Stage | Estimated Time |
|---|---|
| Sensor Gateway: receipt → normalization | ~5ms |
| Gateway → Incident Manager (local queue) | ~10ms |
| Annotation buffer update | ~2ms |
| MACE VMP inference (10 sensors, 5 categories) | ~50–150ms |
| Threat score + WebSocket push | ~5ms |
| **Total P50** | **~75ms** |
| **Total P99 (GC, queue contention)** | **~400ms** |

The P99 estimate fits within the NFR-01 requirement of ≤500ms.

### 9.3 Graceful Degradation

| Failure | System Behavior |
|---|---|
| Inference Service unavailable | Sensor Gateway buffers events in Redis queue; re-queues when service recovers |
| Belief Store unavailable | Inference runs with default `Beta(1,1)` priors (no learned reliability) |
| Single sensor type offline | Missing annotation (-1) handled naturally by MACE sparse matrix |
| VMP non-convergence | Fallback to majority-vote of discretized annotations; log warning |
| Operator console offline | Alert messages queued in Redis; delivered on reconnect |

---

## 10. Open Questions and Future Work

1. **Continuous threat score:** Current design uses 5 discrete levels. A continuous score could be extracted as the posterior expected value `E[T[i]]` or entropy-weighted mean, providing finer granularity for risk scoring.

2. **Hierarchical sensor reliability:** Current design tracks `θ` per sensor ID. A hierarchical Bayesian model could have sensor-type-level priors (camera model X is generally 80% reliable) from which individual device reliability is drawn — beneficial for cold-starting new sensors of a known type.

3. **Contextual reliability modulation:** Sensor reliability may vary with environmental context (reverberant room, low-light camera, crowded lobby). Context covariates could dynamically shift the `θ` prior before inference without waiting for feedback.

4. **Multi-incident correlation:** Two simultaneous incidents in adjacent zones may share a common cause (e.g., an active shooter moving through a building). Cross-incident belief propagation is outside single-item MACE but could be addressed with a multi-item inference window covering adjacent zones.

5. **Active sensing:** When a sensor has not reported for an active incident in its zone, the system could explicitly poll it (useful for badge readers: "what was the last access event in this zone?"). This converts a passive model into an active one that fills annotation gaps.

6. **Temporal decay:** Sensor annotations for a long-running incident should decay in influence as time passes (an unknown person detected 10 minutes ago is less informative than 30 seconds ago). A time-weighted annotation scheme could be layered on top of the discretization step.

---

## References

- Hovy, D., Berg-Kirkpatrick, T., Vaswani, A., Hovy, E. (2013). "Learning Whom to Trust with MACE." *NAACL 2013*.
- Microsoft Research. "Infer.NET — a framework for running Bayesian inference in graphical models." https://dotnet.github.io/infer/
- This codebase: Infer.NET MACE implementation, `MACE/MACETrain.cs`, `MACE/MACEBase.cs`, `MACE/ModelData.cs`
