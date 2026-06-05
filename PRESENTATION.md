# ThreatSense: Bayesian Multi-Modal Threat Assessment

**A real-time physical security intelligence system**

---

## The Problem

A modern secured facility generates thousands of sensor events per hour — cameras, microphones, door sensors, badge readers, environmental detectors. A human operator watching a single screen cannot fuse these signals fast enough, and rule-based systems ("alert if camera AND door sensor fire within 30 seconds") are brittle, require per-facility tuning, and have no memory of which sensors have historically been reliable.

Three specific failure modes motivate this system:

| Failure Mode | Consequence |
|---|---|
| A camera flags a shadow as a person | False alarm fatigue; operators start ignoring alerts |
| A glass-break sensor fires but no camera corroborates | Alert suppressed by conservative threshold rules |
| A sensor is installed incorrectly in a reverberant room | It is always wrong; no one knows until an audit |

The deeper issue is that **sensor reliability is unknown, varies per sensor and per context, and must be learned from experience** — not assumed at installation time.

---

## What ThreatSense Does

ThreatSense fuses heterogeneous sensor events into a single probabilistic threat score per incident, while simultaneously learning which sensors are trustworthy. It gets smarter with every operator verdict.

Three properties distinguish it from rule-based systems:

1. **Probabilistic fusion** — produces a full posterior distribution `P(threat_level | observations)`, not a binary alert. Operators see *how confident* the system is, not just *whether* it fired.

2. **Sensor reliability tracking** — every sensor has a learned reliability score that is updated after each incident. A sensor that has repeatedly disagreed with the consensus is automatically down-weighted in future inferences.

3. **Operator-in-the-loop learning** — when an operator confirms or dismisses an alert, those verdicts propagate back as Bayesian prior updates. The system improves without retraining.

---

## Requirements

- Ingest heterogeneous sensor events and fuse them into a per-incident threat score within **500 ms P99**
- Express threat as a **full probability distribution** with per-sensor reliability breakdown, not a binary alert
- **Learn** which sensors are reliable over time; update beliefs from operator feedback without retraining
- Degrade gracefully under load — queue then shed requests, never crash or corrupt state
- Every score must be **explainable**: traceable to the sensors that drove it and their current reliability
- Maintain a tamper-evident audit trail of all inferences and operator actions

---

## Assumptions & Scope

**Current implementation assumes:**
- A single premises with a fixed set of sensor types known at deployment time
- Five discrete threat levels: CLEAR (0) → LOW → MEDIUM → HIGH → CRITICAL (4)
- Raw sensor outputs are discretized to 0–4 before reaching the inference engine
- Operator feedback is binary (TRUE\_ALARM / FALSE\_ALARM)
- Sensor reliability is tracked per *type* (camera, mic…), not per individual device

**These are not fundamental limitations — they are simplifications chosen for the MVP.** The underlying model is a general Bayesian graphical model that can be extended without architectural changes:

| Current simplification | Possible extension |
|---|---|
| Fixed sensor types | Register new sensor types at runtime; assign uninformative priors until evidence accumulates |
| Per-type reliability | Hierarchical model: type-level prior + per-device deviation (useful for large fleets) |
| Binary feedback | Graded verdicts ("mostly correct", "partially triggered"); per-sensor override by operator |
| No temporal decay | Decay annotation weight as incident age grows; parametric half-life per sensor type |
| Single premises | Multi-premises federation; shared type-level priors, separate device-level beliefs |
| Discrete threat levels | Continuous score as posterior expected value `E[T]`; ordinal regression extension |

The model is intentionally minimal. Complexity is added only when evidence from deployment shows it is needed.

---

## The Core Idea: Inference from Disagreeing Sensors

The system models each sensor as a participant that may be **reliable** (its readings reflect reality) or **unreliable** (its readings are noise or systematic bias). Neither is known in advance. The key question on every incident is: *given that these sensors disagree, which ones should we believe?*

The answer comes from a Bayesian graphical model. Each sensor carries a prior belief about its own reliability. On each incident, the model jointly infers:

1. The most probable threat level, given all available sensor readings
2. For each sensor: how likely is it that *this sensor* is behaving reliably *on this incident*

These two questions are solved simultaneously. The sensor readings that agree with the inferred threat level are judged more reliable; those that disagree are judged less reliable. No rules, no thresholds, no manual weights.

**Sensor reliability as a learned quantity.** Each sensor type has a Beta-distributed reliability prior — `Beta(α, β)` — where `α` accumulates evidence of unreliable behavior and `β` accumulates evidence of reliable behavior. Reference points:

| Prior | Interpretation |
|---|---|
| `Beta(1, 9)` — mean fault rate 10% | Highly reliable; rarely disagrees with ground truth |
| `Beta(5, 5)` — mean fault rate 50% | Unknown; neutral cold-start for a new sensor |
| `Beta(9, 1)` — mean fault rate 90% | Highly unreliable; almost always disagrees |

After each incident and operator verdict, these priors are updated. The changes are small per incident but compound over hundreds of incidents into a precise, evidence-based reliability profile for every sensor in the facility.

### The model in brief

Five quantities, explained in plain terms:

| Variable | What it represents | Technical form |
|---|---|---|
| **T[i]** | *"What is the actual threat level of incident i?"* — the thing we're trying to infer | Discrete distribution over 5 levels |
| **A[i,j]** | *"What did sensor j report for incident i?"* — the raw observation (0–4, or absent) | Observed integer |
| **θ[j]** | *"How often is sensor j unreliable, historically?"* — its long-term track record | Beta distribution, updated over time |
| **S[i,j]** | *"Is sensor j behaving reliably right now, on this specific incident?"* — inferred per-call | Bernoulli (yes/no) |
| **φ[j]** | *"When sensor j is unreliable, what does it tend to say?"* — its systematic bias | Dirichlet distribution |

The model assumes: if a sensor is **reliable** on this incident, it will report the true threat level. If it is **unreliable**, it will report something according to its own bias distribution — which may be consistently too low, too high, or random noise.

Given all the sensor readings, Bayesian inference works backwards: it finds the threat level T[i] and the per-sensor reliability indicators S[i,j] that best explain what was collectively observed. Sensors whose readings are consistent with the inferred threat level are rewarded; sensors that contradict it are penalized.

**Online operation.** The model runs in streaming mode — one incident at a time, as sensor events arrive. Each time a new sensor fires for an active incident, inference re-runs immediately with the updated annotation. The result from the previous run is used as the starting point for the next, so convergence is fast for incremental updates. When an incident closes and the operator provides a verdict, the long-term reliability priors (θ[j]) are updated and persist into all future incidents. The inference pod itself is stateless — it receives priors and annotations as inputs and returns posteriors. All persistence lives in the Belief Store.

---

### The consensus property

A critical emergent behavior: **sensors that repeatedly agree with the eventual ground truth become more trusted, while sensors that repeatedly disagree become less trusted — without the system being told which is which in advance.**

Concretely: if a camera and a microphone both report HIGH on 50 incidents that operators later confirm as TRUE\_ALARM, and a badge reader reports LOW on those same 50 incidents, the model will have accumulated strong evidence that the badge reader is unreliable in that context. Its readings will be given minimal weight in future inferences, automatically, from evidence alone.

This is not a fixed weighting rule. It is a posterior that updates on every new piece of evidence.

### Sensor reliability is context-dependent

A sensor that is reliable in one context may be unreliable in another. The model tracks this naturally because reliability is inferred per-incident, not computed globally in isolation.

**Example:** Consider an acoustic sensor (microphone) in two contexts:

*In a quiet server room*, the microphone correctly detects glass-break events and agrees with the camera CV on every incident over three months. Its `β` grows steadily. It becomes one of the most trusted sensors in that zone — mean fault rate < 5%.

*In a public lobby during business hours*, the same microphone model is installed next to an HVAC duct. It triggers repeatedly on ventilation noise. Cameras and door sensors consistently disagree with it. Its `α` grows across dozens of false alarms. In that zone, it accumulates a mean fault rate > 70%. The system effectively mutes it in the lobby while still trusting the identical model in the server room.

No one configured this distinction. No one wrote a rule. The system learned it from the pattern of disagreements and operator verdicts across two different deployment contexts.

This is the property that makes MACE suitable for long-running deployed systems: the model gets better at every facility it is deployed in, and improves throughout its operational lifetime.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Physical Sensor Layer                         │
│   Camera CV   ·   Microphone   ·   Door / Glass-Break            │
│   Badge Reader   ·   Access Control   ·   Time Context           │
└──────────────────────────┬───────────────────────────────────────┘
                           │  MQTT / gRPC / REST
                           ▼
                   ┌───────────────┐
                   │ Sensor Gateway│  normalize · authenticate · discretize
                   └───────┬───────┘
                           │  CanonicalSensorEvent (0–4 label)
                           ▼
                  ┌─────────────────┐
                  │ Incident Manager│  cluster · buffer · trigger
                  └────────┬────────┘
          ┌────────────────┤
          │  priors        │  annotation vector
          ▼                ▼
  ┌──────────────┐  ┌──────────────────────┐
  │ Belief Store │  │ MACE Inference Service│  VMP · pool · stateless
  │  θ[j], φ[j] │◄─│  Infer.NET / gRPC     │
  └──────────────┘  └──────────┬───────────┘
                               │  posteriors
                               ▼
                    ┌─────────────────────┐
                    │ Threat Score Service │  threshold · alert · WebSocket
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Operator Console   │  dashboard · verdict
                    └──────────┬──────────┘
                               │  TRUE_ALARM / FALSE_ALARM
                               ▼
                    ┌──────────────────────┐
                    │  Feedback Processor  │  Beta update · audit log
                    └──────────────────────┘
```

---

## API Surface

The system exposes three layers of API:

### Sensor Ingest (Sensor Gateway)
```
POST /api/v1/sensors/{sensorId}/events          # push a raw sensor reading
POST /api/v1/sensors/{sensorId}/events/batch    # high-frequency batch push
POST /api/v1/sensors                            # register a new sensor
```

### Threat Query (Threat Score Service)
```
GET  /api/v1/incidents/{id}/threat              # full posterior for one incident
GET  /api/v1/zones/{id}/threat                  # aggregated zone score
WS   /ws/v1/threats?zones=lobby&min_level=LOW   # real-time stream
```

### Operator Feedback
```
POST /api/v1/incidents/{id}/feedback            # submit TRUE_ALARM / FALSE_ALARM
```

### Inference Pod (internal gRPC, also testable via FastAPI gateway)
```
Infer         — run VMP, return t_dist + per-sensor reliability
UpdatePriors  — compute updated Beta priors after a verdict
Health        — liveness probe
```

> The inference pod speaks gRPC internally (efficient binary protocol, typed generated stubs). A FastAPI HTTP/JSON gateway is provided for human testing and Swagger UI access.

---

## Component Deep Dives

### Sensor Gateway

Receives raw sensor outputs and transforms them into a canonical 4-bit threat label. The key function is **discretization**: mapping a continuous sensor score to `{0, 1, 2, 3, 4}` via per-sensor configurable thresholds. Examples:

| Sensor | Mapping |
|---|---|
| Camera — P(unknown person) | `[0.30, 0.60, 0.85]` → labels 0/1/2/3 |
| Microphone — P(gunshot) | `[0.20, 0.50, 0.75]` → 0/1/2/3; reverb flag caps at 2 |
| Glass-break detector | Confidence > 0.5 → label 3 (HIGH); always |
| Badge reader | Valid + business hours → 0; Invalid + off-hours → 3 |
| Time context | Business hours → 0; late night → 2 (ambient risk) |

A per-sensor **token bucket** prevents burst floods from misconfigured sensors from saturating the inference queue downstream.

---

### Incident Manager

Groups sensor events into incidents using a **temporal-spatial window**: events in the same zone within 60 seconds of each other are part of the same incident. The annotation matrix per incident is a fixed-length integer array (one slot per sensor type), updated with `MAX` when multiple events from the same sensor arrive — a conservative policy that escalates rather than averages.

**Re-inference trigger policy** avoids thrashing:
- High-priority events (glass break, forced door, weapon detection): immediate inference
- Standard events: debounced at 200 ms — at most one VMP call per incident per 200 ms regardless of event burst rate

---

### Adding New Sensor Types

The system is designed to accommodate new sensor types without code changes to the inference engine.

**Steps to add a new sensor type** (e.g. thermal camera, radar, or LiDAR):

1. Register the sensor type in the Sensor Gateway with a discretization function (how do raw scores map to 0–4?) and an initial prior — `Beta(1, 9)` if the hardware is well-characterized and known to be accurate, `Beta(5, 5)` if unknown.
2. Extend the annotation vector length from 7 to 8 (or however many types now exist). All existing inferences pass `-1` for the new slot; the model treats absence as missing data and is unaffected.
3. As incidents accumulate, the new sensor's prior updates from evidence like any other.

New sensors start at 50% reliability (neutral prior) and earn trust — or lose it — purely from their record. There is no privileged position for any sensor type. A cheap door sensor that consistently agrees with outcomes can become more trusted than an expensive CV system that fires on shadows.

The only constraint: the number of sensor types must be fixed per deployment and known at pod startup (it determines the size of the compiled factor graph). Changing it requires a pod restart with a new configuration. Dynamic hot-addition of sensor types is a future extension.

---

### MACE Inference Service

The stateless gRPC pod that runs the Bayesian inference. Design decisions worth explaining:

**Stateless by design.** Every call receives all the data it needs: the annotation vector and the current Beta/Dirichlet priors loaded by the caller from the Belief Store. The pod stores nothing between calls. This means it can be scaled horizontally, restarted without data loss, and tested with synthetic inputs without a database connection.

**Concurrent requests via an engine pool.** Inference is CPU-intensive and not safely shareable across concurrent calls. A fixed pool of pre-warmed inference engines is maintained at startup. Each request borrows one exclusively, uses it, and returns it. Pool depth = maximum concurrent incidents the pod can serve simultaneously.

**Pool saturation is graceful.** When all engines are busy, new requests queue. If a slot is not available within the configured timeout (default 5 s), the pod returns `UNAVAILABLE`. Callers can retry; no crash, no data corruption, no stale state.

**Warm-starting.** When the same incident is re-inferred as new sensors fire, the probability distribution from the previous call can be passed back as a starting point. The inference algorithm converges faster when initialized near the previous answer, typically reducing wall time by 2–4×.

**What the pod deliberately does not do:** connect to any database, manage incident state, discretize sensor readings, or trigger alerts. These are all upstream/downstream concerns. The pod's contract is: *given annotations and priors, return posteriors*.

---

### Threat Score Service

Converts the `T[i]` posterior into operator-facing quantities:

```
threat_level = argmax(T[i])          most probable threat level
confidence   = max(T[i])             probability mass on argmax
entropy      = H(T[i])               uncertainty — high entropy = conflicting signals
```

**Alert suppression rules** prevent alert fatigue:
- Threat level must meet or exceed the zone's configured threshold
- Confidence must exceed `confidence_min` (default: 0.70)
- Incident must have been open for at least 3 seconds (anti-flicker)
- Re-alert only on level *increase*, not on repeated inference at the same level

---

### Feedback Processor & Belief Store

After an operator closes an incident, the Feedback Processor updates the Beta prior `θ[j] = Beta(α_j, β_j)` for each sensor that participated:

| Condition | Update | Interpretation |
|---|---|---|
| TRUE\_ALARM + sensor flagged (annotation ≥ MEDIUM) | `β += lr × (1 − fault_prob)` | Reliable sensor: reinforce |
| FALSE\_ALARM + sensor flagged | `α += lr × fault_prob` | Sensor cried wolf: penalize |
| TRUE\_ALARM + sensor missed (annotation < MEDIUM) | `α += lr × 0.3` | Sensor missed threat: penalize slightly |
| FALSE\_ALARM + sensor stayed quiet | `β += lr × 0.3` | Sensor correctly quiet: reinforce slightly |

`lr` (learning rate, default 0.5) controls how aggressively a single incident shifts the global prior. Updated priors are written back to the Belief Store and used in all subsequent inferences.

The Belief Store also maintains a **posterior snapshot log** — an append-only record of every inference result, every operator action, and every prior change, forming a complete auditable history of the system's reasoning.

---

## System Behavior: A Walk-Through Scenario

*02:30 AM. Financial office building. This scenario illustrates both real-time threat scoring and the long-term evolution of sensor reliability.*

### The incident (90 seconds)

**Step 1 — First signal** *(badge reader + time context only)*
- Badge: LOW (unusual access hour) · Time context: HIGH (02:30 AM)
- Result: threat=HIGH, confidence=**52%**, entropy=1.10 — *Watch and wait*

With only 2 of 7 sensors present, and both individually weak evidence, the system correctly refuses to commit. Badge reader is already flagged 69% likely-unreliable; time context 52%. Both are questioned by the model because neither alone is sufficient to triangulate a true threat.

**Step 2 — Escalation** *(camera and microphone fire)*
- Camera: HIGH · Microphone: MEDIUM · Door sensor: LOW
- Result: threat=HIGH, confidence=**72%**, entropy=0.75 — *Alert operator*

Camera and time context agree on HIGH. The microphone reported MEDIUM — one level below the consensus. MACE immediately marks the microphone as **94% likely-spamming on this incident**: it disagrees with the majority, so its reading is down-weighted. The door sensor reported LOW (missed the threat) and is similarly discounted. The alert fires.

**Step 3 — Operator confirms TRUE\_ALARM**

After the operator verdict, priors update:
- Camera `β` increases significantly — it correctly flagged HIGH with low fault probability
- Microphone `β` barely increases — it flagged the threat (annotation ≥ MEDIUM) but was judged 94% likely faulty, so it receives almost no credit
- Door sensor `α` increases — it missed the threat entirely

**Step 4 — Adjacent zone: glass break** *(separate incident)*
- Camera: CRITICAL · Glass-break detector: CRITICAL · Microphone: HIGH
- Result: threat=CRITICAL, confidence=**78%**, entropy=0.53 — *Dispatch immediately*

Camera and glass-break agree on CRITICAL; microphone under-reports at HIGH. Microphone is discounted again. Despite the disagreement, the posterior is decisive.

**Step 5 — All-clear sweep** *(security has cleared the corridor)*
- Camera: CLEAR · Microphone: CLEAR · Door: CLEAR · Time context: HIGH (still 02:30)
- Result: threat=CLEAR, confidence=**99.96%**, entropy=0.003

Three now-trusted physical sensors agree on CLEAR. Time context insists HIGH. The model judges time context **99.97% likely faulty** for this incident — it structurally disagrees with the authoritative physical sensors — and overrides it. The system stands down.

---

### What changes over months of operation

The more interesting story is what happens to sensor priors across hundreds of incidents.

**The microphone's reliability history in the north corridor:**

After 3 months and ~120 incidents, the north corridor microphone has a pattern: on clear-cut alarms confirmed by cameras, it consistently reports one level below the camera. Operators always confirm TRUE\_ALARM. Each time, the microphone is judged a low-credit participant — it flagged something, but disagreed with the consensus level. Its `β` grows slowly, `α` never declines much. After 120 incidents its mean fault rate settles around **35%** — the system has learned that this microphone *tends to under-read*, but is not completely dismissed.

**The same microphone model in the server room:**

In the server room — a quiet, carpeted space — the same microphone model accurately detects glass-break and forced-door events, agreeing with the camera and glass-break sensor every time. After 120 incidents its mean fault rate settles around **8%**. It is one of the most trusted sensors in that zone.

**Nobody configured this difference.** No one wrote a rule that "the lobby mic is less reliable than the server room mic." The system inferred it from the pattern of agreements and disagreements across incidents in each zone, cross-validated by operator verdicts.

**Adding a new sensor type** (say, a thermal camera) is straightforward: register it with a neutral prior `Beta(5, 5)` — 50% assumed fault rate — and it immediately participates in inference. Its weight starts low, reflecting genuine uncertainty. Within 20–30 confirmed incidents it will have accumulated enough evidence to settle into a reliable prior, either trusted or discounted, based purely on how well its readings correlate with eventual outcomes.

---

## Performance Profile

Measurements on a single development machine (Docker, Apple Silicon equivalent):

| Configuration | Throughput | Latency P50 | Saturation point |
|---|---|---|---|
| Pool = 1 | ~5 req/s | 0.35 s | 32 concurrent |
| Pool = 4 | ~13 req/s | 0.39 s | 96 concurrent |
| Pool = 4, sustained 3 min | 13–14 req/s | — | 0 errors, flat memory |

**Scaling is 2.5× for 4× pool slots** — not linear, because the FastAPI HTTP→gRPC gateway and the single-core gRPC network layer become co-bottlenecks. CPU scales nearly linearly (4× pool ≈ 4× CPU cores used) because each VMP inference slot runs its own Intel MKL linear algebra threads independently.

**Memory** is dominated by the Infer.NET compiled factor graph (~1 GB for pool=1; ~160 MB per additional slot). This is a one-time startup cost, fixed regardless of incident volume. No per-incident data is retained in the inference pod.

**NFR-01 (≤500 ms P99):** Met at pool=4 for up to 32 concurrent incidents per pod. Higher concurrency requires additional pod instances (horizontal scaling).

---

## What Is Built Today

| Component | Status |
|---|---|
| MACE Inference Pod (gRPC, Infer.NET VMP, online mode) | ✓ Complete |
| FastAPI HTTP↔gRPC test gateway with Swagger UI | ✓ Complete |
| Prometheus metrics (latency histogram, pool gauge, request counters) | ✓ Complete |
| Grafana dashboard (4-row: overview, traffic, latency, system health) | ✓ Complete |
| Docker Compose stack (`docker-compose up` → all four services) | ✓ Complete |
| Dockerfiles (inference pod + test client) | ✓ Complete |
| Sensor Gateway, Incident Manager, Threat Score Service | Design only |
| Feedback Processor, Belief Store, Operator Console | Design only |

The built components constitute the **inference core** of ThreatSense — the most algorithmically novel part, and the one with the highest technical risk. The remaining services are primarily software engineering work building on well-understood patterns (MQTT ingestion, PostgreSQL persistence, WebSocket fan-out).

---

## Open Questions

1. **Latency at 7 sensor types.** Current VMP runs in 300–800 ms. The design target is ≤500 ms P99. Warm-starting typically brings this to 300–500 ms. Under adversarial load (pool saturation) P99 approaches 5 s. Mitigation: increase pool size or add pod replicas.

2. **Cold-start sensor reliability.** A newly installed sensor starts with a neutral prior `Beta(5,5)` — 50% assumed fault rate — until evidence accumulates. For high-stakes sensors (glass-break), it may be desirable to seed them with an informative prior `Beta(1,9)` based on manufacturer specs.

3. **Temporal decay.** A sensor annotation from 10 minutes ago should carry less weight than one from 10 seconds ago. This is not modeled. One approach: decay annotations toward -1 (absent) as they age, or weight them in a pre-processing step.

4. **Multi-incident correlation.** An active shooter moving through a building creates simultaneous incidents in adjacent zones. Standard MACE treats each incident independently. A multi-item inference window across zones could capture this correlation but significantly increases complexity.

5. **Feedback quality.** The update rules assume operator verdicts are accurate. An operator who habitually confirms false alarms would degrade sensor reliability estimates over time. A confidence weighting on operator feedback (based on the operator's own track record) would mitigate this.

---

*Bayesian model: Hovy et al. "Learning Whom to Trust with MACE," NAACL 2013.*
