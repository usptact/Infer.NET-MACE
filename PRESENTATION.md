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

### Functional

| | Requirement |
|---|---|
| **FR-1** | Ingest sensor events via MQTT, gRPC, and REST webhook |
| **FR-2** | Cluster events into incidents by spatial zone and time window |
| **FR-3** | Score each active incident within 500 ms of a new sensor event |
| **FR-4** | Express threat as a posterior distribution, scalar score, and per-sensor reliability breakdown |
| **FR-5** | Accept operator TRUE\_ALARM / FALSE\_ALARM verdicts; update sensor priors accordingly |
| **FR-6** | Emit real-time alerts to operator consoles (WebSocket) and external systems (webhook) |
| **FR-7** | Maintain a tamper-evident audit trail of all inferences and operator actions |

### Non-Functional

| | Requirement |
|---|---|
| **NFR-1** | Threat score latency ≤ 500 ms P99 from sensor event receipt |
| **NFR-2** | ≥ 1,000 sensor events / second per deployment node |
| **NFR-3** | Inference service degrades gracefully under load (queues, then sheds, never crashes) |
| **NFR-4** | Stateless inference pod — horizontally scalable, restartable without data loss |
| **NFR-5** | Explainable — every threat score traceable to contributing sensors and their weights |

---

## Assumptions & Scope

**In scope (MVP):**
- Single physical premises, fixed set of sensor types defined at deployment time
- Five discrete threat levels: CLEAR (0) → LOW → MEDIUM → HIGH → CRITICAL (4)
- Sensor raw scores are discretized to 0–4 by a normalization layer before entering the system
- Operator feedback is always binary (TRUE\_ALARM / FALSE\_ALARM) and eventually provided for all significant incidents

**Assumed away for MVP:**
- Multi-premises correlation (incidents across buildings)
- Continuous threat scores (extension to posterior expected value is straightforward)
- Dynamic sensor registration at runtime
- Active sensing (polling silent sensors mid-incident)
- Temporal decay of old annotations within a long-running incident

---

## The Core Idea: MACE as a Sensor Fusion Engine

The algorithmic heart of ThreatSense is **MACE** (Multi-Annotator Competence Estimation, Hovy et al. NAACL 2013), a Bayesian model originally designed for crowdsourcing — aggregating noisy human judgements of unknown quality to recover a ground truth label.

The mapping to physical security is exact:

| MACE (Crowdsourcing) | ThreatSense (Physical Security) |
|---|---|
| Worker `j` | Sensor type `j` (camera, mic, badge reader…) |
| Item `i` | Active incident `i` |
| Annotation `A[i,j]` | Sensor `j`'s discretized threat reading for incident `i` |
| True label `T[i]` | Latent threat level for incident `i` |
| Spammer probability `θ[j]` | Sensor `j`'s base false-alarm rate |
| Spammer preference `φ[j]` | Sensor `j`'s bias direction when unreliable |
| Missing annotation | Sensor offline or not in the zone |

### Why this model fits so well

- **Sparse observations are first-class.** Not every sensor sees every incident. MACE handles missing annotations naturally — they simply don't contribute to the posterior.
- **Conflicting signals are resolved probabilistically.** If three sensors say HIGH and one says CLEAR, MACE doesn't average them. It infers that the outlier is probably unreliable for this incident, without requiring anyone to configure that rule.
- **Reliability is latent, not declared.** `θ[j]` is inferred from evidence, not set at installation time. A sensor that consistently disagrees with the consensus accumulates evidence of unreliability in its Beta prior.

### Model structure

```
θ[j] ~ Beta(α_j, β_j)           spammer probability for sensor j
φ[j] ~ Dirichlet(ψ_j)           label bias when spamming

For each incident i:
  T[i] ~ Discrete(1/K, …, 1/K)  true threat level (uniform prior)

  For each sensor j:
    S[i,j] ~ Bernoulli(θ[j])    is sensor j spamming on this incident?

    A[i,j] = T[i]                if S[i,j] = 0  (reliable: copies true label)
    A[i,j] ~ Discrete(φ[j])     if S[i,j] = 1  (spamming: random from bias)
```

Inference is performed with **Variational Message Passing** (VMP) via Infer.NET, which iteratively refines beliefs until convergence. In the online setting (`numItems = 1`), this typically converges in 5–15 iterations, taking 300–800 ms per incident update on current hardware.

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

### MACE Inference Service

The stateless gRPC pod that runs VMP inference. Design decisions worth explaining:

**Stateless by design.** Every call receives all the data it needs: the annotation vector and the current Beta/Dirichlet priors loaded by the caller from the Belief Store. The pod stores nothing between calls. This means it can be scaled horizontally, restarted without data loss, and tested with synthetic inputs without a database connection.

**Thread-safety via object pool.** Infer.NET's inference engine mutates internal state and is not thread-safe. Rather than one engine per request (expensive — factor graph compilation takes ~1–3 s), a fixed pool of pre-warmed engines is maintained. Each concurrent request borrows an engine from the pool exclusively, uses it, and returns it. Pool depth = maximum concurrent incidents the pod can serve.

**Pool saturation is graceful.** When all engines are busy and a new request cannot acquire one within the configured timeout (default 5 s), the pod returns gRPC `UNAVAILABLE`. Callers can retry; no crash, no data corruption.

**Warm-starting.** When the same incident is re-inferred as new sensors fire, the `t_dist` from the previous call can be passed back as `warm_start`. VMP initialises from this posterior rather than a random point, typically reducing wall time by 2–4× on the second and subsequent calls.

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
| TRUE\_ALARM + sensor flagged (annotation ≥ MEDIUM) | `β += lr × (1 − spammer_prob)` | Reliable sensor: reinforce |
| FALSE\_ALARM + sensor flagged | `α += lr × spammer_prob` | Sensor cried wolf: penalize |
| TRUE\_ALARM + sensor missed (annotation < MEDIUM) | `α += lr × 0.3` | Sensor missed threat: penalize slightly |
| FALSE\_ALARM + sensor stayed quiet | `β += lr × 0.3` | Sensor correctly quiet: reinforce slightly |

`lr` (learning rate, default 0.5) controls how aggressively a single incident shifts the global prior. Updated priors are written back to the Belief Store and used in all subsequent inferences.

The Belief Store also maintains a **posterior snapshot log** — an append-only record of every inference result, every operator action, and every prior change, forming a complete auditable history of the system's reasoning.

---

## System Behavior: A Walk-Through Scenario

*02:30 AM. Financial office building. The scenario unfolds over 90 seconds.*

**Step 1 — First signal** (badge reader + time context only)
- Badge reader: LOW (unusual access hour)
- Time context: HIGH (late night)
- Result: threat=HIGH, confidence=52%, entropy=1.10 — *Watch and wait*

MACE already questions both sensors (badge reader: 69% spammer, time context: 52%) because each alone is weak evidence. The model correctly refuses to commit with only 2 of 7 sensors.

**Step 2 — Escalation** (camera and microphone fire)
- Camera: HIGH; Microphone: MEDIUM; Door sensor: LOW
- Result: threat=HIGH, confidence=72%, entropy=0.75 — *Alert operator*

Camera and time context agree on HIGH; microphone reports MEDIUM. MACE marks the microphone as 94% likely-spamming for this incident — it disagrees with the consensus and is down-weighted automatically.

**Step 3 — Operator confirms TRUE\_ALARM**
Camera's `β` increases (correctly flagged, reliably); microphone's `β` barely increases (flagged but was judged unreliable). Door sensor's `α` increases (missed the threat). These updates carry forward to all future incidents.

**Step 4 — Adjacent zone: glass break**
- Camera: CRITICAL; Glass-break detector: CRITICAL; Microphone: HIGH
- Result: threat=CRITICAL, confidence=78%, entropy=0.53 — *Dispatch immediately*

Camera and glass-break agree on CRITICAL; microphone reports HIGH. Microphone is again discounted for under-reading. The posterior is decisive despite the disagreement.

**Step 5 — All-clear sweep** (security has cleared the area)
- Camera: CLEAR; Microphone: CLEAR; Door: CLEAR; Time context: HIGH (still 02:30)
- Result: threat=CLEAR, confidence=99.96%, entropy=0.003

Three trusted sensors agree on CLEAR. Time context (still insisting HIGH) is judged 99.97% spammer and overridden. The system stands down with near-certainty.

**The key insight from this scenario:** MACE's self-consistency property means that sensors which agree with the eventual consensus gain reliability, while sensors that consistently disagree lose it — regardless of whether we know the ground truth in advance.

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

2. **Cold-start sensor reliability.** A newly installed sensor starts with a neutral prior `Beta(5,5)` — 50% assumed spammer rate — until evidence accumulates. For high-stakes sensors (glass-break), it may be desirable to seed them with an informative prior `Beta(1,9)` based on manufacturer specs.

3. **Temporal decay.** A sensor annotation from 10 minutes ago should carry less weight than one from 10 seconds ago. This is not modeled. One approach: decay annotations toward -1 (absent) as they age, or weight them in a pre-processing step.

4. **Multi-incident correlation.** An active shooter moving through a building creates simultaneous incidents in adjacent zones. Standard MACE treats each incident independently. A multi-item inference window across zones could capture this correlation but significantly increases complexity.

5. **Feedback quality.** The update rules assume operator verdicts are accurate. An operator who habitually confirms false alarms would degrade sensor reliability estimates over time. A confidence weighting on operator feedback (based on the operator's own track record) would mitigate this.

---

*Built on Infer.NET / Microsoft.ML.Probabilistic 0.4. Bayesian model: Hovy et al. "Learning Whom to Trust with MACE," NAACL 2013.*
