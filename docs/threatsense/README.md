# ThreatSense

Reference material for the deployment this model was built into, carried over
from the `online-mace` branch.

**None of this is deployable from this repository.** The manifests and
Dockerfiles here build from source — `src/SensorGateway`, `src/IncidentManager`
and so on — that lives in the ThreatSense repositories, not this one. They are
kept because they document how the model was operated, and because the design
decisions behind them still inform the service that is here.

## What ThreatSense was

A security-monitoring system in which MACE estimated sensor reliability instead
of annotator reliability. Six services in the `threatsense` namespace:

| Service | Role |
| --- | --- |
| `sensor-gateway` | Ingest. Publishes sensor events to the Redis stream `ts:sensor-events`. |
| `incident-manager` | Correlation. Groups events into incidents over a 60s window and assembles the annotation matrix. |
| `mace-inference` | **Built from this repository.** Consumes inference requests, returns posteriors. |
| `threat-score-svc` | Turns a posterior into a threat score and an alerting decision. |
| `operator-console` | Where operators review and resolve incidents. |
| `feedback-processor` | Applies operator verdicts to the sensor reliability priors. |

Underneath: MQTT for sensor transport, Postgres for incidents and verdicts,
Redis for streams and for the belief store, a private registry, and
default-deny network policies with an explicit allow-matrix.

The `incident-manager` configmap is the Rosetta stone. Its `SENSOR_TYPES` list —
`CAMERA_CV, MICROPHONE, ACCESS_CONTROL, DOOR_SENSOR, GLASS_BREAK, BADGE_READER,
TIME_CONTEXT` — is annotated *index determines column in annotation matrix*.
That list is the worker roster of the deployment: the seven "workers" were
sensor types.

## Why it is worth keeping

`feedback-processor` runs a single replica, commented *"single writer for prior
updates — avoids concurrent Beta update races"*. That is the same hazard
`BeliefStore` in this repository handles with a lock.

More usefully, it explains a design choice that otherwise looks arbitrary. The
original `mace-inference` was stateless and took priors in and out on every
call, because the belief store lived in Redis and `feedback-processor` owned it
as sole writer. That separation is what let inference run at `replicas: 2`.

The service in this repository folds the belief store into the process, which is
simpler and honest at one replica, and is why its deployment is pinned there. If
horizontal scaling is ever needed, the split documented here is the proven shape,
and it is a `BeliefStore` implementation swap rather than a redesign.

## Contents

- `infra/` — Kubernetes manifests, k3s bootstrap scripts, Helm values, the
  full-stack Makefile and compose file
- `dockerfiles/` — images for the five services not built here
- [`THREATSENSE_DESIGN.md`](THREATSENSE_DESIGN.md) — system design
- [`MACE_SERVICE_DESIGN.md`](MACE_SERVICE_DESIGN.md) — rationale for the inference service
- [`INFRASTRUCTURE.md`](INFRASTRUCTURE.md) — deployment guide for the rack
- [`SCENARIO.md`](SCENARIO.md) — a worked end-to-end example
- [`PRESENTATION.md`](PRESENTATION.md) — overview for a technical audience
- [`ISSUES.md`](ISSUES.md) — model audit. Its Issue 3, no forgetting or decay,
  is resolved in this repository by `PriorUpdateService`'s `retention`.

Each document opens with a banner recording which of its claims still hold. Two
are corrected there rather than in the text: inference is expectation
propagation and cannot be VMP, and the gRPC surface has changed.
