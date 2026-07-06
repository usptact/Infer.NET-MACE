# Issues: MACE / ThreatSense Reliability Learning

This document catalogs shortcomings and likely bugs found while auditing the
MACE / ThreatSense codebase. Each issue is grounded with concrete file
locations, code, model details, and — where relevant — standard Bayesian-modeling
practice.

> **Status (branch `fix-b-em-belief-update`).** Issues **1, 2, 4, and 5 are
> resolved** by Fix B, which replaces the θ-only heuristic with the single-incident
> **EM step** over θ and φ (`PriorUpdateService.UpdateBeliefs`; see
> THREATSENSE_DESIGN.md §7.5). Still open: **Issue 3** (no forgetting/decay — now
> applies to φ pseudocounts too), **Issue 6** (discretizer), and **Issue 7**
> (duplicated posterior post-processing). See the per-issue Status lines below.

## Background needed to read the issues

The system learns **sensor reliability**, encoded as two per-sensor-type
parameters:

- **θ[j]** — `Beta(α, β)`, the sensor's fault / "spammer" probability
  (`MACE/MACEBase.cs:28` `_thetaPriors`, `:32` `_theta`).
- **φ[j]** — `Dirichlet`, the sensor's bias distribution *when* it is faulty
  (`MACE/MACEBase.cs:29` `_phiPriors`, `:33` `_phi`).

These are moved by two mechanisms:

- **Inference-time (VMP):** `MACETrain.InferOnline` (`MACE/MACETrain.cs:160`)
  runs Variational Message Passing on the current incident; posteriors are
  propagated as the next call's priors.
- **Feedback-time (heuristic, not Bayes):** `PriorUpdateService.UpdateTheta`
  (`MACE/Services/PriorUpdateService.cs:34`) nudges α/β after an operator
  verdict.

The issues below concern how these two mechanisms are implemented.

---

## Issue 1 — φ (fault-bias Dirichlet) is never updated by operator feedback

**Severity: High (design/implementation gap; likely a real bug)**

**Status: ✅ Resolved by Fix B** — `UpdateBeliefs` now returns both θ and φ, and the `UpdatePriors` RPC carries `current_phi` / returns `updated_phis`. The conjugate update `φ[reading] += lr·r` learns the sensor's fault-bias from operator ground truth.

The design explicitly states that operator feedback updates **both** θ and φ:

- `THREATSENSE_DESIGN.md` data-flow step 6:
  *"the **Feedback Processor** updates `θ[j]` and `φ[j]` priors in the Belief
  Store for the relevant sensor types."*
- FR-18: *"Update sensor reliability priors based on operator verdicts."*

The implementation updates **only** θ:

- `PriorUpdateService.UpdateTheta` returns a `BetaParameters` and mutates only
  `alpha`/`beta` (`MACE/Services/PriorUpdateService.cs:34–68`). There is no φ
  code path.
- The gRPC feedback response carries only thetas
  (`MACE/Services/MaceInferenceGrpcService.cs:205–210`):

  ```csharp
  response.UpdatedThetas.Add(new UpdatedTheta
  {
      SensorTypeIndex = sensor.SensorTypeIndex,
      Alpha           = updated.Alpha,
      Beta            = updated.Beta
  });
  ```

- The wire contract confirms the omission — `UpdatePriorsResponse` has no φ
  field (`MACE/Protos/*.proto`):

  ```proto
  message UpdatePriorsResponse {
    repeated UpdatedTheta updated_thetas = 1;
  }
  ```

**Consequence.** φ[j] — the direction in which a sensor is wrong when it is
unreliable — only ever adapts through inference-time posterior propagation, and
never from operator ground truth. In MACE, φ is precisely what lets the model
distinguish "this sensor is noisy" from "this sensor is systematically biased
toward a particular label"; leaving it out of the supervised feedback loop
discards a core part of the model's discriminative power.

---

## Issue 2 — Feedback rule discounts surprising evidence (confirmation bias)

**Severity: High (learning dynamics)**

**Status: ✅ Resolved by Fix B** — the responsibility `r` is now recomputed from the gold true level, not the stale `faultProbMean`. A reading that disagrees with gold gives `r = 1` (full penalty) regardless of prior trust, so surprising evidence is no longer discounted.

The two model-weighted branches of `UpdateTheta` scale the update by the model's
*current* belief that the sensor was faulty, `faultProbMean`
(`MACE/Services/PriorUpdateService.cs:48–65`):

```csharp
if (verdict == Verdict.TrueAlarm)
{
    if (flaggedThreat)
        // Sensor correctly flagged → reinforce reliability (increase β)
        beta  += learningRate * (1.0 - faultProbMean);
    else
        alpha += learningRate * 0.3;
}
else  // FalseAlarm
{
    if (flaggedThreat)
        // Sensor contributed to false alarm → penalise (increase α)
        alpha += learningRate * faultProbMean;
    else
        beta  += learningRate * 0.3;
}
```

`faultProbMean` is documented as *"Mean of the fault indicator S[0][j] from the
most recent Infer call"* (`MACE/Services/PriorUpdateService.cs:24–26`), i.e. a
posterior derived from the very θ prior being updated.

**Problem.** Consider a sensor the model already trusts (low `faultProbMean`)
that then contributes to a **FALSE_ALARM**. The penalty is
`alpha += lr * faultProbMean ≈ 0` — almost nothing — exactly when the outcome is
most *surprising* and therefore most informative. Symmetrically, a distrusted
sensor (high `faultProbMean`) that turns out correct on a **TRUE_ALARM** is
reinforced by `beta += lr * (1 - faultProbMean) ≈ 0`.

The rule therefore **amplifies confirming evidence and mutes contradicting
evidence**. Because `faultProbMean` is itself a function of the θ prior under
update, the loop is partially self-reinforcing (circular). This is the opposite
of the standard Bayesian expectation that a *surprising* observation should move
the posterior more, not less.

---

## Issue 3 — No forgetting / decay: priors saturate and stop adapting

**Severity: High (contradicts the stated "learn over time" goal)**

`UpdateTheta` only ever *adds* to α or β
(`MACE/Services/PriorUpdateService.cs:44–67`); it never rescales or decays them.
Consequently `α + β` grows without bound across incidents.

For `Beta(α, β)`:
- mean `= α / (α + β)`
- variance `= αβ / ((α+β)²(α+β+1))` → shrinks like `1/(α+β)`.

So the "effective sample size" `α + β` is a monotonically increasing pseudo-count
of confidence. After hundreds of incidents the Beta becomes razor-sharp and a
single `lr * (…)` nudge (with `lr` defaulting to `0.5`,
`MACE/Services/PriorUpdateService.cs:39`) barely moves the mean.

**Consequence.** A sensor whose true reliability *changes* over time — e.g. the
motivating "sensor installed incorrectly in a reverberant room" case
(`PRESENTATION.md:17`) or a camera that drifts out of focus — will adapt
agonizingly slowly, because old evidence is never down-weighted. This directly
undercuts THREATSENSE goal #2, *"Learn over time which sensors are credible"*
(`THREATSENSE_DESIGN.md:12`).

**Standard practice.** Online reliability estimation typically applies a
forgetting factor / exponential decay (e.g. multiply `(α, β)` by `γ ∈ (0,1)`
before each additive update) or caps the effective sample size so recent
evidence retains leverage. Neither exists here.

**Note (post-Fix B).** This remains open, and now applies to **φ as well**:
`UpdateBeliefs` accumulates `φ` pseudocounts (`m[reading] += lr·r`) with no decay,
so the fault-bias distribution saturates the same way `θ` does. A single decay
factor should be applied to both `(α, β)` and the `φ` vector before each update.

---

## Issue 4 — `MediumThreshold` is hardcoded and coupled to the 5-level scale

**Severity: Medium**

**Status: ✅ Resolved by Fix B** — `MediumThreshold` is deleted. The update compares the reading directly against the operator-resolved gold level; there is no fixed "flagged" cutoff.

The flagged / not-flagged decision inside the feedback rule uses a magic
constant (`MACE/Services/PriorUpdateService.cs:16–17`, `:46`):

```csharp
// Annotations >= this value are treated as "flagged a threat"
private const int MediumThreshold = 2;
...
bool flaggedThreat = annotation >= MediumThreshold;
```

The model, however, is parameterized by `numThreatLevels`
(`MACE/MACETrain.cs:35`, `:56`), and the level count is validated as a free
parameter (`MACE/MACETrain.cs:47–50`). The cutoff `2` is only correct for the
canonical 5-level scale `{0=CLEAR … 4=CRITICAL}`
(`THREATSENSE_DESIGN.md:42`). If the level scheme changes, or a coarser sensor
scale is introduced, "flagged" is silently miscomputed.

This also violates the configurability requirement NFR-10:
*"Sensor thresholds, incident time windows, alert thresholds, and prior
hyperparameters configurable without code changes"*
(`THREATSENSE_DESIGN.md:103`) — this threshold requires a code change.

---

## Issue 5 — The `0.3` penalty constant is unexplained and asymmetric

**Severity: Medium**

**Status: ✅ Resolved by Fix B** — the `0.3` constant is deleted. Both θ and φ updates are now driven by the single conjugate responsibility `r`, so the two branches are on the same principled footing.

The "missed a threat" and "correctly stayed quiet" branches use a fixed
`learningRate * 0.3` with no model weighting
(`MACE/Services/PriorUpdateService.cs:55`, `:64`):

```csharp
else
    // Sensor missed a real threat → penalise slightly (increase α)
    alpha += learningRate * 0.3;
...
else
    // Sensor correctly stayed quiet → reinforce (increase β)
    beta  += learningRate * 0.3;
```

This is asymmetric with the flagged branches, which scale by `faultProbMean` /
`(1 - faultProbMean)` (Issue 2). As a result the two axes of the update are not
on the same footing: model-derived confidence drives one pair of cases but a
bare magic number drives the other. The value `0.3` has no documented
justification in either the code or `THREATSENSE_DESIGN.md` §7.5. If the
asymmetry is deliberate it should be a named, configurable parameter with a
rationale; as written it reads as an unexplained constant.

---

## Issue 6 — Discretizer layer described in design but absent from code

**Severity: Medium (design/implementation divergence)**

`THREATSENSE_DESIGN.md` §7.1 / §6.5 describe an `IDiscretizer` per sensor type
that maps a raw score to the `0–4` label using the configurable discretization
thresholds. No such component exists in the codebase — a search for
`discretiz` / `IDiscretizer` / `rawScore` across `*.cs` returns zero matches.

Instead, this pod receives readings that are *already* discretized. The wire
type is `int32` and the pod treats the value as a final label
(`MACE/Protos/*.proto`, `InferRequest`):

```proto
// Discretized sensor readings for this incident.
repeated int32 sensor_readings = 2;   // -1 = sensor absent
```

`MACETrain.InferOnline` consumes them directly
(`MACE/MACETrain.cs:151–153`, `:183`). This is not a bug in the inference pod
per se, but the discretization stage described in the design has no counterpart
in this repo, so the raw-score-to-label mapping is effectively an untracked
upstream dependency rather than a documented, tested component of the system.

---

## Issue 7 — Duplicated posterior post-processing (drift risk)

**Severity: Low**

The threat posterior is reduced to `argmax` / confidence / entropy by hand
inside `InferOnline` (`MACE/MACETrain.cs:186–205`):

```csharp
var probs = threatDist.GetProbs();
int threatLevel = 0; double confidence = 0.0;
for (int i = 0; i < probs.Count; i++)
    if (probs[i] > confidence) { confidence = probs[i]; threatLevel = i; }

double entropy = 0.0;
for (int i = 0; i < probs.Count; i++)
{ double p = probs[i]; if (p > 0.0) entropy -= p * Math.Log(p); }
```

Separately, `BuildInferResponse` re-derives per-sensor reliability from the fault
posterior (`MACE/Services/MaceInferenceGrpcService.cs:316–327`):

```csharp
double sp = result.FaultDist[j].GetProbTrue();
resp.SensorReliability.Add(new SensorReliability { ... FaultProb = sp, Reliability = 1.0 - sp });
```

The two reductions live in different classes and hand-roll marginal
post-processing. There is no single, tested helper for "summarize a posterior,"
so the definitions of confidence/entropy/reliability can drift independently
over time. Consolidating into one utility would remove the duplication and make
these definitions authoritative.

---

## Priority summary

| # | Issue | Severity | Type |
|---|---|---|---|
| 1 | φ never updated by feedback | High | Design/impl gap, likely bug |
| 2 | Feedback rule discounts surprising evidence | High | Learning dynamics |
| 3 | No forgetting/decay; priors saturate | High | Learning dynamics |
| 4 | `MediumThreshold` hardcoded to 5-level scale | Medium | Configurability |
| 5 | Unexplained/asymmetric `0.3` constant | Medium | Modeling clarity |
| 6 | Discretizer designed but not implemented | Medium | Design divergence |
| 7 | Duplicated posterior post-processing | Low | Maintainability |

If the goal is reliability learning, **Issue 1** (φ never learned from feedback)
and **Issue 3** (no forgetting) are the highest-leverage fixes.
