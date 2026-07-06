"""Integration tests for the FastAPI gateway (`main.py`) against a running stack.

These exercise the HTTP → gRPC bridge end-to-end, so they require the inference
service and gateway to be up (e.g. `docker compose up -d`). Point the tests at a
different gateway with the MACE_GATEWAY_URL env var. If the gateway is not
reachable the whole module is skipped rather than failed, so this is safe to run
in environments where the stack isn't running.

    cd test-client && pip install pytest && pytest -v      # or: make test

The service is configured for 7 sensor types and 5 threat levels
(0=CLEAR .. 4=CRITICAL); the assertions below assume that configuration.
"""
import json
import os
import urllib.error
import urllib.request

import pytest

BASE = os.environ.get("MACE_GATEWAY_URL", "http://localhost:8000")
TOL = 1e-4


def _call(path, body=None, method="POST"):
    """POST/GET JSON to the gateway; return (status_code, parsed_body)."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        raw = e.read().decode() or "{}"
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"raw": raw}


def _gateway_reachable():
    try:
        status, _ = _call("/health", method="GET")
        return status == 200
    except Exception:
        return False


# Skip the whole module (once) when the gateway isn't up.
pytestmark = pytest.mark.skipif(
    not _gateway_reachable(),
    reason=f"gateway not reachable at {BASE} (start the stack: docker compose up -d)",
)


def _uphi():
    """A uniform Dirichlet phi prior of length 5 (num threat levels)."""
    return {"pseudocounts": [1, 1, 1, 1, 1]}


def _theta(a, b):
    return {"alpha": a, "beta": b}


# ── /infer ────────────────────────────────────────────────────────────────────

def test_infer_returns_normalised_threat_dist():
    status, r = _call("/infer", {
        "incident_id": "gw-test",
        "sensor_readings": [3, 3, -1, -1, 2, -1, 0],
        "theta_priors": [_theta(1, 9), _theta(1, 9), _theta(5, 5), _theta(5, 5),
                         _theta(2, 8), _theta(5, 5), _theta(5, 5)],
        "phi_priors": [_uphi()] * 7,
    })
    assert status == 200
    assert r["threat_level"] == 3                      # camera+mic HIGH dominate
    assert abs(sum(r["threat_dist"]) - 1.0) < 1e-6
    assert len(r["threat_dist"]) == 5


# ── /update-priors — EM update happy paths ─────────────────────────────────────

def test_update_returns_theta_and_phi_with_em_math():
    """Agreeing sensor gets a small responsibility; disagreeing sensor gets r=1,
    and both theta AND phi come back (Fix B)."""
    status, r = _call("/update-priors", {
        "verdict": "TRUE_ALARM", "learning_rate": 0.5, "true_threat_level": 3,
        "sensors": [
            {"sensor_type_index": 0, "sensor_reading": 3,   # == gold  → agrees
             "current_theta": _theta(1, 9), "current_phi": _uphi()},
            {"sensor_type_index": 1, "sensor_reading": 2,   # != gold  → disagrees
             "current_theta": _theta(1, 9), "current_phi": _uphi()},
        ],
    })
    assert status == 200
    assert len(r["updated_thetas"]) == 2
    assert len(r["updated_phis"]) == 2

    cam_t, mic_t = r["updated_thetas"]
    cam_p, mic_p = r["updated_phis"]

    # Agreeing camera: small responsibility r ≈ 0.02174 → tiny alpha bump, beta grows.
    assert abs(cam_t["alpha"] - 1.010870) < TOL
    assert abs(cam_t["beta"] - 9.489130) < TOL
    assert abs(cam_p["pseudocounts"][3] - 1.010870) < TOL

    # Disagreeing mic: responsibility = 1 → alpha += lr, beta unchanged, phi[reading] += lr.
    assert abs(mic_t["alpha"] - 1.5) < TOL
    assert abs(mic_t["beta"] - 9.0) < TOL
    assert abs(mic_p["pseudocounts"][2] - 1.5) < TOL


def test_false_alarm_pins_gold_to_clear():
    """FALSE_ALARM forces gold=CLEAR(0) regardless of true_threat_level: a sensor
    that read CLEAR agrees, one that read HIGH disagrees (alpha += lr)."""
    status, r = _call("/update-priors", {
        "verdict": "FALSE_ALARM", "learning_rate": 1.0, "true_threat_level": 4,
        "sensors": [
            {"sensor_type_index": 0, "sensor_reading": 0,   # == CLEAR → agrees
             "current_theta": _theta(1, 9), "current_phi": _uphi()},
            {"sensor_type_index": 1, "sensor_reading": 3,   # != CLEAR → disagrees
             "current_theta": _theta(1, 9), "current_phi": _uphi()},
        ],
    })
    assert status == 200
    agree, disagree = r["updated_thetas"]
    # Agreeing sensor: r ≈ 0.02174, lr=1.0 → alpha ≈ 1.02174 (small bump).
    assert abs(agree["alpha"] - 1.021739) < TOL
    # Disagreeing sensor: r=1, lr=1.0 → alpha = 2.0.
    assert abs(disagree["alpha"] - 2.0) < TOL


def test_absent_sensor_echoes_priors_unchanged():
    status, r = _call("/update-priors", {
        "verdict": "FALSE_ALARM", "learning_rate": 0.5,
        "sensors": [{"sensor_type_index": 2, "sensor_reading": -1,
                     "current_theta": _theta(2, 8), "current_phi": _uphi()}],
    })
    assert status == 200
    assert r["updated_thetas"][0] == {"sensor_type_index": 2, "alpha": 2.0, "beta": 8.0}
    assert r["updated_phis"][0]["pseudocounts"] == [1, 1, 1, 1, 1]


# ── /update-priors — back-compat of the deprecated field ───────────────────────

def test_fault_prob_mean_optional():
    """fault_prob_mean is deprecated and may be omitted entirely."""
    status, _ = _call("/update-priors", {
        "verdict": "TRUE_ALARM", "learning_rate": 0.5, "true_threat_level": 3,
        "sensors": [{"sensor_type_index": 0, "sensor_reading": 3,
                     "current_theta": _theta(1, 9), "current_phi": _uphi()}],
    })
    assert status == 200


def test_legacy_fault_prob_mean_is_ignored():
    """A supplied fault_prob_mean must not affect the EM result (reading==gold ⇒ beta grows)."""
    status, r = _call("/update-priors", {
        "verdict": "TRUE_ALARM", "learning_rate": 0.5, "true_threat_level": 3,
        "sensors": [{"sensor_type_index": 0, "sensor_reading": 3, "fault_prob_mean": 0.9,
                     "current_theta": _theta(1, 9), "current_phi": _uphi()}],
    })
    assert status == 200
    assert r["updated_thetas"][0]["beta"] > 9.0   # driven by gold, not fault_prob_mean


# ── /update-priors — validation ────────────────────────────────────────────────

def test_missing_current_phi_rejected():
    """current_phi is required by the EM update (Pydantic → 422)."""
    status, _ = _call("/update-priors", {
        "verdict": "TRUE_ALARM", "true_threat_level": 3,
        "sensors": [{"sensor_type_index": 0, "sensor_reading": 3,
                     "current_theta": _theta(1, 9)}],
    })
    assert status == 422


def test_learning_rate_out_of_range_rejected():
    """learning_rate must be in (0, 1] (Pydantic → 422)."""
    status, _ = _call("/update-priors", {
        "verdict": "TRUE_ALARM", "learning_rate": 2.0, "true_threat_level": 3,
        "sensors": [{"sensor_type_index": 0, "sensor_reading": 3,
                     "current_theta": _theta(1, 9), "current_phi": _uphi()}],
    })
    assert status == 422


def test_gold_level_out_of_range_rejected():
    """true_threat_level outside [0, NumThreatLevels) is rejected by the service (gRPC → 400)."""
    status, _ = _call("/update-priors", {
        "verdict": "TRUE_ALARM", "learning_rate": 0.5, "true_threat_level": 99,
        "sensors": [{"sensor_type_index": 0, "sensor_reading": 3,
                     "current_theta": _theta(1, 9), "current_phi": _uphi()}],
    })
    assert status == 400


def test_unknown_verdict_rejected():
    status, _ = _call("/update-priors", {
        "verdict": "MAYBE_ALARM", "learning_rate": 0.5, "true_threat_level": 3,
        "sensors": [{"sensor_type_index": 0, "sensor_reading": 3,
                     "current_theta": _theta(1, 9), "current_phi": _uphi()}],
    })
    assert status == 400
