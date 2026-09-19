"""Integration tests for the FastAPI gateway (`main.py`) against a running stack.

These exercise the HTTP to gRPC bridge end-to-end, so they need the inference
service and this gateway to be up (e.g. `docker compose up -d`). Point them at a
different gateway with the MACE_GATEWAY_URL env var. If the gateway is not
reachable the whole module is skipped rather than failed, so this is safe to run
where the stack is not running.

    cd test-client && pip install pytest && pytest -v      # or: make test

These assume the service's default configuration of 8 workers and 3 categories.

Note that feedback mutates server state. The tests below are written so that
order does not matter: they assert relative outcomes (this worker looks worse
than that one) rather than absolute values, because an earlier test's feedback
is still in effect when a later one runs.
"""
import json
import math
import os
import urllib.error
import urllib.request

import pytest

BASE = os.environ.get("MACE_GATEWAY_URL", "http://localhost:8000")
TOL = 1e-4

NUM_CATEGORIES = int(os.environ.get("MACE_NUM_CATEGORIES", "3"))


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


def _gateway_is_up():
    try:
        status, _ = _call("/health", method="GET")
        return status == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _gateway_is_up(),
    reason=f"Gateway not reachable at {BASE}; start the stack with `docker compose up -d`.",
)


def _annotations(*pairs):
    return [{"worker": w, "label": l} for w, l in pairs]


# ── /infer ────────────────────────────────────────────────────────────────────

def test_infer_returns_a_normalised_posterior():
    status, r = _call("/infer", {
        "item_id": "gw-001",
        "annotations": _annotations((0, 1), (1, 1), (2, 1)),
    })

    assert status == 200, r
    assert r["item_id"] == "gw-001"
    assert len(r["label_probabilities"]) == NUM_CATEGORIES
    assert abs(sum(r["label_probabilities"]) - 1.0) < TOL
    assert 0.0 <= r["confidence"] <= 1.0


def test_infer_agrees_with_unanimous_workers():
    status, r = _call("/infer", {
        "item_id": "gw-002",
        "annotations": _annotations((0, 2), (1, 2), (2, 2), (3, 2)),
    })

    assert status == 200, r
    assert r["label"] == 2
    assert r["confidence"] > 0.8


def test_infer_reports_only_contributing_workers():
    status, r = _call("/infer", {
        "item_id": "gw-003",
        "annotations": _annotations((1, 0), (4, 0)),
    })

    assert status == 200, r
    assert r["contributing_workers"] == [1, 4]
    assert [a["worker"] for a in r["worker_assessments"]] == [1, 4]


def test_infer_with_no_annotations_returns_maximum_entropy():
    status, r = _call("/infer", {"item_id": "gw-004", "annotations": []})

    assert status == 200, r
    assert r["contributing_workers"] == []
    # No evidence means the posterior is the prior, which is uniform.
    assert abs(r["entropy"] - math.log(NUM_CATEGORIES)) < 1e-3


def test_infer_is_repeatable():
    body = {"item_id": "gw-005", "annotations": _annotations((0, 0), (1, 1), (2, 0))}

    first_status, first = _call("/infer", body)
    second_status, second = _call("/infer", body)

    assert first_status == second_status == 200
    # Inference must not move worker reliability, so the same request twice in a
    # row has to give exactly the same answer.
    assert first["label_probabilities"] == second["label_probabilities"]


# ── /infer — validation ───────────────────────────────────────────────────────

def test_infer_rejects_duplicate_worker():
    status, _ = _call("/infer", {
        "item_id": "gw-006",
        "annotations": _annotations((1, 0), (1, 2)),
    })
    assert status == 400


def test_infer_rejects_unknown_worker():
    status, _ = _call("/infer", {
        "item_id": "gw-007",
        "annotations": _annotations((999, 0)),
    })
    assert status == 400


def test_infer_rejects_label_out_of_range():
    status, _ = _call("/infer", {
        "item_id": "gw-008",
        "annotations": _annotations((0, 99)),
    })
    assert status == 400


# ── /feedback ─────────────────────────────────────────────────────────────────

def test_feedback_returns_only_the_workers_it_moved():
    status, r = _call("/feedback", {
        "item_id": "gw-010",
        "annotations": _annotations((0, 0), (1, 1)),
        "true_label": 0,
        "learning_rate": 1.0,
    })

    assert status == 200, r
    assert r["item_id"] == "gw-010"
    assert sorted(u["worker"] for u in r["updated"]) == [0, 1]


def test_feedback_penalises_the_worker_who_contradicted_the_truth():
    status, r = _call("/feedback", {
        "item_id": "gw-011",
        "annotations": _annotations((2, 1), (3, 0)),
        "true_label": 1,
        "learning_rate": 1.0,
    })

    assert status == 200, r
    agreed = next(u for u in r["updated"] if u["worker"] == 2)
    disagreed = next(u for u in r["updated"] if u["worker"] == 3)

    # An honest worker reports the truth by definition, so contradicting it is
    # the stronger evidence of the two.
    assert disagreed["spammer_probability"] > agreed["spammer_probability"]


def test_feedback_accumulates_evidence():
    body = {
        "item_id": "gw-012",
        "annotations": _annotations((5, 0)),
        "true_label": 0,
        "learning_rate": 1.0,
    }

    _, first = _call("/feedback", body)
    _, second = _call("/feedback", body)

    before = next(u for u in first["updated"] if u["worker"] == 5)["evidence"]
    after = next(u for u in second["updated"] if u["worker"] == 5)["evidence"]

    assert after > before


def test_feedback_leaves_absent_workers_alone():
    _, before = _call("/workers", method="GET")
    baseline = {w["worker"]: w["evidence"] for w in before["workers"]}

    _, r = _call("/feedback", {
        "item_id": "gw-013",
        "annotations": _annotations((6, 0)),
        "true_label": 0,
        "learning_rate": 1.0,
    })
    assert [u["worker"] for u in r["updated"]] == [6]

    _, after = _call("/workers", method="GET")
    for w in after["workers"]:
        if w["worker"] != 6:
            assert abs(w["evidence"] - baseline[w["worker"]]) < TOL


# ── /feedback — validation ────────────────────────────────────────────────────

def test_feedback_rejects_true_label_out_of_range():
    status, _ = _call("/feedback", {
        "item_id": "gw-020",
        "annotations": _annotations((0, 0)),
        "true_label": 99,
    })
    assert status == 400


def test_feedback_rejects_non_positive_learning_rate():
    status, _ = _call("/feedback", {
        "item_id": "gw-021",
        "annotations": _annotations((0, 0)),
        "true_label": 0,
        "learning_rate": 0.0,
    })
    # Rejected by the gateway's own schema (gt=0) before it reaches the service.
    assert status in (400, 422)


def test_feedback_rejects_retention_above_one():
    status, _ = _call("/feedback", {
        "item_id": "gw-022",
        "annotations": _annotations((0, 0)),
        "true_label": 0,
        "retention": 1.5,
    })
    assert status in (400, 422)


# ── /workers ──────────────────────────────────────────────────────────────────

def test_workers_reports_every_worker():
    status, r = _call("/workers", method="GET")

    assert status == 200, r
    assert len(r["workers"]) >= 1
    for w in r["workers"]:
        assert 0.0 <= w["spammer_probability"] <= 1.0
        assert len(w["spam_preferences"]) == NUM_CATEGORIES
        assert abs(sum(w["spam_preferences"]) - 1.0) < TOL


def test_health_reports_the_pool():
    status, r = _call("/health", method="GET")

    assert status == 200, r
    assert r["status"] == "ready"
    assert r["total"] >= 1
