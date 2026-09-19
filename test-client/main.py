"""
MACE Inference Test Gateway
===========================
A FastAPI application that exposes the gRPC MaceInference service as plain
HTTP/JSON endpoints, so any HTTP client (curl, Postman, browser) can exercise
the service without a gRPC-aware client.

Start:
    uvicorn main:app --reload --port 8000

Then browse to http://localhost:8000/docs for the interactive Swagger UI.

Environment variables:
    MACE_GRPC_TARGET   gRPC address        (default: localhost:8080)
    MACE_HTTP_TARGET   metrics/health base (default: http://localhost:9090)

Note on state: the service owns worker reliability. A caller sends annotations
and gets a label back; it does not pass priors in and out. Reliability changes
only through /feedback, when an item's true label is actually known.
"""

import os
import sys
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from typing import List, Optional

import grpc
import grpc.aio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Generated stubs are in ./generated/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "generated"))
try:
    import mace_inference_pb2 as pb2
    import mace_inference_pb2_grpc as pb2_grpc
except ImportError:
    raise SystemExit(
        "Generated gRPC stubs not found.\n"
        "Run:  make generate\n"
        "or:   bash generate_stubs.sh"
    )

GRPC_TARGET = os.getenv("MACE_GRPC_TARGET", "localhost:8080")
HTTP_TARGET = os.getenv("MACE_HTTP_TARGET", "http://localhost:9090")


# ---------------------------------------------------------------------------
# Shared gRPC channel (created once at startup)
# ---------------------------------------------------------------------------

_channel: Optional[grpc.aio.Channel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _channel
    _channel = grpc.aio.insecure_channel(GRPC_TARGET)
    print(f"gRPC channel opened -> {GRPC_TARGET}")
    yield
    await _channel.close()
    print("gRPC channel closed.")


def stub() -> "pb2_grpc.MaceInferenceStub":
    assert _channel is not None, "Channel not initialised"
    return pb2_grpc.MaceInferenceStub(_channel)


app = FastAPI(
    title="MACE Inference Test Gateway",
    description=(
        "REST to gRPC bridge for the MACE inference service.\n\n"
        "Workers are identified by a zero-based index fixed at service startup. "
        "Labels are integers in [0, num_categories)."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AnnotationModel(BaseModel):
    worker: int = Field(..., description="Zero-based worker index.", examples=[0])
    label: int = Field(..., description="Label the worker assigned.", examples=[2])


class InferRequest(BaseModel):
    item_id: str = Field("test-001", description="Caller-defined item identifier, echoed back.")
    annotations: List[AnnotationModel] = Field(
        ...,
        description=(
            "At most one annotation per worker. An empty list is accepted and returns the "
            "prior, with contributing_workers empty to say the answer carries no evidence."
        ),
    )


class WorkerAssessmentModel(BaseModel):
    worker: int
    spammer_probability: float


class InferResponse(BaseModel):
    item_id: str
    label: int
    confidence: float
    entropy: float = Field(
        ...,
        description=(
            "Shannon entropy in nats, 0 (certain) to ln(num_categories) (no information). "
            "Separates one strong runner-up from an even spread, which confidence cannot."
        ),
    )
    label_probabilities: List[float]
    contributing_workers: List[int]
    worker_assessments: List[WorkerAssessmentModel]


class FeedbackRequest(BaseModel):
    item_id: str = Field("test-001", description="Caller-defined item identifier, echoed back.")
    annotations: List[AnnotationModel] = Field(
        ..., description="The annotations as they were when the item was inferred."
    )
    true_label: int = Field(..., description="The established true label for the item.")
    learning_rate: Optional[float] = Field(
        None, gt=0, description="Weight given to this item. Omit for the service default."
    )
    retention: Optional[float] = Field(
        None,
        gt=0,
        le=1,
        description=(
            "Share of accumulated evidence surviving this update. Omit for the service "
            "default. 1.0 never forgets, which stops a worker from being re-learned."
        ),
    )


class WorkerReliabilityModel(BaseModel):
    worker: int
    spammer_probability: float
    spam_preferences: List[float]
    evidence: float = Field(
        ..., description="Evidence accumulated for this worker, as a count of items."
    )


class FeedbackResponse(BaseModel):
    item_id: str
    updated: List[WorkerReliabilityModel]


class WorkersResponse(BaseModel):
    workers: List[WorkerReliabilityModel]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _annotations(models: List[AnnotationModel]):
    return [pb2.Annotation(worker=a.worker, label=a.label) for a in models]


def _reliability(item) -> WorkerReliabilityModel:
    return WorkerReliabilityModel(
        worker=item.worker,
        spammer_probability=item.spammer_probability,
        spam_preferences=list(item.spam_preferences),
        evidence=item.evidence,
    )


@app.post("/infer", response_model=InferResponse, tags=["Inference"])
async def infer(req: InferRequest):
    """
    Infer one item's label from the annotations supplied.

    Reads worker reliability; never changes it. Send the same item twice and you
    get the same answer.
    """
    grpc_req = pb2.InferLabelRequest(
        item_id=req.item_id,
        annotations=_annotations(req.annotations),
    )

    try:
        resp = await stub().InferLabel(grpc_req)
    except grpc.aio.AioRpcError as e:
        _handle_grpc_error(e)

    return InferResponse(
        item_id=resp.item_id,
        label=resp.label,
        confidence=resp.confidence,
        entropy=resp.entropy,
        label_probabilities=list(resp.label_probabilities),
        contributing_workers=list(resp.contributing_workers),
        worker_assessments=[
            WorkerAssessmentModel(
                worker=a.worker, spammer_probability=a.spammer_probability
            )
            for a in resp.worker_assessments
        ],
    )


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def feedback(req: FeedbackRequest):
    """
    Update worker reliability from an item whose true label is now known.

    Closed-form arithmetic, no Infer.NET: a worker who contradicted the true
    label takes the full weight, and one who agreed takes the share a spammer
    could have earned by chance.
    """
    grpc_req = pb2.SubmitFeedbackRequest(
        item_id=req.item_id,
        annotations=_annotations(req.annotations),
        true_label=req.true_label,
    )

    # The proto fields are optional, so leave them unset to take the service default.
    if req.learning_rate is not None:
        grpc_req.learning_rate = req.learning_rate
    if req.retention is not None:
        grpc_req.retention = req.retention

    try:
        resp = await stub().SubmitFeedback(grpc_req)
    except grpc.aio.AioRpcError as e:
        _handle_grpc_error(e)

    return FeedbackResponse(
        item_id=resp.item_id,
        updated=[_reliability(u) for u in resp.updated],
    )


@app.get("/workers", response_model=WorkersResponse, tags=["Feedback"])
async def workers(worker: Optional[List[int]] = None):
    """Current reliability for every worker, or for the workers listed."""
    grpc_req = pb2.GetWorkerReliabilityRequest(workers=worker or [])

    try:
        resp = await stub().GetWorkerReliability(grpc_req)
    except grpc.aio.AioRpcError as e:
        _handle_grpc_error(e)

    return WorkersResponse(workers=[_reliability(w) for w in resp.workers])


@app.get("/health", tags=["Operations"])
async def health():
    """
    Proxy the service's readiness endpoint.

    Health lives on the service's HTTP/1.1 port, not on gRPC: a plaintext gRPC
    listener only speaks HTTP/2, which probes and scrapers cannot.
    """
    try:
        with urllib.request.urlopen(f"{HTTP_TARGET}/readyz", timeout=10) as r:
            import json

            return json.load(r)
    except urllib.error.URLError as e:
        raise HTTPException(status_code=503, detail=f"Service unreachable: {e}")


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "MACE Test Gateway",
        "docs": "/docs",
        "grpc_target": GRPC_TARGET,
        "http_target": HTTP_TARGET,
    }


# ---------------------------------------------------------------------------
# Error helper
# ---------------------------------------------------------------------------

def _handle_grpc_error(e: grpc.aio.AioRpcError):
    code = e.code()
    if code == grpc.StatusCode.INVALID_ARGUMENT:
        raise HTTPException(status_code=400, detail=e.details())
    if code == grpc.StatusCode.UNAVAILABLE:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {e.details()}")
    if code == grpc.StatusCode.DEADLINE_EXCEEDED:
        raise HTTPException(status_code=504, detail=f"Deadline exceeded: {e.details()}")
    raise HTTPException(status_code=500, detail=f"{code.name}: {e.details()}")
