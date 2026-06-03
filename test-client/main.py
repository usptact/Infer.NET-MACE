"""
MACE Inference Test Gateway
===========================
A FastAPI application that exposes the gRPC MaceInference service as plain
HTTP/JSON endpoints, so any HTTP client (curl, Postman, browser) can exercise
the inference pod without a gRPC-aware client.

Start:
    uvicorn main:app --reload --port 8000

Then browse to http://localhost:8000/docs for the interactive Swagger UI.

Environment variables:
    MACE_GRPC_TARGET   gRPC server address (default: localhost:8080)
"""

import os
import sys
import math
import time
from contextlib import asynccontextmanager
from typing import Optional

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


# ---------------------------------------------------------------------------
# Shared gRPC channel (created once at startup)
# ---------------------------------------------------------------------------

_channel: grpc.aio.Channel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _channel
    _channel = grpc.aio.insecure_channel(GRPC_TARGET)
    print(f"gRPC channel opened → {GRPC_TARGET}")
    yield
    await _channel.close()
    print("gRPC channel closed.")


def stub() -> pb2_grpc.MaceInferenceStub:
    assert _channel is not None, "Channel not initialised"
    return pb2_grpc.MaceInferenceStub(_channel)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MACE Inference Test Gateway",
    description=(
        "REST→gRPC bridge for the MACE Inference Service.\n\n"
        f"Connecting to gRPC server at **{GRPC_TARGET}** "
        "(override with `MACE_GRPC_TARGET` env var)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class BetaParams(BaseModel):
    alpha: float = Field(1.0, gt=0, description="Beta α parameter")
    beta:  float = Field(9.0, gt=0, description="Beta β parameter")


class DirichletParams(BaseModel):
    pseudocounts: list[float] = Field(
        default=[1.0, 1.0, 1.0, 1.0, 1.0],
        description="Dirichlet concentration parameters, one per threat level"
    )


class InferRequest(BaseModel):
    incident_id: str = Field("test-001", description="Caller-defined incident identifier")
    annotations: list[int] = Field(
        ...,
        description=(
            "One entry per sensor type in fixed order. "
            "-1 = sensor absent. "
            "0=CLEAR, 1=LOW, 2=MEDIUM, 3=HIGH, 4=CRITICAL."
        ),
        example=[-1, 2, -1, 3, -1, 0, 1]
    )
    theta_priors: list[BetaParams] = Field(
        ..., description="Beta prior for each sensor type (indexed by sensor-type order)"
    )
    phi_priors: list[DirichletParams] = Field(
        ..., description="Dirichlet prior for each sensor type"
    )
    warm_start: Optional[list[float]] = Field(
        None,
        description=(
            "t_dist from a previous /infer call on the same incident. "
            "Omit on first call; supply on subsequent calls for faster VMP convergence."
        )
    )


class SensorReliabilityItem(BaseModel):
    sensor_type_index: int
    annotation:        int
    spammer_prob:      float
    reliability:       float


class InferResponse(BaseModel):
    incident_id:        str
    t_dist:             list[float]
    threat_level:       int
    confidence:         float
    entropy:            float
    sensor_reliability: list[SensorReliabilityItem]
    num_observations:   int
    inference_ms:       int


class SensorPriorUpdate(BaseModel):
    sensor_type_index: int
    annotation:        int   = Field(..., description="-1 if sensor was absent")
    spammer_prob_mean: float = Field(..., ge=0, le=1)
    current_theta:     BetaParams


class UpdatePriorsRequest(BaseModel):
    verdict:       str   = Field(..., description='"TRUE_ALARM" or "FALSE_ALARM"')
    learning_rate: float = Field(0.5, gt=0, le=1)
    sensors:       list[SensorPriorUpdate]


class UpdatedThetaItem(BaseModel):
    sensor_type_index: int
    alpha:             float
    beta:              float


class UpdatePriorsResponse(BaseModel):
    updated_thetas: list[UpdatedThetaItem]


class HealthResponse(BaseModel):
    status:         str
    pool_available: int
    pool_total:     int
    uptime_seconds: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/infer", response_model=InferResponse, tags=["Inference"])
async def infer(req: InferRequest):
    """
    Run VMP inference for one active incident.

    Pass the current sensor priors (from your Belief Store) and the annotation
    vector for the incident. Optionally supply `warm_start` from a previous
    call on the same incident to reduce VMP iterations.
    """
    grpc_req = pb2.InferRequest(
        incident_id=req.incident_id,
        annotations=req.annotations,
        theta_priors=[pb2.BetaParams(alpha=t.alpha, beta=t.beta)
                      for t in req.theta_priors],
        phi_priors=[pb2.DirichletParams(pseudocounts=p.pseudocounts)
                    for p in req.phi_priors],
        warm_start=req.warm_start or [],
    )

    try:
        resp = await stub().Infer(grpc_req)
    except grpc.aio.AioRpcError as e:
        _handle_grpc_error(e)

    return InferResponse(
        incident_id=resp.incident_id,
        t_dist=list(resp.t_dist),
        threat_level=resp.threat_level,
        confidence=resp.confidence,
        entropy=resp.entropy,
        sensor_reliability=[
            SensorReliabilityItem(
                sensor_type_index=sr.sensor_type_index,
                annotation=sr.annotation,
                spammer_prob=sr.spammer_prob,
                reliability=sr.reliability,
            )
            for sr in resp.sensor_reliability
        ],
        num_observations=resp.num_observations,
        inference_ms=resp.inference_ms,
    )


@app.post("/update-priors", response_model=UpdatePriorsResponse, tags=["Priors"])
async def update_priors(req: UpdatePriorsRequest):
    """
    Compute updated Beta priors after an operator verdict.

    Does not use Infer.NET — pure arithmetic. Supply the spammer posteriors
    from the most recent `/infer` response for the closed incident.
    """
    grpc_req = pb2.UpdatePriorsRequest(
        verdict=req.verdict,
        learning_rate=req.learning_rate,
        sensors=[
            pb2.SensorPriorUpdate(
                sensor_type_index=s.sensor_type_index,
                annotation=s.annotation,
                spammer_prob_mean=s.spammer_prob_mean,
                current_theta=pb2.BetaParams(
                    alpha=s.current_theta.alpha,
                    beta=s.current_theta.beta,
                ),
            )
            for s in req.sensors
        ],
    )

    try:
        resp = await stub().UpdatePriors(grpc_req)
    except grpc.aio.AioRpcError as e:
        _handle_grpc_error(e)

    return UpdatePriorsResponse(
        updated_thetas=[
            UpdatedThetaItem(
                sensor_type_index=t.sensor_type_index,
                alpha=t.alpha,
                beta=t.beta,
            )
            for t in resp.updated_thetas
        ]
    )


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health():
    """Proxy the gRPC Health RPC. Useful for scripted smoke tests."""
    try:
        resp = await stub().Health(pb2.HealthRequest())
    except grpc.aio.AioRpcError as e:
        _handle_grpc_error(e)

    return HealthResponse(
        status=resp.status,
        pool_available=resp.pool_available,
        pool_total=resp.pool_total,
        uptime_seconds=resp.uptime_seconds,
    )


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "MACE Test Gateway", "docs": "/docs", "grpc_target": GRPC_TARGET}


# ---------------------------------------------------------------------------
# Error helper
# ---------------------------------------------------------------------------

def _handle_grpc_error(e: grpc.aio.AioRpcError):
    code = e.code()
    if code == grpc.StatusCode.INVALID_ARGUMENT:
        raise HTTPException(status_code=400, detail=e.details())
    if code == grpc.StatusCode.UNAVAILABLE:
        raise HTTPException(status_code=503, detail=e.details())
    raise HTTPException(status_code=502, detail=f"gRPC {code.name}: {e.details()}")
