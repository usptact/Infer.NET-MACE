# MACE Inference Test Gateway

A FastAPI app that bridges HTTP/JSON → gRPC for the MACE Inference Service.
Use it for manual testing, curl scripts, or the Swagger UI during development.

## Quick start

```bash
# 1. Install Python dependencies
make install

# 2. Generate gRPC stubs from the shared .proto file
make generate

# 3. Start the gateway (assumes MACE pod is already running on localhost:8080)
make run

# 4. Open the Swagger UI
open http://localhost:8000/docs
```

If the MACE pod is on a different host or port:
```bash
make run TARGET=192.168.10.100:8080
```

## Smoke tests

```bash
make curl-health    # GET /health
make curl-infer     # POST /infer  (camera=HIGH + mic=HIGH + badge=MEDIUM + time=CLEAR)
make curl-update    # POST /update-priors  (TRUE_ALARM, reinforce camera + mic)
```

## Automated tests

`test_gateway.py` drives the gateway end-to-end (HTTP → gRPC): the EM θ/φ update,
`FALSE_ALARM` gold-pinning, absent-sensor echo, deprecated-field back-compat, and
validation (422/400). It requires a running stack and **skips itself** if the
gateway is unreachable.

```bash
docker compose up -d           # from the repo root
make test                      # or: MACE_GATEWAY_URL=http://host:port pytest -v test_gateway.py
```

## Direct curl examples

### POST /infer
```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "incident_id": "lobby-001",
    "annotations": [3, 3, -1, -1, 2, -1, 0],
    "theta_priors": [
      {"alpha":1,"beta":9},{"alpha":1,"beta":9},{"alpha":5,"beta":5},
      {"alpha":5,"beta":5},{"alpha":2,"beta":8},{"alpha":5,"beta":5},
      {"alpha":5,"beta":5}
    ],
    "phi_priors": [
      {"pseudocounts":[1,1,1,1,1]},{"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]},{"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]},{"pseudocounts":[1,1,1,1,1]},
      {"pseudocounts":[1,1,1,1,1]}
    ]
  }'
```

### POST /update-priors (after operator confirms TRUE_ALARM)
```bash
curl -X POST http://localhost:8000/update-priors \
  -H "Content-Type: application/json" \
  -d '{
    "verdict": "TRUE_ALARM",
    "learning_rate": 0.5,
    "true_threat_level": 3,
    "sensors": [
      {"sensor_type_index":0, "sensor_reading":3,
       "current_theta":{"alpha":1,"beta":9},
       "current_phi":{"pseudocounts":[1,1,1,1,1]}},
      {"sensor_type_index":1, "sensor_reading":3,
       "current_theta":{"alpha":1,"beta":9},
       "current_phi":{"pseudocounts":[1,1,1,1,1]}}
    ]
  }'
```

## Annotation index → sensor type mapping

The default 7 sensor types (matches `Inference:NumSensorTypes` in `appsettings.json`):

| Index | Sensor type     | Threat labels           |
|-------|-----------------|-------------------------|
| 0     | CAMERA_CV       | 0=clear … 3=high person |
| 1     | MICROPHONE      | 0=quiet … 3=gunshot     |
| 2     | ACCESS_CONTROL  | 0=normal … 3=forced     |
| 3     | DOOR_SENSOR     | 0=closed … 3=forced     |
| 4     | GLASS_BREAK     | 0=none, 3=detected      |
| 5     | BADGE_READER    | 0=valid … 3=invalid OOH |
| 6     | TIME_CONTEXT    | 0=business … 2=3am      |

## Environment variables

| Variable          | Default         | Description                     |
|-------------------|-----------------|---------------------------------|
| `MACE_GRPC_TARGET` | `localhost:8080` | gRPC server address and port    |
