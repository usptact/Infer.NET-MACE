PYTHON  ?= python3
UVICORN ?= uvicorn
PORT    ?= 8000
TARGET  ?= localhost:8080

.PHONY: install generate run test curl-health curl-infer curl-feedback curl-workers

install:
	$(PYTHON) -m pip install -r requirements.txt

generate:
	bash generate_stubs.sh

# ── Automated gateway tests (requires a running service + this gateway) ────
# Override the target gateway with MACE_GATEWAY_URL; the suite skips itself if
# the gateway is unreachable.
test:
	MACE_GATEWAY_URL=http://localhost:$(PORT) $(PYTHON) -m pytest -v test_gateway.py

run: generate
	MACE_GRPC_TARGET=$(TARGET) $(UVICORN) main:app --reload --port $(PORT)

# ── Quick smoke tests (requires a running service + this gateway) ──────────

curl-health:
	curl -s http://localhost:$(PORT)/health | $(PYTHON) -m json.tool

curl-infer:
	curl -s -X POST http://localhost:$(PORT)/infer \
	  -H "Content-Type: application/json" \
	  -d '{"item_id":"smoke-001","annotations":[{"worker":0,"label":2},{"worker":1,"label":2},{"worker":3,"label":0}]}' \
	  | $(PYTHON) -m json.tool

# Teaches the service that worker 3 contradicted an established truth.
curl-feedback:
	curl -s -X POST http://localhost:$(PORT)/feedback \
	  -H "Content-Type: application/json" \
	  -d '{"item_id":"smoke-001","annotations":[{"worker":0,"label":2},{"worker":1,"label":2},{"worker":3,"label":0}],"true_label":2,"learning_rate":1.0}' \
	  | $(PYTHON) -m json.tool

curl-workers:
	curl -s http://localhost:$(PORT)/workers | $(PYTHON) -m json.tool
