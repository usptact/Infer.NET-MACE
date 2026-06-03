PYTHON  ?= python3
UVICORN ?= uvicorn
PORT    ?= 8000
TARGET  ?= localhost:8080

.PHONY: install generate run curl-health curl-infer curl-update

install:
	$(PYTHON) -m pip install -r requirements.txt

generate:
	bash generate_stubs.sh

run: generate
	MACE_GRPC_TARGET=$(TARGET) $(UVICORN) main:app --reload --port $(PORT)

# ── Quick smoke tests (requires a running MACE pod + this gateway) ──────────

curl-health:
	curl -s http://localhost:$(PORT)/health | python3 -m json.tool

curl-infer:
	curl -s -X POST http://localhost:$(PORT)/infer \
	  -H "Content-Type: application/json" \
	  -d '{ \
	    "incident_id": "smoke-001", \
	    "annotations": [3, 3, -1, -1, 2, -1, 0], \
	    "theta_priors": [ \
	      {"alpha":1,"beta":9},{"alpha":1,"beta":9},{"alpha":5,"beta":5}, \
	      {"alpha":5,"beta":5},{"alpha":2,"beta":8},{"alpha":5,"beta":5}, \
	      {"alpha":5,"beta":5} \
	    ], \
	    "phi_priors": [ \
	      {"pseudocounts":[1,1,1,1,1]},{"pseudocounts":[1,1,1,1,1]}, \
	      {"pseudocounts":[1,1,1,1,1]},{"pseudocounts":[1,1,1,1,1]}, \
	      {"pseudocounts":[1,1,1,1,1]},{"pseudocounts":[1,1,1,1,1]}, \
	      {"pseudocounts":[1,1,1,1,1]} \
	    ] \
	  }' | python3 -m json.tool

curl-update:
	curl -s -X POST http://localhost:$(PORT)/update-priors \
	  -H "Content-Type: application/json" \
	  -d '{ \
	    "verdict": "TRUE_ALARM", \
	    "learning_rate": 0.5, \
	    "sensors": [ \
	      {"sensor_type_index":0,"annotation":3,"spammer_prob_mean":0.09, \
	       "current_theta":{"alpha":1,"beta":9}}, \
	      {"sensor_type_index":1,"annotation":3,"spammer_prob_mean":0.12, \
	       "current_theta":{"alpha":1,"beta":9}} \
	    ] \
	  }' | python3 -m json.tool
