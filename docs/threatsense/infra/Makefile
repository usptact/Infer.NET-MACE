REGISTRY  := 192.168.10.201:5000
TAG       := $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
NAMESPACE := threatsense
SERVICES  := sensor-gateway incident-manager mace-inference \
             threat-score-svc feedback-processor operator-console

.PHONY: build push deploy rollback status logs clean

## Build all service images
build:
	@for svc in $(SERVICES); do \
	  echo "==> Building $$svc:$(TAG)"; \
	  docker build -t $(REGISTRY)/threatsense/$$svc:$(TAG) \
	               -t $(REGISTRY)/threatsense/$$svc:latest \
	               ../dockerfiles/$$svc/; \
	done

## Push all images to private registry
push:
	@for svc in $(SERVICES); do \
	  echo "==> Pushing $$svc:$(TAG)"; \
	  docker push $(REGISTRY)/threatsense/$$svc:$(TAG); \
	  docker push $(REGISTRY)/threatsense/$$svc:latest; \
	done

## Rolling update all deployments to TAG
deploy:
	@for svc in $(SERVICES); do \
	  kubectl set image deployment/$$svc \
	    $$svc=$(REGISTRY)/threatsense/$$svc:$(TAG) \
	    -n $(NAMESPACE) 2>/dev/null || true; \
	done
	@kubectl rollout status deployment -n $(NAMESPACE) --timeout=300s

## Roll back all deployments one revision
rollback:
	@for svc in $(SERVICES); do \
	  kubectl rollout undo deployment/$$svc -n $(NAMESPACE); \
	done

## Show pod status
status:
	kubectl get pods,svc,pvc -n $(NAMESPACE)

## Tail logs for a specific service: make logs SVC=mace-inference
logs:
	kubectl logs -l app=$(SVC) -n $(NAMESPACE) --tail=200 -f

## Apply all manifests (initial install or config sync)
apply:
	kubectl apply -f k8s/00-namespace.yaml
	kubectl apply -f k8s/01-infrastructure/
	kubectl apply -f k8s/02-registry/
	kubectl apply -f k8s/04-services/
	kubectl apply -f k8s/03-network/

## Remove all ThreatSense workloads (preserves PVCs and namespace)
clean:
	kubectl delete deployments,statefulsets --all -n $(NAMESPACE)
