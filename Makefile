VENV_DIR ?= .venv
UVICORN_APP ?= app.api:app
UVICORN_HOST ?= 0.0.0.0
UVICORN_PORT ?= 8000
DOCKER_COMPOSE ?= docker compose

.PHONY: help dev docker-up docker-down docker-logs docker-clean activate

help:
	@echo "Available targets:"
	@echo "  make activate      Open a subshell with the virtualenv activated"
	@echo "  make dev           Run the FastAPI app locally with uvicorn"
	@echo "  make docker-up     Build and start the docker compose stack"
	@echo "  make docker-down   Stop the docker compose stack"
	@echo "  make docker-logs   Follow logs from the docker compose stack"
	@echo "  make docker-clean  Stop containers and remove volumes"

activate:
	@echo "Launching new shell with $(VENV_DIR) activated (exit to return)..."
	@bash -c 'source $(VENV_DIR)/bin/activate && exec $$SHELL'

dev:
	$(VENV_DIR)/bin/uvicorn $(UVICORN_APP) --reload --host $(UVICORN_HOST) --port $(UVICORN_PORT)

docker-up:
	$(DOCKER_COMPOSE) up --build -d

docker-down:
	$(DOCKER_COMPOSE) down

docker-logs:
	$(DOCKER_COMPOSE) logs -f

docker-clean:
	$(DOCKER_COMPOSE) down -v
