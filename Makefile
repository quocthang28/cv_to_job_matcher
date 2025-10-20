PYTHON ?= python3
VENV_DIR ?= .venv
UVICORN_APP ?= app.api:app
UVICORN_HOST ?= 0.0.0.0
UVICORN_PORT ?= 8000
DOCKER_COMPOSE ?= docker compose

.PHONY: help venv dev docker-up docker-down docker-logs activate clean

help:
	@echo "Available targets:"
	@echo "  make venv        Create virtual environment and install dependencies"
	@echo "  make activate    Open a subshell with the virtualenv activated"
	@echo "  make dev         Run the FastAPI app locally with uvicorn"
	@echo "  make docker-up   Build and start the docker compose stack"
	@echo "  make docker-down Stop the docker compose stack"
	@echo "  make docker-logs Follow logs from the docker compose stack"
	@echo "  make clean       Remove the virtualenv directory"

$(VENV_DIR)/.venv_ready: requirements.txt
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r requirements.txt
	touch $(VENV_DIR)/.venv_ready

venv: $(VENV_DIR)/.venv_ready

activate: venv
	@echo "Launching new shell with $(VENV_DIR) activated (exit to return)..."
	@bash -c 'source $(VENV_DIR)/bin/activate && exec $$SHELL'

dev: venv
	$(VENV_DIR)/bin/uvicorn $(UVICORN_APP) --reload --host $(UVICORN_HOST) --port $(UVICORN_PORT)

docker-up:
	$(DOCKER_COMPOSE) up --build

docker-down:
	$(DOCKER_COMPOSE) down

docker-logs:
	$(DOCKER_COMPOSE) logs -f

clean:
	rm -rf $(VENV_DIR)
