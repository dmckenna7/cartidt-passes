PY ?= python

.PHONY: help install dev lint type test smoke docker clean

help:
	@echo "targets:"
	@echo "  install  - editable install of the cartidt package"
	@echo "  dev      - editable install with the dev extras (ruff/black/isort/mypy/pytest)"
	@echo "  lint     - ruff + black --check + isort --check-only"
	@echo "  type     - mypy --strict on the cartidt package"
	@echo "  test     - run the full pytest suite"
	@echo "  smoke    - one optimiser step on the tiny unit config"
	@echo "  docker   - build the cartidt:latest image"
	@echo "  clean    - remove caches, runs, and build artefacts"

install:
	$(PY) -m pip install -e .

dev:
	$(PY) -m pip install -e ".[dev]"

lint:
	$(PY) -m ruff check .
	$(PY) -m black --check .
	$(PY) -m isort --check-only .

type:
	$(PY) -m mypy --strict cartidt

test:
	$(PY) -m pytest -q

smoke:
	$(PY) -m cartidt.driver.train --config configs/_unittest.yaml --out ./runs/smoke --steps 1

docker:
	docker build -t cartidt:latest .

clean:
	rm -rf ./runs ./eval build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache
