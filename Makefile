# Reproduce every number in the README.
#   make all   runs the full pipeline

PYTHON ?= python
WORKERS ?= 12

.PHONY: help install test frozenlake cartpole sweep compare figures all clean

help:
	@echo "install     install the package and dev dependencies"
	@echo "test        run the test suite (37 tests)"
	@echo "frozenlake  train against the exact dynamic-programming optimum"
	@echo "cartpole    train on the discretised continuous task"
	@echo "sweep       discretisation resolution over five seeds (parallel)"
	@echo "compare     Q-learning vs SARSA, paired over eight seeds (parallel)"
	@echo "figures     regenerate every figure in the README"
	@echo "all         test -> frozenlake -> cartpole -> sweep -> compare -> figures"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

frozenlake:
	$(PYTHON) scripts/train_frozenlake.py

cartpole:
	$(PYTHON) scripts/train_cartpole.py

sweep:
	$(PYTHON) scripts/sweep_resolution.py --workers $(WORKERS)

compare:
	$(PYTHON) scripts/compare_agents.py --workers $(WORKERS)

figures:
	$(PYTHON) scripts/make_figures.py

all: test frozenlake cartpole sweep compare figures

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
