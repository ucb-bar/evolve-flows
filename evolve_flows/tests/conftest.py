"""Shared pytest fixtures and constants for EvolverNode unit and integration tests."""

import os
from typing import Any, Dict

import pytest
import yaml

from evolve_flows.evolver.types import EvolverInput

SEED_PROGRAM = "def solve(x):\n    return x + 1\n"
MOCK_LLM_CODE = "def solve(x):\n    return x ** 2 + 1\n"


@pytest.fixture()
def make_config_yaml(tmp_path: Any):
    """Factory that writes a minimal skydiscover YAML config and returns its path.

    Accepts keyword overrides shallow-merged into the defaults before dumping.
    """

    def _factory(**overrides: Any) -> str:
        defaults: Dict[str, Any] = {
            "max_iterations": 5,
            "diff_based_generation": False,
            "monitor": {"enabled": False},
            "search": {"type": "topk"},
            "llm": {
                "models": [
                    {
                        "name": "fake-model",
                        "api_key": "fake",
                        "api_base": "http://localhost:1",
                    }
                ]
            },
        }
        defaults.update(overrides)
        path = os.path.join(str(tmp_path), "test_config.yaml")
        with open(path, "w") as f:
            yaml.dump(defaults, f)
        return path

    return _factory


def make_evolver_input(*, config_path: str, initial_program: str) -> EvolverInput:
    """Construct an EvolverInput for tests."""
    return EvolverInput(config_path=config_path, initial_program=initial_program)
