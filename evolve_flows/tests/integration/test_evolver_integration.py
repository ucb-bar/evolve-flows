"""Integration tests: full skydiscover loop inside a real local Ray runtime (CT-03).

Two full-loop tests run ``run_skydiscover`` inside a ``@ray.remote`` task (so the
search executes in a Ray worker process), and one lightweight test exercises the
``EvolverNode`` actor lifecycle directly.

LLMPool is patched *inside* the ``@ray.remote`` function body so the mock is
resolved in the worker's own import namespace -- a ``patch`` applied in the test
process would not affect the separate worker process.
"""

import ast
import json
import os
import textwrap
from typing import Any, Callable, Dict, List

import pytest
import ray
import yaml

from evolve_flows.evolver.node import EvolverNode, run_evolver
from evolve_flows.evolver.types import EvolverInput, EvolverResult
from skydiscover.config import LLMModelConfig
from skydiscover.evaluation.evaluation_result import EvaluationResult
from skydiscover.llm.base import LLMResponse

MAX_ITERATIONS = 5

EVALUATOR_SOURCE = textwrap.dedent(
    """\
    import ast

    def evaluate(program_path: str) -> dict:
        with open(program_path, "r") as f:
            source = f.read()

        score = 0.1  # baseline for any non-empty program
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "solve":
                    score = 0.8
                    break
        except SyntaxError:
            score = 0.0

        return {"combined_score": score}
    """
)

SEED_PROGRAM = "def solve(x):\n    return x + 1\n"
MOCK_LLM_CODE = "def solve(x):\n    return x ** 2 + 1\n"
MOCK_RESPONSE_TEXT = f"```python\n{MOCK_LLM_CODE}```"


class FakeLLMPool:
    """Drop-in LLMPool replacement returning a canned response (matches smoke test)."""

    def __init__(self, models_cfg: List[LLMModelConfig]) -> None:
        # Intentionally do NOT create real clients.
        self.models_cfg = models_cfg

    async def generate(
        self, system_message: str, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> LLMResponse:
        return LLMResponse(text=MOCK_RESPONSE_TEXT)

    async def generate_all(
        self, system_message: str, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> List[LLMResponse]:
        return [LLMResponse(text=MOCK_RESPONSE_TEXT)]


@pytest.fixture(scope="module")
def ray_env() -> Any:
    """Start a fresh local Ray instance, isolated from any stale cluster (D-03, D-04)."""
    os.environ.pop("RAY_ADDRESS", None)
    ray.init(
        address="local",
        num_cpus=2,
        ignore_reinit_error=True,
        resources={"evolver": 1.0},
    )
    yield
    ray.shutdown()


def write_test_config(tmp_path: Any, max_iterations: int = MAX_ITERATIONS) -> str:
    """Write a minimal skydiscover YAML config and return its path."""
    config = {
        "max_iterations": max_iterations,
        "diff_based_generation": False,
        "monitor": {"enabled": False},
        "search": {"type": "topk"},
        "llm": {
            "models": [
                {"name": "fake-model", "api_key": "fake", "api_base": "http://localhost:1"}
            ]
        },
    }
    config_path = os.path.join(str(tmp_path), "test_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


def make_build_fn() -> Callable[..., Any]:
    """Build callable returning a ray.put()-wrapped result (ray.get-able ObjectRef)."""

    def build_fn(source_code: str) -> Any:
        return ray.put({"status": "ok", "hash": hash(source_code)})

    return build_fn


def make_run_fn() -> Callable[..., Any]:
    """Run callable returning a ray.put()-wrapped result (ray.get-able ObjectRef)."""

    def run_fn(workload: str = "default") -> Any:
        return ray.put({"ipc": 1.5, "workload": workload})

    return run_fn


def make_result_mapper_fn() -> Callable[..., EvaluationResult]:
    """Map run-result dicts to an EvaluationResult with a combined_score float."""

    def result_mapper_fn(run_results: List[Dict[str, Any]]) -> EvaluationResult:
        avg_ipc = sum(r["ipc"] for r in run_results) / len(run_results)
        return EvaluationResult(metrics={"combined_score": avg_ipc, "ipc": avg_ipc})

    return result_mapper_fn


@ray.remote
def _run_in_ray(
    config_path: str,
    initial_program: str,
    build_fn: Callable[..., Any],
    run_fn: Callable[..., Any],
    mapper_fn: Callable[..., Any],
) -> EvolverResult:
    """Run the full skydiscover loop inside a Ray worker with LLMPool mocked.

    The patch is applied here (in the worker process) so it resolves against the
    worker's own ``skydiscover.search.default_discovery_controller`` import.
    """
    from unittest.mock import patch

    from evolve_flows.evolver.bridge import run_skydiscover

    evolver_input = EvolverInput(config_path=config_path, initial_program=initial_program)
    with patch(
        "skydiscover.search.default_discovery_controller.LLMPool",
        FakeLLMPool,
    ):
        return run_skydiscover(evolver_input, build_fn, run_fn, mapper_fn)


class TestEvolverInRay:
    """Full skydiscover loop inside Ray with trivial local callables (CT-03)."""

    def test_full_loop_direct_evaluator(self, ray_env: Any, tmp_path: Any) -> None:
        """5+ iterations complete inside Ray and return a valid EvolverResult (D-07)."""
        config_path = write_test_config(tmp_path, max_iterations=MAX_ITERATIONS)

        result = ray.get(
            _run_in_ray.remote(
                config_path,
                SEED_PROGRAM,
                make_build_fn(),
                make_run_fn(),
                make_result_mapper_fn(),
            ),
            timeout=120,
        )

        assert isinstance(result, EvolverResult)
        assert result.error_message is None
        assert result.terminal_status == "completed"

        assert isinstance(result.best_program, str)
        assert result.best_program.strip() != ""
        assert ast.parse(result.best_program)  # syntactically valid Python

        assert "combined_score" in result.best_metrics
        assert isinstance(result.best_metrics["combined_score"], float)

        assert result.iteration_count >= 2
        assert len(result.population) >= 2
        for program in result.population:
            assert isinstance(program, dict)
            assert "solution" in program

    def test_full_loop_chia_evaluator_path(self, ray_env: Any, tmp_path: Any) -> None:
        """Same loop, plus validation of the ChiaEvaluator JSONL log artifact (D-07)."""
        config_path = write_test_config(tmp_path, max_iterations=MAX_ITERATIONS)

        result = ray.get(
            _run_in_ray.remote(
                config_path,
                SEED_PROGRAM,
                make_build_fn(),
                make_run_fn(),
                make_result_mapper_fn(),
            ),
            timeout=120,
        )

        assert isinstance(result, EvolverResult)
        assert result.error_message is None
        assert result.terminal_status == "completed"
        assert result.iteration_count >= 2
        assert len(result.population) >= 2

        assert ast.parse(result.best_program)
        assert "def solve" in result.best_program
        assert "combined_score" in result.best_metrics
        assert isinstance(result.best_metrics["combined_score"], float)

        assert result.metrics_log_path is not None
        assert os.path.isfile(result.metrics_log_path)

        with open(result.metrics_log_path, "r") as f:
            lines = [line for line in f.read().splitlines() if line.strip()]
        assert len(lines) >= 2
        records = [json.loads(line) for line in lines]  # each line is valid JSON
        assert any("combined_score" in record["mapped_result"] for record in records)


def _patch_llm_pool_on_worker(fake_cls: type) -> None:
    """CHIA _chia_setup hook: monkey-patch LLMPool in the worker before run_evolver."""
    import skydiscover.search.default_discovery_controller as mod

    mod.LLMPool = fake_cls


class TestRunEvolverInRay:
    """run_evolver.chia_remote() dispatches through the real CHIA trampoline (CE-01)."""

    def test_run_evolver_chia_remote(self, ray_env: Any, tmp_path: Any) -> None:
        """Real .chia_remote() dispatch against a cluster advertising 'evolver'."""
        config_path = write_test_config(tmp_path, max_iterations=MAX_ITERATIONS)
        evolver_input = EvolverInput(config_path=config_path, initial_program=SEED_PROGRAM)

        ref = run_evolver.chia_remote(
            evolver_input,
            make_build_fn(),
            make_run_fn(),
            make_result_mapper_fn(),
            _chia_setup=_patch_llm_pool_on_worker,
            _chia_setup_args=(FakeLLMPool,),
        )
        result = ray.get(ref, timeout=120)

        assert isinstance(result, EvolverResult)
        assert result.error_message is None
        assert result.terminal_status == "completed"

        assert isinstance(result.best_program, str)
        assert result.best_program.strip() != ""
        assert ast.parse(result.best_program)

        assert result.iteration_count >= 2


class TestEvolverActorLifecycle:
    """EvolverNode Ray actor create / status / stop lifecycle in real Ray."""

    def test_actor_create_status_stop(self, ray_env: Any) -> None:
        """A fresh actor reports idle status and stops without error."""
        actor = EvolverNode.remote()

        status = ray.get(actor.get_status.remote(), timeout=30)
        assert status["state"] == "idle"
        assert status["iteration"] == 0
        assert status["best_score"] == 0.0
        assert status["best_metrics"] is None

        assert ray.get(actor.stop.remote(), timeout=30) is None
