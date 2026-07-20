"""EvolverNode — Ray actor wrapping skydiscover evolutionary search for CHIA."""

import logging
import os
from typing import Any, Callable, Dict, Optional

import ray
from chia.base.ChiaFunction import ChiaFunction

from evolve_flows.evolver.bridge import run_skydiscover
from evolve_flows.evolver.types import EvolverInput, EvolverResult, EvolverStatus
from skydiscover.evaluation.chia_evaluator import ChiaEvaluator

logger = logging.getLogger(__name__)


@ChiaFunction(resources={"evolver": 1.0})
def run_evolver(
    evolver_input: EvolverInput,
    build_fn: Callable[..., Any],
    run_fn: Callable[..., Any],
    result_mapper_fn: Callable[..., Any],
) -> EvolverResult:
    """CHIA-native entry point for evolutionary search.

    Dispatches via .chia_remote() with default resource tag {'evolver': 1.0}.
    Override resources at call time via
    run_evolver.options(resources={...}).chia_remote(...).

    One-shot function (no status polling / stop control) — use the EvolverNode
    actor when stateful progress polling is required.
    """
    if not evolver_input.config_content and not os.path.isfile(evolver_input.config_path):
        raise ValueError(
            f"config_path does not exist: {evolver_input.config_path}"
        )
    if not evolver_input.initial_program or not evolver_input.initial_program.strip():
        raise ValueError("initial_program must be a non-empty string")

    return run_skydiscover(evolver_input, build_fn, run_fn, result_mapper_fn)


@ray.remote(max_concurrency=2)
class EvolverNode:
    """Stateful Ray actor that runs skydiscover search with CHIA evaluation.

    Callables (build_fn, run_fn, result_mapper_fn) are passed as method
    arguments to run_search(), not as constructor args or EvolverInput fields,
    to avoid pickle issues with Ray actor state.
    """

    def __init__(self) -> None:
        logging.basicConfig(level=logging.INFO, force=True)
        self._status = EvolverStatus(state="idle")
        self._stop_requested = False
        self._result: Optional[EvolverResult] = None

    def run_search(
        self,
        evolver_input: EvolverInput,
        build_fn: Callable[..., Any],
        run_fn: Callable[..., Any],
        result_mapper_fn: Callable[..., Any],
        evaluator: Optional[ChiaEvaluator] = None,
    ) -> EvolverResult:
        """Execute an evolutionary search run.

        Args:
            evolver_input: Config path and initial program.
            build_fn: CHIA simulator build callable.
            run_fn: CHIA simulator run callable.
            result_mapper_fn: Maps raw run results to EvaluationResult.
            evaluator: Optional pre-configured evaluator (e.g. ChampSimEvaluator
                with binary-capture logic). If provided, the bridge uses it
                instead of creating a plain ChiaEvaluator.

        Returns:
            EvolverResult with search outcome.
        """
        if not os.path.isfile(evolver_input.config_path):
            raise ValueError(
                f"config_path does not exist: {evolver_input.config_path}"
            )
        if not evolver_input.initial_program or not evolver_input.initial_program.strip():
            raise ValueError("initial_program must be a non-empty string")

        self._stop_requested = False
        self._status = EvolverStatus(state="running", iteration=0)

        def status_callback(status: EvolverStatus) -> None:
            self._status = status

        def stop_check() -> bool:
            return self._stop_requested

        try:
            result = run_skydiscover(
                evolver_input,
                build_fn,
                run_fn,
                result_mapper_fn,
                evaluator=evaluator,
                status_callback=status_callback,
                stop_check=stop_check,
            )
            self._result = result

            if result.terminal_status == "error":
                self._status = EvolverStatus(
                    state="error",
                    iteration=result.iteration_count,
                    best_score=0.0,
                )
            else:
                best_score = 0.0
                if result.best_metrics:
                    from skydiscover.utils.metrics import get_score

                    best_score = get_score(result.best_metrics)
                self._status = EvolverStatus(
                    state="completed",
                    iteration=result.iteration_count,
                    best_score=best_score,
                    best_metrics=result.best_metrics,
                )

            return result

        except Exception as exc:
            logger.error("EvolverNode.run_search failed: %s", exc, exc_info=True)
            self._status = EvolverStatus(state="error")
            raise

    def get_status(self) -> Dict[str, Any]:
        """Return current search status as a plain dict for trivial Ray serialization."""
        return {
            "state": self._status.state,
            "iteration": self._status.iteration,
            "best_score": self._status.best_score,
            "best_metrics": self._status.best_metrics,
            "best_program": self._status.best_program,
        }

    def stop(self) -> None:
        """Request graceful shutdown of the running search."""
        logger.info("EvolverNode stop requested")
        self._stop_requested = True
