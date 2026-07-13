"""Async bridge: wraps skydiscover Runner with ChiaEvaluator injection.

Replicates Runner.run()'s initialization sequence so that ChiaEvaluator can be
swapped in AFTER controller creation but BEFORE initial program evaluation.
Signal handlers are guarded for Ray worker thread safety.
"""

import asyncio
import logging
import os
import signal
import tempfile
import threading
from typing import Any, Callable, Dict, Optional

from evolve_flows.evolver.types import EvolverInput, EvolverResult, EvolverStatus
from skydiscover.evaluation.chia_evaluator import ChiaEvaluator
from skydiscover.search.default_discovery_controller import DiscoveryControllerInput
from skydiscover.search.registry import get_program
from skydiscover.search.route import get_discovery_controller
from skydiscover.utils.metrics import get_score

logger = logging.getLogger(__name__)

_TERMINAL_COMPLETED = "completed"
_TERMINAL_MAX_ITERATIONS = "max_iterations"
_TERMINAL_ERROR = "error"
_TERMINAL_STOPPED = "stopped"


def _write_dummy_evaluator() -> str:
    """Create a minimal temp evaluator file that satisfies Runner.__init__."""
    fd, path = tempfile.mkstemp(suffix=".py", prefix="gsd_dummy_eval_")
    with os.fdopen(fd, "w") as f:
        f.write(
            'def evaluate(program_path):\n'
            '    return {"combined_score": 0.0}\n'
        )
    return path


def _guard_signal_handlers(runner: Any) -> None:
    """Monkey-patch Runner._install_signal_handlers to be a no-op off main thread."""
    original = runner._install_signal_handlers

    def guarded() -> None:
        if threading.current_thread() is not threading.main_thread():
            logger.debug("Skipping signal handler install (not main thread)")
            return
        original()

    runner._install_signal_handlers = guarded


async def _run_search_async(
    evolver_input: EvolverInput,
    build_fn: Callable[..., Any],
    run_fn: Callable[..., Any],
    result_mapper_fn: Callable[..., Any],
    *,
    evaluator: Optional[ChiaEvaluator] = None,
    status_callback: Optional[Callable[[EvolverStatus], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
) -> EvolverResult:
    """Core async implementation of the search bridge."""
    from skydiscover.config import load_config
    from skydiscover.runner import Runner
    from skydiscover.search.base_database import Program

    dummy_path = _write_dummy_evaluator()
    config_tmp = None
    try:
        if evolver_input.config_content:
            fd, config_tmp = tempfile.mkstemp(suffix=".yaml", prefix="gsd_cfg_")
            with os.fdopen(fd, "w") as f:
                f.write(evolver_input.config_content)
            config = load_config(config_tmp)
        else:
            config = load_config(evolver_input.config_path)
        runner = Runner(
            evaluation_file=dummy_path,
            initial_program_path=None,
            config=config,
        )
        runner.initial_program_solution = evolver_input.initial_program

        _guard_signal_handlers(runner)

        # --- Replicate Runner.run() flow with evaluator swap ---
        max_iterations = config.max_iterations
        start_iteration = runner.database.last_iteration

        controller_input = DiscoveryControllerInput(
            config=config,
            evaluation_file=dummy_path,
            database=runner.database,
            file_suffix=config.file_suffix,
            output_dir=runner.output_dir,
            evaluator_env_vars=runner.evaluator_env_vars,
        )
        runner.discovery_controller = get_discovery_controller(controller_input)

        # EVALUATOR SWAP — after controller creation, before initial eval
        if evaluator is not None:
            chia_evaluator = evaluator
        else:
            chia_evaluator = ChiaEvaluator(
                build_fn=build_fn,
                run_fn=run_fn,
                result_mapper_fn=result_mapper_fn,
                workloads=["default"],
                output_dir=runner.output_dir,
            )
        runner.discovery_controller.evaluator = chia_evaluator

        # Add initial program
        should_add_initial = (
            start_iteration == 0
            and len(runner.database.programs) == 0
            and runner.initial_program_solution is not None
        )
        if should_add_initial:
            await runner._add_initial_program(start_iteration)

        if status_callback:
            status_callback(EvolverStatus(state="running", iteration=0))

        # Start monitor, install (guarded) signal handlers, run discovery
        monitor_server = None
        try:
            monitor_server = runner._start_monitor(max_iterations)
            runner._setup_human_feedback(monitor_server)
            runner._setup_monitor_summary(monitor_server)
            runner._push_existing_to_monitor()
            runner._install_signal_handlers()

            discovery_start = start_iteration + 1 if should_add_initial else start_iteration
            runner.database.log_status()

            # Status updates via monitor_callback (fires every iteration),
            # separate from checkpoint_callback (fires every N iterations).
            original_monitor_cb = runner.discovery_controller.monitor_callback

            def _status_monitor_cb(program: Any, iteration: int) -> None:
                if original_monitor_cb:
                    try:
                        original_monitor_cb(program, iteration)
                    except Exception:
                        pass
                if status_callback:
                    best = runner._get_best_program()
                    score = get_score(best.metrics) if best and best.metrics else 0.0
                    metrics = best.metrics if best else None
                    status_callback(
                        EvolverStatus(
                            state="running",
                            iteration=iteration,
                            best_score=score,
                            best_metrics=metrics,
                        )
                    )
                if stop_check and stop_check():
                    runner.discovery_controller.request_shutdown()

            runner.discovery_controller.monitor_callback = _status_monitor_cb

            def checkpoint_cb(iteration: int) -> None:
                runner._sync_database()
                runner._save_checkpoint(iteration)

            await runner.discovery_controller.run_discovery(
                discovery_start,
                max_iterations,
                checkpoint_callback=checkpoint_cb,
            )

            runner._sync_database()

        finally:
            if runner.discovery_controller is not None:
                runner.discovery_controller.close()
            if monitor_server:
                try:
                    monitor_server.push_event(
                        {"type": "discovery_complete", "reason": "completed"}
                    )
                    monitor_server.stop()
                except Exception:
                    logger.debug("Failed to stop monitor server", exc_info=True)

        # Collect results
        best = runner._get_best_program()
        was_stopped = stop_check() if stop_check else False

        if was_stopped:
            terminal = _TERMINAL_STOPPED
        elif best is not None:
            terminal = _TERMINAL_COMPLETED
        else:
            terminal = _TERMINAL_MAX_ITERATIONS

        population = [p.to_dict() for p in runner.database.programs.values()]
        log_path = getattr(chia_evaluator, "_log_path", None)

        return EvolverResult(
            best_program=best.solution if best else "",
            best_metrics=dict(best.metrics) if best and best.metrics else {},
            iteration_count=len(runner.database.programs),
            terminal_status=terminal,
            population=population,
            metrics_log_path=log_path,
        )

    except Exception as exc:
        logger.error("run_skydiscover failed: %s", exc, exc_info=True)
        return EvolverResult(
            best_program="",
            best_metrics={},
            iteration_count=0,
            terminal_status=_TERMINAL_ERROR,
            population=[],
            error_message=str(exc),
        )
    finally:
        if os.path.exists(dummy_path):
            os.unlink(dummy_path)
        if config_tmp and os.path.exists(config_tmp):
            os.unlink(config_tmp)


def run_skydiscover(
    evolver_input: EvolverInput,
    build_fn: Callable[..., Any],
    run_fn: Callable[..., Any],
    result_mapper_fn: Callable[..., Any],
    *,
    evaluator: Optional[ChiaEvaluator] = None,
    status_callback: Optional[Callable[[EvolverStatus], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
) -> EvolverResult:
    """Bridge skydiscover's async Runner into a synchronous call for Ray actors.

    Handles event loop detection: uses asyncio.run() when no loop exists,
    or loop.run_until_complete() when called from an existing loop context.
    """
    coro = _run_search_async(
        evolver_input,
        build_fn,
        run_fn,
        result_mapper_fn,
        evaluator=evaluator,
        status_callback=status_callback,
        stop_check=stop_check,
    )
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        return asyncio.run(coro)
    else:
        return loop.run_until_complete(coro)
