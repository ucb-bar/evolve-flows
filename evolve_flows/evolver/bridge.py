"""Async bridge: wraps skydiscover Runner with ChiaEvaluator injection.

Replicates Runner.run()'s initialization sequence so that ChiaEvaluator can be
swapped in AFTER controller creation but BEFORE initial program evaluation.
Signal handlers are guarded for Ray worker thread safety.

For external backends (AlphaEvolve, OpenEvolve, etc.) the bridge writes a
thin evaluator shim that translates the file-based ``evaluate(path)`` contract
into ``ChiaEvaluator.evaluate_program(source_code)``, keeping the entire
evaluation pipeline (build → run → map) identical to the native path.
"""

import asyncio
import logging
import os
import signal
import tempfile
import threading
import uuid
from typing import Any, Callable, Dict, Optional

import yaml

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

# Registry for passing ChiaEvaluator instances to file-based evaluator shims.
# Keyed by a unique ID so concurrent runs don't collide.
_EVALUATOR_REGISTRY: Dict[str, ChiaEvaluator] = {}


def _write_dummy_evaluator() -> str:
    """Create a minimal temp evaluator file that satisfies Runner.__init__."""
    fd, path = tempfile.mkstemp(suffix=".py", prefix="ef_dummy_eval_")
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


# ---------------------------------------------------------------------------
# External-backend evaluator shim
# ---------------------------------------------------------------------------

_SHIM_TEMPLATE = '''\
"""Auto-generated evaluator shim — delegates to ChiaEvaluator via bridge registry."""
import asyncio
import threading

_KEY = {key!r}

def evaluate(program_path):
    from evolve_flows.evolver.bridge import _EVALUATOR_REGISTRY
    evaluator = _EVALUATOR_REGISTRY.get(_KEY)
    if evaluator is None:
        return {{"combined_score": 0.0}}
    with open(program_path, "r") as f:
        source_code = f.read()
    # Run the async evaluate_program in a dedicated thread with its own
    # event loop so we never nest inside the external backend's loop.
    result_box = [None]
    exc_box = [None]
    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_box[0] = loop.run_until_complete(
                evaluator.evaluate_program(source_code)
            )
        except Exception as e:
            exc_box[0] = e
        finally:
            loop.close()
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join()
    if exc_box[0] is not None:
        return {{"combined_score": 0.0, "error": str(exc_box[0])}}
    if result_box[0] is None:
        return {{"combined_score": 0.0}}
    return result_box[0].metrics
'''


def _write_chia_evaluator_shim(key: str) -> str:
    """Write a temp Python evaluator file that delegates to a registered ChiaEvaluator."""
    fd, path = tempfile.mkstemp(suffix=".py", prefix="chia_eval_shim_")
    with os.fdopen(fd, "w") as f:
        f.write(_SHIM_TEMPLATE.format(key=key))
    return path


def _is_external_search(search_type: str) -> bool:
    """Check whether *search_type* is handled by an external backend."""
    try:
        from skydiscover.extras.external import is_external
        return is_external(search_type)
    except ImportError:
        return False


async def _run_external_search(
    evolver_input: EvolverInput,
    config: Any,
    chia_evaluator: ChiaEvaluator,
    *,
    status_callback: Optional[Callable[[EvolverStatus], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
) -> EvolverResult:
    """Run search using an external backend while evaluating through ChiaEvaluator."""
    from skydiscover.extras.external import get_runner

    search_type = config.search.type

    # Config.from_dict drops unknown YAML sections (like ``alphaevolve:``).
    # Re-parse the raw YAML to extract the backend-specific section and
    # attach it to the config object so the backend's config resolver finds it.
    if evolver_input.config_content:
        raw = yaml.safe_load(evolver_input.config_content)
        backend_section = raw.get(search_type)
        if backend_section and isinstance(backend_section, dict):
            setattr(config, search_type, backend_section)

    # Propagate system prompt so external backends can use it as
    # problem_description (they read config.system_prompt_override).
    if not getattr(config, "system_prompt_override", None):
        cb = getattr(config, "context_builder", None)
        if cb and getattr(cb, "system_message", None):
            config.system_prompt_override = cb.system_message

    # Write seed program to a temp file (external backends expect a path).
    fd, program_path = tempfile.mkstemp(
        suffix=getattr(config, "file_suffix", ".py"),
        prefix="seed_",
    )
    with os.fdopen(fd, "w") as f:
        f.write(evolver_input.initial_program)

    # Register evaluator and write the shim file.
    shim_key = str(uuid.uuid4())
    _EVALUATOR_REGISTRY[shim_key] = chia_evaluator
    evaluator_path = _write_chia_evaluator_shim(shim_key)

    if status_callback:
        status_callback(EvolverStatus(state="running", iteration=0))

    _best_score = [0.0]
    _best_metrics: Dict[str, Any] = {}
    _best_program = [""]

    def _monitor_cb(program: Any, iteration: int) -> None:
        score = get_score(program.metrics) if program and program.metrics else 0.0
        if score > _best_score[0]:
            _best_score[0] = score
            _best_metrics.update(program.metrics)
            _best_program[0] = program.solution if program else ""

    def _status_from_eval(evaluated: int) -> None:
        if status_callback:
            status_callback(EvolverStatus(
                state="running",
                iteration=evaluated,
                best_score=_best_score[0],
                best_metrics=_best_metrics or None,
                best_program=_best_program[0] or None,
            ))

    try:
        result = await get_runner(search_type)(
            program_path=program_path,
            evaluator_path=evaluator_path,
            config_obj=config,
            iterations=config.max_iterations,
            output_dir=chia_evaluator.output_dir,
            monitor_callback=_monitor_cb,
            feedback_reader=None,
            stop_check=stop_check,
            status_callback=_status_from_eval,
        )

        best_code = result.best_solution or ""
        metrics = dict(result.metrics) if result.metrics else {}
        best_score = result.best_score
        log_path = getattr(chia_evaluator, "_log_path", None)

        if status_callback:
            status_callback(EvolverStatus(
                state="completed",
                iteration=0,
                best_score=best_score,
                best_metrics=metrics,
                best_program=best_code,
            ))

        return EvolverResult(
            best_program=best_code,
            best_metrics=metrics,
            iteration_count=0,
            terminal_status=_TERMINAL_COMPLETED,
            population=[],
            metrics_log_path=log_path,
        )

    except Exception as exc:
        logger.error("External backend '%s' failed: %s", search_type, exc, exc_info=True)
        return EvolverResult(
            best_program="",
            best_metrics={},
            iteration_count=0,
            terminal_status=_TERMINAL_ERROR,
            population=[],
            error_message=str(exc),
        )
    finally:
        _EVALUATOR_REGISTRY.pop(shim_key, None)
        for p in (program_path, evaluator_path):
            try:
                os.unlink(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Native search path
# ---------------------------------------------------------------------------


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
            fd, config_tmp = tempfile.mkstemp(suffix=".yaml", prefix="ef_cfg_")
            with os.fdopen(fd, "w") as f:
                f.write(evolver_input.config_content)
            config = load_config(config_tmp)
        else:
            config = load_config(evolver_input.config_path)

        # Route external backends (alphaevolve, openevolve, …) through their
        # own SDK while still evaluating via the same ChiaEvaluator pipeline.
        search_type = getattr(config.search, "type", None)
        if search_type and _is_external_search(search_type):
            logger.info("Detected external backend '%s' — routing through shim evaluator", search_type)
            if evaluator is not None:
                chia_evaluator = evaluator
            else:
                chia_evaluator = ChiaEvaluator(
                    build_fn=build_fn,
                    run_fn=run_fn,
                    result_mapper_fn=result_mapper_fn,
                    workloads=["default"],
                    output_dir=os.path.join(tempfile.gettempdir(), "chia_eval"),
                )
            return await _run_external_search(
                evolver_input,
                config,
                chia_evaluator,
                status_callback=status_callback,
                stop_check=stop_check,
            )

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
            best = runner._get_best_program()
            score = get_score(best.metrics) if best and best.metrics else 0.0
            status_callback(EvolverStatus(
                state="running",
                iteration=0,
                best_score=score,
                best_metrics=best.metrics if best else None,
                best_program=best.solution if best else None,
            ))

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
            _max_iteration = 0

            def _status_monitor_cb(program: Any, iteration: int) -> None:
                nonlocal _max_iteration
                _max_iteration = max(_max_iteration, iteration)
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
                            iteration=_max_iteration,
                            best_score=score,
                            best_metrics=metrics,
                            best_program=best.solution if best else None,
                        )
                    )
                if stop_check and stop_check():
                    runner.discovery_controller.request_shutdown()

            runner.discovery_controller.monitor_callback = _status_monitor_cb

            def checkpoint_cb(iteration: int) -> None:
                runner._sync_database()
                runner._save_checkpoint(iteration)

            async def _poll_stop() -> None:
                while True:
                    await asyncio.sleep(5)
                    if stop_check and stop_check():
                        runner.discovery_controller.request_shutdown()
                        return

            stop_task = asyncio.create_task(_poll_stop()) if stop_check else None
            try:
                await runner.discovery_controller.run_discovery(
                    discovery_start,
                    max_iterations,
                    checkpoint_callback=checkpoint_cb,
                )
            finally:
                if stop_task and not stop_task.done():
                    stop_task.cancel()
                    try:
                        await stop_task
                    except asyncio.CancelledError:
                        pass

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
