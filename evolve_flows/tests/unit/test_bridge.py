"""Unit tests for evolve_flows.evolver.bridge helper and entry functions."""

import ast
import asyncio
import importlib
import os
from unittest.mock import MagicMock, patch

import pytest

from evolve_flows.evolver import bridge
from evolve_flows.evolver.bridge import (
    _EVALUATOR_REGISTRY,
    _guard_signal_handlers,
    _is_external_search,
    _write_chia_evaluator_shim,
    _write_dummy_evaluator,
    run_skydiscover,
)
from evolve_flows.evolver.types import EvolverResult

from evolve_flows.tests.conftest import SEED_PROGRAM, make_evolver_input


class TestWriteDummyEvaluator:
    """The dummy evaluator file is a valid, well-formed Python module."""

    def test_creates_file(self) -> None:
        """A real file is written containing a def evaluate."""
        path = _write_dummy_evaluator()
        try:
            assert os.path.isfile(path)
            with open(path) as f:
                contents = f.read()
            assert "def evaluate" in contents
        finally:
            os.unlink(path)

    def test_file_is_valid_python(self) -> None:
        """The written file parses without SyntaxError."""
        path = _write_dummy_evaluator()
        try:
            with open(path) as f:
                contents = f.read()
            ast.parse(contents)
        finally:
            os.unlink(path)


class TestGuardSignalHandlers:
    """The signal-handler guard respects the main-thread check."""

    def test_skips_off_main_thread(self) -> None:
        """The original installer is NOT called when off the main thread."""
        runner = MagicMock()
        original = runner._install_signal_handlers
        _guard_signal_handlers(runner)

        with patch.object(bridge, "threading") as mock_threading:
            mock_threading.current_thread.return_value = MagicMock(name="worker")
            mock_threading.main_thread.return_value = MagicMock(name="main")
            runner._install_signal_handlers()

        original.assert_not_called()

    def test_allows_on_main_thread(self) -> None:
        """The original installer IS called once when on the main thread."""
        runner = MagicMock()
        original = runner._install_signal_handlers
        _guard_signal_handlers(runner)

        with patch.object(bridge, "threading") as mock_threading:
            same_thread = MagicMock(name="main")
            mock_threading.current_thread.return_value = same_thread
            mock_threading.main_thread.return_value = same_thread
            runner._install_signal_handlers()

        original.assert_called_once_with()


class TestRunSkydiscover:
    """The synchronous bridge handles errors and event-loop detection."""

    def test_error_returns_error_result(self, make_config_yaml) -> None:
        """A failure inside _run_search_async returns an error EvolverResult."""
        config_path = make_config_yaml()
        evolver_input = make_evolver_input(
            config_path=config_path, initial_program=SEED_PROGRAM
        )

        with patch(
            "skydiscover.config.load_config", side_effect=RuntimeError("boom")
        ):
            result = run_skydiscover(
                evolver_input, MagicMock(), MagicMock(), MagicMock()
            )

        assert isinstance(result, EvolverResult)
        assert result.terminal_status == "error"
        assert result.error_message is not None
        assert "boom" in result.error_message

    def test_event_loop_detection(self, make_config_yaml) -> None:
        """With no running loop, the bridge dispatches through asyncio.run."""
        config_path = make_config_yaml()
        evolver_input = make_evolver_input(
            config_path=config_path, initial_program=SEED_PROGRAM
        )
        sentinel = EvolverResult(
            best_program="",
            best_metrics={},
            iteration_count=0,
            terminal_status="completed",
            population=[],
        )

        async def _fake_async(*args, **kwargs):
            return sentinel

        with patch.object(bridge, "_run_search_async", side_effect=_fake_async):
            with patch.object(
                bridge.asyncio, "run", wraps=asyncio.run
            ) as mock_run:
                result = run_skydiscover(
                    evolver_input, MagicMock(), MagicMock(), MagicMock()
                )

        mock_run.assert_called_once()
        assert result is sentinel


class TestChiaEvaluatorShim:
    """The evaluator shim delegates evaluate(path) to a registered ChiaEvaluator."""

    def test_shim_is_valid_python(self) -> None:
        path = _write_chia_evaluator_shim("test-key")
        try:
            with open(path) as f:
                contents = f.read()
            ast.parse(contents)
            assert "def evaluate" in contents
        finally:
            os.unlink(path)

    def test_shim_calls_evaluate_program(self, tmp_path) -> None:
        """The shim reads source from file and calls evaluator.evaluate_program."""
        from skydiscover.evaluation.evaluation_result import EvaluationResult

        mock_eval = MagicMock()
        expected_result = EvaluationResult(
            metrics={"combined_score": 1.5, "ipc": 1.5},
        )
        async def _fake_eval(src):
            return expected_result

        mock_eval.evaluate_program = _fake_eval

        key = "test-shim-call"
        _EVALUATOR_REGISTRY[key] = mock_eval
        shim_path = _write_chia_evaluator_shim(key)

        program_file = tmp_path / "candidate.cc"
        program_file.write_text("int main() {}")

        try:
            spec = importlib.util.spec_from_file_location("_shim", shim_path)
            shim_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(shim_mod)

            result = shim_mod.evaluate(str(program_file))
            assert result["combined_score"] == 1.5
        finally:
            _EVALUATOR_REGISTRY.pop(key, None)
            os.unlink(shim_path)

    def test_shim_missing_evaluator_returns_zero(self, tmp_path) -> None:
        """When the registry key is gone, the shim returns score 0."""
        key = "missing-key"
        shim_path = _write_chia_evaluator_shim(key)

        program_file = tmp_path / "candidate.cc"
        program_file.write_text("int main() {}")

        try:
            spec = importlib.util.spec_from_file_location("_shim2", shim_path)
            shim_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(shim_mod)

            result = shim_mod.evaluate(str(program_file))
            assert result["combined_score"] == 0.0
        finally:
            os.unlink(shim_path)


class TestIsExternalSearch:
    """_is_external_search correctly identifies external backends."""

    def test_known_external_when_registered(self) -> None:
        """Simulate a registered external backend."""
        with patch("skydiscover.extras.external.is_external", return_value=True):
            assert _is_external_search("alphaevolve") is True

    def test_native_search(self) -> None:
        assert _is_external_search("adaevolve") is False
        assert _is_external_search("topk") is False

    def test_unknown_search(self) -> None:
        assert _is_external_search("nonexistent_backend_xyz") is False
