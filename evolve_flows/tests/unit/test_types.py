"""Unit tests for evolve_flows.evolver.types dataclasses."""

import pickle

from evolve_flows.evolver.types import EvolverInput, EvolverResult, EvolverStatus


class TestEvolverInput:
    """Field assignment and pickle round-trip for EvolverInput."""

    def test_fields(self) -> None:
        """config_path and initial_program are stored verbatim."""
        inp = EvolverInput(config_path="/tmp/c.yaml", initial_program="def f(): pass")
        assert inp.config_path == "/tmp/c.yaml"
        assert inp.initial_program == "def f(): pass"

    def test_pickle_roundtrip(self) -> None:
        """Pickling preserves all field values (Ray actor serialization)."""
        inp = EvolverInput(config_path="/tmp/c.yaml", initial_program="def f(): pass")
        restored = pickle.loads(pickle.dumps(inp))
        assert restored.config_path == inp.config_path
        assert restored.initial_program == inp.initial_program


class TestEvolverResult:
    """Required/optional fields and pickle round-trip for EvolverResult."""

    def test_required_fields(self) -> None:
        """All five required fields are stored verbatim."""
        result = EvolverResult(
            best_program="def solve(x): return x",
            best_metrics={"combined_score": 0.9},
            iteration_count=3,
            terminal_status="completed",
            population=[{"id": "p1"}],
        )
        assert result.best_program == "def solve(x): return x"
        assert result.best_metrics == {"combined_score": 0.9}
        assert result.iteration_count == 3
        assert result.terminal_status == "completed"
        assert result.population == [{"id": "p1"}]

    def test_optional_defaults(self) -> None:
        """metrics_log_path and error_message default to None."""
        result = EvolverResult(
            best_program="",
            best_metrics={},
            iteration_count=0,
            terminal_status="max_iterations",
            population=[],
        )
        assert result.metrics_log_path is None
        assert result.error_message is None

    def test_pickle_roundtrip(self) -> None:
        """Pickling preserves all fields including non-None optionals."""
        result = EvolverResult(
            best_program="code",
            best_metrics={"combined_score": 0.5},
            iteration_count=2,
            terminal_status="error",
            population=[{"id": "p1"}],
            metrics_log_path="/tmp/log.jsonl",
            error_message="boom",
        )
        restored = pickle.loads(pickle.dumps(result))
        assert restored.best_program == result.best_program
        assert restored.best_metrics == result.best_metrics
        assert restored.iteration_count == result.iteration_count
        assert restored.terminal_status == result.terminal_status
        assert restored.population == result.population
        assert restored.metrics_log_path == result.metrics_log_path
        assert restored.error_message == result.error_message


class TestEvolverStatus:
    """Default values and full-field construction for EvolverStatus."""

    def test_defaults(self) -> None:
        """Only state is required; numeric fields default and metrics is None."""
        status = EvolverStatus(state="idle")
        assert status.state == "idle"
        assert status.iteration == 0
        assert status.best_score == 0.0
        assert status.best_metrics is None

    def test_all_fields(self) -> None:
        """All fields are stored verbatim when explicitly set."""
        status = EvolverStatus(
            state="running",
            iteration=7,
            best_score=1.5,
            best_metrics={"combined_score": 1.5},
        )
        assert status.state == "running"
        assert status.iteration == 7
        assert status.best_score == 1.5
        assert status.best_metrics == {"combined_score": 1.5}
