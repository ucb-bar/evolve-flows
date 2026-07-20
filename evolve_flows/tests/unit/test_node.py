"""Unit tests for EvolverNode method behavior (Ray decorator stripped)."""

import os
import pickle
from unittest.mock import MagicMock

import pytest

from evolve_flows.evolver.node import EvolverNode, run_evolver
from evolve_flows.evolver.types import EvolverInput


def _unwrap_node_class():
    """Return the raw Python class behind the @ray.remote ActorClass.

    Ray 2.x stores the original class at __ray_metadata__.modified_class.
    """
    metadata = getattr(EvolverNode, "__ray_metadata__", None)
    if metadata is not None and hasattr(metadata, "modified_class"):
        return metadata.modified_class
    raise RuntimeError("Could not unwrap EvolverNode actor class for unit testing")


NodeClass = _unwrap_node_class()


class TestRunSearchValidation:
    """run_search rejects invalid inputs before launching a search."""

    def test_missing_config_path(self) -> None:
        """A nonexistent config_path raises ValueError."""
        node = NodeClass()
        evolver_input = EvolverInput(
            config_path="/nonexistent/path.yaml", initial_program="code"
        )
        with pytest.raises(ValueError, match="config_path does not exist"):
            node.run_search(
                evolver_input, MagicMock(), MagicMock(), MagicMock()
            )

    def test_empty_initial_program(self, tmp_path) -> None:
        """An empty initial_program raises ValueError."""
        node = NodeClass()
        config_path = os.path.join(str(tmp_path), "c.yaml")
        with open(config_path, "w") as f:
            f.write("max_iterations: 1\n")
        evolver_input = EvolverInput(config_path=config_path, initial_program="")
        with pytest.raises(
            ValueError, match="initial_program must be a non-empty string"
        ):
            node.run_search(
                evolver_input, MagicMock(), MagicMock(), MagicMock()
            )

    def test_whitespace_initial_program(self, tmp_path) -> None:
        """A whitespace-only initial_program raises ValueError."""
        node = NodeClass()
        config_path = os.path.join(str(tmp_path), "c.yaml")
        with open(config_path, "w") as f:
            f.write("max_iterations: 1\n")
        evolver_input = EvolverInput(config_path=config_path, initial_program="   ")
        with pytest.raises(
            ValueError, match="initial_program must be a non-empty string"
        ):
            node.run_search(
                evolver_input, MagicMock(), MagicMock(), MagicMock()
            )


class TestGetStatus:
    """get_status returns a plain serializable dict."""

    def test_initial_status(self) -> None:
        """A fresh node reports the idle status snapshot."""
        node = NodeClass()
        assert node.get_status() == {
            "state": "idle",
            "iteration": 0,
            "best_score": 0.0,
            "best_metrics": None,
            "best_program": None,
        }

    def test_status_keys(self) -> None:
        """get_status exposes exactly the documented keys."""
        node = NodeClass()
        status = node.get_status()
        assert isinstance(status, dict)
        assert set(status.keys()) == {
            "state",
            "iteration",
            "best_score",
            "best_metrics",
            "best_program",
        }


class TestStop:
    """stop requests graceful shutdown via the stop flag."""

    def test_sets_stop_flag(self) -> None:
        """stop flips _stop_requested from False to True."""
        node = NodeClass()
        assert node._stop_requested is False
        node.stop()
        assert node._stop_requested is True


class TestRunEvolver:
    """run_evolver is the @ChiaFunction CHIA-native search entry point."""

    def test_is_chia_function(self) -> None:
        """run_evolver carries the ChiaFunction dispatch surface."""
        assert hasattr(run_evolver, "chia_remote")
        assert hasattr(run_evolver, "options")
        assert hasattr(run_evolver, "_chia_options")

    def test_default_resource_tag(self) -> None:
        """The decorator default matches the CHIA {'evolver': 1.0} convention."""
        assert run_evolver._chia_options.get("resources") == {"evolver": 1.0}

    def test_options_override_resources(self) -> None:
        """.options() returns a dispatch handle with overridden resources."""
        handle = run_evolver.options(resources={"custom_gpu": 2.0})
        assert hasattr(handle, "chia_remote")

    def test_validates_missing_config_path(self) -> None:
        """A nonexistent config_path raises ValueError (via _chia_original)."""
        evolver_input = EvolverInput(
            config_path="/nonexistent/path.yaml", initial_program="code"
        )
        with pytest.raises(ValueError, match="config_path does not exist"):
            run_evolver._chia_original(
                evolver_input, MagicMock(), MagicMock(), MagicMock()
            )

    def test_validates_empty_initial_program(self, tmp_path) -> None:
        """An empty initial_program raises ValueError (via _chia_original)."""
        config_path = os.path.join(str(tmp_path), "c.yaml")
        with open(config_path, "w") as f:
            f.write("max_iterations: 1\n")
        evolver_input = EvolverInput(config_path=config_path, initial_program="")
        with pytest.raises(
            ValueError, match="initial_program must be a non-empty string"
        ):
            run_evolver._chia_original(
                evolver_input, MagicMock(), MagicMock(), MagicMock()
            )


class TestLoggingSetup:
    """EvolverNode.__init__ configures logging for the Ray actor process."""

    def test_logging_configured_on_init(self) -> None:
        """basicConfig(force=True) ensures at least one root handler exists."""
        import logging

        node = NodeClass()
        root = logging.getLogger()
        assert len(root.handlers) > 0, "EvolverNode.__init__ should configure logging"


class TestResourceTag:
    """EvolverInput.resource_tag carries a caller-configured CHIA resource tag."""

    def test_resource_tag_default(self) -> None:
        """resource_tag defaults to None when unset."""
        evolver_input = EvolverInput(config_path="x", initial_program="y")
        assert evolver_input.resource_tag is None

    def test_resource_tag_custom(self) -> None:
        """resource_tag stores a caller-provided mapping."""
        evolver_input = EvolverInput(
            config_path="x", initial_program="y", resource_tag={"chipyard": 1.0}
        )
        assert evolver_input.resource_tag == {"chipyard": 1.0}

    def test_resource_tag_pickle_roundtrip(self) -> None:
        """resource_tag survives a pickle round-trip for Ray dispatch."""
        evolver_input = EvolverInput(
            config_path="x", initial_program="y", resource_tag={"evolver": 1.0}
        )
        restored = pickle.loads(pickle.dumps(evolver_input))
        assert restored == evolver_input
