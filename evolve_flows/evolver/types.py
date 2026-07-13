"""Pickle-safe data types for EvolverNode I/O and status reporting."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvolverInput:
    """Input contract for an evolutionary search run.

    Callables (build_fn, run_fn, result_mapper_fn) are intentionally excluded —
    they are passed as separate method arguments to avoid pickle issues.

    resource_tag is an optional CHIA resource tag for placement group scheduling.
    """

    config_path: str
    initial_program: str
    config_content: Optional[str] = None
    resource_tag: Optional[Dict[str, float]] = None


@dataclass
class EvolverResult:
    """Output of a completed (or failed) evolutionary search run."""

    best_program: str
    best_metrics: Dict[str, Any]
    iteration_count: int
    terminal_status: str
    population: List[Dict[str, Any]]
    metrics_log_path: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class EvolverStatus:
    """Snapshot of a running search for progress polling."""

    state: str
    iteration: int = 0
    best_score: float = 0.0
    best_metrics: Optional[Dict[str, Any]] = None
