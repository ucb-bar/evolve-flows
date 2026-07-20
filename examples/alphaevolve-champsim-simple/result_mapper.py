"""Result mapper: ChampSimRunResult list -> EvaluationResult with geomean metrics."""

import math
from typing import Any, Dict, List

from skydiscover.evaluation.evaluation_result import EvaluationResult


def _geomean(values: List[float]) -> float:
    positive = [v for v in values if v > 0]
    if not positive:
        return 0.0
    return math.exp(sum(math.log(v) for v in positive) / len(positive))


def map_champsim_results(run_results: List[Any]) -> EvaluationResult:
    """Map ChampSimRunResult list to a single EvaluationResult.

    combined_score = geomean IPC (primary objective).
    """
    ipcs: List[float] = []
    mpkis: List[float] = []
    accuracies: List[float] = []
    coverages: List[float] = []
    per_workload: Dict[str, float] = {}
    failed_runs: List[str] = []

    for i, result in enumerate(run_results):
        if not getattr(result, "success", False):
            tail = getattr(result, "stdout_tail", "")
            failed_runs.append(f"workload_{i}: {tail[:200]}")
            continue

        ipc = getattr(result, "ipc", 0.0)
        ipcs.append(ipc)
        per_workload[f"workload_{i}"] = ipc

        cache_stats = getattr(result, "cache_stats", {})
        l2c = cache_stats.get("L2C")
        if l2c is not None:
            pf = getattr(l2c, "prefetch", None)
            if pf is not None:
                if pf.mpki is not None and pf.mpki > 0:
                    mpkis.append(pf.mpki)
                if pf.accuracy is not None and pf.accuracy > 0:
                    accuracies.append(pf.accuracy)
                if pf.coverage is not None and pf.coverage > 0:
                    coverages.append(pf.coverage)

    metrics: Dict[str, float] = {
        "combined_score": _geomean(ipcs),
        "geomean_ipc": _geomean(ipcs),
        "geomean_mpki": _geomean(mpkis),
        "geomean_accuracy": _geomean(accuracies),
        "geomean_coverage": _geomean(coverages),
        "successful_runs": float(len(ipcs)),
        "total_runs": float(len(run_results)),
    }

    artifacts: Dict[str, Any] = {"per_workload_ipc": per_workload}
    if failed_runs:
        artifacts["failed_runs"] = failed_runs

    return EvaluationResult(metrics=metrics, artifacts=artifacts)
