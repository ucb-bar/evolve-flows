"""ChampSimEvaluator -- bridges ChampSim's build->binary->run pattern."""

import contextvars
from typing import Any, List, Optional, Union

from chia.simulators.champsim import ChampSimNode
from result_mapper import map_champsim_results
from skydiscover.evaluation.chia_evaluator import ChiaEvaluator
from skydiscover.evaluation.evaluation_result import EvaluationResult

# Per-task binary so concurrent evaluate_program calls don't race.
_eval_binary: contextvars.ContextVar[Optional[bytes]] = contextvars.ContextVar(
    "_eval_binary", default=None,
)


class ChampSimEvaluator(ChiaEvaluator):
    """ChiaEvaluator subclass that captures the compiled binary from builds
    and injects it into subsequent runs."""

    def __init__(
        self,
        build_node: ChampSimNode,
        run_node: ChampSimNode,
        champsim_root: str,
        module_name: str,
        cache_level: str,
        workloads: List[str],
        output_dir: str,
        warmup_instructions: int,
        simulation_instructions: int,
        **kwargs: Any,
    ) -> None:
        self._build_node = build_node
        self._run_node = run_node
        self._champsim_root = champsim_root
        self._module_name = module_name
        self._cache_level = cache_level
        self._warmup = warmup_instructions
        self._sim_instr = simulation_instructions

        super().__init__(
            build_fn=self._build,
            run_fn=self._run,
            result_mapper_fn=map_champsim_results,
            workloads=workloads,
            output_dir=output_dir,
            **kwargs,
        )

    def _build(self, program_solution: str) -> Any:
        return self._build_node.build_champsim.options(
            resources={"champsim_build": 1.0},
            num_cpus=8,
        ).chia_remote(
            self._champsim_root,
            program_solution,
            self._module_name,
            cache_level=self._cache_level,
            timeout_s=1800,
            incremental=True,
        )

    def _run(self, *, workload: str) -> Any:
        binary = _eval_binary.get()
        if binary is None:
            raise RuntimeError("No binary available -- build must succeed first")
        return self._run_node.run_champsim.chia_remote(
            binary=binary,
            trace=workload,
            warmup_instructions=self._warmup,
            simulation_instructions=self._sim_instr,
            timeout_s=3600,
        )

    async def _dispatch_build(
        self,
        program_solution: str,
        label: str,
    ) -> Union[Any, EvaluationResult]:
        _eval_binary.set(None)
        result = await super()._dispatch_build(program_solution, label)

        if isinstance(result, EvaluationResult):
            return result
        if result is None:
            return None

        if not getattr(result, "success", False):
            diagnostics = getattr(result, "build_diagnostics", "")
            stdout = getattr(result, "stdout_tail", "")
            return EvaluationResult(
                metrics={"error": 0.0, "combined_score": 0.0},
                artifacts={
                    "failure_stage": "build",
                    "error_type": "BuildFailure",
                    "stderr": diagnostics or stdout,
                },
            )

        _eval_binary.set(result.binary)
        return result
