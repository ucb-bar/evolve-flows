#!/usr/bin/env python3
"""AlphaEvolve prefetcher search flow for ChampSim (simplified example).

Wires EvolverNode (AdaEvolve evolutionary search) to ChampSimNode
(build + run) via a ChampSim-specialized ChiaEvaluator.

Usage:
    python run_flow.py [--ray-address RAY_ADDRESS] [--output-dir DIR]
                       [--champsim-root PATH] [--config PATH]

    python run_flow.py stop [--ray-address RAY_ADDRESS]
    python run_flow.py status [--ray-address RAY_ADDRESS]
"""

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import ray

from champsim_evaluator import ChampSimEvaluator
from chia.simulators.champsim import ChampSimNode
from evolve_flows.evolver.node import EvolverNode
from evolve_flows.evolver.types import EvolverInput
from result_mapper import map_champsim_results

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

FLOW_DIR = Path(__file__).resolve().parent

_ENV_VARS_TO_FORWARD = [
    "GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "DEEPSEEK_API_KEY", "MISTRAL_API_KEY", "COHERE_API_KEY",
]

RUNTIME_ENV = {
    "working_dir": str(FLOW_DIR),
    "excludes": [
        "__pycache__",
        "*.pyc",
    ],
    "env_vars": {k: os.environ[k] for k in _ENV_VARS_TO_FORWARD if k in os.environ},
}

DEFAULT_CONFIG = str(FLOW_DIR / "config_adaevolve.yaml")
DEFAULT_CHAMPSIM_ROOT = "/home/ray/champsim"
DEFAULT_OUTPUT_BASE = "./results"
DEFAULT_RAY_ADDRESS = "HEAD_NODE:6379"

EVOLVER_ACTOR_NAME = "adaevolve-prefetcher-simple-evolver"

MODULE_NAME = "evolved_pf"
CACHE_LEVEL = "L2C"
WARMUP_INSTRUCTIONS = 5_000_000
SIMULATION_INSTRUCTIONS = 25_000_000

TRACE_DIR = os.environ.get("TRACE_DIR", "/path/to/traces/1C")
WORKLOADS = [
    os.path.join(TRACE_DIR, "602.gcc_s-734B.champsimtrace.xz"),
    os.path.join(TRACE_DIR, "605.mcf_s-472B.champsimtrace.xz"),
    os.path.join(TRACE_DIR, "623.xalancbmk_s-10B.champsimtrace.xz"),
]

SEED_PROGRAM = '''\
#include <cstdint>
#include "address.h"
#include "modules.h"

struct evolved_pf : public champsim::modules::prefetcher {
  using prefetcher::prefetcher;

  uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address ip,
                                     uint8_t cache_hit, bool useful_prefetch,
                                     access_type type, uint32_t metadata_in) {
    champsim::block_number pf_addr{addr};
    prefetch_line(champsim::address{pf_addr + 1}, true, metadata_in);
    return metadata_in;
  }

  uint32_t prefetcher_cache_fill(champsim::address addr, long set, long way,
                                  uint8_t prefetch, champsim::address evicted_addr,
                                  uint32_t metadata_in) {
    return metadata_in;
  }
};
'''


# ── Flow entry point ─────────────────────────────────────────────────

def run_flow(
    ray_address: str = DEFAULT_RAY_ADDRESS,
    output_dir: Optional[str] = None,
    champsim_root: str = DEFAULT_CHAMPSIM_ROOT,
    config_path: str = DEFAULT_CONFIG,
) -> None:
    """Launch the AlphaEvolve prefetcher search flow."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if output_dir is None:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(DEFAULT_OUTPUT_BASE, f"run_{run_tag}")

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    logger.info("Connecting to Ray cluster at %s", ray_address)
    if not ray.is_initialized():
        ray.init(address=ray_address, namespace="default", runtime_env=RUNTIME_ENV)

    logger.info("Creating ChampSimNodes (build: champsim_build resource, run: uncolocated)")
    build_node = ChampSimNode(require_colocated=False)
    run_node = ChampSimNode(require_colocated=False)

    logger.info("Creating ChampSimEvaluator with %d workloads", len(WORKLOADS))
    evaluator = ChampSimEvaluator(
        build_node=build_node,
        run_node=run_node,
        champsim_root=champsim_root,
        module_name=MODULE_NAME,
        cache_level=CACHE_LEVEL,
        workloads=WORKLOADS,
        output_dir=output_dir,
        warmup_instructions=WARMUP_INSTRUCTIONS,
        simulation_instructions=SIMULATION_INSTRUCTIONS,
        timeout=3600.0,
        max_retries=1,
    )

    try:
        stale = ray.get_actor(EVOLVER_ACTOR_NAME)
        logger.warning(
            "Found stale evolver actor '%s' from a previous run -- killing it",
            EVOLVER_ACTOR_NAME,
        )
        ray.kill(stale)
    except ValueError:
        pass

    logger.info("Creating EvolverNode actor '%s'", EVOLVER_ACTOR_NAME)
    evolver = EvolverNode.options(
        name=EVOLVER_ACTOR_NAME,
        lifetime="detached",
        resources={"evolver": 1.0},
        runtime_env=RUNTIME_ENV,
    ).remote()

    with open(config_path) as f:
        config_content = f.read()

    evolver_input = EvolverInput(
        config_path=config_path,
        initial_program=SEED_PROGRAM,
        config_content=config_content,
    )

    stop_requested = False

    def _signal_handler(signum: int, frame: Any) -> None:
        nonlocal stop_requested
        sig_name = signal.Signals(signum).name
        if stop_requested:
            logger.warning("Second %s received -- forcing exit", sig_name)
            sys.exit(1)
        stop_requested = True
        logger.info(
            "%s received -- requesting graceful stop "
            "(current iteration will complete, then search exits)",
            sig_name,
        )
        try:
            ray.get(evolver.stop.remote(), timeout=10.0)
        except Exception as e:
            logger.warning("Failed to send stop to evolver: %s", e)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    logger.info(
        "Launching AlphaEvolve search: config=%s, max_iterations=50", config_path
    )
    result_ref = evolver.run_search.remote(
        evolver_input,
        build_fn=evaluator._build,
        run_fn=evaluator._run,
        result_mapper_fn=map_champsim_results,
        evaluator=evaluator,
    )

    last_saved_score = 0.0
    best_path = os.path.join(output_dir, "best_prefetcher.cc")

    while True:
        ready, _ = ray.wait([result_ref], timeout=30.0)
        if ready:
            break
        try:
            status = ray.get(evolver.get_status.remote())
            cur_score = status.get("best_score", 0.0)
            logger.info(
                "Status: state=%s  iteration=%s  best_score=%.6f",
                status.get("state", "?"),
                status.get("iteration", "?"),
                cur_score,
            )
            best_prog = status.get("best_program")
            if best_prog and cur_score > last_saved_score:
                with open(best_path, "w") as f:
                    f.write(best_prog)
                last_saved_score = cur_score
                logger.info("Saved best program (score=%.6f) to %s", cur_score, best_path)
        except Exception:
            pass

    result = ray.get(result_ref)

    logger.info("Search complete: terminal_status=%s", result.terminal_status)
    logger.info("Iterations: %d", result.iteration_count)
    if result.best_metrics:
        logger.info("Best metrics: %s", result.best_metrics)
    if result.best_program:
        with open(best_path, "w") as f:
            f.write(result.best_program)
        logger.info("Best program saved to %s", best_path)
    if result.error_message:
        logger.error("Error: %s", result.error_message)

    evaluator.close()
    build_node.close()
    run_node.close()
    ray.kill(evolver)
    logger.info("Flow complete")


# ── Remote stop / status commands ──────────────────────────────────

def _connect_ray(ray_address: str) -> None:
    if not ray.is_initialized():
        ray.init(address=ray_address, namespace="default")


def _get_evolver(ray_address: str) -> Any:
    _connect_ray(ray_address)
    try:
        return ray.get_actor(EVOLVER_ACTOR_NAME)
    except ValueError:
        logger.error(
            "No running evolver actor '%s' found on cluster %s",
            EVOLVER_ACTOR_NAME, ray_address,
        )
        sys.exit(1)


def stop_flow(ray_address: str = DEFAULT_RAY_ADDRESS) -> None:
    """Send a graceful stop to the running evolver actor."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    evolver = _get_evolver(ray_address)
    logger.info("Sending stop to '%s'...", EVOLVER_ACTOR_NAME)
    ray.get(evolver.stop.remote(), timeout=30.0)
    logger.info(
        "Stop accepted -- evolver will finish the current iteration and exit"
    )


def status_flow(ray_address: str = DEFAULT_RAY_ADDRESS) -> None:
    """Print the current evolver status."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    evolver = _get_evolver(ray_address)
    status = ray.get(evolver.get_status.remote(), timeout=30.0)
    print(
        f"state={status.get('state', '?')}  "
        f"iteration={status.get('iteration', '?')}  "
        f"best_score={status.get('best_score', 0.0):.6f}"
    )
    if status.get("best_metrics"):
        for k, v in status["best_metrics"].items():
            print(f"  {k}: {v}")


# ── CLI ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AlphaEvolve prefetcher search for ChampSim (simple example)",
    )
    sub = parser.add_subparsers(dest="command")

    parser.add_argument(
        "--ray-address", default=DEFAULT_RAY_ADDRESS,
        help="Ray cluster address (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Results output directory (default: timestamped subdir under "
             f"{DEFAULT_OUTPUT_BASE})",
    )
    parser.add_argument(
        "--champsim-root", default=DEFAULT_CHAMPSIM_ROOT,
        help="Path to ChampSim checkout (default: %(default)s)",
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help="AdaEvolve config YAML (default: %(default)s)",
    )

    stop_parser = sub.add_parser("stop", help="Gracefully stop a running search")
    stop_parser.add_argument(
        "--ray-address", default=DEFAULT_RAY_ADDRESS,
        help="Ray cluster address (default: %(default)s)",
    )

    status_parser = sub.add_parser("status", help="Show current search status")
    status_parser.add_argument(
        "--ray-address", default=DEFAULT_RAY_ADDRESS,
        help="Ray cluster address (default: %(default)s)",
    )

    args = parser.parse_args()

    if args.command == "stop":
        stop_flow(ray_address=args.ray_address)
    elif args.command == "status":
        status_flow(ray_address=args.ray_address)
    else:
        run_flow(
            ray_address=args.ray_address,
            output_dir=args.output_dir,
            champsim_root=args.champsim_root,
            config_path=args.config,
        )


if __name__ == "__main__":
    main()
