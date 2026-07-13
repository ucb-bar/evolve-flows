# evolve-flows

CHIA integration layer for evolutionary search.

## Install

```bash
git clone --recurse-submodules git@github.com:ucb-bar/evolve-flows.git
cd evolve-flows
conda create -n chia_env python=3.10 -y
conda activate chia_env
pip install -e ./skydiscover
pip install -e .
```

## Usage

```python
from evolve_flows import EvolverNode, run_evolver

node = EvolverNode(config_path="path/to/config.yaml")
```

## License

BSD 3-Clause. See [LICENSE](LICENSE).
