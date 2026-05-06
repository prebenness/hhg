# hhg

Research repo for heterogeneous and heterophilic graph learning. Uses four standard benchmarks to evaluate models against published baselines.

## Setup

Requires Python 3.9 and a CUDA-capable GPU.

```bash
git clone --recurse-submodules https://github.com/prebenness/hhg.git
cd hhg
python3.9 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Project structure

```
hhg/
  main.py              # H2GB entry point
  models/              # Custom models (registered via @register_network)
  configs/             # H2GB experiment configs (YAML)
  scripts/run.sh       # H2GB batch runner
  benchmarks/
    HGB/               # Lv et al. (KDD 2021) — git submodule
    heterophilous-graphs/  # Platonov et al. (ICLR 2023) — git submodule
    ogb/               # Hu et al. (NeurIPS 2020) — git submodule
  data/                # Downloaded datasets (gitignored)
  results/             # Experiment outputs (gitignored)
```

## Benchmarks

### H2GB (Lin et al., KDD 2025)

Lin, Huang, Giraldo, Baek, Hou, Koutis, Ribeiro. [When Heterophily Meets Heterogeneity](https://arxiv.org/abs/2411.14830). KDD 2025.

Installed as a pip dependency. Run experiments via `main.py` with YAML configs:

```bash
.venv/bin/python main.py --cfg configs/ogbn-mag/ogbn-mag-MLP.yaml --repeat 1 name_tag MLP wandb.use False
```

Custom models go in `models/` using the `@register_network` decorator, then reference the model name in a YAML config.

### HGB (Lv et al., KDD 2021)

Lv, Ding, Yang, Hou, Wu, Wang, Liu, Tang. [Are We Really Making Much Progress? Revisiting, Benchmarking and Refining Heterogeneous Graph Neural Networks](https://arxiv.org/abs/2112.14936). KDD 2021.

Per-model scripts under `benchmarks/HGB/NC/benchmark/methods/`. Datasets download automatically (or manually from [Google Drive](https://drive.google.com/drive/folders/10-pf2ADCjq_kpJKFHHLHxr_czNNCJ3aX)).

```bash
cd benchmarks/HGB/NC/benchmark/methods/baseline
../../../../../../.venv/bin/python run_new.py --dataset DBLP --num-layers 2
```

### Platonov et al. (ICLR 2023)

Platonov, Kuznedelev, Diskin, Babenko, Prokhorenkova. [A Critical Look at the Evaluation of GNNs under Heterophily: Are We Really Making Progress?](https://arxiv.org/abs/2302.11640). ICLR 2023.

Unified entry point. Datasets are bundled in the repo.

```bash
cd benchmarks/heterophilous-graphs
../../.venv/bin/python train.py --dataset roman-empire --model GCN --num_layers 5 --num_steps 1000 --num_runs 10
```

### OGB (Hu et al., NeurIPS 2020)

Hu, Fey, Zitnik, Dong, Ren, Liu, Catasta, Leskovec. [Open Graph Benchmark: Datasets for Machine Learning on Graphs](https://arxiv.org/abs/2005.00687). NeurIPS 2020.

Per-dataset scripts under `benchmarks/ogb/examples/nodeproppred/`. Datasets download on first run.

```bash
cd benchmarks/ogb/examples/nodeproppred/arxiv
../../../../../.venv/bin/python mlp.py --epochs 500 --runs 10
```

## Replication time estimates

Estimated wall-clock time for full replication on a single GPU (RTX 3080 Laptop, sequential runs, single seed).

### H2GB

All configs use 500 epochs with no early stopping.

| Dataset | Models | Est. time |
|---|---|---|
| ogbn-mag | 31 | ~49 hrs |
| oag-chem | 30 | ~47 hrs |
| oag-cs | 27 | ~43 hrs |
| oag-eng | 30 | ~47 hrs |
| mag-year | 28 | ~44 hrs |
| RCDD | 30 | ~47 hrs |
| IEEE-CIS | 27 | ~43 hrs |
| Pokec | 28 | ~44 hrs |
| PDNS | 29 | ~46 hrs |
| **Total** | **260** | **~410 hrs (~17 days)** |

### HGB

Small datasets with early stopping. Fast iterations (~0.3-1s/epoch).

| Dataset | Models | Est. time |
|---|---|---|
| ACM | 10 | ~15-20 min |
| DBLP | 10 | ~40-70 min |
| IMDB | 10 | ~15-20 min |
| Freebase | 10 | ~20-30 min |
| **Total** | **~35** | **~2-3 hrs** |

### Platonov et al.

1000 steps per run, 10 runs per experiment.

| Dataset | Experiments | Est. time |
|---|---|---|
| roman-empire | 45 | ~2.5 hrs |
| amazon-ratings | 45 | ~2.5 hrs |
| minesweeper | 45 | ~2.5 hrs |
| tolokers | 45 | ~2.5 hrs |
| questions | 45 | ~2.5 hrs |
| **Total** | **225** | **~12-19 hrs** |

### OGB

Variable epoch counts (30-1000), 10 runs default.

| Dataset | Models | Est. time | Notes |
|---|---|---|---|
| arxiv | 2 | ~30-40 min | |
| mag | 6 | ~4-5 hrs | RGCN may OOM (~14GB VRAM needed) |
| products | 6 | ~3-4 hrs | |
| proteins | 3 | ~4-5 hrs | 1000 epochs |
| papers100M | 3 | ~10-15 min | May OOM on 8GB GPU |
| **Total** | **~20** | **~12-15 hrs** |
