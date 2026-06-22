# Emergency Vehicle Routing in Kanpur using Classical Shortest Paths and Deep RL

This repository contains the final project notebook and supporting artifacts for an emergency-routing study on the Kanpur, Uttar Pradesh road network. The project compares exact graph-search methods for single hospital-to-incident routing with a Deep Reinforcement Learning (DRL) approach for multi-stop incident ordering.

The final notebook is here:

- `notebooks/kanpur_emergency_routing_final.ipynb`

## Project Overview

The work is organized into two main phases.

| Phase | Goal | Main methods | Outcome |
|---|---|---|---|
| Phase 1 | Fastest path from one hospital to one incident | Two-Q, Dijkstra, A* on OpenStreetMap road graph | All methods find the same optimal path; A* is fastest on average |
| Phase 2 | Best order for visiting multiple incidents | DRL4VRP-style actor-critic model with road-network reward | RL can learn valid multi-stop policies, but simple baselines remain strong at this budget |

The project deliberately separates two tasks:

1. **Path finding:** A* / Dijkstra find the exact shortest road path between two fixed nodes.
2. **Stop ordering:** The RL model decides the order in which multiple incidents should be visited.

This separation is important because classical shortest-path algorithms solve point-to-point routing exactly, while multi-stop ordering is a combinatorial planning problem.

## Key Results

### Phase 1: Single-Pair Emergency Routing

Thirty random hospital-to-incident pairs were benchmarked on the Kanpur road graph.

| Algorithm | Mean travel time | Mean execution time | Trials |
|---|---:|---:|---:|
| Two-Q | 22.439 min | 242.138 ms | 30 |
| Dijkstra | 22.439 min | 295.436 ms | 30 |
| A* | 22.439 min | 198.239 ms | 30 |

All three algorithms found identical route costs. A* was the fastest on average because the geographic heuristic reduces unnecessary node expansion.

### Phase 2: Multi-Stop Routing

The final notebook evaluates a 5-incident routing setting using true road-network distances. A single showcased test instance produced:

| Method | Tour time | Gap vs true optimal |
|---|---:|---:|
| DRL4VRP greedy | 8706.0 s | +1.5% |
| Nearest neighbour | 10075.2 s | +17.5% |
| True optimal A* ordering | 8575.4 s | Baseline |

Across 50 held-out instances:

| Method | Mean time | Mean gap vs true optimal |
|---|---:|---:|
| DRL4VRP greedy | 7753.4 s | 19.0% +/- 13.0 |
| Nearest neighbour | 6898.6 s | 4.9% +/- 5.1 |
| True optimal | 6576.6 s | 0.0% |

The honest conclusion is that the upgraded GNN + Node2Vec + attention architecture is functional and well-diagnosed, but at the tested training budget the simpler GRU/lat-lon model and nearest-neighbour baseline are more sample-efficient.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── kanpur_emergency_routing_final.ipynb
├── artifacts/
│   └── final_run/
│       ├── kanpur_road_network.png
│       ├── drl_training_curves.png
│       ├── single_route_comparison.html
│       ├── drl_vs_classical_routes.html
│       ├── kanpur_hospitals.html
│       └── notebook_audit.json
├── docs/
│   ├── RESULTS.md
│   ├── REPRODUCIBILITY.md
│   └── SUBMISSION_NOTES.md
├── runs/
│   └── prior audited notebook runs and generated outputs
└── tools/
    └── audit and diagnostic helper scripts
```

## Artifacts

Important generated artifacts are included under `artifacts/final_run/`:

- `kanpur_road_network.png`: static visualization of the Kanpur road network.
- `single_route_comparison.html`: interactive comparison of Two-Q, Dijkstra, and A* on one route.
- `kanpur_hospitals.html`: interactive map of hospital locations snapped to the road graph.
- `drl_vs_classical_routes.html`: interactive comparison of RL, nearest-neighbour, and optimal multi-stop tours.
- `drl_training_curves.png`: training curves from the DRL experiment.
- `notebook_audit.json`: audit metadata from a previous successful no-error execution.

## How to Read This Project

For grading or review, start with:

1. `notebooks/kanpur_emergency_routing_final.ipynb`
2. `docs/RESULTS.md`
3. `docs/REPRODUCIBILITY.md`

The notebook has already-run outputs. Re-running the full notebook is not necessary for review and can require significant compute, especially the Node2Vec and DRL training sections.

## Reproducibility Note

The code downloads live OpenStreetMap data. Exact graph contents can shift over time as OSM is edited. The included notebook outputs and artifacts preserve the completed run used for this submission.
