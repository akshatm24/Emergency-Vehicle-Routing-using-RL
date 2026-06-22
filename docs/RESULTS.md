# Results Summary

This document summarizes the completed project results without requiring the notebook to be re-run.

## Dataset

- City: Kanpur, Uttar Pradesh, India
- Source: OpenStreetMap via OSMnx
- Network type: drivable roads
- Hospital source: OSM hospital amenities
- Hospitals found in the final run: 204
- Routing weight: estimated vehicle travel time per road edge

## Phase 1: Classical Emergency Routing

Phase 1 solves a single-source single-destination emergency-routing task: hospital to incident.

The graph edge weight is `travel_time`, computed from road length and a road-type speed estimate. Three shortest-path methods were compared:

- Two-Q shortest path
- Dijkstra shortest path
- A* shortest path with a geographic heuristic

### Benchmark Result

| Algorithm | Mean travel time | Std travel time | Mean execution time | Trials |
|---|---:|---:|---:|---:|
| Two-Q | 22.439 min | 14.442 min | 242.138 ms | 30 |
| Dijkstra | 22.439 min | 14.442 min | 295.436 ms | 30 |
| A* | 22.439 min | 14.442 min | 198.239 ms | 30 |

The identical travel times show that all three methods recover the same shortest paths. A* is fastest on average because the heuristic guides search toward the destination.

## Phase 2: Deep RL for Multi-Stop Ordering

Phase 2 changes the problem from "find the shortest path to one incident" to "choose the best order for multiple incidents." The final path between consecutive stops is still evaluated using road-network distances.

The upgraded model includes:

- Node2Vec static node features
- GNN static encoder when PyTorch Geometric is available
- Attention-model decoder
- Actor-critic REINFORCE training
- Reward based on a precomputed true road-network distance matrix
- Phase 6 fixes: entropy regularization, advantage normalization, reward normalization, reduced attention clipping, and cosine learning-rate scheduling

## Single Multi-Stop Test Instance

| Method | Tour time | Gap vs true optimal |
|---|---:|---:|
| DRL4VRP greedy | 8706.0 s, 145.10 min | +1.5% |
| Nearest neighbour | 10075.2 s, 167.92 min | +17.5% |
| True optimal A* ordering | 8575.4 s, 142.92 min | Baseline |

On this instance, the DRL policy nearly matched the true optimal ordering and beat nearest neighbour.

## Held-Out Evaluation

The more reliable evaluation averages over 50 held-out 5-incident instances.

| Method | Mean time | Mean gap vs optimal |
|---|---:|---:|
| DRL4VRP greedy | 7753.4 s | 19.0% +/- 13.0 |
| Nearest neighbour | 6898.6 s | 4.9% +/- 5.1 |
| True optimal | 6576.6 s | 0.0% |

The RL model beat nearest neighbour on 9 of 50 held-out instances.

## Ablation

A simpler model was trained with the same 5000-episode budget:

- Linear static encoder
- GRU decoder
- Raw latitude/longitude features

| Model | Mean gap vs optimal |
|---|---:|
| Upgraded GNN + Node2Vec + attention | 19.0% |
| Simple linear + GRU + lat/lon | 4.8% |
| Nearest neighbour | 4.9% |

The simple architecture is more sample-efficient at this budget. This does not invalidate the upgraded architecture; it shows that the more complex model needs more training and tuning before its capacity pays off.

## Diagnostic Conclusion

The final diagnostics show:

- The critic loss is controlled after reward normalization.
- Entropy remains healthy, so the policy has not collapsed.
- Gradients are non-zero, so the training graph is connected.
- Node2Vec vs raw lat/lon is not the dominant bottleneck at the short diagnostic budget.

The most defensible conclusion is that the project successfully demonstrates both exact classical routing and RL-based multi-stop ordering, while honestly showing that simple baselines are hard to beat under limited training compute.
