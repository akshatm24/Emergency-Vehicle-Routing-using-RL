# Submission Notes

## What This Repository Shows

This project is not just a shortest-path demo. It shows a full pipeline:

1. Download and preprocess the Kanpur road network from OpenStreetMap.
2. Extract real hospital locations and snap them to road intersections.
3. Assign vehicle speeds and edge travel-time weights.
4. Compare exact shortest-path algorithms for emergency routing.
5. Formulate a multi-incident routing problem as an RL task.
6. Train and evaluate DRL policies using true road-network travel times.
7. Run ablations and diagnostics to explain model behavior honestly.

## Main Academic Takeaway

Classical algorithms are exact and efficient for single-pair emergency routing. For multiple incidents, the difficult part is choosing the visit order. RL can learn this ordering policy, but at the tested training budget, a simpler GRU baseline and nearest-neighbour heuristic remain very competitive.

## Why the Result Is Still Valuable

The project is valuable because it includes:

- Real map data instead of a toy graph
- Real hospital extraction
- Travel-time edge weights instead of only geometric distance
- Classical baselines
- True optimal evaluation for small multi-stop instances
- Ablation studies
- Diagnostic analysis of why the upgraded architecture underperformed
- Practical next steps for GPU training

## Recommended Reviewer Path

Open these files in order:

1. `README.md`
2. `notebooks/kanpur_emergency_routing_final.ipynb`
3. `docs/RESULTS.md`
4. `artifacts/final_run/drl_vs_classical_routes.html`
5. `artifacts/final_run/kanpur_hospitals.html`

The notebook already contains the important outputs, so no GPU rerun is needed for grading.
