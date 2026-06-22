#!/usr/bin/env python3
"""Probe Claude ablation tensor shapes and gradient flow."""

import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = Path("/Users/anshikamittal/Downloads/project_final_v3.ipynb")
RUN_DIR = ROOT / "runs" / "ablation_gradient_probe"
SOURCE_CACHE_DIR = ROOT / "runs" / "project_final_v3_audit" / "cache"


def cell_source(nb, idx):
    return "".join(nb["cells"][idx]["source"])


def exec_cell(ns, nb, idx):
    print(f"\n--- executing source cell {idx} ---")
    exec(compile(cell_source(nb, idx), f"cell_{idx}", "exec"), ns)


def grad_summary(module):
    rows = []
    total_norm = 0.0
    none = 0
    zero = 0
    for name, param in module.named_parameters():
        if param.grad is None:
            none += 1
            rows.append((name, tuple(param.shape), None))
            continue
        norm = float(param.grad.detach().norm().cpu())
        total_norm += norm
        if norm == 0.0:
            zero += 1
        rows.append((name, tuple(param.shape), norm))
    return rows, total_norm, none, zero


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if SOURCE_CACHE_DIR.exists() and not (RUN_DIR / "cache").exists():
        shutil.copytree(SOURCE_CACHE_DIR, RUN_DIR / "cache")
    os.chdir(RUN_DIR)

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    ns = {"__name__": "__main__"}

    for idx in [4, 6, 8, 12, 14, 16, 18, 37]:
        exec_cell(ns, nb, idx)
    ns["all_nodes"] = list(ns["city_graph"].nodes())
    for idx in [39, 41, 50, 51]:
        exec_cell(ns, nb, idx)

    reward_cell = cell_source(nb, 58)
    reward_def = reward_cell[: reward_cell.index('print("Reward function comparison')]
    exec(compile(reward_def, "cell_58_reward_defs", "exec"), ns)

    # Pull only the class/sampler definitions from the ablation cell, not its full training loop.
    ablation = cell_source(nb, 70)
    start = ablation.index("class SimpleDRL4VRP")
    end = ablation.index("torch.manual_seed(SEED + 1)")
    exec(compile(ablation[start:end], "cell_70_ablation_defs", "exec"), ns)

    ns["HIDDEN_SIZE"] = 128
    ns["BATCH_SIZE"] = 32
    ns["torch"].manual_seed(ns["SEED"] + 101)
    ns["np"].random.seed(ns["SEED"] + 101)
    ns["random"].seed(ns["SEED"] + 101)

    actor = ns["SimpleDRL4VRP"](2, ns["HIDDEN_SIZE"]).to(ns["device"])
    critic = ns["SimpleCritic"](2, ns["HIDDEN_SIZE"]).to(ns["device"])
    actor.train()
    critic.train()

    static, dynamic, dist_mats = ns["sample_instance_batch_simple"](ns["NUM_INCIDENTS"], 4)
    print("\n=== ablation probe shapes ===")
    print(f"static      {tuple(static.shape)}")
    print(f"dynamic     {tuple(dynamic.shape)}")
    print(f"dist_mats   {tuple(dist_mats.shape)}")

    tours, log_probs = actor(static, dynamic.clone())
    tour_lengths = ns["compute_tour_time_matrix"](tours, dist_mats)
    rewards = -tour_lengths
    baseline = critic(static)
    actor_loss = -((rewards - baseline).detach() * log_probs).mean()
    critic_loss = ns["F"].mse_loss(critic(static), rewards.detach())

    print(f"tours       {tuple(tours.shape)} dtype={tours.dtype} values={tours.detach().cpu().tolist()}")
    print(f"log_probs   {tuple(log_probs.shape)} finite={bool(ns['torch'].isfinite(log_probs).all())}")
    print(f"tour_lengths {tuple(tour_lengths.shape)} finite={bool(ns['torch'].isfinite(tour_lengths).all())}")
    print(f"baseline    {tuple(baseline.shape)} finite={bool(ns['torch'].isfinite(baseline).all())}")
    print(f"actor_loss  {float(actor_loss.detach().cpu()):.6f}")
    print(f"critic_loss {float(critic_loss.detach().cpu()):.6f}")

    actor.zero_grad(set_to_none=True)
    actor_loss.backward()
    critic.zero_grad(set_to_none=True)
    critic_loss.backward()

    for label, module in [("actor", actor), ("critic", critic)]:
        rows, total_norm, none, zero = grad_summary(module)
        print(f"\n=== {label} gradients ===")
        print(f"params={len(rows)} grad_none={none} grad_zero={zero} total_grad_norm={total_norm:.6f}")
        for name, shape, norm in rows:
            norm_text = "None" if norm is None else f"{norm:.6f}"
            print(f"{name:<42} shape={shape!s:<18} grad_norm={norm_text}")


if __name__ == "__main__":
    main()
