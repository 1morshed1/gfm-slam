#!/usr/bin/env python3
"""§6.3 / Phase-4 bit allocation from a desk sensitivity profile.

Allocates which units stay FP16 under a budget; the rest get fake int WO (via
ptq.fake_except). Contrasts geometry-driven scores vs a weight-magnitude proxy.

Examples:
  python scripts/allocate_bits.py --method greedy --k 7
  python scripts/allocate_bits.py --method magnitude --k 7 --write-protect results/protect_mag_k7.txt
  python scripts/allocate_bits.py --method ilp --budget-params auto --k-ref 7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_sensitivity(path: Path) -> dict[str, float]:
    rec = json.loads(path.read_text())
    deltas = rec["metrics"]["ate_rel_delta_per_unit"]
    return {k: float(v) for k, v in deltas.items()}


def is_head(unit: str) -> bool:
    return "downstream_head" in unit


def trunk_units(deltas: dict[str, float]) -> list[str]:
    return [u for u in deltas if not is_head(u)]


def always_protect_heads() -> list[str]:
    return ["downstream_head1", "downstream_head2"]


def score_magnitude(model) -> dict[str, float]:
    """Per-unit weight L1 mass (LLM-style size/magnitude proxy)."""
    from collections import defaultdict
    from torch import nn
    from ptq import classify_unit

    mass: dict[str, float] = defaultdict(float)
    for fqn, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        u = classify_unit(fqn)
        if u is None:
            continue
        mass[u] += float(module.weight.detach().abs().sum().item())
    return dict(mass)


def unit_param_counts(model) -> dict[str, int]:
    from collections import defaultdict
    from torch import nn
    from ptq import classify_unit

    counts: dict[str, int] = defaultdict(int)
    for fqn, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        u = classify_unit(fqn)
        if u is None:
            continue
        counts[u] += int(module.weight.numel())
        if module.bias is not None:
            counts[u] += int(module.bias.numel())
    return dict(counts)


def allocate_greedy(scores: dict[str, float], k: int) -> list[str]:
    trunk = [u for u in scores if not is_head(u)]
    ranked = sorted(trunk, key=lambda u: scores[u], reverse=True)
    return ranked[:k]


def allocate_ilp(
    scores: dict[str, float],
    costs: dict[str, int],
    budget: int,
) -> list[str]:
    """Maximise sum(score) s.t. sum(cost) <= budget; heads excluded from decision."""
    import pulp

    trunk = [u for u in scores if not is_head(u) and u in costs]
    prob = pulp.LpProblem("slam_aware_bits", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("prot", trunk, cat="Binary")
    prob += pulp.lpSum(scores[u] * x[u] for u in trunk)
    prob += pulp.lpSum(costs[u] * x[u] for u in trunk) <= budget
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return [u for u in trunk if pulp.value(x[u]) > 0.5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sens",
        default=str(ROOT / "results/20260923-phase2-sens-w4-full-fr1desk.json"),
    )
    ap.add_argument("--method", choices=["greedy", "magnitude", "ilp"], required=True)
    ap.add_argument("--k", type=int, default=7, help="protect top-K trunk units")
    ap.add_argument(
        "--budget-params",
        default=None,
        help="'auto' = param sum of greedy top-K; or an integer",
    )
    ap.add_argument(
        "--k-ref",
        type=int,
        default=7,
        help="when budget-params=auto, match greedy top-k-ref param mass",
    )
    ap.add_argument("--write-protect", type=str, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    deltas = load_sensitivity(Path(args.sens))
    heads = always_protect_heads()

    need_model = args.method in ("magnitude", "ilp") or args.budget_params is not None
    model = None
    mag = {}
    costs = {}
    if need_model:
        import os

        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
        # load from MASt3R-SLAM cwd expectation
        ext = ROOT / "ext/MASt3R-SLAM"
        os.chdir(ext)
        from mast3r_slam.mast3r_utils import load_mast3r

        model = load_mast3r(device=args.device)
        mag = score_magnitude(model)
        costs = unit_param_counts(model)

    if args.method == "greedy":
        chosen = allocate_greedy(deltas, args.k)
        score_name = "ate_rel_delta"
        scores_used = {u: deltas[u] for u in chosen}
    elif args.method == "magnitude":
        chosen = allocate_greedy(mag, args.k)
        score_name = "weight_l1"
        scores_used = {u: mag[u] for u in chosen}
    else:  # ilp
        if args.budget_params is None or args.budget_params == "auto":
            ref = allocate_greedy(deltas, args.k_ref)
            budget = sum(costs[u] for u in ref)
        else:
            budget = int(args.budget_params)
        chosen = allocate_ilp(deltas, costs, budget)
        score_name = "ate_rel_delta"
        scores_used = {u: deltas[u] for u in chosen}
        print(f"ILP budget_params={budget} n_protect_trunk={len(chosen)}")

    protect = sorted(set(chosen) | set(heads))
    print(f"method={args.method} score={score_name}")
    print(f"trunk_protect ({len(chosen)}): {','.join(chosen)}")
    print(f"full_protect  ({len(protect)}): {','.join(protect)}")
    for u in chosen:
        extra = f" cost={costs.get(u)}" if costs else ""
        print(f"  {u:30s} {score_name}={scores_used[u]:.6g}{extra}")

    out = {
        "method": args.method,
        "score": score_name,
        "k": args.k,
        "trunk_protect": chosen,
        "protect": protect,
        "sens_path": str(args.sens),
    }
    if args.write_protect:
        p = Path(args.write_protect)
        if not p.is_absolute():
            p = ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(",".join(protect) + "\n")
        meta = p.with_suffix(".json")
        meta.write_text(json.dumps(out, indent=2))
        print(f"wrote {p} and {meta}")
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
