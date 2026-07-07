from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from stage_b_plan import DEFAULT_NONLOCAL_REWARD, load_seeded_configs, normalize_stage_b_config, parse_seed_list


def ablations(base_credit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    full = dict(base_credit)
    full["kernel"] = "gated_evidence"

    no_evidence = dict(full)
    no_evidence["nonlocal_evidence_weight"] = 0.0
    no_evidence["nonlocal_recency_weight"] = 1.0

    no_terminal_mix = dict(full)
    no_terminal_mix["terminal_success_uniform_weight"] = 0.0
    no_terminal_mix["terminal_success_evidence_weight"] = 0.0
    no_terminal_mix["terminal_success_recency_weight"] = 1.0
    no_terminal_mix["terminal_failure_recency_weight"] = 1.0
    no_terminal_mix["terminal_failure_evidence_weight"] = 0.0

    top1 = dict(full)
    top1["evidence_top_k"] = 1
    top1["evidence_temperature"] = 0.7

    soft_evidence = dict(full)
    soft_evidence["evidence_top_k"] = 0
    soft_evidence["evidence_temperature"] = 1.0

    return {
        "gated_full": full,
        "gated_no_evidence": no_evidence,
        "gated_no_terminal_mix": no_terminal_mix,
        "gated_top1": top1,
        "gated_soft_evidence": soft_evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="configs/hf_lora_stage_b_fair/gated")
    parser.add_argument("--pattern", default="gated_seed*.json")
    parser.add_argument("--out-dir", default="configs/hf_lora_stage_b_ablation")
    parser.add_argument("--out-root", default="runs/hf_lora_stage_b_ablation")
    parser.add_argument("--lag", type=int, default=2)
    parser.add_argument("--nonlocal-reward", type=float, default=DEFAULT_NONLOCAL_REWARD)
    parser.add_argument("--seeds", nargs="*", default=None)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    base_paths = sorted(base_dir.glob(args.pattern))
    if not base_paths:
        raise SystemExit(f"No base configs matched {base_dir / args.pattern}")

    for seed, base_config, base_path in load_seeded_configs(base_paths, parse_seed_list(args.seeds)):
        for name, credit in ablations(dict(base_config.get("credit", {}))).items():
            config = deepcopy(base_config)
            normalize_stage_b_config(
                config,
                seed=seed,
                lag=args.lag,
                reward=args.nonlocal_reward,
            )
            config["experiment_name"] = f"hf_lora_stage_b_ablation_{name}_seed{seed}"
            config["output_dir"] = f"{args.out_root}/{name}/seed={seed}"
            config["credit"] = credit

            target_dir = out_dir / name
            target_dir.mkdir(parents=True, exist_ok=True)
            out_path = target_dir / f"{name}_seed{seed}.json"
            out_path.write_text(
                json.dumps(config, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
            print(f"{base_path} -> {out_path}")


if __name__ == "__main__":
    main()
