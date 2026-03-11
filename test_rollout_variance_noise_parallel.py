#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare rollout intermediate (variance) noise across parallel configs.

Runs SDE and CPS with the same prompt/seed under different parallel configurations
(tp/sp/cfg), collects trajectory_variance_noises from each run, and reports whether
the per-step variance noises match across configs (determinism check).
"""

from __future__ import annotations

import argparse
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

from sglang.multimodal_gen import DiffGenerator


@dataclass(frozen=True)
class ParallelConfig:
    name: str
    tp_size: int
    sp_degree: int
    enable_cfg_parallel: bool


def parse_size(size: str) -> tuple[int, int]:
    w, h = size.strip().lower().split("x")
    return int(w), int(h)


def create_generator(
    *,
    model: str,
    num_gpus: int,
    tp_size: int,
    sp_degree: int,
    enable_cfg_parallel: bool,
    trust_remote_code: bool,
    output_path: Path,
) -> DiffGenerator:
    return DiffGenerator.from_pretrained(
        local_mode=True,
        model_path=model,
        num_gpus=num_gpus,
        tp_size=tp_size,
        sp_degree=sp_degree,
        enable_cfg_parallel=enable_cfg_parallel,
        trust_remote_code=trust_remote_code,
        output_path=str(output_path),
    )


def default_parallel_configs(num_gpus: int) -> list[ParallelConfig]:
    configs: list[ParallelConfig] = []
    for tp in range(1, num_gpus + 1):
        if num_gpus % tp == 0:
            sp = num_gpus // tp
            configs.append(
                ParallelConfig(
                    f"tp{tp}_sp{sp}_cfg0",
                    tp_size=tp,
                    sp_degree=sp,
                    enable_cfg_parallel=False,
                )
            )
    if num_gpus % 2 == 0:
        half = num_gpus // 2
        for tp in range(1, half + 1):
            if half % tp == 0:
                sp = half // tp
                configs.append(
                    ParallelConfig(
                        f"tp{tp}_sp{sp}_cfg1",
                        tp_size=tp,
                        sp_degree=sp,
                        enable_cfg_parallel=True,
                    )
                )
    dedup: dict[str, ParallelConfig] = {}
    for c in configs:
        dedup.setdefault(c.name, c)
    return list(dedup.values())


def to_numpy(x: Any) -> np.ndarray:
    if x is None:
        raise ValueError("trajectory_variance_noises is None")
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        t = x.detach().cpu()
        # numpy does not support bfloat16/float16; convert to float32 first
        t = t.float()
        return t.numpy()
    return np.asarray(x)


def generate_with_variance_noises(
    generator: DiffGenerator,
    *,
    prompt: str,
    seed: int,
    size: str,
    mode: str,
    noise_level: float,
    num_inference_steps: int | None,
    guidance_scale: float | None,
    cfg_guidance_scale: float,
    log_prob_no_const: bool,
    negative_prompt: str | None = None,
) -> list[np.ndarray] | None:
    """Run one generation with rollout; return list of per-step variance noises (numpy)."""
    width, height = parse_size(size)
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "seed": seed,
        "width": width,
        "height": height,
        "rollout": True,
        "rollout_sde_type": mode,
        "rollout_noise_level": noise_level,
        "rollout_log_prob_no_const": log_prob_no_const,
        "return_file_paths_only": True,
    }
    if num_inference_steps is not None:
        kwargs["num_inference_steps"] = num_inference_steps
    if guidance_scale is not None:
        kwargs["guidance_scale"] = guidance_scale
    if negative_prompt is not None:
        kwargs["negative_prompt"] = negative_prompt
    if guidance_scale is None or guidance_scale <= 1.0:
        kwargs["guidance_scale"] = cfg_guidance_scale

    result = generator.generate(sampling_params_kwargs=kwargs)
    if result is None:
        return None
    if isinstance(result, list):
        result = result[0] if result else None
    if result is None:
        return None
    noises = getattr(result, "trajectory_variance_noises", None)
    if noises is None or len(noises) == 0:
        return None
    return [to_numpy(t) for t in noises]


def run_comparison(args: argparse.Namespace, out_root: Path) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    modes = ("sde", "cps")
    requested = getattr(args, "parallel_gpu_count", None)
    visible = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
    if "0" in os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        visible = max(1, visible)
    effective_gpus = visible if requested is None else min(requested, visible)
    configs = default_parallel_configs(effective_gpus)

    # Collect (config, mode) -> list of numpy arrays (per step)
    data: dict[str, dict[str, list[np.ndarray]]] = {"sde": {}, "cps": {}}
    failures: list[str] = []
    cfg_scale = getattr(args, "cfg_guidance_scale", 3.0)

    for cfg in configs:
        gen: DiffGenerator | None = None
        try:
            gen = create_generator(
                model=args.model,
                num_gpus=effective_gpus,
                tp_size=cfg.tp_size,
                sp_degree=cfg.sp_degree,
                enable_cfg_parallel=cfg.enable_cfg_parallel,
                trust_remote_code=getattr(args, "trust_remote_code", False),
                output_path=out_root,
            )
            gs = args.guidance_scale
            if cfg.enable_cfg_parallel and (gs is None or gs <= 1.0):
                gs = cfg_scale
            for mode in modes:
                noises = generate_with_variance_noises(
                    gen,
                    prompt=args.prompt,
                    seed=args.seed,
                    size=args.size,
                    mode=mode,
                    noise_level=args.noise_level,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=gs,
                    cfg_guidance_scale=cfg_scale,
                    log_prob_no_const=args.logprob_no_const,
                    negative_prompt="low quality",
                )
                if noises is not None:
                    data[mode][cfg.name] = noises
                else:
                    failures.append(f"{cfg.name}_{mode}: no trajectory_variance_noises")
        except Exception as e:
            failures.append(f"{cfg.name}: {e}")
            traceback.print_exc()
        finally:
            if gen is not None:
                gen.shutdown()

    # Compare per mode: ref = first config, others vs ref
    report: dict[str, list[dict[str, Any]]] = {"sde": [], "cps": []}
    for mode in modes:
        configs_with_data = list(data[mode].keys())
        if not configs_with_data:
            continue
        ref_name = configs_with_data[0]
        ref_list = data[mode][ref_name]
        ref_len = len(ref_list)
        ref_shapes = [tuple(a.shape) for a in ref_list]

        for cname in configs_with_data:
            curr = data[mode][cname]
            same_len = len(curr) == ref_len
            same_shapes = same_len and all(
                tuple(curr[i].shape) == ref_shapes[i] for i in range(ref_len)
            )
            step_max_diffs: list[float] = []
            step_allclose: list[bool] = []
            if same_shapes:
                for i in range(ref_len):
                    diff = np.abs(curr[i].astype(np.float64) - ref_list[i].astype(np.float64))
                    step_max_diffs.append(float(np.max(diff)) if diff.size > 0 else 0.0)
                    step_allclose.append(
                        bool(np.allclose(curr[i], ref_list[i], rtol=1e-5, atol=1e-5))
                    )
            else:
                step_max_diffs = [float("inf")] * max(len(curr), ref_len)
                step_allclose = [False] * max(len(curr), ref_len)

            report[mode].append(
                {
                    "config": cname,
                    "same_num_steps": same_len,
                    "same_shapes": same_shapes,
                    "num_steps": len(curr),
                    "ref_num_steps": ref_len,
                    "step_max_diff": step_max_diffs,
                    "step_allclose": step_allclose,
                    "all_steps_match": all(step_allclose) if step_allclose else False,
                }
            )

    return {
        "data": data,
        "report": report,
        "failures": failures,
        "effective_gpus": effective_gpus,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare rollout variance noise across parallel configs (SDE/CPS)."
    )
    parser.add_argument("--model", type=str, default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=str, default="1024x1024")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--cfg-guidance-scale", type=float, default=3.0)
    parser.add_argument("--noise-level", type=float, default=0.5)
    parser.add_argument("--logprob-no-const", action="store_true")
    parser.add_argument("--parallel-gpu-count", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/variance_noise_compare")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    result = run_comparison(args, out_root)

    # Print report
    print("=== Rollout variance noise comparison (SDE / CPS) ===\n")
    print(f"Effective GPUs: {result['effective_gpus']}")
    if result["failures"]:
        print("Failures:")
        for f in result["failures"]:
            print(f"  - {f}")
        print()

    for mode in ("sde", "cps"):
        print(f"--- {mode.upper()} ---")
        for row in result["report"][mode]:
            print(f"  config: {row['config']}")
            print(f"    same_num_steps: {row['same_num_steps']}, same_shapes: {row['same_shapes']}")
            print(f"    num_steps: {row['num_steps']} (ref: {row['ref_num_steps']})")
            if row["step_max_diff"]:
                max_over_steps = max(row["step_max_diff"])
                print(f"    max |diff| over steps: {max_over_steps:.8f}")
                print(f"    all_steps_match (allclose): {row['all_steps_match']}")
            print()
        print()

    report_md = out_root / "variance_noise_report.md"
    lines = [
        "# Rollout variance noise comparison",
        "",
        f"- Prompt: {args.prompt[:80]}...",
        f"- Seed: {args.seed}, noise_level: {args.noise_level}",
        f"- Effective GPUs: {result['effective_gpus']}",
        "",
    ]
    if result["failures"]:
        lines.append("## Failures")
        lines.extend(f"- {f}" for f in result["failures"])
        lines.append("")
    for mode in ("sde", "cps"):
        lines.append(f"## {mode.upper()}")
        lines.append("")
        lines.append("| config | same_num_steps | same_shapes | all_steps_match | max|diff| ")
        lines.append("|---|---|---:|---:|---:|")
        for row in result["report"][mode]:
            max_d = max(row["step_max_diff"]) if row["step_max_diff"] else float("nan")
            max_d_str = "inf" if max_d == float("inf") else f"{max_d:.8f}"
            lines.append(
                f"| {row['config']} | {row['same_num_steps']} | {row['same_shapes']} | "
                f"{row['all_steps_match']} | {max_d_str} |"
            )
        lines.append("")
    report_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to: {report_md}")


if __name__ == "__main__":
    main()
