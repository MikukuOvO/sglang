#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare SDE/CPS rollout trajectory debug (prev_sample_mean, noise_std_dev, variance_noise) across parallel configs.

All tensors are converted to float32 before numpy (bf16-safe). Reports per-step and overall
max absolute difference and allclose for each quantity.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Prefer repo python/ so rollout and scheduler fixes are used (client and server subprocess)
_script_dir = Path(__file__).resolve().parent
_repo_python = _script_dir / "python"
if _repo_python.is_dir():
    sys.path.insert(0, str(_repo_python))
    _existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(_repo_python) + (os.pathsep + _existing if _existing else "")

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


def to_numpy_bf16_safe(x: Any) -> np.ndarray:
    """Convert tensor to numpy; use float32 to avoid bf16/fp16 unsupported by numpy."""
    if x is None:
        raise ValueError("tensor is None")
    if isinstance(x, np.ndarray):
        return x.astype(np.float32) if x.dtype == np.float16 else x
    if hasattr(x, "detach"):
        t = x.detach().cpu().float()
        return t.numpy()
    return np.asarray(x, dtype=np.float32)


def to_step_list(x: Any) -> list[Any] | None:
    """Normalize trajectory debug payload into a per-step list."""
    if x is None:
        return None
    if isinstance(x, list):
        return x

    # Tensor-like path: expected layout is [B, T, ...]
    if hasattr(x, "ndim") and int(x.ndim) >= 2:
        if hasattr(x, "unbind"):
            return list(x.unbind(dim=1))
        if isinstance(x, np.ndarray):
            return [x[:, i, ...] for i in range(x.shape[1])]

    # Fallback to single-step payload
    return [x]


def run_one(
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
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]] | None:
    """Run one generation with rollout; return (variance_noises, prev_sample_means, noise_std_devs) as numpy lists or None."""
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
    gs = guidance_scale if guidance_scale is not None and guidance_scale > 1.0 else cfg_guidance_scale
    kwargs["guidance_scale"] = gs
    if negative_prompt is not None:
        kwargs["negative_prompt"] = negative_prompt

    result = generator.generate(sampling_params_kwargs=kwargs)
    if result is None:
        return None
    if isinstance(result, list):
        result = result[0] if result else None
    if result is None:
        return None
    v = getattr(result, "trajectory_variance_noises", None)
    p = getattr(result, "trajectory_prev_sample_means", None)
    s = getattr(result, "trajectory_noise_std_devs", None)
    v_steps = to_step_list(v)
    p_steps = to_step_list(p)
    s_steps = to_step_list(s)
    if not v_steps or not p_steps or not s_steps:
        return None
    return (
        [to_numpy_bf16_safe(t) for t in v_steps],
        [to_numpy_bf16_safe(t) for t in p_steps],
        [to_numpy_bf16_safe(t) for t in s_steps],
    )


def compare_lists(
    ref_v: list[np.ndarray],
    ref_p: list[np.ndarray],
    ref_s: list[np.ndarray],
    cur_v: list[np.ndarray],
    cur_p: list[np.ndarray],
    cur_s: list[np.ndarray],
) -> dict[str, Any]:
    """Compare current (cur_*) to reference (ref_*). Return metrics for each quantity."""
    def metrics(ref_list: list[np.ndarray], cur_list: list[np.ndarray], name: str) -> dict[str, Any]:
        same_len = len(cur_list) == len(ref_list)
        if not same_len:
            return {
                f"{name}_same_len": False,
                f"{name}_same_shapes": False,
                f"{name}_max_abs_diff": float("inf"),
                f"{name}_all_steps_match": False,
            }
        same_shapes = all(c.shape == r.shape for c, r in zip(cur_list, ref_list))
        if not same_shapes:
            return {
                f"{name}_same_len": True,
                f"{name}_same_shapes": False,
                f"{name}_max_abs_diff": float("inf"),
                f"{name}_all_steps_match": False,
            }
        diffs = [np.abs(cur_list[i].astype(np.float64) - ref_list[i].astype(np.float64)) for i in range(len(ref_list))]
        max_abs = float(max(np.max(d) for d in diffs)) if diffs else 0.0
        all_match = all(
            np.allclose(cur_list[i], ref_list[i], rtol=1e-5, atol=1e-5)
            for i in range(len(ref_list))
        )
        return {
            f"{name}_same_len": True,
            f"{name}_same_shapes": True,
            f"{name}_max_abs_diff": max_abs,
            f"{name}_all_steps_match": all_match,
        }

    out: dict[str, Any] = {}
    out.update(metrics(ref_v, cur_v, "variance_noise"))
    out.update(metrics(ref_p, cur_p, "prev_sample_mean"))
    out.update(metrics(ref_s, cur_s, "noise_std_dev"))
    return out


def format_array_full(arr: np.ndarray) -> str:
    """Return a full, non-truncated textual dump for numpy array."""
    return np.array2string(
        arr,
        threshold=np.inf,
        max_line_width=200,
        separator=", ",
        precision=8,
        floatmode="maxprec_equal",
    )


def write_tensor_dump_file(
    *,
    out_root: Path,
    data: dict[str, dict[str, tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]]],
    effective_gpus: int,
    args: argparse.Namespace,
    failures: list[str],
) -> Path:
    """Write all compared intermediate tensors in plain text."""
    lines: list[str] = [
        "# Rollout trajectory debug tensor dump",
        "",
        f"Effective GPUs: {effective_gpus}",
        f"Prompt: {args.prompt!r}, seed={args.seed}, noise_level={args.noise_level}",
        "",
    ]
    if failures:
        lines.append("## Failures")
        for f in failures:
            lines.append(f"- {f}")
        lines.append("")

    for mode in ("sde", "cps"):
        lines.append(f"## Mode: {mode}")
        lines.append("")
        if not data[mode]:
            lines.append("(no data)")
            lines.append("")
            continue

        for config_name, (v_steps, p_steps, s_steps) in data[mode].items():
            lines.append(f"### Config: {config_name}")
            lines.append("")
            max_steps = max(len(v_steps), len(p_steps), len(s_steps))
            for step_idx in range(max_steps):
                lines.append(f"Step {step_idx}:")
                lines.append("")

                if step_idx < len(v_steps):
                    v = v_steps[step_idx]
                    lines.append(f"- variance_noise shape={v.shape}")
                    lines.append(format_array_full(v))
                    lines.append("")
                else:
                    lines.append("- variance_noise: <missing>")
                    lines.append("")

                if step_idx < len(p_steps):
                    p = p_steps[step_idx]
                    lines.append(f"- prev_sample_mean shape={p.shape}")
                    lines.append(format_array_full(p))
                    lines.append("")
                else:
                    lines.append("- prev_sample_mean: <missing>")
                    lines.append("")

                if step_idx < len(s_steps):
                    s = s_steps[step_idx]
                    lines.append(f"- noise_std_dev shape={s.shape}")
                    lines.append(format_array_full(s))
                    lines.append("")
                else:
                    lines.append("- noise_std_dev: <missing>")
                    lines.append("")
            lines.append("")

    dump_path = out_root / "trajectory_debug_tensors.txt"
    dump_path.write_text("\n".join(lines), encoding="utf-8")
    return dump_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare SDE/CPS trajectory debug (prev_sample_mean, noise_std_dev, variance_noise) across parallel configs."
    )
    parser.add_argument("--model", type=str, default="Tongyi-MAI/Z-Image-Turbo", help="Model path.")
    parser.add_argument("--prompt", type=str, default="A cat", help="Prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--size", type=str, default="1024x1024", help="Image size.")
    parser.add_argument("--noise-level", type=float, default=0.5, help="Rollout noise level.")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--cfg-guidance-scale", type=float, default=3.0)
    parser.add_argument("--logprob-no-const", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--parallel-gpu-count", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    if args.output_dir:
        out_root = Path(args.output_dir)
    else:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_root = Path("outputs/rollout_trajectory_debug_compare") / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_root}")

    visible = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")) if os.environ.get("CUDA_VISIBLE_DEVICES") else 1
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        try:
            import torch
            visible = int(torch.cuda.device_count()) if torch.cuda.is_available() else 1
        except Exception:
            visible = 1
    effective_gpus = visible if args.parallel_gpu_count is None else min(args.parallel_gpu_count, visible)
    configs = default_parallel_configs(effective_gpus)

    # (mode -> config_name -> (v_list, p_list, s_list))
    data: dict[str, dict[str, tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]]] = {
        "sde": {},
        "cps": {},
    }
    failures: list[str] = []

    for cfg in configs:
        gen: DiffGenerator | None = None
        try:
            gen = create_generator(
                model=args.model,
                num_gpus=effective_gpus,
                tp_size=cfg.tp_size,
                sp_degree=cfg.sp_degree,
                enable_cfg_parallel=cfg.enable_cfg_parallel,
                trust_remote_code=args.trust_remote_code,
                output_path=out_root,
            )
            gs = args.guidance_scale
            if cfg.enable_cfg_parallel and (gs is None or gs <= 1.0):
                gs = args.cfg_guidance_scale
            for mode in ("sde", "cps"):
                tri = run_one(
                    gen,
                    prompt=args.prompt,
                    seed=args.seed,
                    size=args.size,
                    mode=mode,
                    noise_level=args.noise_level,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=gs,
                    cfg_guidance_scale=args.cfg_guidance_scale,
                    log_prob_no_const=args.logprob_no_const,
                    negative_prompt="low quality",
                )
                if tri is not None:
                    data[mode][cfg.name] = tri
                else:
                    failures.append(f"{cfg.name}_{mode}: no trajectory debug data")
        except Exception as e:
            failures.append(f"{cfg.name}: {e}")
            traceback.print_exc()
        finally:
            if gen is not None:
                gen.shutdown()

    # Build report: for each mode, ref = first config, compare others
    report: dict[str, list[dict[str, Any]]] = {"sde": [], "cps": []}
    for mode in ("sde", "cps"):
        configs_with_data = list(data[mode].keys())
        if not configs_with_data:
            continue
        ref_name = configs_with_data[0]
        ref_v, ref_p, ref_s = data[mode][ref_name]
        for cname in configs_with_data:
            cur_v, cur_p, cur_s = data[mode][cname]
            row = {"config": cname}
            row.update(
                compare_lists(ref_v, ref_p, ref_s, cur_v, cur_p, cur_s)
            )
            report[mode].append(row)

    # Print and write report
    print("=== Rollout trajectory debug (prev_sample_mean, noise_std_dev, variance_noise) ===\n")
    print(f"Effective GPUs: {effective_gpus}")
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  - {f}")
    print()
    lines = [
        "# Rollout trajectory debug comparison",
        "",
        f"Effective GPUs: {effective_gpus}",
        f"Prompt: {args.prompt!r}, seed={args.seed}, noise_level={args.noise_level}",
        "",
    ]
    if failures:
        lines.append("## Failures")
        for f in failures:
            lines.append(f"- {f}")
        lines.append("")
    for mode in ("sde", "cps"):
        if not report[mode]:
            continue
        lines.append(f"## Mode: {mode}")
        lines.append("")
        lines.append("| config | variance_noise max_abs_diff | variance_noise all_match | prev_sample_mean max_abs_diff | prev_sample_mean all_match | noise_std_dev max_abs_diff | noise_std_dev all_match |")
        lines.append("|--------|-----------------------------|--------------------------|-------------------------------|---------------------------|---------------------------|-------------------------|")
        for row in report[mode]:
            v_diff = row.get("variance_noise_max_abs_diff", "")
            v_ok = row.get("variance_noise_all_steps_match", "")
            p_diff = row.get("prev_sample_mean_max_abs_diff", "")
            p_ok = row.get("prev_sample_mean_all_steps_match", "")
            s_diff = row.get("noise_std_dev_max_abs_diff", "")
            s_ok = row.get("noise_std_dev_all_steps_match", "")
            lines.append(f"| {row['config']} | {v_diff} | {v_ok} | {p_diff} | {p_ok} | {s_diff} | {s_ok} |")
        lines.append("")
    report_path = out_root / "trajectory_debug_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {report_path}")
    dump_path = write_tensor_dump_file(
        out_root=out_root,
        data=data,
        effective_gpus=effective_gpus,
        args=args,
        failures=failures,
    )
    print(f"Tensor dump written to {dump_path}")


if __name__ == "__main__":
    main()
