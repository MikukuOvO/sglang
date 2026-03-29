#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path


ROOT = Path("/root/miles/sglang")
MODEL_PATH = "stabilityai/stable-diffusion-3-medium-diffusers"
PROMPT = "A red ceramic mug on a wooden table, soft natural light, high detail"
DEFAULT_RESOLUTIONS = [512, 1024]
DEFAULT_NUM_OUTPUTS = [1, 2, 4, 8, 12, 16]
RUN_FILE_RE = re.compile(
    r"^(?P<resolution>\d+x\d+)_n(?P<num_outputs>\d+)_r(?P<repeat>\d+)_a(?P<attempt>\d+)\.json$"
)


@dataclass
class RunRecord:
    resolution: str
    width: int
    height: int
    num_outputs_per_prompt: int
    repeat: int
    attempt: int
    success: bool
    exit_code: int
    output_count: int
    wall_time_s: float
    total_duration_ms: float | None
    peak_reserved_mb: float | None
    perf_path: str
    log_path: str
    image_dir: str
    command: str
    error: str | None = None


def logical_key(record: RunRecord) -> tuple[str, int, int]:
    return (record.resolution, record.num_outputs_per_prompt, record.repeat)


def prepend_pythonpath(env: dict[str, str], value: str) -> dict[str, str]:
    updated = env.copy()
    current = updated.get("PYTHONPATH")
    updated["PYTHONPATH"] = value if not current else f"{value}:{current}"
    return updated


def load_perf_metrics(perf_path: Path) -> tuple[float | None, float | None]:
    with perf_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    total_duration_ms = data.get("total_duration_ms")
    checkpoints = data.get("memory_checkpoints") or data.get("memory_snapshots") or {}

    peak_reserved_mb = None
    if isinstance(checkpoints, dict):
        if "after_forward" in checkpoints:
            peak_reserved_mb = checkpoints["after_forward"].get("peak_reserved_mb")
        elif checkpoints:
            peak_values = [
                snapshot.get("peak_reserved_mb")
                for snapshot in checkpoints.values()
                if isinstance(snapshot, dict) and snapshot.get("peak_reserved_mb") is not None
            ]
            if peak_values:
                peak_reserved_mb = max(peak_values)

    return total_duration_ms, peak_reserved_mb


def get_run_label(resolution: str, n: int, repeat: int, attempt: int) -> str:
    return f"{resolution}_n{n}_r{repeat}_a{attempt}"


def get_image_dir(output_root: Path, resolution: str, n: int, repeat: int) -> Path:
    return output_root / "images" / resolution / f"n{n}" / f"repeat{repeat}"


def count_run_outputs(image_dir: Path, run_label: str) -> int:
    if not image_dir.exists():
        return 0
    return len(list(image_dir.glob(f"{run_label}*.png")))


def latest_successful_records(records: list[RunRecord]) -> list[RunRecord]:
    latest: dict[tuple[str, int, int], RunRecord] = {}
    for record in records:
        if not record.success:
            continue
        if record.total_duration_ms is None or record.peak_reserved_mb is None:
            continue
        key = logical_key(record)
        previous = latest.get(key)
        if previous is None or record.attempt > previous.attempt:
            latest[key] = record
    return sorted(
        latest.values(),
        key=lambda record: (
            int(record.resolution.split("x")[0]),
            record.num_outputs_per_prompt,
            record.repeat,
        ),
    )


def next_attempt_number(
    records: list[RunRecord],
    *,
    output_root: Path,
    resolution: str,
    n: int,
    repeat: int,
) -> int:
    attempts = [
        record.attempt
        for record in records
        if record.resolution == resolution
        and record.num_outputs_per_prompt == n
        and record.repeat == repeat
    ]
    run_prefix = f"{resolution}_n{n}_r{repeat}_a"
    for directory, suffix in ((output_root / "perf", ".json"), (output_root / "logs", ".log")):
        if not directory.exists():
            continue
        for path in directory.glob(f"{run_prefix}*{suffix}"):
            match = RUN_FILE_RE.match(path.with_suffix(".json").name)
            if match:
                attempts.append(int(match.group("attempt")))
    return (max(attempts) + 1) if attempts else 1


def load_existing_records(output_root: Path) -> list[RunRecord]:
    perf_dir = output_root / "perf"
    if not perf_dir.exists():
        return []

    records: list[RunRecord] = []
    for perf_path in sorted(perf_dir.glob("*.json")):
        match = RUN_FILE_RE.match(perf_path.name)
        if not match:
            continue

        resolution = match.group("resolution")
        n = int(match.group("num_outputs"))
        repeat = int(match.group("repeat"))
        attempt = int(match.group("attempt"))
        width, height = (int(value) for value in resolution.split("x"))
        run_label = perf_path.stem
        image_dir = get_image_dir(output_root, resolution, n, repeat)
        log_path = output_root / "logs" / f"{run_label}.log"
        output_count = count_run_outputs(image_dir, run_label)

        total_duration_ms = None
        peak_reserved_mb = None
        success = False
        error = None
        try:
            total_duration_ms, peak_reserved_mb = load_perf_metrics(perf_path)
        except Exception as exc:  # noqa: BLE001
            error = f"failed to parse perf dump: {exc}"

        if error is None:
            if total_duration_ms is None:
                error = "missing total_duration_ms"
            elif peak_reserved_mb is None:
                error = "missing peak_reserved_mb"
            elif output_count != n:
                error = f"expected {n} outputs, found {output_count}"
            else:
                success = True

        records.append(
            RunRecord(
                resolution=resolution,
                width=width,
                height=height,
                num_outputs_per_prompt=n,
                repeat=repeat,
                attempt=attempt,
                success=success,
                exit_code=0 if success else -1,
                output_count=output_count,
                wall_time_s=0.0,
                total_duration_ms=total_duration_ms,
                peak_reserved_mb=peak_reserved_mb,
                perf_path=str(perf_path),
                log_path=str(log_path),
                image_dir=str(image_dir),
                command="[loaded_from_disk]",
                error=error,
            )
        )

    return records


def format_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def format_ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}x"


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def svg_line_chart(
    title: str,
    x_label: str,
    y_label: str,
    points: list[tuple[float, float]],
    output_path: Path,
) -> None:
    width = 920
    height = 560
    margin_left = 90
    margin_right = 40
    margin_top = 70
    margin_bottom = 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x = min(xs)
    max_x = max(xs)
    min_y = 0.0
    max_y = max(ys) if ys else 1.0
    if max_y <= min_y:
        max_y = min_y + 1.0

    def map_x(value: float) -> float:
        if max_x == min_x:
            return margin_left + plot_width / 2
        ratio = (value - min_x) / (max_x - min_x)
        return margin_left + ratio * plot_width

    def map_y(value: float) -> float:
        ratio = (value - min_y) / (max_y - min_y)
        return margin_top + plot_height - ratio * plot_height

    grid_lines = []
    y_ticks = 5
    for tick in range(y_ticks + 1):
        value = min_y + (max_y - min_y) * tick / y_ticks
        y = map_y(value)
        grid_lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" '
            'stroke="#d8d8d8" stroke-width="1" />'
        )
        grid_lines.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            'font-family="sans-serif" font-size="14" fill="#444">'
            f"{value:.1f}</text>"
        )

    for x in xs:
        mapped_x = map_x(x)
        grid_lines.append(
            f'<line x1="{mapped_x:.2f}" y1="{margin_top}" x2="{mapped_x:.2f}" y2="{height - margin_bottom}" '
            'stroke="#efefef" stroke-width="1" />'
        )
        grid_lines.append(
            f'<text x="{mapped_x:.2f}" y="{height - margin_bottom + 28}" text-anchor="middle" '
            'font-family="sans-serif" font-size="14" fill="#444">'
            f"{int(x)}</text>"
        )

    polyline_points = " ".join(f"{map_x(x):.2f},{map_y(y):.2f}" for x, y in points)
    markers = []
    labels = []
    for x, y in points:
        cx = map_x(x)
        cy = map_y(y)
        markers.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="5" fill="#1473e6" stroke="white" stroke-width="2" />'
        )
        labels.append(
            f'<text x="{cx:.2f}" y="{cy - 12:.2f}" text-anchor="middle" '
            'font-family="sans-serif" font-size="13" fill="#1473e6">'
            f"{y:.1f}</text>"
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" fill="white" />
  <text x="{width / 2}" y="36" text-anchor="middle" font-family="sans-serif" font-size="24" fill="#111">{escape(title)}</text>
  <text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="16" fill="#444">{escape(x_label)}</text>
  <text x="24" y="{height / 2}" text-anchor="middle" font-family="sans-serif" font-size="16" fill="#444" transform="rotate(-90 24 {height / 2})">{escape(y_label)}</text>
  {''.join(grid_lines)}
  <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#555" stroke-width="2" />
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#555" stroke-width="2" />
  <polyline fill="none" stroke="#1473e6" stroke-width="3" points="{polyline_points}" />
  {''.join(markers)}
  {''.join(labels)}
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def build_markdown(
    summary_rows: list[dict[str, float | int | str]],
    output_path: Path,
    *,
    model_path: str,
    prompt: str,
    attention_backend: str,
    steps: int,
    guidance_scale: float,
    seed: int,
    repeats: int,
) -> None:
    lines = [
        "# SD3 num_outputs_per_prompt Benchmark",
        "",
        f"- Model: `{model_path}`",
        "- Backend: `sglang`",
        f"- Attention backend: `{attention_backend}`",
        f"- Prompt: `{prompt}`",
        f"- Steps: `{steps}`",
        f"- Guidance scale: `{guidance_scale}`",
        f"- Seed: `{seed}`",
        f"- Repeats per setting: `{repeats}`",
        "",
        "| Resolution | num_outputs_per_prompt | Median total time (ms) | Median peak reserved memory (MB) | Median time per image (ms) | Speedup vs n=1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in summary_rows:
        lines.append(
            "| "
            f"{row['resolution']} | "
            f"{row['num_outputs_per_prompt']} | "
            f"{row['median_total_duration_ms']:.2f} | "
            f"{row['median_peak_reserved_mb']:.2f} | "
            f"{row['median_time_per_image_ms']:.2f} | "
            f"{row['speedup_vs_n1']:.2f}x |"
        )

    lines.extend(["", "## Conclusion", ""])
    by_resolution: dict[str, list[dict[str, float | int | str]]] = {}
    for row in summary_rows:
        by_resolution.setdefault(str(row["resolution"]), []).append(row)

    for resolution, rows in by_resolution.items():
        rows = sorted(rows, key=lambda row: int(row["num_outputs_per_prompt"]))
        base = rows[0]
        last = rows[-1]
        last_n = int(last["num_outputs_per_prompt"])
        base_tpi = float(base["median_time_per_image_ms"])
        last_tpi = float(last["median_time_per_image_ms"])
        base_mem = float(base["median_peak_reserved_mb"])
        last_mem = float(last["median_peak_reserved_mb"])
        tpi_reduction = (1.0 - last_tpi / base_tpi) * 100.0 if base_tpi else 0.0
        mem_growth = last_mem / base_mem if base_mem else 0.0
        lines.append(
            f"- `{resolution}`: `n={last_n}` completed successfully. Median time per image improved from "
            f"`{base_tpi:.2f} ms` at `n=1` to `{last_tpi:.2f} ms` at `n={last_n}` "
            f"({tpi_reduction:.1f}% lower), while median peak reserved memory increased "
            f"from `{base_mem:.2f} MB` to `{last_mem:.2f} MB` ({mem_growth:.2f}x)."
        )

    lines.append(
        "- The SD3 runs scaled with `num_outputs_per_prompt` via the full batched path: a single request reused the pre-denoising pipeline and denoised the latent batch timestep-by-timestep."
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_results(records: list[RunRecord]) -> list[dict[str, float | int | str]]:
    records = latest_successful_records(records)
    grouped: dict[tuple[str, int], list[RunRecord]] = {}
    for record in records:
        if record.success and record.total_duration_ms is not None and record.peak_reserved_mb is not None:
            grouped.setdefault((record.resolution, record.num_outputs_per_prompt), []).append(record)

    summary_rows: list[dict[str, float | int | str]] = []
    baseline_tpi: dict[str, float] = {}

    for resolution in sorted({resolution for resolution, _ in grouped.keys()}, key=lambda value: int(value.split("x")[0])):
        key = (resolution, 1)
        values = grouped.get(key, [])
        if not values:
            continue
        baseline_tpi[resolution] = median(
            [record.total_duration_ms / record.num_outputs_per_prompt for record in values if record.total_duration_ms is not None]
        )

    for (resolution, n), runs in sorted(
        grouped.items(),
        key=lambda item: (int(item[0][0].split("x")[0]), item[0][1]),
    ):
        median_total = median([record.total_duration_ms for record in runs if record.total_duration_ms is not None])
        median_peak = median([record.peak_reserved_mb for record in runs if record.peak_reserved_mb is not None])
        median_tpi = median(
            [
                record.total_duration_ms / record.num_outputs_per_prompt
                for record in runs
                if record.total_duration_ms is not None
            ]
        )
        base = baseline_tpi.get(resolution)
        speedup = base / median_tpi if base and median_tpi else None
        summary_rows.append(
            {
                "resolution": resolution,
                "num_outputs_per_prompt": n,
                "median_total_duration_ms": median_total,
                "median_peak_reserved_mb": median_peak,
                "median_time_per_image_ms": median_tpi,
                "speedup_vs_n1": speedup or 0.0,
                "successful_runs": len(runs),
            }
        )

    return summary_rows


def write_resolution_csvs(
    summary_rows: list[dict[str, float | int | str]],
    output_root: Path,
) -> None:
    fieldnames = [
        "resolution",
        "num_outputs_per_prompt",
        "median_total_duration_ms",
        "median_peak_reserved_mb",
        "median_time_per_image_ms",
        "speedup_vs_n1",
        "successful_runs",
    ]
    by_resolution: dict[str, list[dict[str, float | int | str]]] = {}
    for row in summary_rows:
        by_resolution.setdefault(str(row["resolution"]), []).append(row)

    for resolution, rows in by_resolution.items():
        csv_path = output_root / f"benchmark_summary_{resolution}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in sorted(rows, key=lambda item: int(item["num_outputs_per_prompt"])):
                writer.writerow(row)


def run_once(
    args: argparse.Namespace,
    records: list[RunRecord],
    output_root: Path,
    size: int,
    n: int,
    repeat: int,
) -> RunRecord:
    resolution = f"{size}x{size}"
    attempt = next_attempt_number(
        records,
        output_root=output_root,
        resolution=resolution,
        n=n,
        repeat=repeat,
    )
    run_label = get_run_label(resolution, n, repeat, attempt)
    perf_path = output_root / "perf" / f"{run_label}.json"
    log_path = output_root / "logs" / f"{run_label}.log"
    image_dir = get_image_dir(output_root, resolution, n, repeat)
    image_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "sglang.multimodal_gen.runtime.entrypoints.cli.main",
        "generate",
        "--backend",
        "sglang",
        "--attention-backend",
        args.attention_backend,
        "--model-path",
        args.model_path,
        "--prompt",
        args.prompt,
        "--width",
        str(size),
        "--height",
        str(size),
        "--num-inference-steps",
        str(args.steps),
        "--guidance-scale",
        str(args.guidance_scale),
        "--num-outputs-per-prompt",
        str(n),
        "--output-path",
        str(image_dir),
        "--output-file-name",
        f"{run_label}.png",
        "--seed",
        str(args.seed),
        "--num-gpus",
        "1",
        "--perf-dump-path",
        str(perf_path),
    ]

    env = prepend_pythonpath(os.environ, str(ROOT / "python"))
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    flashinfer_base = output_root / ".runtime_cache" / "flashinfer_home"
    flashinfer_base.mkdir(parents=True, exist_ok=True)
    env["FLASHINFER_WORKSPACE_BASE"] = str(flashinfer_base)

    started = time.monotonic()
    proc = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    wall_time_s = time.monotonic() - started

    output_count = count_run_outputs(image_dir, run_label)
    total_duration_ms = None
    peak_reserved_mb = None
    error = None

    if proc.returncode == 0 and perf_path.exists():
        try:
            total_duration_ms, peak_reserved_mb = load_perf_metrics(perf_path)
        except Exception as exc:  # noqa: BLE001
            error = f"failed to parse perf dump: {exc}"
    elif proc.returncode == 0:
        error = "perf dump file missing"
    else:
        error = f"command failed with exit code {proc.returncode}"

    expected_outputs = n
    success = (
        proc.returncode == 0
        and perf_path.exists()
        and total_duration_ms is not None
        and peak_reserved_mb is not None
        and output_count == expected_outputs
    )
    if not success and error is None and output_count != expected_outputs:
        error = f"expected {expected_outputs} outputs, found {output_count}"

    log_lines = [
        f"# Command\n{' '.join(command)}\n",
        f"# CUDA_VISIBLE_DEVICES={args.gpu}\n",
        f"# Started: {datetime.now(UTC).isoformat()}\n",
        f"# Wall time (s): {wall_time_s:.2f}\n",
        f"# Exit code: {proc.returncode}\n",
        f"# Output count: {output_count}\n",
        f"# total_duration_ms: {format_ms(total_duration_ms)}\n",
        f"# peak_reserved_mb: {format_ms(peak_reserved_mb)}\n",
    ]
    if error:
        log_lines.append(f"# Error: {error}\n")
    log_lines.append("\n# Combined stdout/stderr\n")
    log_lines.append(proc.stdout)
    log_path.write_text("".join(log_lines), encoding="utf-8")

    return RunRecord(
        resolution=resolution,
        width=size,
        height=size,
        num_outputs_per_prompt=n,
        repeat=repeat,
        attempt=attempt,
        success=success,
        exit_code=proc.returncode,
        output_count=output_count,
        wall_time_s=wall_time_s,
        total_duration_ms=total_duration_ms,
        peak_reserved_mb=peak_reserved_mb,
        perf_path=str(perf_path),
        log_path=str(log_path),
        image_dir=str(image_dir),
        command=" ".join(command),
        error=error,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SD3 num_outputs_per_prompt scaling.")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--attention-backend", default="torch_sdpa")
    parser.add_argument("--resolutions", nargs="+", type=int, default=DEFAULT_RESOLUTIONS)
    parser.add_argument("--num-outputs", nargs="+", type=int, default=DEFAULT_NUM_OUTPUTS)
    parser.add_argument("--run-name", default=datetime.now(UTC).strftime("sd3_num_outputs_bench_%Y%m%dT%H%M%SZ"))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = ROOT / "tmp" / args.run_name
    for subdir in ["perf", "logs", "images", "plots"]:
        (output_root / subdir).mkdir(parents=True, exist_ok=True)

    print(f"[info] output_root={output_root}")
    print(f"[info] gpu={args.gpu} resolutions={args.resolutions} num_outputs={args.num_outputs} repeats={args.repeats}")
    print(f"[info] prompt={args.prompt}")

    records: list[RunRecord] = []
    if args.resume:
        records = load_existing_records(output_root)
        loaded_successes = len(latest_successful_records(records))
        print(
            f"[info] resume enabled: loaded {len(records)} existing attempt(s), "
            f"{loaded_successes} successful logical point(s)"
        )

    existing_successes = {
        logical_key(record): record for record in latest_successful_records(records)
    }
    total_runs = len(args.resolutions) * len(args.num_outputs) * args.repeats
    current = 0

    for size in args.resolutions:
        for n in args.num_outputs:
            for repeat in range(1, args.repeats + 1):
                current += 1
                print(f"[run {current}/{total_runs}] resolution={size}x{size} n={n} repeat={repeat}")
                resolution = f"{size}x{size}"
                existing = existing_successes.get((resolution, n, repeat))
                if existing is not None:
                    print(
                        "[skip] "
                        f"resolution={existing.resolution} n={existing.num_outputs_per_prompt} "
                        f"repeat={existing.repeat} attempt={existing.attempt} "
                        f"total_duration_ms={format_ms(existing.total_duration_ms)} "
                        f"peak_reserved_mb={format_ms(existing.peak_reserved_mb)} "
                        f"outputs={existing.output_count}"
                    )
                    continue

                last_record: RunRecord | None = None
                for _ in range(1, args.max_attempts + 1):
                    record = run_once(args, records, output_root, size, n, repeat)
                    records.append(record)
                    last_record = record
                    status = "ok" if record.success else "failed"
                    print(
                        "[result] "
                        f"resolution={record.resolution} n={record.num_outputs_per_prompt} repeat={record.repeat} "
                        f"attempt={record.attempt} status={status} "
                        f"total_duration_ms={format_ms(record.total_duration_ms)} "
                        f"peak_reserved_mb={format_ms(record.peak_reserved_mb)} "
                        f"outputs={record.output_count}"
                    )
                    if record.success:
                        existing_successes[logical_key(record)] = record
                        break
                    print(f"[retry] reason={record.error}")
                    time.sleep(2.0)

                if last_record is not None and not last_record.success:
                    print(
                        f"[warn] failed after {args.max_attempts} attempts: "
                        f"{last_record.resolution} n={last_record.num_outputs_per_prompt} repeat={last_record.repeat}"
                    )

    csv_path = output_root / "benchmark_runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(RunRecord.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    summary_rows = aggregate_results(records)
    write_resolution_csvs(summary_rows, output_root)
    successful_logical_records = latest_successful_records(records)
    summary_json_path = output_root / "benchmark_summary.json"
    summary_json_path.write_text(
        json.dumps(
            {
                "output_root": str(output_root),
                "model_path": args.model_path,
                "prompt": args.prompt,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
                "gpu": args.gpu,
                "repeats": args.repeats,
                "resolutions": args.resolutions,
                "num_outputs": args.num_outputs,
                "summary_rows": summary_rows,
                "successful_logical_runs": len(successful_logical_records),
                "successful_attempts": len([record for record in records if record.success]),
                "total_attempts": len(records),
                "failed_attempts": len([record for record in records if not record.success]),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if summary_rows:
        build_markdown(
            summary_rows,
            output_root / "summary.md",
            model_path=args.model_path,
            prompt=args.prompt,
            attention_backend=args.attention_backend,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            repeats=args.repeats,
        )

    by_resolution: dict[str, list[dict[str, float | int | str]]] = {}
    for row in summary_rows:
        by_resolution.setdefault(str(row["resolution"]), []).append(row)

    for resolution, rows in by_resolution.items():
        rows = sorted(rows, key=lambda row: int(row["num_outputs_per_prompt"]))
        size = int(resolution.split("x")[0])
        total_points = [
            (float(row["num_outputs_per_prompt"]), float(row["median_total_duration_ms"]))
            for row in rows
        ]
        tpi_points = [
            (float(row["num_outputs_per_prompt"]), float(row["median_time_per_image_ms"]))
            for row in rows
        ]
        memory_points = [
            (float(row["num_outputs_per_prompt"]), float(row["median_peak_reserved_mb"]))
            for row in rows
        ]
        svg_line_chart(
            title=f"SD3 {size}x{size}: total duration vs num_outputs_per_prompt",
            x_label="num_outputs_per_prompt",
            y_label="Median total duration (ms)",
            points=total_points,
            output_path=output_root / "plots" / f"{resolution}_total_duration.svg",
        )
        svg_line_chart(
            title=f"SD3 {size}x{size}: time per image vs num_outputs_per_prompt",
            x_label="num_outputs_per_prompt",
            y_label="Median time per image (ms)",
            points=tpi_points,
            output_path=output_root / "plots" / f"{resolution}_time_per_image.svg",
        )
        svg_line_chart(
            title=f"SD3 {size}x{size}: peak reserved memory vs num_outputs_per_prompt",
            x_label="num_outputs_per_prompt",
            y_label="Median peak reserved memory (MB)",
            points=memory_points,
            output_path=output_root / "plots" / f"{resolution}_peak_reserved_memory.svg",
        )

    failed_runs = [record for record in records if not record.success]
    if failed_runs:
        print(f"[done] completed with failures. output_root={output_root}")
        for record in failed_runs:
            print(
                f"[failure] resolution={record.resolution} n={record.num_outputs_per_prompt} "
                f"repeat={record.repeat} attempt={record.attempt} error={record.error}"
            )
        return 1

    print(f"[done] completed successfully. output_root={output_root}")
    print(f"[artifacts] summary_md={output_root / 'summary.md'}")
    print(f"[artifacts] summary_json={summary_json_path}")
    print(f"[artifacts] runs_csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
