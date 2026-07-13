#!/usr/bin/env python3
"""Parse training logs, emit CSV/summary/plots under baseline_experiments/meeting_summary."""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    np = None  # type: ignore
    plt = None  # type: ignore

ROOT = Path("/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/Humanoid/yangky")
BE = ROOT / "baseline_experiments"
LOGS = BE / "logs"
OUT_ROOT = BE / "meeting_summary" / "artifacts"

STEP_RE = re.compile(
    r"Training steps:.*?\|?\s*(\d+)\s*/\s*(\d+)\s*\["
    r"([^<\]]+)<([^,\]]+),\s*([^\]]+)\]"
    r".*?loss=([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)"
    r".*?lr=([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)",
    re.DOTALL,
)
CKPT_RE = re.compile(r"ckpt_(\d+)")


def parse_log(path: Path) -> tuple[list[dict], list[int]]:
    raw = path.read_bytes().replace(b"\r", b"\n").decode("utf-8", errors="ignore")
    by_step: dict[int, dict] = {}
    for m in STEP_RE.finditer(raw):
        step = int(m.group(1))
        total = int(m.group(2))
        by_step[step] = {
            "step": step,
            "total_steps": total,
            "progress_percent": round(100.0 * step / total, 4) if total else 0.0,
            "elapsed": m.group(3).strip(),
            "eta": m.group(4).strip(),
            "loss": float(m.group(6)),
            "lr": float(m.group(7)),
        }
    rows = [by_step[k] for k in sorted(by_step)]
    ckpts = sorted({int(x) for x in CKPT_RE.findall(raw)})
    return rows, ckpts


def rolling(values: list[float], window: int) -> list[float]:
    out: list[float] = []
    s = 0.0
    q: list[float] = []
    for v in values:
        q.append(v)
        s += v
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def summarize(name: str, log_path: Path, rows: list[dict], ckpts: list[int], status: str) -> dict:
    losses = [r["loss"] for r in rows]
    latest = rows[-1] if rows else None
    total = latest["total_steps"] if latest else None
    return {
        "model_run": name,
        "status": status,
        "source_log": str(log_path),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_logged_steps": len(rows),
        "first_step": rows[0]["step"] if rows else None,
        "latest_step": latest["step"] if latest else None,
        "total_steps": total,
        "computed_progress_percent": round(100.0 * latest["step"] / total, 2) if latest and total else None,
        "latest_loss": latest["loss"] if latest else None,
        "latest_lr": latest["lr"] if latest else None,
        "min_loss": min(losses) if losses else None,
        "max_loss": max(losses) if losses else None,
        "mean_loss": float(statistics.fmean(losses)) if losses else None,
        "median_loss": float(statistics.median(losses)) if losses else None,
        "last_100_mean_loss": float(statistics.fmean(losses[-100:])) if losses else None,
        "last_500_mean_loss": float(statistics.fmean(losses[-500:])) if losses else None,
        "checkpoints_observed": ckpts,
        "notes": [
            "Parsed from tqdm training log; W&B run may be offline on remote.",
            "This is training progress, not simulator success-rate evaluation.",
        ],
    }


def write_csv(path: Path, rows: list[dict], with_rolling: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if with_rolling:
        losses = [r["loss"] for r in rows]
        r100 = rolling(losses, 100)
        r500 = rolling(losses, 500)
        fieldnames = [
            "step",
            "total_steps",
            "progress_percent",
            "elapsed",
            "eta",
            "loss",
            "lr",
            "loss_rolling_100",
            "loss_rolling_500",
        ]
        out_rows = []
        for r, a, b in zip(rows, r100, r500):
            d = dict(r)
            d["loss_rolling_100"] = a
            d["loss_rolling_500"] = b
            out_rows.append(d)
    else:
        fieldnames = ["step", "total_steps", "progress_percent", "elapsed", "eta", "loss", "lr"]
        out_rows = rows
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r[k] for k in fieldnames})


def plot_curves(out_dir: Path, prefix: str, title: str, rows: list[dict]) -> None:
    if not rows:
        return
    if not HAS_MPL:
        print(f"WARN {prefix}: matplotlib unavailable, skip plots")
        return
    steps = np.array([r["step"] for r in rows], dtype=float)
    losses = np.array([r["loss"] for r in rows], dtype=float)
    lrs = np.array([r["lr"] for r in rows], dtype=float)
    r100 = np.array(rolling(losses.tolist(), 100), dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=140)
    ax.plot(steps, losses, color="#4C78A8", alpha=0.25, linewidth=0.6, label="loss")
    ax.plot(steps, r100, color="#F58518", linewidth=1.5, label="rolling mean@100")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"{title} — training loss")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_loss_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=140)
    ax.plot(steps, lrs, color="#54A24B", linewidth=1.4)
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    ax.set_title(f"{title} — learning rate")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_lr_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
    clipped = losses[losses < np.quantile(losses, 0.99)] if len(losses) > 50 else losses
    ax.hist(clipped, bins=60, color="#72B7B2", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("loss")
    ax.set_ylabel("count")
    ax.set_title(f"{title} — loss histogram (≤ p99)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_loss_histogram.png")
    plt.close(fig)


def process_run(
    key: str,
    title: str,
    log_name: str,
    status_name: str,
    *,
    skip_plots: bool = False,
) -> dict | None:
    log_path = LOGS / log_name
    if not log_path.exists() or log_path.stat().st_size == 0:
        print(f"SKIP {key}: missing/empty {log_path}")
        return None
    status_path = BE / "state" / status_name
    status = status_path.read_text().strip() if status_path.exists() else "unknown"
    rows, ckpts = parse_log(log_path)
    if not rows:
        print(f"SKIP {key}: no tqdm rows parsed from {log_path}")
        return None
    out_dir = OUT_ROOT / key
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / f"{key}_training_metrics.csv", rows, with_rolling=False)
    write_csv(out_dir / f"{key}_training_metrics_with_rolling.csv", rows, with_rolling=True)
    summary = summarize(key, log_path, rows, ckpts, status)
    (out_dir / f"{key}_training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not skip_plots:
        plot_curves(out_dir, key, title, rows)
    print(
        f"OK {key}: steps={summary['latest_step']}/{summary['total_steps']} "
        f"progress={summary['computed_progress_percent']}% "
        f"loss={summary['latest_loss']} status={status}"
    )
    return summary


def write_overview(summaries: dict[str, dict]) -> None:
    status_map = {}
    for p in sorted((BE / "state").glob("*.status")):
        status_map[p.name] = p.read_text().strip()
    overview = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "remote_root": str(ROOT),
        "statuses": status_map,
        "runs": summaries,
    }
    out = OUT_ROOT / "overview.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(overview, indent=2) + "\n")
    print(f"Wrote {out}")


def write_meeting_md(summaries: dict[str, dict]) -> None:
    psi = summaries.get("psi0_simple")
    dp = summaries.get("diffusion_policy_simple")
    act_path = BE / "state" / "act_simple.status"
    act_status = act_path.read_text().strip() if act_path.exists() else "unknown"
    lines = [
        "# 组会汇报速览：SIMPLE / DexJoCo 多任务 Baseline 实验",
        "",
        f"更新时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
        "",
        "## 一句话结论",
        "",
    ]
    if psi and psi.get("latest_step") == psi.get("total_steps"):
        lines.append(
            f"Psi0 / SIMPLE 多任务 full 训练已完成（{psi['latest_step']}/{psi['total_steps']}），"
            f"最终 train loss≈{psi['latest_loss']}。"
            + (
                f" Diffusion Policy / SIMPLE full 正在跑（{dp['latest_step']}/{dp['total_steps']}，约 {dp['computed_progress_percent']}%）。"
                if dp
                else ""
            )
            + " 当前仍是训练指标；正式 success rate 需等 SIMPLE 仿真闭环 rollout。"
        )
    else:
        lines.append("训练矩阵推进中，详见下表。")
    lines += [
        "",
        "## 当前可汇报进展",
        "",
        "| 模块 | 状态 | 说明 |",
        "|---|---|---|",
        "| SIMPLE 数据 | 已准备 | 17 canonical tasks，540,664 training samples |",
        "| DexJoCo 数据 | 已归一化 | 11 tasks，1,100 episodes，523,763 frames |",
        f"| Psi0 / SIMPLE full | {(psi or {}).get('status', 'unknown')} | "
        f"{(psi or {}).get('latest_step')} / {(psi or {}).get('total_steps')}，"
        f"loss={(psi or {}).get('latest_loss')}，ckpt={(psi or {}).get('checkpoints_observed')} |",
        f"| Diffusion Policy / SIMPLE | {(dp or {}).get('status', 'unknown')} | "
        f"{(dp or {}).get('latest_step')} / {(dp or {}).get('total_steps')}，"
        f"loss={(dp or {}).get('latest_loss')} |",
        f"| ACT / SIMPLE | {act_status} | ResNet18 权重已缓存；等 DP 完成后自动重跑 |",
        "| ABot-M0.5 | blocked | 官方代码/权重未发布 |",
        "| Demo 视频库 | 已保存 | `videos/demo_gallery`：SIMPLE 17 + DexJoCo 11；另有 `SIMPLE_longest` |",
        "| SIMPLE 仿真安装 | install_failed | 节点 DNS 不可用，闭环 rollout 暂缓 |",
        "",
        "## 训练曲线与指标",
        "",
    ]
    if psi:
        lines += [
            "### Psi0 / SIMPLE",
            "",
            "![Psi0 SIMPLE loss curve](artifacts/psi0_simple/psi0_simple_loss_curve.png)",
            "",
            "![Psi0 SIMPLE learning-rate curve](artifacts/psi0_simple/psi0_simple_lr_curve.png)",
            "",
            "![Psi0 SIMPLE loss histogram](artifacts/psi0_simple/psi0_simple_loss_histogram.png)",
            "",
            "| 指标 | 值 |",
            "|---|---:|",
            f"| latest step | {psi['latest_step']} / {psi['total_steps']} |",
            f"| progress | {psi['computed_progress_percent']}% |",
            f"| latest loss | {psi['latest_loss']} |",
            f"| latest lr | {psi['latest_lr']} |",
            f"| min / max / mean / median loss | {psi['min_loss']} / {psi['max_loss']} / {round(psi['mean_loss'],4)} / {psi['median_loss']} |",
            f"| last100 / last500 mean loss | {round(psi['last_100_mean_loss'],4)} / {round(psi['last_500_mean_loss'],4)} |",
            f"| checkpoints | {', '.join(str(x) for x in psi['checkpoints_observed'])} |",
            "",
            "- [psi0_simple_training_summary.json](artifacts/psi0_simple/psi0_simple_training_summary.json)",
            "- [psi0_simple_training_metrics.csv](artifacts/psi0_simple/psi0_simple_training_metrics.csv)",
            "",
        ]
    if dp:
        lines += [
            "### Diffusion Policy / SIMPLE",
            "",
            "![DP SIMPLE loss curve](artifacts/diffusion_policy_simple/diffusion_policy_simple_loss_curve.png)",
            "",
            "![DP SIMPLE learning-rate curve](artifacts/diffusion_policy_simple/diffusion_policy_simple_lr_curve.png)",
            "",
            "![DP SIMPLE loss histogram](artifacts/diffusion_policy_simple/diffusion_policy_simple_loss_histogram.png)",
            "",
            "| 指标 | 值 |",
            "|---|---:|",
            f"| latest step | {dp['latest_step']} / {dp['total_steps']} |",
            f"| progress | {dp['computed_progress_percent']}% |",
            f"| latest loss | {dp['latest_loss']} |",
            f"| latest lr | {dp['latest_lr']} |",
            f"| min / max / mean / median loss | {dp['min_loss']} / {dp['max_loss']} / {round(dp['mean_loss'],4)} / {dp['median_loss']} |",
            f"| last100 / last500 mean loss | {round(dp['last_100_mean_loss'],4)} / {round(dp['last_500_mean_loss'],4)} |",
            f"| checkpoints | {', '.join(str(x) for x in dp['checkpoints_observed'])} |",
            "",
            "- [diffusion_policy_simple_training_summary.json](artifacts/diffusion_policy_simple/diffusion_policy_simple_training_summary.json)",
            "- [diffusion_policy_simple_training_metrics.csv](artifacts/diffusion_policy_simple/diffusion_policy_simple_training_metrics.csv)",
            "",
        ]
    lines += [
        "## 说明",
        "",
        "- 指标来自远端 tqdm 训练日志解析；W&B 多为 offline。",
        "- 报告值仅为 train loss / lr / progress，不是仿真 success rate。",
        "- 闭环 rollout 视频需先恢复节点 DNS 或离线安装 SIMPLE/IsaacSim。",
        "",
        "## 证据路径",
        "",
        f"- `{BE / 'logs'}`",
        f"- `{OUT_ROOT}`",
        f"- `{BE / 'videos' / 'demo_gallery'}`",
        "",
    ]
    md = BE / "meeting_summary" / "00_组会汇报速览.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {md}")


def plot_from_csv(csv_path: Path, out_dir: Path, prefix: str, title: str) -> None:
    rows = []
    with csv_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"step": int(r["step"]), "loss": float(r["loss"]), "lr": float(r["lr"])})
    plot_curves(out_dir, prefix, title, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--plots-only-from", type=Path)
    args = parser.parse_args()
    if args.plots_only_from:
        root = args.plots_only_from
        for key, title in [
            ("psi0_simple", "Psi0 / SIMPLE"),
            ("diffusion_policy_simple", "Diffusion Policy / SIMPLE"),
        ]:
            csv_path = root / key / f"{key}_training_metrics.csv"
            if csv_path.exists():
                plot_from_csv(csv_path, root / key, key, title)
                print(f"PLOTTED {key}")
        return

    skip = args.skip_plots or not HAS_MPL
    summaries: dict[str, dict] = {}
    for key, title, log_name, status_name in [
        ("psi0_simple", "Psi0 / SIMPLE", "psi0_simple_full.log", "psi0_simple.status"),
        (
            "diffusion_policy_simple",
            "Diffusion Policy / SIMPLE",
            "diffusion_policy_simple_full.log",
            "diffusion_policy_simple.status",
        ),
    ]:
        s = process_run(key, title, log_name, status_name, skip_plots=skip)
        if s:
            summaries[key] = s
    write_overview(summaries)
    write_meeting_md(summaries)


if __name__ == "__main__":
    main()
