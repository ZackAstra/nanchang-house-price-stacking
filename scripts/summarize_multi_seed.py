"""
多种子模型指标汇总（评审修复 · 任务1）

读取多个种子的 model_metrics.csv，汇总每个模型的
MAE 均值/标准差/平均名次/第一名次数（按各种子内 MAE 升序排名）。

真实种子为 {42, 7, 21, 84, 168}（论文曾误写为 {42,51,4,68,100}）。
seed 从目录名解析；空目录名（数据根目录）对应 seed 42 主实验。

本仓库仅随附 seed 42 的 model_metrics.csv（data/expected_outputs/model_metrics.csv），
其余种子目录需在获得完整研究输出（见 README 数据获取）后放入
data/expected_outputs/<review5_seed_N>/model_metrics.csv 方可参与汇总。

产物（data/expected_outputs/）：
    multi_seed_summary.csv        每模型一行：mae_mean/mae_std/rank_mean/best_count 等
    multi_seed_model_metrics.csv  长表：每 (model, seed) 一行，含 seed 与种子内名次

同时在 stdout 打印与已提交的 multi_seed_summary.csv（论文 5 种子汇总）
的数值复核差异。

用法：
    uv run python scripts/summarize_multi_seed.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = PROJECT_ROOT / "data" / "expected_outputs"
OUT_DIR = RESEARCH_DIR

# (目录名, 真实种子)。空目录名表示数据根目录下的 model_metrics.csv（seed 42 主实验）。
SEED_DIRS = [
    ("", 42),
    ("review5_seed_7", 7),
    ("review5_seed_21", 21),
    ("review5_seed_84", 84),
    ("review5_seed_168", 168),
]

OLD_SUMMARY = RESEARCH_DIR / "multi_seed_summary.csv"


def parse_seed_from_dirname(name: str) -> int | None:
    """从目录名解析种子；reviewX_main_seed42 / review_full_final 目录约定为 seed 42。"""
    m = re.search(r"review\d*_seed_(\d+?)(?:_|$)", name)
    if m:
        return int(m.group(1))
    if name.startswith("review_full_final") or re.match(r"review\d+_main", name):
        return 42
    return None


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    long_frames: list[pd.DataFrame] = []
    for dirname, seed in SEED_DIRS:
        metrics_path = RESEARCH_DIR / "model_metrics.csv" if dirname == "" else RESEARCH_DIR / dirname / "model_metrics.csv"
        if not metrics_path.exists():
            print(f"[WARN] 缺少 {metrics_path}，跳过", file=sys.stderr)
            continue
        parsed = seed if dirname == "" else parse_seed_from_dirname(dirname)
        assert parsed == seed, f"目录名种子解析不一致: {dirname} -> {parsed} != {seed}"
        df = pd.read_csv(metrics_path, encoding="utf-8-sig")
        df["seed"] = seed
        df["run_id"] = dirname
        long_frames.append(df)

    long_df = pd.concat(long_frames, ignore_index=True)
    # 种子内按 MAE 升序排名（并列取最小名次）
    long_df["mae_rank"] = long_df.groupby("seed")["mae"].rank(method="min", ascending=True)

    summary = (
        long_df.groupby("model_name")
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mae_min=("mae", "min"),
            mae_max=("mae", "max"),
            r2_mean=("r2", "mean"),
            rank_mean=("mae_rank", "mean"),
            best_count=("mae_rank", lambda s: int((s == 1).sum())),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
        .sort_values("mae_mean")
    )

    long_out = OUT_DIR / "multi_seed_model_metrics.csv"
    summary_out = OUT_DIR / "multi_seed_summary.csv"
    long_df.to_csv(long_out, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_out, index=False, encoding="utf-8-sig")
    print(f"[OK] {long_out}  ({len(long_df)} 行)")
    print(f"[OK] {summary_out}  ({len(summary)} 行)")

    print("\n=== 多种子汇总（按 MAE 均值升序）===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- 复核旧产物 ----
    print("\n=== 与旧产物 multi_seed_summary.csv 复核 ===")
    if OLD_SUMMARY.exists():
        old = pd.read_csv(OLD_SUMMARY, encoding="utf-8-sig")
        merged = old.merge(summary, on="model_name", suffixes=("_old", "_new"), how="outer")
        diff_found = False
        for col in ["mae_mean", "mae_std", "rank_mean", "best_count"]:
            o, n = f"{col}_old", f"{col}_new"
            if o in merged.columns and n in merged.columns:
                diff = (merged[o].fillna(-9e9) - merged[n].fillna(-9e9)).abs()
                max_diff = diff.max()
                if max_diff > 1e-9:
                    diff_found = True
                    worst = merged.loc[diff.idxmax(), "model_name"]
                    print(f"  [差异] {col}: 最大绝对差 {max_diff:.6g}（模型 {worst}）")
                else:
                    print(f"  [一致] {col}")
        if not diff_found:
            print("  结论：新汇总与旧产物数值完全一致。")
    else:
        print(f"  旧产物不存在: {OLD_SUMMARY}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
