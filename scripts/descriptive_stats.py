"""
描述性统计（评审修复 · 任务2）

基于 cleaned_features.csv 生成论文 §2.1 需要的数据摘要。
cleaned_features.csv 为主实验清洗后特征文件，本仓库未随附（见 README 数据获取），
需放入 data/expected_outputs/cleaned_features.csv 后运行。

产物（data/expected_outputs/）：
    descriptive_stats_numeric.csv      目标变量与核心数值特征的
                                       count/mean/std/min/p25/median/p75/max/skewness
    descriptive_stats_categorical.csv  region_slug 分布、house_layout Top10、价格分段分布
    descriptive_stats.md               markdown 表，便于直接进论文

用法：
    uv run python scripts/descriptive_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "expected_outputs" / "cleaned_features.csv"
OUT_DIR = PROJECT_ROOT / "data" / "expected_outputs"

NUMERIC_COLS = [
    "total_price_wan",
    "area_sqm",
    "room_count",
    "hall_count",
    "bath_count",
    "total_floors",
    "house_certificate_years",
    "poi_bank_count",
    "poi_bus_count",
    "poi_subway_count",
    "poi_school_count",
    "poi_restaurant_count",
    "poi_shop_count",
    "poi_hospital_count",
    "accessibility_index",
    "poi_balance_score",
]

# 价格分段（万元）
PRICE_BINS = [0, 60, 100, 150, float("inf")]
PRICE_LABELS = ["<60万", "60-100万", "100-150万", ">150万"]


def _fmt(x: float) -> str:
    return f"{x:.4f}" if abs(x) < 1000 else f"{x:.2f}"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    n = len(df)
    print(f"[INFO] cleaned_features.csv: {n} 行")

    # ---- 数值特征 ----
    rows = []
    for col in NUMERIC_COLS:
        if col not in df.columns:
            print(f"[WARN] 缺少列 {col}，跳过", file=sys.stderr)
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "feature": col,
                "count": int(s.notna().sum()),
                "mean": s.mean(),
                "std": s.std(),
                "min": s.min(),
                "p25": s.quantile(0.25),
                "median": s.median(),
                "p75": s.quantile(0.75),
                "max": s.max(),
                "skewness": s.skew(),
            }
        )
    numeric = pd.DataFrame(rows)
    numeric_out = OUT_DIR / "descriptive_stats_numeric.csv"
    numeric.to_csv(numeric_out, index=False, encoding="utf-8-sig", float_format="%.6f")
    print(f"[OK] {numeric_out}  ({len(numeric)} 行)")

    # ---- 类别/分布 ----
    cat_frames = []

    region = (
        df["region_slug"]
        .value_counts()
        .rename_axis("category")
        .reset_index(name="count")
    )
    region.insert(0, "variable", "region_slug")
    region["pct"] = region["count"] / n * 100
    cat_frames.append(region)

    layout = (
        df["house_layout"]
        .value_counts()
        .head(10)
        .rename_axis("category")
        .reset_index(name="count")
    )
    layout.insert(0, "variable", "house_layout_top10")
    layout["pct"] = layout["count"] / n * 100
    cat_frames.append(layout)

    # house_certificate_years 已数值化（0/2/5 年，pipeline._parse_certificate_years），
    # 此处仅补充分布计数（0 = 未满档/未标注）
    cert = (
        df["house_certificate_years"]
        .astype("string")
        .fillna("unknown")
        .value_counts()
        .rename_axis("category")
        .reset_index(name="count")
    )
    cert.insert(0, "variable", "house_certificate_years")
    cert["pct"] = cert["count"] / n * 100
    cat_frames.append(cert)

    price_seg = pd.cut(df["total_price_wan"], bins=PRICE_BINS, labels=PRICE_LABELS, right=False)
    seg = price_seg.value_counts().reindex(PRICE_LABELS).rename_axis("category").reset_index(name="count")
    seg.insert(0, "variable", "price_segment")
    seg["pct"] = seg["count"] / n * 100
    cat_frames.append(seg)

    categorical = pd.concat(cat_frames, ignore_index=True)
    categorical_out = OUT_DIR / "descriptive_stats_categorical.csv"
    categorical.to_csv(categorical_out, index=False, encoding="utf-8-sig", float_format="%.4f")
    print(f"[OK] {categorical_out}  ({len(categorical)} 行)")

    # ---- markdown ----
    lines = [
        "# 描述性统计（论文 §2.1 数据摘要）",
        "",
        f"- 数据源：`data/expected_outputs/cleaned_features.csv`",
        f"- 样本量：{n}",
        "",
        "## 数值特征",
        "",
        "| 特征 | count | mean | std | min | p25 | median | p75 | max | skewness |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in numeric.iterrows():
        lines.append(
            f"| {r['feature']} | {r['count']} | {_fmt(r['mean'])} | {_fmt(r['std'])} "
            f"| {_fmt(r['min'])} | {_fmt(r['p25'])} | {_fmt(r['median'])} "
            f"| {_fmt(r['p75'])} | {_fmt(r['max'])} | {_fmt(r['skewness'])} |"
        )

    def md_dist(frame: pd.DataFrame, title: str) -> list[str]:
        out = ["", f"## {title}", "", "| 类别 | 计数 | 占比(%) |", "|---|---:|---:|"]
        for _, r in frame.iterrows():
            out.append(f"| {r['category']} | {int(r['count'])} | {r['pct']:.2f} |")
        return out

    lines += md_dist(region, "区域分布（region_slug）")
    lines += md_dist(layout, "户型 Top10（house_layout）")
    lines += md_dist(cert, "产权年限状态分布（house_certificate_years，类别特征）")
    lines += md_dist(seg, "价格分段分布（total_price_wan，万元）")

    md_out = OUT_DIR / "descriptive_stats.md"
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] {md_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
