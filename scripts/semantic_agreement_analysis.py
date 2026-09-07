"""规则语义 vs LLM 语义 · 样本级一致性分析（评审修复 · E4 / Comment 5b）

回答评审："分析 rule-based 与 LLM 标签在样本级的重叠与分歧（而非仅聚合阳性率）"。

输入（本仓库随附的示例数据，可直接运行）：
    data/samples/houses_sample.jsonl           （200 条分层示例：title + has_elevator + decoration_type 等）
    data/samples/llm_semantic_labels_sample.csv （对应 200 条 house_id 的 LLM 28 标签）
规则标注：复用 src.research.llm_semantic_vs_rule_runner 的 RULE_SEMANTIC_KEYWORDS
           从 title 确定性重生成（与实验同口径）。

产物（data/expected_outputs/）：
    semantic_agreement.csv   逐标签 Cohen's κ + 四格表 + 阳性率
    semantic_agreement.md    论文可用 markdown（含分歧画像与冗余统计）

注：论文结果基于全量 7,140 条标注；在示例数据上运行得到的是 200 条抽样口径的结果。

用法：
    uv run python scripts/semantic_agreement_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.research.llm_semantic_vs_rule_runner import (  # noqa: E402
    LLM_SEMANTIC_LABELS,
    RULE_SEMANTIC_FEATURES,
    _extract_rule_semantic_features,
)

HOUSES = PROJECT_ROOT / "data" / "samples" / "houses_sample.jsonl"
LLM_CSV = PROJECT_ROOT / "data" / "samples" / "llm_semantic_labels_sample.csv"
OUT_DIR = PROJECT_ROOT / "data" / "expected_outputs"


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """两 0/1 序列的 Cohen's κ。"""
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a == b).mean())
    pa1, pb1 = float(a.mean()), float(b.mean())
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if abs(1 - pe) < 1e-12:
        return float("nan")
    return (po - pe) / (1 - pe)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    houses = pd.read_json(HOUSES, lines=True)
    rule_df = _extract_rule_semantic_features(houses)
    llm_df = pd.read_csv(LLM_CSV)
    df = rule_df.merge(llm_df, on="house_id", how="inner", suffixes=("_rule", ""))
    # 并入画像字段
    profile_cols = ["house_id", "total_price_wan", "area_sqm", "region_slug", "decoration_type", "has_elevator"]
    df = df.merge(houses[profile_cols], on="house_id", how="left")
    n = len(df)
    print(f"[INFO] 合并样本: {n}")

    overlap = sorted(set(RULE_SEMANTIC_FEATURES) & set(LLM_SEMANTIC_LABELS))
    rows = []
    for name in RULE_SEMANTIC_FEATURES:
        rule_col = df[f"{name}_rule"] if name in overlap else df[name]
        llm_col = df[name] if name in LLM_SEMANTIC_LABELS else pd.Series(0, index=df.index)
        both_one = int(((rule_col == 1) & (llm_col == 1)).sum())
        rule_only = int(((rule_col == 1) & (llm_col == 0)).sum())
        llm_only = int(((rule_col == 0) & (llm_col == 1)).sum())
        both_zero = int(((rule_col == 0) & (llm_col == 0)).sum())
        rows.append(
            {
                "label": name,
                "rule_positive_rate": float(rule_col.mean()),
                "llm_positive_rate": float(llm_col.mean()),
                "agreement": float((rule_col == llm_col).mean()),
                "cohens_kappa": cohen_kappa(rule_col, llm_col),
                "both_one": both_one,
                "rule_only": rule_only,
                "llm_only": llm_only,
                "both_zero": both_zero,
                "disagreement_rate": (rule_only + llm_only) / n,
            }
        )
    result = pd.DataFrame(rows)
    out_csv = OUT_DIR / "semantic_agreement.csv"
    result.to_csv(out_csv, index=False, encoding="utf-8-sig", float_format="%.6f")
    print(f"[OK] {out_csv}")

    # ---- 冗余统计：semantic 标签 vs 结构化字段 ----
    redundancy = []
    if "has_elevator" in df.columns and "elevator_rule" in df.columns:
        he = pd.to_numeric(df["has_elevator"], errors="coerce").fillna(-1)
        er = df["elevator_rule"]
        redundancy.append(
            f"- elevator(规则) 与 has_elevator(结构化)：规则阳性中结构化也为真的比例 "
            f"{float((he[er == 1] == 1).mean()):.4f}（重合度越高说明语义标签越冗余）"
        )
    if "decoration_type" in df.columns and "luxury_decor_rule" in df.columns:
        dec = df["decoration_type"].astype("string").fillna("")
        ld = df["luxury_decor_rule"]
        luxury_like = dec.str.contains("豪|精", na=False)
        redundancy.append(
            f"- luxury_decor(规则) 与 decoration_type 含豪/精：规则阳性中结构化匹配比例 "
            f"{float(luxury_like[ld == 1].mean()):.4f}"
        )

    # ---- 分歧画像：不一致样本的价格/区域分布（取分歧率 Top5 标签）----
    top_div = result.nlargest(5, "disagreement_rate")["label"].tolist()
    profile_lines = []
    for name in top_div:
        rule_col = df[f"{name}_rule"] if name in overlap else df[name]
        llm_col = df[name] if name in LLM_SEMANTIC_LABELS else pd.Series(0, index=df.index)
        div = rule_col != llm_col
        price_div = pd.to_numeric(df.loc[div, "total_price_wan"], errors="coerce")
        price_all = pd.to_numeric(df["total_price_wan"], errors="coerce")
        region_top = df.loc[div, "region_slug"].value_counts().head(3)
        profile_lines.append(
            f"- {name}：分歧 {int(div.sum())} 条（{float(div.mean())*100:.1f}%）；"
            f"分歧样本价格中位 {price_div.median():.1f} vs 全样本 {price_all.median():.1f}；"
            f"分歧 Top 区域：{', '.join(f'{k}({v})' for k, v in region_top.items())}"
        )

    kappa_vals = result["cohens_kappa"].dropna()
    md = [
        "# 规则 vs LLM 语义 · 样本级一致性分析（Comment 5b）",
        "",
        f"- 样本量：{n}（规则标注由 title 确定性重生成，与实验同口径；LLM = Qwen3-32B 全量标注）",
        f"- Cohen's κ：mean={kappa_vals.mean():.4f}，median={kappa_vals.median():.4f}，"
        f"range=[{kappa_vals.min():.4f}, {kappa_vals.max():.4f}]",
        "- κ 解读：<0.2 几乎无一致；0.2-0.4 弱；0.4-0.6 中等；>0.6 强",
        "",
        "## 逐标签一致性（按 κ 降序，前 15）",
        "",
        "| 标签 | 规则阳性率 | LLM阳性率 | agreement | κ | both1 | rule_only | llm_only |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in result.sort_values("cohens_kappa", ascending=False).head(15).iterrows():
        md.append(
            f"| {r['label']} | {r['rule_positive_rate']:.4f} | {r['llm_positive_rate']:.4f} "
            f"| {r['agreement']:.4f} | {r['cohens_kappa']:.4f} "
            f"| {int(r['both_one'])} | {int(r['rule_only'])} | {int(r['llm_only'])} |"
        )
    md += [
        "",
        "## 与结构化特征的冗余统计",
        "",
        *(redundancy or ["- （无可用结构化对照列）"]),
        "",
        "## 分歧样本画像（分歧率 Top5 标签）",
        "",
        *profile_lines,
        "",
        "注：LLM 标注中约 40.2% 全零行包含 API 失败回退与『真未提及』两类，不可区分（固有局限，论文已声明）。",
    ]
    out_md = OUT_DIR / "semantic_agreement.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[OK] {out_md}")
    print(f"[STAT] κ mean={kappa_vals.mean():.4f} median={kappa_vals.median():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
