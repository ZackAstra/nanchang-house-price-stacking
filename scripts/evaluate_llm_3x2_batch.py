# -*- coding: utf-8 -*-
"""DeepSeek-V4-Flash 3×2 批量标注结果解析与评估（评审修复 · Comment 5a 跨模型维度）

输入：data/research/llm_3x3_dataset/batch_output/batch_{prompt}_t{temp}.jsonl
      （SiliconFlow Batch output，每行 {custom_id, response.body.choices[0].message.content, error}）

产物（data/research/llm_3x3_dataset/batch_parsed/）：
    labels_{config}.csv          6 组 0/1 标签（parse_ok 显式标记）
    eval_summary.csv             6 组汇总（解析率/覆盖率/阳性率/与规则 κ/与 Qwen 桥接 κ）
    label_rate_corr.csv          6 组逐标签阳性率相关矩阵
    eval_report.md               论文可用报告
    best_config_full_input.jsonl 最优配置的全量 7,140 条批量输入（供用户上传做 MAE 验证）

用法：common/Scripts/python.exe scripts/evaluate_llm_3x2_batch.py
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.research.llm_semantic_extractor import SEMANTIC_LABELS  # noqa: E402
from src.research.llm_semantic_vs_rule_runner import _extract_rule_semantic_features  # noqa: E402
from scripts.semantic_agreement_analysis import cohen_kappa  # noqa: E402

BASE = PROJECT_ROOT / "data" / "full" / "llm_3x3_dataset"  # 完整数据集需联系作者获取（见 README）
OUT_DIR = BASE / "batch_output"
PARSED_DIR = BASE / "batch_parsed"
SAMPLE_CSV = BASE / "batch_input" / "annotation_sample_2000.csv"
BRIDGE_DIR = BASE / "qwen_bridge"
FULL_INPUT = BASE / "annotation_input.csv"

CONFIGS = [
    ("baseline", "0p1"), ("baseline", "0p3"),
    ("fewshot", "0p1"), ("fewshot", "0p3"),
    ("strict", "0p1"), ("strict", "0p3"),
]


def parse_output_file(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        hid = rec["custom_id"]
        err = rec.get("error")
        raw = ""
        if not err:
            try:
                raw = rec["response"]["body"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                err = {"message": "missing content"}
        labels = _parse_labels(raw) if raw else None
        row = {"house_id": hid, "parse_ok": labels is not None,
               "api_error": json.dumps(err, ensure_ascii=False) if err else ""}
        row.update(labels or {k: pd.NA for k in SEMANTIC_LABELS})
        rows.append(row)
    return pd.DataFrame(rows)


def _parse_labels(raw: str) -> dict[str, int] | None:
    try:
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        obj = json.loads(text[start : end + 1])
        return {k: int(bool(obj.get(k, 0))) for k in SEMANTIC_LABELS}
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(SAMPLE_CSV, encoding="utf-8-sig")
    sample["title"] = sample["title"].fillna("").astype(str)
    rule_df = _extract_rule_semantic_features(sample)
    rule_df = rule_df.rename(columns={c: f"{c}_rule" for c in rule_df.columns if c != "house_id"})

    # 1. 解析 6 组
    parsed: dict[str, pd.DataFrame] = {}
    for prompt, temp in CONFIGS:
        name = f"{prompt}_t{temp}"
        path = OUT_DIR / f"batch_{name}.jsonl"
        df = parse_output_file(path)
        df.to_csv(PARSED_DIR / f"labels_{name}.csv", index=False, encoding="utf-8-sig")
        parsed[name] = df
        ok = int(df["parse_ok"].sum())
        err = int((df["api_error"] != "").sum())
        print(f"[PARSE] {name}: {len(df)} 行, parse_ok={ok}, api_error={err}")

    # 2. 汇总指标
    summary_rows = []
    rates = {}
    for name, df in parsed.items():
        valid = df[df["parse_ok"]]
        rate = valid[SEMANTIC_LABELS].mean()
        rates[name] = rate
        merged = valid.merge(rule_df, on="house_id", how="left")
        kappas = []
        for label in SEMANTIC_LABELS:
            rule_col = f"{label}_rule"
            if rule_col in merged.columns:
                kappas.append(cohen_kappa(merged[label].astype(int), merged[rule_col].fillna(0).astype(int)))
        summary_rows.append({
            "config": name,
            "n": len(df),
            "parse_ok_rate": float(df["parse_ok"].mean()),
            "any_label_coverage": float(valid[SEMANTIC_LABELS].any(axis=1).mean()),
            "mean_positive_labels": float(valid[SEMANTIC_LABELS].sum(axis=1).mean()),
            "kappa_vs_rule_mean": float(np.nanmean(kappas)),
            "kappa_vs_rule_median": float(np.nanmedian(kappas)),
        })
    summary = pd.DataFrame(summary_rows)

    # 3. 与 Qwen 桥接在同一样本（500 条桥接子集）上的跨模型对照
    qwen_labels = pd.read_csv(BRIDGE_DIR / "labels_baseline.csv", encoding="utf-8-sig")
    qwen_valid = qwen_labels[qwen_labels["parse_ok"] == True]  # noqa: E712
    bridge_ids = set(qwen_valid["house_id"])
    xmodel_rows = []
    for name, df in parsed.items():
        sub = df[df["house_id"].isin(bridge_ids) & df["parse_ok"]]
        joint = sub.merge(qwen_valid, on="house_id", suffixes=("", "_qwen"))
        kappas = [cohen_kappa(joint[l].astype(int), joint[f"{l}_qwen"].astype(int)) for l in SEMANTIC_LABELS]
        xmodel_rows.append({"config": name, "n_overlap": len(joint),
                            "kappa_vs_qwen_mean": float(np.nanmean(kappas)),
                            "kappa_vs_qwen_median": float(np.nanmedian(kappas))})
    xmodel = pd.DataFrame(xmodel_rows)
    summary = summary.merge(xmodel, on="config", how="left")

    summary.to_csv(PARSED_DIR / "eval_summary.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    # 4. 逐标签阳性率相关矩阵（6 组间稳定性）
    rate_df = pd.DataFrame(rates)
    corr = rate_df.corr()
    corr.to_csv(PARSED_DIR / "label_rate_corr.csv", encoding="utf-8-sig", float_format="%.4f")
    off_diag = [corr.loc[a, b] for a, b in combinations(corr.columns, 2)]
    print(f"[STAT] 6 组阳性率相关 min={min(off_diag):.4f} mean={np.mean(off_diag):.4f}")

    # 5. 最优配置选择：parse_ok 优先，其次 κ_vs_rule，再看覆盖率合理性
    best = summary.sort_values(["parse_ok_rate", "kappa_vs_rule_mean"], ascending=False).iloc[0]
    best_name = str(best["config"])
    print(f"[BEST] {best_name} (parse_ok={best['parse_ok_rate']:.4f}, κ_rule={best['kappa_vs_rule_mean']:.4f})")

    # 6. 生成最优配置的全量批量输入（供 MAE 验证 / 彻底替换 Qwen 数据）
    full = pd.read_csv(FULL_INPUT, encoding="utf-8-sig")
    full["title"] = full["title"].fillna("").astype(str)
    prompt_name, temp_tag = best_name.rsplit("_t", 1)
    temp = float(temp_tag.replace("p", "."))
    from scripts.build_llm_batch_jsonl import (  # noqa: E402
        MODEL_PLACEHOLDER, STRICT_PROMPT, STRICT_USER_TEMPLATE,
        SYSTEM_PROMPT, USER_TEMPLATE, build_fewshot_prompt,
    )
    sys_prompt = {"baseline": SYSTEM_PROMPT, "fewshot": build_fewshot_prompt(), "strict": STRICT_PROMPT}[prompt_name]
    user_tpl = STRICT_USER_TEMPLATE if prompt_name == "strict" else USER_TEMPLATE
    # 平台单文件 ≤5000 行 → 全量 7140 拆 2 个文件
    parts = [full.iloc[:5000], full.iloc[5000:]]
    for i, part in enumerate(parts, start=1):
        out_path = PARSED_DIR / f"best_config_full_input_part{i}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for _, row in part.iterrows():
                line = {
                    "custom_id": str(row["house_id"]),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": MODEL_PLACEHOLDER,
                        "messages": [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_tpl.format(title=row["title"])},
                        ],
                        "temperature": temp,
                        "max_tokens": 500,
                    },
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(f"[OK] {out_path.name} ({len(part)} 行)")

    # 7. 报告
    md = [
        "# DeepSeek-V4-Flash 3×2 批量标注评估报告", "",
        f"- 样本：分层 2,000 条（13 区 × 价格段）；模型：DeepSeek-V4-Flash（SiliconFlow Batch）",
        f"- 最优配置：**{best_name}**", "",
        "| 配置 | 解析率 | 覆盖率 | 平均阳性标签数 | κ vs 规则(mean) | κ vs Qwen(mean) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        md.append(f"| {r['config']} | {r['parse_ok_rate']:.4f} | {r['any_label_coverage']:.4f} "
                  f"| {r['mean_positive_labels']:.2f} | {r['kappa_vs_rule_mean']:.4f} "
                  f"| {r['kappa_vs_qwen_mean']:.4f} |")
    md += ["", f"- 6 组逐标签阳性率相关：min={min(off_diag):.4f}，mean={np.mean(off_diag):.4f}",
           "- κ vs Qwen 为与 Qwen 桥接 500 条重叠子集的同样本跨模型对照", ""]
    (PARSED_DIR / "eval_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[OK] {PARSED_DIR}/eval_report.md")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
