# -*- coding: utf-8 -*-
"""解析 DeepSeek-V4-Flash 全量批量标注输出（strict_t0p3 最优配置，7,140 条）

输入：data/research/llm_3x3_dataset/batch_output/best_config_full_input_part{1,2}.jsonl
产物：
    data/llm_semantic_features_deepseek_full.csv   runner 兼容格式（house_id + 28 标签；解析失败行记 0 并在报告中计数）
    data/research/llm_3x3_dataset/batch_parsed/full_parse_report.json  完整性/解析率报告

用法：common/Scripts/python.exe scripts/parse_deepseek_full_labels.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.research.llm_semantic_extractor import SEMANTIC_LABELS  # noqa: E402
from scripts.evaluate_llm_3x2_batch import _parse_labels  # noqa: E402

BASE = PROJECT_ROOT / "data" / "full" / "llm_3x3_dataset"  # 完整数据集需联系作者获取（见 README）
OUT_CSV = PROJECT_ROOT / "data" / "llm_semantic_features_deepseek_full.csv"
REPORT_JSON = BASE / "batch_parsed" / "full_parse_report.json"
FULL_INPUT = BASE / "annotation_input.csv"


def main() -> int:
    rows = []
    n_err = n_parse_fail = 0
    for i in (1, 2):
        path = BASE / "batch_output" / f"best_config_full_input_part{i}.jsonl"
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
            if err:
                n_err += 1
            labels = _parse_labels(raw) if raw else None
            if labels is None:
                n_parse_fail += 1
                labels = {k: 0 for k in SEMANTIC_LABELS}
            rows.append({"house_id": hid, **labels})

    df = pd.DataFrame(rows).drop_duplicates(subset=["house_id"])
    expected = pd.read_csv(FULL_INPUT, encoding="utf-8-sig")["house_id"].astype(str)
    got = set(df["house_id"])
    missing = set(expected) - got
    extra = got - set(expected)

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    valid = df[SEMANTIC_LABELS].sum(axis=1) > 0
    report = {
        "total_records": int(len(df)),
        "expected": int(len(expected)),
        "missing_house_ids": len(missing),
        "extra_house_ids": len(extra),
        "api_errors": n_err,
        "parse_failures": n_parse_fail,
        "any_label_coverage": float(valid.mean()),
        "mean_positive_labels": float(df[SEMANTIC_LABELS].sum(axis=1).mean()),
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[OK] {OUT_CSV}")
    return 0 if not missing and not extra else 1


if __name__ == "__main__":
    sys.exit(main())
