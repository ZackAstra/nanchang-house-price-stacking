# -*- coding: utf-8 -*-
"""Qwen3-32B prompt 桥接对照（评审修复 · Comment 5a 缺口闭合）

目的：堵住"3×2 换了模型，同模型 prompt 敏感性未验证"的质疑——
在原提取模型 Qwen/Qwen3-32B 上，用与 DeepSeek 3×2 完全相同的 3 个 prompt
（baseline / fewshot / strict）、温度 0.1、500 条抽样做实时 API 对照。

设计要点：
- 输入：从 data/llm/batch_input/annotation_sample_2000.csv 中再分层抽 500 条（seed 42）；
- 原始响应逐条留存（raw_responses_{prompt}.jsonl）；
- 解析失败不再静默归 0：以 parse_ok 列显式标记，与"真未提及"区分；
- 评估：各 prompt 的覆盖率、与规则标注的 Cohen's κ。

环境要求：SILICONFLOW_API_KEY 环境变量；需已由 build_llm_batch_jsonl.py 生成
annotation_sample_2000.csv（依赖完整数据集）。

用法：
    冒烟：uv run python scripts/run_qwen_prompt_bridge.py --smoke 5
    全量：uv run python scripts/run_qwen_prompt_bridge.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.research.llm_semantic_extractor import SEMANTIC_LABELS  # noqa: E402
from scripts.build_llm_batch_jsonl import (  # noqa: E402
    STRICT_PROMPT,
    STRICT_USER_TEMPLATE,
    USER_TEMPLATE,
    build_fewshot_prompt,
)
from src.research.llm_semantic_extractor import SYSTEM_PROMPT  # noqa: E402

API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL = "Qwen/Qwen3-32B"
TEMPERATURE = 0.1
MAX_TOKENS = 500
BRIDGE_N = 500
SEED = 42

SAMPLE_CSV = PROJECT_ROOT / "data" / "llm" / "batch_input" / "annotation_sample_2000.csv"
OUT_DIR = PROJECT_ROOT / "data" / "llm" / "qwen_bridge"


def _prompts() -> dict[str, tuple[str, str]]:
    return {
        "baseline": (SYSTEM_PROMPT, USER_TEMPLATE),
        "fewshot": (build_fewshot_prompt(), USER_TEMPLATE),
        "strict": (STRICT_PROMPT, STRICT_USER_TEMPLATE),
    }


def _call(title: str, sys_prompt: str, user_tpl: str, api_key: str, max_retries: int = 3) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_tpl.format(title=title)},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_err = ""
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return {"ok": True, "raw": content, "error": ""}
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            if attempt < max_retries - 1:
                time.sleep(1 + attempt)
    return {"ok": False, "raw": "", "error": last_err}


def _parse_labels(raw: str) -> dict[str, int] | None:
    """解析 JSON 标签；失败返回 None（不静默归 0）。"""
    try:
        text = raw.strip()
        # 容忍 ```json 包裹
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        obj = json.loads(text[start : end + 1])
        return {label: int(bool(obj.get(label, 0))) for label in SEMANTIC_LABELS}
    except Exception:  # noqa: BLE001
        return None


def _stratified_subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    price = pd.to_numeric(df["total_price_wan"], errors="coerce")
    seg = pd.cut(price, bins=[0, 60, 100, 150, float("inf")], labels=["<60", "60-100", "100-150", ">150"], right=False)
    work = df.assign(_seg=seg.astype(str))
    parts = []
    for _, g in work.groupby(["region_slug", "_seg"], observed=True):
        k = max(1, round(len(g) * n / len(work)))
        parts.append(g.sample(min(k, len(g)), random_state=seed))
    out = pd.concat(parts).reset_index(drop=True)
    if len(out) > n:
        out = out.sample(n=n, random_state=seed).reset_index(drop=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3-32B prompt 桥接对照")
    parser.add_argument("--smoke", type=int, default=0, help="每 prompt 只跑 N 条（冒烟）")
    args = parser.parse_args()

    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        print("Error: SILICONFLOW_API_KEY 未设置。请先 export SILICONFLOW_API_KEY=...", file=sys.stderr)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SAMPLE_CSV, encoding="utf-8-sig")
    df["title"] = df["title"].fillna("").astype(str)
    sample = _stratified_subsample(df, BRIDGE_N, SEED)
    if args.smoke:
        sample = sample.head(args.smoke)
    n = len(sample)
    print(f"[INFO] 桥接样本 {n} 条（区域 {sample['region_slug'].nunique()} 个），3 prompts × t={TEMPERATURE}")

    for prompt_name, (sys_prompt, user_tpl) in _prompts().items():
        raw_path = OUT_DIR / f"raw_responses_{prompt_name}.jsonl"
        done_ids = set()
        if raw_path.exists():
            for line in raw_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    done_ids.add(json.loads(line)["house_id"])
        todo = sample[~sample["house_id"].isin(done_ids)]
        print(f"[INFO] {prompt_name}: 待跑 {len(todo)} / {n}")
        started = time.time()
        with raw_path.open("a", encoding="utf-8") as raw_f, ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(_call, r["title"], sys_prompt, user_tpl, api_key): r["house_id"]
                for _, r in todo.iterrows()
            }
            for fut in as_completed(futures):
                hid = futures[fut]
                result = fut.result()
                raw_f.write(
                    json.dumps(
                        {"house_id": hid, "prompt": prompt_name, "ok": result["ok"],
                         "raw": result["raw"], "error": result["error"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"[OK] {prompt_name} 完成，耗时 {time.time()-started:.0f}s → {raw_path.name}")

    # 汇总：解析 → 标签 CSV + 覆盖率 + 与规则标注 κ
    from scripts.semantic_agreement_analysis import cohen_kappa  # noqa: E402
    from src.research.llm_semantic_vs_rule_runner import _extract_rule_semantic_features  # noqa: E402

    rule_df = _extract_rule_semantic_features(sample.rename(columns={"title": "title"}))
    # 与 runner 同口径：规则列统一加 _rule 后缀，避免与 LLM 列名冲突
    rule_df = rule_df.rename(columns={c: f"{c}_rule" for c in rule_df.columns if c != "house_id"})
    summary_rows = []
    for prompt_name in _prompts():
        raw_path = OUT_DIR / f"raw_responses_{prompt_name}.jsonl"
        records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = []
        for rec in records:
            labels = _parse_labels(rec["raw"]) if rec["ok"] else None
            row = {"house_id": rec["house_id"], "parse_ok": labels is not None}
            row.update(labels or {label: pd.NA for label in SEMANTIC_LABELS})
            rows.append(row)
        labels_df = pd.DataFrame(rows)
        labels_df.to_csv(OUT_DIR / f"labels_{prompt_name}.csv", index=False, encoding="utf-8-sig")

        merged = labels_df.merge(rule_df, on="house_id", how="left")
        valid = merged[merged["parse_ok"]]
        coverage = float(valid[SEMANTIC_LABELS].any(axis=1).mean()) if len(valid) else 0.0
        kappas = []
        overlap = [c for c in SEMANTIC_LABELS if f"{c}_rule" in valid.columns or c in valid.columns]
        for label in SEMANTIC_LABELS:
            rule_col = f"{label}_rule" if f"{label}_rule" in valid.columns else label
            if rule_col in valid.columns and label in valid.columns:
                kappas.append(cohen_kappa(valid[label].fillna(0).astype(int), valid[rule_col].fillna(0).astype(int)))
        import numpy as np

        summary_rows.append(
            {
                "prompt": prompt_name,
                "n_total": len(records),
                "parse_ok_rate": float(len(valid) / max(len(records), 1)),
                "any_label_coverage": coverage,
                "kappa_vs_rule_mean": float(np.nanmean(kappas)),
                "kappa_vs_rule_median": float(np.nanmedian(kappas)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "bridge_summary.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    print(summary.to_string(index=False))
    print(f"[OK] {OUT_DIR}/bridge_summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
