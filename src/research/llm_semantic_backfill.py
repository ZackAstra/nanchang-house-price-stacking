"""
LLM 语义特征补充提取器

功能：对缺失的 house_id 单独调用 LLM API 补充提取，避免全量重写。

用法：
    python -m src.research.llm_semantic_backfill \
        --missing-ids data/llm_missing_house_ids.json \
        --houses data/full/nanchang_houses.jsonl \
        --output data/llm_semantic_features_full.csv \
        --max-workers 10
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from src.research.llm_semantic_extractor import (
    API_URL,
    SEMANTIC_LABELS,
    MODEL,
    SYSTEM_PROMPT,
    extract_llm_semantic_features,
)


def _call_llm(title: str, api_key: str, max_retries: int = 3) -> dict[str, int]:
    """调用 LLM API 提取语义标签，带重试。"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'Property title: "{title}"\n\nExtract labels as JSON only.'},
        ],
        "temperature": 0.1,
        "max_tokens": 500,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            result = json.loads(content)
            return {label: int(result.get(label, 0)) for label in SEMANTIC_LABELS}
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1 + attempt)
            else:
                return {label: 0 for label in LLM_SEMANTIC_LABELS}

    return {label: 0 for label in LLM_SEMANTIC_LABELS}


def backfill_missing_records(
    missing_ids_path: Path,
    houses_path: Path,
    output_path: Path,
    api_key: str,
    max_workers: int = 10,
) -> Path:
    """补充提取缺失记录的 LLM 语义特征。"""

    # 1. 读取缺失的 house_id
    with missing_ids_path.open("r", encoding="utf-8") as f:
        missing_ids: set[str] = set(json.load(f))
    print(f"Missing records to backfill: {len(missing_ids)}")

    # 2. 读取房源数据，构建 house_id -> title 映射
    title_map: dict[str, str] = {}
    with houses_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                hid = obj.get("house_id")
                if hid in missing_ids:
                    title_map[hid] = obj.get("title", "")
            except json.JSONDecodeError:
                continue

    pending = [{"house_id": hid, "title": title_map.get(hid, "")} for hid in missing_ids]
    print(f"Found titles for {len(pending)} missing records")

    # 3. 补充提取
    results: list[dict] = []
    errors = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_record = {
            executor.submit(_call_llm, r["title"], api_key): r
            for r in pending
        }

        for future in as_completed(future_to_record):
            record = future_to_record[future]
            try:
                labels = future.result()
                row = {"house_id": record["house_id"], **labels}
                results.append(row)
            except Exception:
                errors += 1
                row = {"house_id": record["house_id"], **{label: 0 for label in SEMANTIC_LABELS}}
                results.append(row)

            if len(results) % 10 == 0:
                print(f"  Progress: {len(results)}/{len(pending)} (errors: {errors})")

    # 4. 追加到现有 CSV
    df_new = pd.DataFrame(results)
    if output_path.exists():
        df_existing = pd.read_csv(output_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        # 去重：同一 house_id 保留最后一次出现
        df_combined = df_combined.drop_duplicates(subset=["house_id"], keep="last")
        df_combined.to_csv(output_path, index=False)
    else:
        df_new.to_csv(output_path, index=False)

    print(f"Done. Backfilled {len(results)} records, {errors} errors.")
    print(f"Total unique house_ids in CSV: {df_combined['house_id'].nunique()}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM 语义特征补充提取")
    parser.add_argument("--missing-ids", required=True, help="缺失 house_id JSON 路径")
    parser.add_argument("--houses", required=True, help="房源 JSONL 路径")
    parser.add_argument("--output", required=True, help="输出 CSV 路径")
    parser.add_argument("--max-workers", type=int, default=10, help="并发数")
    args = parser.parse_args()

    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        print("Error: SILICONFLOW_API_KEY environment variable not set")
        return

    backfill_missing_records(
        missing_ids_path=Path(args.missing_ids),
        houses_path=Path(args.houses),
        output_path=Path(args.output),
        api_key=api_key,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
