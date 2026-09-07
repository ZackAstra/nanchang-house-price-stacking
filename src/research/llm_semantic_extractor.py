"""
LLM 语义特征提取器

调用 SiliconFlow API (Qwen3-32B) 从房源标题中提取结构化语义标签。
支持并发调用、断点续传、错误重试。

用法：
    python -m src.research.llm_semantic_extractor \
        --houses data/full/nanchang_houses.jsonl \
        --output data/llm_semantic_features.csv \
        --max-workers 10

环境变量：
    SILICONFLOW_API_KEY - API 密钥
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL = "Qwen/Qwen3-32B"

SEMANTIC_LABELS = [
    "subway",
    "school_district",
    "school_nearby",
    "river_view",
    "park_nearby",
    "high_floor_view",
    "luxury_decor",
    "simple_decor",
    "rough",
    "urgent_sale",
    "below_market",
    "sincere",
    "quality_community",
    "new_community",
    "old_community",
    "low_density",
    "good_lighting",
    "elevator",
    "shopping",
    "direct_seller",
    "price_negotiable",
    "garage",
    "good_floor",
    "duplex",
    "garden",
    "furnished",
    "double_bath",
    "board_building",
]

SYSTEM_PROMPT = (
    "You are a real estate information extraction assistant. "
    "Extract semantic labels from the given property title. "
    "Return ONLY a JSON object with 0/1 values, no explanation.\n\n"
    "Fields (0=not mentioned, 1=mentioned):\n"
    + "\n".join(
        [
            "- subway: metro/subway/transit",
            "- school_district: school district/zone/elite school",
            "- school_nearby: school/kindergarten/primary/middle",
            "- river_view: river/lake/view/waterside",
            "- park_nearby: park/green/ecology",
            "- high_floor_view: view/unblocked/scenery",
            "- luxury_decor: luxury/fine/decoration/renovated",
            "- simple_decor: simple/basic decoration",
            "- rough: bare/unfinished",
            "- urgent_sale: urgent/quick sale/special price",
            "- below_market: below market/bargain",
            "- sincere: sincere/genuine seller",
            "- quality_community: quality/high-end/luxury complex",
            "- new_community: new/nearly-new complex",
            "- old_community: old/aging complex",
            "- low_density: low density/spacious",
            "- good_lighting: good lighting/sunny/south-facing",
            "- elevator: elevator/lift",
            "- shopping: shopping mall/commercial/convenient",
            "- direct_seller: owner direct sale",
            "- price_negotiable: negotiable price",
            "- garage: garage/parking space",
            "- good_floor: good floor level/high floor",
            "- duplex: duplex/loft/multi-level",
            "- garden: garden/terrace/balcony",
            "- furnished: furnished/move-in ready",
            "- double_bath: double bathroom",
            "- board_building: slab building",
        ]
    )
    + '\n\nReturn format: {"subway":0,"school_district":1,...}'
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
            # 尝试解析 JSON
            result = json.loads(content)
            # 确保所有标签都存在
            return {label: int(result.get(label, 0)) for label in SEMANTIC_LABELS}
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1 + attempt)
            else:
                # 全部重试失败，返回全 0
                return {label: 0 for label in SEMANTIC_LABELS}

    return {label: 0 for label in SEMANTIC_LABELS}


def extract_llm_semantic_features(
    houses_path: Path,
    output_path: Path,
    api_key: str,
    max_workers: int = 10,
    sample_limit: int | None = None,
) -> Path:
    """提取 LLM 语义特征并保存。"""
    # 1. 读取房源数据
    records: list[dict] = []
    with houses_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append({
                    "house_id": obj.get("house_id"),
                    "title": obj.get("title", ""),
                })
            except json.JSONDecodeError:
                continue

    if sample_limit and len(records) > sample_limit:
        import numpy as np
        rng = np.random.RandomState(42)
        indices = rng.choice(len(records), size=sample_limit, replace=False)
        records = [records[i] for i in indices]

    print(f"Total titles to process: {len(records)}")

    # 2. 检查断点续传
    completed_ids: set[str] = set()
    if output_path.exists():
        import csv
        with output_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed_ids.add(row["house_id"])
        print(f"Resuming: {len(completed_ids)} already processed")

    pending_records = [r for r in records if r["house_id"] not in completed_ids]
    print(f"Pending: {len(pending_records)}")

    if not pending_records:
        print("All records already processed.")
        return output_path

    # 3. 并发调用 API
    results: list[dict] = []
    errors = 0

    mode = "a" if completed_ids else "w"
    with output_path.open(mode, encoding="utf-8", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=["house_id"] + SEMANTIC_LABELS)
        if not completed_ids:
            writer.writeheader()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_record = {
                executor.submit(_call_llm, r["title"], api_key): r
                for r in pending_records
            }

            for future in as_completed(future_to_record):
                record = future_to_record[future]
                try:
                    labels = future.result()
                    row = {"house_id": record["house_id"], **labels}
                    writer.writerow(row)
                    f.flush()
                    results.append(row)
                except Exception:
                    errors += 1
                    row = {"house_id": record["house_id"], **{label: 0 for label in SEMANTIC_LABELS}}
                    writer.writerow(row)
                    f.flush()

                if (len(results) + len(completed_ids)) % 50 == 0:
                    print(f"  Progress: {len(results) + len(completed_ids)}/{len(records)} (errors: {errors})")

    print(f"Done. Processed {len(results)} new records, {errors} errors.")
    print(f"Output: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM 语义特征提取")
    parser.add_argument("--houses", required=True, help="房源 JSONL 路径")
    parser.add_argument("--output", required=True, help="输出 CSV 路径")
    parser.add_argument("--max-workers", type=int, default=10, help="并发数")
    parser.add_argument("--sample-limit", type=int, help="抽样上限")
    args = parser.parse_args()

    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        print("Error: SILICONFLOW_API_KEY environment variable not set")
        return

    extract_llm_semantic_features(
        houses_path=Path(args.houses),
        output_path=Path(args.output),
        api_key=api_key,
        max_workers=args.max_workers,
        sample_limit=args.sample_limit,
    )


if __name__ == "__main__":
    main()
