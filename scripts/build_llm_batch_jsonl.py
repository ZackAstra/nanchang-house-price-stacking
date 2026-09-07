# -*- coding: utf-8 -*-
"""生成 SiliconFlow 批量推理（Batch）输入 JSONL —— LLM 3×2 对照标注实验（方案甲·分层 2,000 条）

依据官方文档：https://api-docs.siliconflow.cn/docs/userguide/guides/batch
- 每行必须含 custom_id（文件内唯一）+ body.messages（以 user 消息结尾）；
- 单文件行数 ≤ 5000（本实验 2000 行/文件，合规）；
- 模型在平台上手动选择（DeepSeek-V4-Flash），或创建任务时 extra_body={"replace":{"model": ...}} 覆盖；
  文件内每行 model 字段仅为占位（平台覆盖为准）。

6 组配置 = 3 prompt（baseline / fewshot / strict）× 2 温度（0.1 / 0.3），
共用同一份分层抽样 2,000 条输入（保证组间可比）。

输入：data/llm/annotation_input.csv —— 全量标注输入（house_id/title/region_slug/total_price_wan/area_sqm），
     由完整数据集生成（本仓库未随附，见 README 数据获取）。

产物（data/llm/batch_input/）：
    annotation_sample_2000.csv            分层抽样输入（含区域/价格供事后分析）
    batch_{prompt}_t{temp}.jsonl          6 个批量输入文件
    README_UPLOAD.md                      平台操作步骤（上传→建任务→下载）

3 个 prompt 全文（baseline/fewshot/strict）写入仓库公开层 llm_prompts/。

用法：uv run python scripts/build_llm_batch_jsonl.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.research.llm_semantic_extractor import SEMANTIC_LABELS, SYSTEM_PROMPT  # noqa: E402

INPUT_CSV = PROJECT_ROOT / "data" / "llm" / "annotation_input.csv"
OUT_DIR = PROJECT_ROOT / "data" / "llm" / "batch_input"
PROMPT_DIR = PROJECT_ROOT / "llm_prompts"

SAMPLE_N = 2000
SEED = 42
MODEL_PLACEHOLDER = "deepseek-ai/DeepSeek-V4-Flash"  # 平台手动选择/覆盖为准
MAX_TOKENS = 500
TEMPERATURES = [0.1, 0.3]

USER_TEMPLATE = 'Property title: "{title}"\n\nExtract labels as JSON only.'


def _full_label_json(positive: list[str]) -> str:
    """按 SEMANTIC_LABELS 全量键生成紧凑 JSON 文本（few-shot 示例用）。"""
    return json.dumps({k: (1 if k in positive else 0) for k in SEMANTIC_LABELS}, ensure_ascii=False)


def build_fewshot_prompt() -> str:
    """A 版：few-shot 示例增强（在原 system prompt 后追加 2 个判定示例，提升召回）。"""
    ex1_title = "满五唯一 地铁口 学区房 精装修 南北通透 大三房 急售"
    ex1_json = _full_label_json(
        ["subway", "school_district", "luxury_decor", "good_lighting", "urgent_sale", "elevator"]
    )
    ex2_title = "江景房 高层视野开阔 带车位 满两年 业主直售 价格可谈"
    ex2_json = _full_label_json(
        ["river_view", "high_floor_view", "good_floor", "garage", "direct_seller", "price_negotiable"]
    )
    return (
        SYSTEM_PROMPT
        + "\n\nExamples:\n"
        + f'Property title: "{ex1_title}"\n{ex1_json}\n\n'
        + f'Property title: "{ex2_title}"\n{ex2_json}'
    )


# B 版：结构化严格版（中文指令、逐标签判定规则、不确定一律 0、仅输出 JSON）
STRICT_PROMPT = (
    "你是房产信息抽取助手。请从给定的【中文房源标题】中逐条判定下列 28 个语义标签。\n"
    "严格规则：\n"
    "1. 仅输出一个 JSON 对象，禁止任何解释、注释或多余文本；\n"
    "2. 每个标签取 0 或 1：标题明确提及或为同义表述（如「近地铁」=subway、「满五唯一」不影响标签）取 1；\n"
    "3. 未提及或无法确定的一律取 0，不得猜测；\n"
    "4. 28 个键必须全部出现，不多不少。\n\n"
    "标签定义：\n"
    + "\n".join(
        [
            "- subway: 地铁/轨道交通/换乘",
            "- school_district: 学区房/学位/名校/重点学校",
            "- school_nearby: 临近学校/幼儿园/小学/中学（非学区表述）",
            "- river_view: 江景/河景/湖景/水岸",
            "- park_nearby: 公园/绿地/生态",
            "- high_floor_view: 视野/无遮挡/景观好",
            "- luxury_decor: 豪装/精装/豪华装修/全新装修",
            "- simple_decor: 简装/普通装修",
            "- rough: 毛坯",
            "- urgent_sale: 急售/速卖/特价/降价急出",
            "- below_market: 低于市场价/捡漏",
            "- sincere: 诚心出售/诚意卖",
            "- quality_community: 品质小区/高端社区",
            "- new_community: 新小区/次新房",
            "- old_community: 老小区/旧社区",
            "- low_density: 低密度/容积率低/楼间距大",
            "- good_lighting: 采光好/朝南/南北通透",
            "- elevator: 电梯房/有电梯",
            "- shopping: 商场/商圈/商业配套/购物方便",
            "- direct_seller: 业主直售/房东自卖",
            "- price_negotiable: 价格可谈/可议价",
            "- garage: 车位/车库",
            "- good_floor: 楼层好/黄金楼层/中高层",
            "- duplex: 复式/跃层/LOFT",
            "- garden: 花园/露台/阳台",
            "- furnished: 拎包入住/家电齐全",
            "- double_bath: 双卫/两卫",
            "- board_building: 板楼",
        ]
    )
    + '\n\n输出格式：{"subway":0,"school_district":1,...（共 28 键）}'
)
STRICT_USER_TEMPLATE = '房源标题："{title}"\n\n请仅输出 JSON。'

PROMPTS = {
    "baseline": (SYSTEM_PROMPT, USER_TEMPLATE),
    "fewshot": (None, USER_TEMPLATE),  # 运行时由 build_fewshot_prompt() 生成
    "strict": (STRICT_PROMPT, STRICT_USER_TEMPLATE),
}


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """按 区域 × 价格段 分层等比抽样，保证组间覆盖。"""
    price = pd.to_numeric(df["total_price_wan"], errors="coerce")
    seg = pd.cut(price, bins=[0, 60, 100, 150, float("inf")], labels=["<60", "60-100", "100-150", ">150"], right=False)
    work = df.assign(_seg=seg.astype(str))
    parts = []
    for _, g in work.groupby(["region_slug", "_seg"], observed=True):
        k = max(1, round(len(g) * n / len(work)))
        parts.append(g.sample(min(k, len(g)), random_state=seed))
    sampled = pd.concat(parts).reset_index(drop=True)
    # 调整至精确 n（多退少补）
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed).reset_index(drop=True)
    elif len(sampled) < n:
        rest = work.drop(index=sampled.index, errors="ignore")
        # index 已重置，改用 house_id 差集
        rest = work[~work["house_id"].isin(sampled["house_id"])]
        sampled = pd.concat([sampled, rest.sample(n=n - len(sampled), random_state=seed)]).reset_index(drop=True)
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # 打乱


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    df["title"] = df["title"].fillna("").astype(str)
    sample = stratified_sample(df, SAMPLE_N, SEED)
    sample_path = OUT_DIR / "annotation_sample_2000.csv"
    sample.to_csv(sample_path, index=False, encoding="utf-8-sig")
    print(f"[OK] {sample_path} ({len(sample)} 行, 区域 {sample['region_slug'].nunique()} 个)")

    prompts = dict(PROMPTS)
    prompts["fewshot"] = (build_fewshot_prompt(), USER_TEMPLATE)

    # 留存 prompt 全文（PUBLIC 层）
    for name, (sys_prompt, _) in prompts.items():
        (PROMPT_DIR / f"prompt_{name}.md").write_text(
            f"# Prompt: {name}\n\n## system\n\n{sys_prompt}\n\n## user template\n\n{prompts[name][1]}\n",
            encoding="utf-8",
        )

    for name, (sys_prompt, user_tpl) in prompts.items():
        for temp in TEMPERATURES:
            tag = f"{name}_t{str(temp).replace('.', 'p')}"
            out_path = OUT_DIR / f"batch_{tag}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for _, row in sample.iterrows():
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
                            "max_tokens": MAX_TOKENS,
                        },
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
            size_kb = out_path.stat().st_size // 1024
            print(f"[OK] {out_path.name} (2000 行, {size_kb} KB)")

    readme = OUT_DIR / "README_UPLOAD.md"
    readme.write_text(
        "# SiliconFlow 批量推理上传步骤（LLM 3×2 对照标注）\n\n"
        "官方文档：https://api-docs.siliconflow.cn/docs/userguide/guides/batch\n\n"
        "## 文件清单\n\n"
        "- 6 个输入文件：`batch_{baseline,fewshot,strict}_t{0p1,0p3}.jsonl`（各 2000 行，custom_id = house_id）\n"
        "- 共用抽样输入：`annotation_sample_2000.csv`\n\n"
        "## 平台操作（每个文件一次）\n\n"
        "1. 上传输入文件（purpose=batch），记录返回的 file id；\n"
        "2. 创建批量任务：endpoint=`/v1/chat/completions`，completion_window=`24h`，"
        "**模型手动选择 DeepSeek-V4-Flash**（或以 extra_body `{\"replace\":{\"model\":\"...\"}}` 覆盖——文件内 model 字段仅为占位）；\n"
        "3. 任务完成后（通常 24h 内）下载 output 文件（**结果仅保留 30 天，务必及时转存**），"
        "命名存为 `data/llm/batch_output/output_{同名}.jsonl`；\n"
        "4. 全部 6 组下载后通知执行方进行解析与评估（覆盖率/κ/log 基线 MAE 对照）。\n\n"
        "## 注意\n\n"
        "- 单文件 2000 行 < 5000 行上限，文件内每行 model 一致（占位值），无需拆分；\n"
        "- 批量价约为实时价 50%，且不占用在线速率限额；\n"
        "- 失败请求会进入 error 文件，解析时需按 custom_id 核对完整性（2000/组）。\n",
        encoding="utf-8",
    )
    print(f"[OK] {readme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
