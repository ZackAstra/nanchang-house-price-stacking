from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path


REQUIRED_FIELDS: tuple[str, str, str, str] = ("house_id", "region", "total_price", "area_sqm")
POI_CATEGORIES: tuple[str, str, str, str, str, str, str] = (
    "bus",
    "subway",
    "school",
    "restaurant",
    "shop",
    "hospital",
    "bank",
)


@dataclass(frozen=True)
class HouseFieldCheck:
    house_id: str
    completeness_rate: float
    missing_fields: tuple[str, ...]
    community_id: str | None
    community_linked: bool


@dataclass(frozen=True)
class PoiSampleCheck:
    community_id: str
    category_count: int
    categories: tuple[str, ...]
    pass_ge_three: bool


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as jsonl_file:
        for raw_line in jsonl_file:
            line = raw_line.strip()
            if line == "":
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"JSONL行不是对象, path={path}")
            records.append(record)
    return records


def _as_dict(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"字段不是对象, field={field_name}")
    return value


def _extract_total_price(house: dict[str, object]) -> float | int | None:
    total_price = house.get("total_price")
    if isinstance(total_price, (int, float)):
        return total_price
    total_price_wan = house.get("total_price_wan")
    if isinstance(total_price_wan, (int, float)):
        return total_price_wan
    return None


def _extract_non_empty_community_ids_from_sidecars(data_dir: Path) -> set[str]:
    community_ids: set[str] = set()
    for sidecar_path in data_dir.glob("*_community.jsonl"):
        for record in _read_jsonl(sidecar_path):
            community_id_raw = record.get("community_id")
            if community_id_raw is None:
                continue
            community_id = str(community_id_raw).strip()
            if community_id == "":
                continue
            community_ids.add(community_id)
    return community_ids


def _check_house_fields_and_links(
    house_records: list[dict[str, object]],
    available_community_ids: set[str],
) -> list[HouseFieldCheck]:
    checks: list[HouseFieldCheck] = []
    for record in house_records:
        house = _as_dict(record.get("house"), "house")
        house_id_raw = house.get("house_id")
        house_id = str(house_id_raw) if house_id_raw is not None else "UNKNOWN"
        region = record.get("region")
        total_price = _extract_total_price(house)
        area_sqm = house.get("area_sqm")
        field_map: dict[str, object] = {
            "house_id": house_id_raw,
            "region": region,
            "total_price": total_price,
            "area_sqm": area_sqm,
        }
        missing_fields: list[str] = []
        for field_name in REQUIRED_FIELDS:
            value = field_map[field_name]
            if value is None:
                missing_fields.append(field_name)
                continue
            if isinstance(value, str) and value.strip() == "":
                missing_fields.append(field_name)
        completeness_rate = (len(REQUIRED_FIELDS) - len(missing_fields)) / len(REQUIRED_FIELDS)
        community_id_raw = house.get("community_id")
        community_id = str(community_id_raw) if community_id_raw is not None else None
        community_linked = community_id in available_community_ids if community_id is not None else False
        checks.append(
            HouseFieldCheck(
                house_id=house_id,
                completeness_rate=completeness_rate,
                missing_fields=tuple(missing_fields),
                community_id=community_id,
                community_linked=community_linked,
            )
        )
    return checks


def _extract_poi_categories(record: dict[str, object]) -> tuple[str, ...]:
    poi_raw = record.get("poi")
    if not isinstance(poi_raw, dict):
        return tuple()
    summary_raw = poi_raw.get("summary")
    details_raw = poi_raw.get("details")
    summary = summary_raw if isinstance(summary_raw, dict) else {}
    details = details_raw if isinstance(details_raw, dict) else {}
    categories: list[str] = []
    for category in POI_CATEGORIES:
        summary_count_raw = summary.get(category)
        if isinstance(summary_count_raw, int) and summary_count_raw > 0:
            categories.append(category)
            continue
        detail_list_raw = details.get(category)
        if isinstance(detail_list_raw, list) and len(detail_list_raw) > 0:
            categories.append(category)
    return tuple(categories)


def _sample_poi_checks(community_records: list[dict[str, object]], sample_size: int, seed: int) -> list[PoiSampleCheck]:
    if len(community_records) == 0:
        return []
    random.seed(seed)
    sampled_records: list[dict[str, object]]
    if len(community_records) >= sample_size:
        sampled_records = random.sample(community_records, sample_size)
    else:
        sampled_records = random.choices(community_records, k=sample_size)
    checks: list[PoiSampleCheck] = []
    for sampled_record in sampled_records:
        community_id_raw = sampled_record.get("community_id")
        community_id = str(community_id_raw) if community_id_raw is not None else "UNKNOWN"
        categories = _extract_poi_categories(sampled_record)
        checks.append(
            PoiSampleCheck(
                community_id=community_id,
                category_count=len(categories),
                categories=categories,
                pass_ge_three=len(categories) >= 3,
            )
        )
    return checks


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M2 readiness 校验脚本")
    parser.add_argument("--house-file", required=True, help="房源主表JSONL路径")
    parser.add_argument("--community-file", required=True, help="小区侧表JSONL路径")
    parser.add_argument("--sample-size", required=False, type=int, default=5, help="POI随机抽样条数")
    parser.add_argument("--seed", required=False, type=int, default=20260326, help="随机种子")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    house_file = Path(args.house_file)
    community_file = Path(args.community_file)
    if not house_file.exists():
        raise FileNotFoundError(f"房源文件不存在: {house_file}")
    if not community_file.exists():
        raise FileNotFoundError(f"小区文件不存在: {community_file}")

    house_records = _read_jsonl(house_file)
    community_records = _read_jsonl(community_file)
    all_sidecar_community_ids = _extract_non_empty_community_ids_from_sidecars(community_file.parent)

    house_checks = _check_house_fields_and_links(house_records, all_sidecar_community_ids)
    if len(house_checks) == 0:
        raise ValueError(f"房源文件无有效记录: {house_file}")
    field_rate_avg = sum(check.completeness_rate for check in house_checks) / len(house_checks)
    field_rate_all_pass = all(check.completeness_rate == 1.0 for check in house_checks)
    community_link_all_pass = all(check.community_linked for check in house_checks)

    poi_checks = _sample_poi_checks(community_records, args.sample_size, args.seed)
    if len(poi_checks) == 0:
        raise ValueError(f"小区文件无有效记录: {community_file}")
    poi_all_pass = all(check.pass_ge_three for check in poi_checks)

    print(f"房源记录数: {len(house_checks)}")
    print(f"字段完整率(平均): {_format_percent(field_rate_avg)}")
    for check in house_checks:
        print(
            "  - house_id={house_id}, 完整率={rate}, 缺失={missing}, community_id={community_id}, linked={linked}".format(
                house_id=check.house_id,
                rate=_format_percent(check.completeness_rate),
                missing=list(check.missing_fields),
                community_id=check.community_id,
                linked=check.community_linked,
            )
        )
    print(f"community_id 关联校验(在 *_community.jsonl 中命中): {community_link_all_pass}")

    print(f"POI 抽样条数: {len(poi_checks)}")
    for index, check in enumerate(poi_checks, start=1):
        print(
            "  - 样本{idx}: community_id={community_id}, 类别数={count}, 类别={categories}, pass={passed}".format(
                idx=index,
                community_id=check.community_id,
                count=check.category_count,
                categories=list(check.categories),
                passed=check.pass_ge_three,
            )
        )

    final_pass = field_rate_all_pass and community_link_all_pass and poi_all_pass
    if final_pass:
        print("M1 验收合格，建议进入 M2 阶段")
        print("M2 阶段任务清单:")
        print("1. 14区域每区域10页小规模全量采集")
        print("2. 每批次执行字段完整率与POI覆盖率自动校验")
        print("3. 输出并审阅 validation_errors.jsonl 异常样本")
        print("4. 统计四级配套非空率，目标 >= 90%")
        print("5. 记录性能基线（页面耗时、失败率、重试率）")
    else:
        print("M1 验收未通过，暂不建议进入 M2 阶段")
        print(
            "失败项: 字段全通过={field_pass}, 关联全通过={link_pass}, POI抽样全通过={poi_pass}".format(
                field_pass=field_rate_all_pass,
                link_pass=community_link_all_pass,
                poi_pass=poi_all_pass,
            )
        )


if __name__ == "__main__":
    main()
