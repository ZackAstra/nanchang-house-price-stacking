from __future__ import annotations

import re
from html import unescape


KEYWORD_MAP: dict[str, str] = {
    "bank": "银行",
    "bus": "公交",
    "subway": "地铁",
    "school": "学校",
    "restaurant": "餐饮",
    "shop": "购物",
    "hospital": "医院",
}

NOISE_NAME_SET: set[str] = {
    "银行",
    "公交",
    "地铁",
    "学校",
    "餐饮",
    "购物",
    "医院",
    "地铁找房",
    "青山湖二手房",
    "生活配套",
    "轨道交通",
}


def _clean_html_text(raw_text: str) -> str:
    no_tags = re.sub(r"<[^>]+>", " ", raw_text)
    normalized = re.sub(r"\s+", " ", unescape(no_tags)).strip()
    return normalized


def _infer_type(item_html: str, item_text: str) -> str:
    source_text = f"{item_html} {item_text}"
    for poi_type, keyword in KEYWORD_MAP.items():
        if keyword in source_text:
            return poi_type
    return "unknown"


def _extract_name(item_html: str, item_text: str) -> str | None:
    title_match = re.search(r"""title=["']([^"']+)["']""", item_html, flags=re.IGNORECASE)
    if title_match is not None:
        return unescape(title_match.group(1)).strip()
    anchor_match = re.search(r"<a[^>]*>(.*?)</a>", item_html, flags=re.IGNORECASE | re.DOTALL)
    if anchor_match is not None:
        anchor_text = _clean_html_text(anchor_match.group(1))
        if anchor_text != "":
            return anchor_text
    text_name_match = re.search(r"([\u4e00-\u9fa5A-Za-z0-9（）()·\-]{2,40})", item_text)
    if text_name_match is not None:
        return text_name_match.group(1)
    return None


def _extract_distance(item_html: str, item_text: str) -> str | None:
    source_text = f"{item_html} {item_text}"
    distance_match = re.search(r"(\d+(?:\.\d+)?\s*(?:米|公里))", source_text)
    if distance_match is None:
        return None
    return distance_match.group(1).replace(" ", "")


def _extract_coordinate(item_html: str) -> str | None:
    coordinate_match = re.search(r"""data-address=["']([^"']+)["']""", item_html, flags=re.IGNORECASE)
    if coordinate_match is None:
        return None
    coordinate_text = unescape(coordinate_match.group(1)).strip()
    if "," in coordinate_text:
        return coordinate_text
    return None


def _extract_sub_type(item_text: str) -> str | None:
    subtype_match = re.search(r"(ATM|幼儿园|小学|中学|三甲|社区医院|卫生站|商场|超市)", item_text)
    if subtype_match is None:
        return None
    return subtype_match.group(1)


def _is_noise_name(name: str) -> bool:
    return name in NOISE_NAME_SET


def _dedupe_details(details: dict[str, list[dict[str, str | None]]]) -> dict[str, list[dict[str, str | None]]]:
    deduped: dict[str, list[dict[str, str | None]]] = {poi_type: [] for poi_type in KEYWORD_MAP}
    for poi_type, poi_items in details.items():
        seen: set[tuple[str, str | None]] = set()
        for poi_item in poi_items:
            name = poi_item["name"]
            distance = poi_item["distance"]
            if name is None or _is_noise_name(name):
                continue
            dedupe_key = (name, distance)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            deduped[poi_type].append(poi_item)
    return deduped


def extract_poi_details(html: str) -> dict[str, list[dict[str, str | None]]]:
    details: dict[str, list[dict[str, str | None]]] = {poi_type: [] for poi_type in KEYWORD_MAP}
    list_items = re.findall(r"<li[^>]*>(.*?)</li>", html, flags=re.IGNORECASE | re.DOTALL)
    for item_html in list_items:
        item_text = _clean_html_text(item_html)
        if item_text == "":
            continue
        poi_type = _infer_type(item_html, item_text)
        if poi_type not in details:
            continue
        name = _extract_name(item_html, item_text)
        if name is None:
            continue
        details[poi_type].append(
            {
                "type": poi_type,
                "name": name,
                "distance": _extract_distance(item_html, item_text),
                "coordinate": _extract_coordinate(item_html),
                "address": None,
                "sub_type": _extract_sub_type(item_text),
            }
        )
    return _dedupe_details(details)


def extract_poi_summary(html: str) -> dict[str, int]:
    details = extract_poi_details(html)
    summary: dict[str, int] = {}
    for poi_type, keyword in KEYWORD_MAP.items():
        detail_count = len(details[poi_type])
        if detail_count > 0:
            summary[poi_type] = detail_count
            continue
        summary[poi_type] = len(re.findall(keyword, html))
    return summary
