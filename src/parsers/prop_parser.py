from __future__ import annotations

import re
from html import unescape
from urllib.parse import urljoin

from src.models.house import HouseRecord


def _extract_first_float(pattern: re.Pattern[str], text: str) -> float | None:
    matched = pattern.search(text)
    if matched is None:
        return None
    return float(matched.group(1))


def _compact_text(html: str) -> str:
    plain = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
    plain = re.sub(r"<style[\s\S]*?</style>", " ", plain, flags=re.IGNORECASE)
    plain = re.sub(r"<[^>]+>", " ", plain)
    plain = unescape(plain)
    return re.sub(r"\s+", " ", plain)


def _extract_title(html: str) -> str | None:
    title_match = re.search(r"<title>\s*(.*?)\s*</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if title_match is None:
        return None
    normalized_title = re.sub(r"\s+", " ", title_match.group(1)).strip()
    if normalized_title == "":
        return None
    return normalized_title.split("_")[0].strip()


def _extract_layout(html: str) -> tuple[int | None, int | None, int | None]:
    layout_match = re.search(r"(\d+)\s*室\s*(\d+)\s*厅\s*(\d+)\s*卫", html)
    if layout_match is None:
        return None, None, None
    return int(layout_match.group(1)), int(layout_match.group(2)), int(layout_match.group(3))


def _extract_maininfo_item_text(html: str, item_no: int) -> tuple[str | None, str | None]:
    pattern = re.compile(
        rf'maininfo-model-item-{item_no}"?[^>]*>[\s\S]{{0,1800}}?'
        rf'maininfo-model-strong[^>]*>([\s\S]*?)</div>\s*'
        rf'<div[^>]*class="[^"]*maininfo-model-weak[^"]*"[^>]*>([\s\S]*?)</div>',
        flags=re.IGNORECASE,
    )
    matched = pattern.search(html)
    if matched is None:
        return None, None
    strong_text = _compact_text(matched.group(1)).strip()
    weak_text = _compact_text(matched.group(2)).strip()
    return (strong_text if strong_text != "" else None), (weak_text if weak_text != "" else None)


def _extract_floor_info(text: str) -> tuple[str | None, int | None]:
    # 常见格式：高层(共33层) / 中层（共18层） / 低楼层(共11层)
    floor_match = re.search(
        r"([低中高][楼层]{0,2}|底层|顶层|中层|高层|低层)\s*[（(]\s*共\s*(\d+)\s*层\s*[)）]",
        text,
    )
    if floor_match is not None:
        return floor_match.group(1).strip(), int(floor_match.group(2))
    return None, None


def _extract_house_certificate_years(text: str) -> str | None:
    matched = re.search(r"(满[二三四五六七八九十\d]+(?:年)?)", text)
    if matched is None:
        return None
    value = matched.group(1).strip()
    return value if value != "" else None


def _extract_unique_housing(text: str) -> bool | None:
    if "非唯一" in text or "不唯一" in text:
        return False
    if "唯一住房" in text or "满五唯一" in text or "唯一" in text:
        return True
    return None


def _extract_has_elevator(text: str) -> bool | None:
    if "无电梯" in text or "电梯:无" in text or "电梯：无" in text:
        return False
    if "有电梯" in text or "电梯:有" in text or "电梯：有" in text:
        return True
    return None


def _extract_community_info(prop_url: str, html: str) -> tuple[str | None, str | None, str | None]:
    anchor_match = re.search(
        r"""<a[^>]+href=["']([^"']*/community/view/[^"']*)["'][^>]*>(.*?)</a>""",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if anchor_match is None:
        return None, None, None

    community_url = urljoin(prop_url, unescape(anchor_match.group(1)))
    community_id_match = re.search(r"/community/view/([^/?#]+)", community_url)
    community_id = community_id_match.group(1) if community_id_match is not None else None

    raw_text = re.sub(r"<[^>]+>", "", anchor_match.group(2))
    normalized_text = re.sub(r"\s+", " ", raw_text).strip()
    community_name = normalized_text if normalized_text != "" else None
    return community_url, community_id, community_name


def parse_prop_page(prop_url: str, html: str, region_slug: str) -> HouseRecord:
    house_id_match = re.search(r"/prop/view/([^/?#]+)", prop_url)
    if house_id_match is None:
        raise ValueError(f"无法从URL提取房源ID, prop_url={prop_url}")
    house_id = house_id_match.group(1)
    text = _compact_text(html)
    item1_strong, item1_weak = _extract_maininfo_item_text(html, 1)
    item2_strong, item2_weak = _extract_maininfo_item_text(html, 2)
    item3_strong, _ = _extract_maininfo_item_text(html, 3)

    total_price_wan = _extract_first_float(re.compile(r"(\d+(?:\.\d+)?)\s*万"), html)
    area_sqm = _extract_first_float(re.compile(r"(\d+(?:\.\d+)?)\s*(?:㎡|平米|平方米)"), item2_strong or text)
    room_count, hall_count, bath_count = _extract_layout(item1_strong or html)
    title = _extract_title(html)
    decoration_type = item2_weak
    floor_level, total_floors = _extract_floor_info(item1_weak or text)
    orientation = item3_strong
    house_certificate_years = _extract_house_certificate_years(text)
    is_unique_housing = _extract_unique_housing(text)
    has_elevator = _extract_has_elevator(text)
    community_url, community_id, community_name = _extract_community_info(prop_url, html)

    required_field_map: dict[str, object | None] = {
        "title": title,
        "total_price_wan": total_price_wan,
        "area_sqm": area_sqm,
        "room_count": room_count,
        "hall_count": hall_count,
        "bath_count": bath_count,
        "decoration_type": decoration_type,
        "house_certificate_years": house_certificate_years,
        "is_unique_housing": is_unique_housing,
        "floor_level": floor_level,
        "total_floors": total_floors,
        "orientation": orientation,
        "has_elevator": has_elevator,
        "community_name": community_name,
        "community_id": community_id,
        "community_url": community_url,
    }
    missing_fields = [
        field_name
        for field_name, field_value in required_field_map.items()
        if field_value is None or (isinstance(field_value, str) and field_value.strip() == "")
    ]
    has_required_fields = len(missing_fields) == 0
    error_message = None
    if not has_required_fields:
        error_message = f"必填字段缺失({','.join(missing_fields)})"

    return HouseRecord(
        house_id=house_id,
        title=title,
        region_slug=region_slug,
        prop_url=prop_url,
        decoration_type=decoration_type,
        total_price_wan=total_price_wan,
        area_sqm=area_sqm,
        room_count=room_count,
        hall_count=hall_count,
        bath_count=bath_count,
        house_certificate_years=house_certificate_years,
        is_unique_housing=is_unique_housing,
        floor_level=floor_level,
        total_floors=total_floors,
        orientation=orientation,
        has_elevator=has_elevator,
        community_id=community_id,
        community_name=community_name,
        community_url=community_url,
        is_valid=has_required_fields,
        error_message=error_message,
    )
