from __future__ import annotations

import re

from src.models.community import CommunityRecord


def parse_community_page(
    community_url: str,
    html: str,
    tab_item_count: int,
    poi_summary: dict[str, int],
    poi_details: dict[str, list[dict[str, str | None]]],
) -> CommunityRecord:
    community_id_match = re.search(r"/community/view/([^/?#]+)", community_url)
    if community_id_match is None:
        raise ValueError(f"无法从URL提取小区ID, community_url={community_url}")
    community_id = community_id_match.group(1)

    title_match = re.search(r"<title>\s*(.*?)\s*</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if title_match is None:
        community_name = f"community_{community_id}"
    else:
        community_name = title_match.group(1).strip().split("_")[0].strip()
        if community_name == "":
            community_name = f"community_{community_id}"

    return CommunityRecord(
        community_id=community_id,
        community_name=community_name,
        community_url=community_url,
        tab_item_count=tab_item_count,
        poi_summary=poi_summary,
        poi_details=poi_details,
    )
