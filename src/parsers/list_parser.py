from __future__ import annotations

import re
from html import unescape
from urllib.parse import urljoin

from pydantic import BaseModel
from pydantic import Field
from pydantic import HttpUrl


class ListPageResult(BaseModel):
    region_slug: str = Field(min_length=1, description="区域英文slug")
    page_no: int = Field(ge=1, description="分页页码")
    prop_urls: list[HttpUrl] = Field(description="房源详情链接列表")


def parse_list_page(base_url: str, html: str, region_slug: str, page_no: int) -> ListPageResult:
    url_pattern = re.compile(r"""href=["']([^"']*(?:/prop/view/)[^"']*)["']""")
    seen_urls: set[str] = set()
    ordered_urls: list[str] = []
    for raw_url in url_pattern.findall(html):
        full_url = urljoin(base_url, unescape(raw_url))
        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)
        ordered_urls.append(full_url)
    return ListPageResult(region_slug=region_slug, page_no=page_no, prop_urls=ordered_urls)
