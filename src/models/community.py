from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field
from pydantic import HttpUrl


class PoiDetail(BaseModel):
    type: str = Field(min_length=1, description="配套类型")
    name: str = Field(min_length=1, description="名称")
    distance: str | None = Field(default=None, description="距离")
    itemcontent: str | None = Field(default=None, description="最小单元-itemcontent")
    iteminfo: str | None = Field(default=None, description="最小单元-iteminfo")
    coordinate: str | None = Field(default=None, description="坐标")
    address: str | None = Field(default=None, description="地址")
    sub_type: str | None = Field(default=None, description="子类型")


class CommunityRecord(BaseModel):
    community_id: str = Field(min_length=1, description="小区ID")
    community_name: str = Field(min_length=1, description="小区名称")
    community_url: HttpUrl = Field(description="小区详情URL(首次命中URL)")
    tab_item_count: int = Field(ge=0, description="tabBox内项目数量")
    poi_summary: dict[str, int] = Field(description="配套分组统计")
    poi_details: dict[str, list[PoiDetail]] = Field(description="配套明细数据")
