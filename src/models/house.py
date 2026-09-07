from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field
from pydantic import HttpUrl


class HouseRecord(BaseModel):
    house_id: str = Field(min_length=1, description="房源ID")
    title: str | None = Field(default=None, description="房源标题")
    region_slug: str = Field(min_length=1, description="区域英文slug")
    prop_url: HttpUrl = Field(description="房源详情URL")
    decoration_type: str | None = Field(default=None, description="装修类型")
    total_price_wan: float | None = Field(default=None, ge=0, description="总价(万元)")
    area_sqm: float | None = Field(default=None, ge=0, description="面积(平方米)")
    room_count: int | None = Field(default=None, ge=0, description="室")
    hall_count: int | None = Field(default=None, ge=0, description="厅")
    bath_count: int | None = Field(default=None, ge=0, description="卫")
    house_certificate_years: str | None = Field(default=None, description="房本年限")
    is_unique_housing: bool | None = Field(default=None, description="是否唯一住房")
    floor_level: str | None = Field(default=None, description="所在楼层")
    total_floors: int | None = Field(default=None, ge=0, description="总楼层")
    orientation: str | None = Field(default=None, description="朝向")
    has_elevator: bool | None = Field(default=None, description="是否电梯")
    community_id: str | None = Field(default=None, description="小区ID")
    community_name: str | None = Field(default=None, description="小区名称")
    community_url: HttpUrl | None = Field(default=None, description="小区详情URL")
    is_valid: bool = Field(description="是否满足必填字段")
    error_message: str | None = Field(default=None, description="异常说明")
