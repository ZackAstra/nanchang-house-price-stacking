from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from pydantic import Field


JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]


class ResearchRunConfig(BaseModel):
    houses_path: Path = Field(description="房源JSONL路径")
    communities_path: Path = Field(description="小区JSONL路径")
    output_root: Path = Field(description="研究产物根目录")
    run_id: str = Field(min_length=1, description="研究运行ID")
    random_state: int = Field(ge=0, description="随机种子")
    test_size: float = Field(gt=0, lt=1, description="留出集比例")
    cv_folds: int = Field(ge=2, description="交叉验证折数")
    sample_limit: int | None = Field(default=None, gt=0, description="抽样上限")
    split_mode: str = Field(default="time", description="划分方式：time=时间留出；random=随机留出（稳健性对照）")


class DatasetAudit(BaseModel):
    raw_house_count: int = Field(ge=0, description="原始房源记录数")
    deduplicated_house_count: int = Field(ge=0, description="去重后房源记录数")
    raw_community_count: int = Field(ge=0, description="原始小区记录数")
    deduplicated_community_count: int = Field(ge=0, description="去重后小区记录数")
    usable_house_count: int = Field(ge=0, description="清洗后可用房源记录数")
    removed_invalid_count: int = Field(ge=0, description="无效记录剔除数")
    removed_outlier_count: int = Field(ge=0, description="异常值剔除数")
    community_join_rate: float = Field(ge=0, le=1, description="小区关联成功率")
    key_field_usable_rate: float = Field(ge=0, le=1, description="关键字段可用率")
    missing_rate: dict[str, float] = Field(description="字段缺失率")
    price_quantile_bounds: dict[str, float] = Field(description="总价分位数边界")
    area_quantile_bounds: dict[str, float] = Field(description="面积分位数边界")


class ModelMetric(BaseModel):
    model_name: str = Field(min_length=1, description="模型名称")
    mae: float = Field(ge=0, description="平均绝对误差")
    rmse: float = Field(ge=0, description="均方根误差")
    r2: float = Field(description="R2")
    mape: float = Field(ge=0, description="平均绝对百分比误差")
    cv_mae_mean: float = Field(ge=0, description="交叉验证MAE均值")
    cv_mae_std: float = Field(ge=0, description="交叉验证MAE标准差")


class ResearchResult(BaseModel):
    run_id: str = Field(min_length=1, description="研究运行ID")
    output_dir: Path = Field(description="研究产物目录")
    audit: DatasetAudit = Field(description="数据审计结果")
    metrics: list[ModelMetric] = Field(description="模型指标")
    best_model_name: str = Field(min_length=1, description="最优模型名称")

