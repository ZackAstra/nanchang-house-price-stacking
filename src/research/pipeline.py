from __future__ import annotations

import json
import re
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import ValidationError
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from scipy import stats

from src.exceptions import ParseError
from src.research.models import DatasetAudit
from src.research.models import JsonObject
from src.research.models import JsonValue
from src.research.models import ModelMetric
from src.research.models import ResearchResult
from src.research.models import ResearchRunConfig


POI_TYPES: tuple[str, ...] = ("bank", "bus", "subway", "school", "restaurant", "shop", "hospital")
BASE_NUMERIC_FEATURES: tuple[str, ...] = (
    "area_sqm",
    "room_count",
    "hall_count",
    "bath_count",
    "total_floors",
    "house_certificate_years",
    "floor_level_score",
    "floor_ratio_proxy",
    "area_per_room",
    "layout_density",
    "poi_bank_count",
    "poi_bus_count",
    "poi_subway_count",
    "poi_school_count",
    "poi_restaurant_count",
    "poi_shop_count",
    "poi_hospital_count",
    "poi_total_count",
    "accessibility_index",
)
STAGE3_NUMERIC_FEATURES: tuple[str, ...] = (
    "poi_balance_score",
    "transit_medical_ratio",
    "education_commerce_ratio",
    "area_layout_efficiency",
    "room_area_interaction",
    "bath_room_ratio",
    "region_accessibility_interaction",
    "floor_area_interaction",
)
NUMERIC_FEATURES: tuple[str, ...] = BASE_NUMERIC_FEATURES + STAGE3_NUMERIC_FEATURES
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "region_slug",
    "decoration_type",
    "orientation",
    "floor_level",
    "has_elevator",
    "is_unique_housing",
    "house_layout",
)
TARGET_COLUMN = "total_price_wan"
OPTIONAL_MODEL_IMPORT_ERRORS: dict[str, str] = {}

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    XGBRegressor = None
    OPTIONAL_MODEL_IMPORT_ERRORS["xgboost"] = str(exc)

try:
    from lightgbm import LGBMRegressor
except ImportError as exc:
    LGBMRegressor = None
    OPTIONAL_MODEL_IMPORT_ERRORS["lightgbm"] = str(exc)

try:
    from catboost import CatBoostRegressor
except ImportError as exc:
    CatBoostRegressor = None
    OPTIONAL_MODEL_IMPORT_ERRORS["catboost"] = str(exc)


def run_price_research(config: ResearchRunConfig) -> ResearchResult:
    output_dir = config.output_root / config.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        _write_run_log(output_dir, "run_started", {"run_id": config.run_id})
        _write_run_log(output_dir, "load_data_started", {})
        houses_df = _load_houses(config.houses_path)
        communities_df = _load_communities(config.communities_path)
        _write_run_log(
            output_dir,
            "load_data_completed",
            {"house_rows": int(len(houses_df)), "community_rows": int(len(communities_df))},
        )

        _write_run_log(output_dir, "clean_join_started", {})
        merged_df, audit = _clean_and_join(houses_df, communities_df)
        _write_run_log(
            output_dir,
            "clean_join_completed",
            {"usable_house_count": audit.usable_house_count, "community_join_rate": audit.community_join_rate},
        )

        _write_run_log(output_dir, "feature_engineering_started", {})
        feature_df = _build_features(merged_df)
        feature_df = _apply_sample_limit(feature_df, config.sample_limit, config.random_state)
        feature_path = output_dir / "cleaned_features.csv"
        feature_df.to_csv(feature_path, index=False, encoding="utf-8-sig")
        _write_feature_generation_log(output_dir)
        _write_run_log(output_dir, "feature_engineering_completed", {"feature_rows": int(len(feature_df))})

        audit_path = output_dir / "data_audit.json"
        audit_path.write_text(
            audit.model_dump_json(indent=2),
            encoding="utf-8",
        )

        _write_run_log(output_dir, "model_training_started", {})
        if getattr(config, "split_mode", "time") == "random":
            train_df, test_df = _split_random(feature_df, config.test_size, config.random_state)
        else:
            train_df, test_df = _split_by_time(feature_df, config.test_size)
        train_df, test_df = _apply_reference_group_medians(
            train_df, test_df, _compute_region_ai_medians(config.communities_path)
        )
        metrics, predictions_df, best_pipeline, predictions_by_model = _train_and_evaluate_models(
            train_df=train_df,
            test_df=test_df,
            config=config,
            output_dir=output_dir,
        )
        _write_run_log(output_dir, "model_training_completed", {"model_count": len(metrics)})

        _write_run_log(output_dir, "artifact_writing_started", {})
        _write_metrics(output_dir, metrics)
        _write_feature_ablation_metrics(output_dir, metrics)
        _write_feature_set_ablation_metrics(output_dir, train_df, test_df, config)
        predictions_df.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
        _write_model_prediction_errors(output_dir, test_df, predictions_by_model)
        _write_significance_tests(output_dir, test_df, predictions_by_model)
        _write_error_stratification(output_dir, test_df, predictions_df)
        _write_temporal_validation_profile(output_dir, feature_df, train_df, test_df)
        _write_feature_importance(output_dir, best_pipeline, train_df, test_df, config.random_state)
        _write_shap_analysis(output_dir, best_pipeline, train_df, test_df, config.random_state)
        _write_plots(output_dir, feature_df, predictions_df)
        _write_report(output_dir, config, audit, metrics)
        _write_run_log(output_dir, "artifact_writing_completed", {})

        best_metric = min(metrics, key=lambda item: item.mae)
        _write_run_log(output_dir, "run_completed", {"best_model_name": best_metric.model_name})
        return ResearchResult(
            run_id=config.run_id,
            output_dir=output_dir,
            audit=audit,
            metrics=metrics,
            best_model_name=best_metric.model_name,
        )
    except Exception as exc:
        _write_error_report(output_dir, config, exc)
        _write_run_log(output_dir, "run_failed", {"error_type": type(exc).__name__, "error_message": str(exc)})
        raise


def _load_houses(path: Path) -> pd.DataFrame:
    records = _read_jsonl(path)
    if len(records) == 0:
        raise ParseError(f"房源数据为空: path={path}")
    return pd.DataFrame(records)


def _load_communities(path: Path) -> pd.DataFrame:
    records = _read_jsonl(path)
    rows: list[dict[str, JsonValue]] = []
    for record in records:
        poi_summary_value = record.get("poi_summary")
        poi_summary = poi_summary_value if isinstance(poi_summary_value, dict) else {}
        row: dict[str, JsonValue] = {
            "community_id": record.get("community_id"),
            "community_name_detail": record.get("community_name"),
            "tab_item_count": record.get("tab_item_count"),
        }
        for poi_type in POI_TYPES:
            count_value = poi_summary.get(poi_type)
            row[f"poi_{poi_type}_count"] = count_value if isinstance(count_value, int | float) else 0
        rows.append(row)
    if len(rows) == 0:
        raise ParseError(f"小区数据为空: path={path}")
    return pd.DataFrame(rows)


def _read_jsonl(path: Path) -> list[JsonObject]:
    if not path.exists():
        raise ParseError(f"JSONL文件不存在: path={path}")

    records: list[JsonObject] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if line == "":
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ParseError(f"JSONL解析失败: path={path}, line_no={line_no}, error={exc}") from exc
            if not isinstance(parsed, dict):
                raise ParseError(f"JSONL行不是对象: path={path}, line_no={line_no}")
            records.append(cast(JsonObject, parsed))
    return records


def _write_run_log(output_dir: Path, event: str, payload: dict[str, JsonValue]) -> None:
    log_record: dict[str, JsonValue] = {
        "event": event,
        "created_at": datetime.now().isoformat(),
        "payload": payload,
    }
    with (output_dir / "run_log.jsonl").open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(log_record, ensure_ascii=False) + "\n")


def _write_error_report(output_dir: Path, config: ResearchRunConfig, exc: Exception) -> None:
    error_report: dict[str, JsonValue] = {
        "run_id": config.run_id,
        "created_at": datetime.now().isoformat(),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
        "houses_path": str(config.houses_path),
        "communities_path": str(config.communities_path),
        "sample_limit": config.sample_limit,
        "cv_folds": config.cv_folds,
        "test_size": config.test_size,
    }
    (output_dir / "error_report.json").write_text(
        json.dumps(error_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _append_model_experiment(output_dir: Path, record: dict[str, JsonValue]) -> None:
    log_record: dict[str, JsonValue] = {
        "created_at": datetime.now().isoformat(),
        **record,
    }
    with (output_dir / "model_experiments.jsonl").open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(log_record, ensure_ascii=False) + "\n")


def _clean_and_join(houses_df: pd.DataFrame, communities_df: pd.DataFrame) -> tuple[pd.DataFrame, DatasetAudit]:
    raw_house_count = int(len(houses_df))
    raw_community_count = int(len(communities_df))

    houses = houses_df.drop_duplicates(subset=["house_id"]).copy()
    communities = communities_df.drop_duplicates(subset=["community_id"]).copy()

    for column_name in ("total_price_wan", "area_sqm", "room_count", "hall_count", "bath_count", "total_floors"):
        if column_name in houses.columns:
            houses[column_name] = pd.to_numeric(houses[column_name], errors="coerce")

    key_columns = ["total_price_wan", "area_sqm", "region_slug", "room_count", "hall_count", "bath_count"]
    key_valid_mask = (
        houses["total_price_wan"].gt(0)
        & houses["area_sqm"].gt(0)
        & houses["region_slug"].notna()
        & houses["room_count"].notna()
        & houses["hall_count"].notna()
        & houses["bath_count"].notna()
    )
    key_field_usable_rate = float(key_valid_mask.mean()) if len(houses) > 0 else 0.0
    valid_houses = houses.loc[key_valid_mask].copy()

    price_low, price_high = _quantile_bounds(valid_houses["total_price_wan"])
    area_low, area_high = _quantile_bounds(valid_houses["area_sqm"])
    outlier_mask = (
        valid_houses["total_price_wan"].between(price_low, price_high)
        & valid_houses["area_sqm"].between(area_low, area_high)
    )
    cleaned_houses = valid_houses.loc[outlier_mask].copy()

    for poi_type in POI_TYPES:
        poi_column = f"poi_{poi_type}_count"
        communities[poi_column] = pd.to_numeric(communities[poi_column], errors="coerce").fillna(0)

    merged = cleaned_houses.merge(communities, on="community_id", how="left", indicator=True)
    community_join_rate = float((merged["_merge"] == "both").mean()) if len(merged) > 0 else 0.0
    merged = merged.drop(columns=["_merge"])

    missing_rate = _build_missing_rate(houses, key_columns)
    audit = DatasetAudit(
        raw_house_count=raw_house_count,
        deduplicated_house_count=int(len(houses)),
        raw_community_count=raw_community_count,
        deduplicated_community_count=int(len(communities)),
        usable_house_count=int(len(merged)),
        removed_invalid_count=int(len(houses) - len(valid_houses)),
        removed_outlier_count=int(len(valid_houses) - len(cleaned_houses)),
        community_join_rate=community_join_rate,
        key_field_usable_rate=key_field_usable_rate,
        missing_rate=missing_rate,
        price_quantile_bounds={"low": float(price_low), "high": float(price_high)},
        area_quantile_bounds={"low": float(area_low), "high": float(area_high)},
    )
    return merged, audit


def _quantile_bounds(series: pd.Series) -> tuple[float, float]:
    low = float(series.quantile(0.01))
    high = float(series.quantile(0.99))
    return low, high


def _build_missing_rate(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    missing_rate: dict[str, float] = {}
    for column_name in columns:
        missing_rate[column_name] = float(df[column_name].isna().mean()) if column_name in df.columns else 1.0
    return missing_rate


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = df.copy()
    for poi_type in POI_TYPES:
        poi_column = f"poi_{poi_type}_count"
        feature_df[poi_column] = pd.to_numeric(feature_df.get(poi_column), errors="coerce").fillna(0)

    feature_df["room_count"] = pd.to_numeric(feature_df["room_count"], errors="coerce")
    feature_df["hall_count"] = pd.to_numeric(feature_df["hall_count"], errors="coerce")
    feature_df["bath_count"] = pd.to_numeric(feature_df["bath_count"], errors="coerce")
    feature_df["area_sqm"] = pd.to_numeric(feature_df["area_sqm"], errors="coerce")
    feature_df["total_floors"] = pd.to_numeric(feature_df["total_floors"], errors="coerce")
    feature_df["extracted_at"] = pd.to_datetime(feature_df.get("extracted_at"), errors="coerce")

    feature_df["house_layout"] = (
        feature_df["room_count"].fillna(0).astype(int).astype(str)
        + "室"
        + feature_df["hall_count"].fillna(0).astype(int).astype(str)
        + "厅"
        + feature_df["bath_count"].fillna(0).astype(int).astype(str)
        + "卫"
    )
    feature_df["area_per_room"] = feature_df["area_sqm"] / feature_df["room_count"].clip(lower=1)
    feature_df["layout_density"] = (
        feature_df["room_count"] + feature_df["hall_count"] + feature_df["bath_count"]
    ) / feature_df["area_sqm"].clip(lower=1)
    feature_df["floor_level_score"] = feature_df["floor_level"].map(_floor_level_score).fillna(0.5)
    feature_df["floor_ratio_proxy"] = feature_df["floor_level_score"]
    feature_df["house_certificate_years"] = feature_df["house_certificate_years"].map(
        _parse_certificate_years
    )
    feature_df["poi_total_count"] = sum(feature_df[f"poi_{poi_type}_count"] for poi_type in POI_TYPES)
    feature_df["accessibility_index"] = _build_accessibility_index(feature_df)
    feature_df = _build_stage3_features(feature_df)

    for column_name in CATEGORICAL_FEATURES:
        feature_df[column_name] = feature_df[column_name].astype("string").fillna("unknown")
    for column_name in NUMERIC_FEATURES:
        feature_df[column_name] = pd.to_numeric(feature_df[column_name], errors="coerce").fillna(0)

    selected_columns = (
        ["house_id", "community_id", "extracted_at", TARGET_COLUMN]
        + list(NUMERIC_FEATURES)
        + list(CATEGORICAL_FEATURES)
    )
    return feature_df[selected_columns].copy()


def _build_stage3_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    enhanced_df = feature_df.copy()
    poi_columns = [f"poi_{poi_type}_count" for poi_type in POI_TYPES]
    poi_matrix = enhanced_df[poi_columns].astype(float)
    poi_total = poi_matrix.sum(axis=1).clip(lower=1)
    poi_share = poi_matrix.div(poi_total, axis=0)
    entropy = -(poi_share * np.log(poi_share.replace(0, np.nan))).sum(axis=1).fillna(0)
    enhanced_df["poi_balance_score"] = entropy / np.log(len(POI_TYPES))
    enhanced_df["transit_medical_ratio"] = (
        enhanced_df["poi_bus_count"] + 2.0 * enhanced_df["poi_subway_count"]
    ) / (enhanced_df["poi_hospital_count"] + 1.0)
    enhanced_df["education_commerce_ratio"] = enhanced_df["poi_school_count"] / (
        enhanced_df["poi_restaurant_count"] + enhanced_df["poi_shop_count"] + 1.0
    )
    layout_median_area = enhanced_df.groupby("house_layout", dropna=False)["area_sqm"].transform("median").clip(lower=1)
    enhanced_df["area_layout_efficiency"] = enhanced_df["area_sqm"] / layout_median_area
    enhanced_df["room_area_interaction"] = enhanced_df["room_count"] * enhanced_df["area_per_room"]
    enhanced_df["bath_room_ratio"] = enhanced_df["bath_count"] / enhanced_df["room_count"].clip(lower=1)
    region_accessibility_median = (
        enhanced_df.groupby("region_slug", dropna=False)["accessibility_index"].transform("median").clip(lower=1e-6)
    )
    enhanced_df["region_accessibility_interaction"] = enhanced_df["accessibility_index"] / region_accessibility_median
    enhanced_df["floor_area_interaction"] = enhanced_df["floor_level_score"] * np.log1p(enhanced_df["area_sqm"])
    return enhanced_df


def _compute_region_ai_medians(communities_path: Path) -> dict[str, float]:
    """从按区域组织的社区采集文件估计各区域 accessibility_index 中位数。

    这是全市社区 POI 底表（外部参考数据，不含任何房源价格标签），
    在部署语义下对所有区域（含训练集中无房源的区域）均合法可得，
    用于 region_accessibility_interaction 的区域基准，避免使用测试集房源估计。

    优先读取随仓库发布的聚合参考文件 data/reference/region_ai_medians.json
    （13 个区域的 AI 中位数，聚合统计、不含样本级数据）；缺失时才回退到
    按区域目录的社区采集文件重新计算（完整数据集场景）。
    """
    reference_path = Path(communities_path).parent.parent / "reference" / "region_ai_medians.json"
    if reference_path.exists():
        return {k: float(v) for k, v in json.loads(reference_path.read_text(encoding="utf-8")).items()}
    data_root = Path(communities_path).parent.parent
    region_values: dict[str, list[float]] = {}
    for community_file in sorted(data_root.glob("*/output/*_communities.jsonl")):
        region = community_file.stem.removesuffix("_communities")
        for record in _read_jsonl(community_file):
            poi_summary = record.get("poi_summary")
            poi_summary = poi_summary if isinstance(poi_summary, dict) else {}
            row = {
                f"poi_{poi_type}_count": (poi_summary.get(poi_type) if isinstance(poi_summary.get(poi_type), int | float) else 0)
                for poi_type in POI_TYPES
            }
            region_values.setdefault(region, []).append(float(_build_accessibility_index(pd.DataFrame([row])).iloc[0]))
    return {region: float(np.median(values)) for region, values in region_values.items() if values}


def _apply_reference_group_medians(
    train_df: pd.DataFrame, test_df: pd.DataFrame, region_ai_medians: dict[str, float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """用无泄漏参考基准重算两个组中位数派生特征（在 _split_by_time 之后调用）。

    - area_layout_efficiency：户型面积中位数由训练集估计（户型是房源级属性，训练集覆盖全部主力户型）；
      未见户型回退训练集全局中位数。
    - region_accessibility_interaction：区域 AI 中位数来自全市社区 POI 底表
      （_compute_region_ai_medians），未见区域回退各区域中位数的中位数。
    原实现（_build_features 内）对含测试集的全量数据计算，存在测试集特征值参与统计的泄漏路径。
    """
    layout_median = train_df.groupby("house_layout", dropna=False)["area_sqm"].median()
    layout_global = float(train_df["area_sqm"].median())
    region_fallback = float(np.median(list(region_ai_medians.values()))) if region_ai_medians else float(
        train_df["accessibility_index"].median()
    )
    for df in (train_df, test_df):
        layout_ref = df["house_layout"].map(layout_median).fillna(layout_global).clip(lower=1)
        df["area_layout_efficiency"] = df["area_sqm"] / layout_ref
        region_ref = df["region_slug"].map(region_ai_medians).fillna(region_fallback).clip(lower=1e-6)
        df["region_accessibility_interaction"] = df["accessibility_index"] / region_ref
    return train_df, test_df


def _apply_sample_limit(df: pd.DataFrame, sample_limit: int | None, random_state: int) -> pd.DataFrame:
    if sample_limit is None or len(df) <= sample_limit:
        return df
    return df.sample(n=sample_limit, random_state=random_state).copy()


def _floor_level_score(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    if "低" in value:
        return 0.2
    if "中" in value:
        return 0.5
    if "高" in value:
        return 0.8
    return None


# 产权年限状态文本 → 持有年数（数值型）。安居客该字段为标准档位徽标：
# 满二/满两年/满2年 → 2.0；满五/满五年 → 5.0；其余档位（如满十）按字面解析。
# 缺失、未满档（平台不展示徽标）或爬取截断噪声（如"满24""满20"，共 7 条）统一归 0.0。
_CERTIFICATE_CN_DIGITS = {
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def _parse_certificate_years(value: object) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    match = re.fullmatch(r"满?([0-9]+|[一二两三四五六七八九十]{1,2})年?", text)
    if not match:
        return 0.0
    token = match.group(1)
    if token.isdigit():
        years = int(token)
    elif token in _CERTIFICATE_CN_DIGITS:
        years = _CERTIFICATE_CN_DIGITS[token]
    elif len(token) == 2 and token[0] == "十" and token[1] in _CERTIFICATE_CN_DIGITS:
        years = 10 + _CERTIFICATE_CN_DIGITS[token[1]]
    else:
        return 0.0
    # 合理持有年限上限 70 年；截断噪声（如 24/20 无"年"字后缀）不具档位语义，归 0
    if years in (2, 5) or (text.endswith("年") and 1 <= years <= 70):
        return float(years)
    return 0.0


def _build_accessibility_index(df: pd.DataFrame) -> pd.Series:
    weights: dict[str, float] = {
        "poi_bank_count": 0.6,
        "poi_bus_count": 1.0,
        "poi_subway_count": 1.5,
        "poi_school_count": 1.4,
        "poi_restaurant_count": 0.7,
        "poi_shop_count": 0.8,
        "poi_hospital_count": 1.2,
    }
    score = pd.Series(np.zeros(len(df)), index=df.index)
    for column_name, weight in weights.items():
        score = score + np.log1p(pd.to_numeric(df[column_name], errors="coerce").fillna(0)) * weight
    return score


def _split_by_time(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_df = df.sort_values(by=["extracted_at", "house_id"], na_position="first").reset_index(drop=True)
    test_count = max(int(len(sorted_df) * test_size), 1)
    if len(sorted_df) - test_count < 2:
        raise ParseError(f"可用样本不足以划分训练和测试集: usable_count={len(sorted_df)}")
    train_df = sorted_df.iloc[:-test_count].copy()
    test_df = sorted_df.iloc[-test_count:].copy()
    return train_df, test_df


def _split_random(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """随机 80/20 划分（稳健性对照用，回应"时间留出实为区域外推"的结构性问题）。"""
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_count = max(int(len(shuffled) * test_size), 1)
    if len(shuffled) - test_count < 2:
        raise ParseError(f"可用样本不足以划分训练和测试集: usable_count={len(shuffled)}")
    train_df = shuffled.iloc[:-test_count].copy()
    test_df = shuffled.iloc[-test_count:].copy()
    return train_df, test_df


def _train_and_evaluate_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ResearchRunConfig,
    output_dir: Path,
) -> tuple[list[ModelMetric], pd.DataFrame, Pipeline, dict[str, np.ndarray]]:
    x_train = train_df[list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)]
    y_train = train_df[TARGET_COLUMN]
    x_test = test_df[list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)]
    y_test = test_df[TARGET_COLUMN]

    models = _build_model_zoo(config.random_state)
    metrics: list[ModelMetric] = []
    predictions_by_model: dict[str, np.ndarray] = {}
    pipelines_by_model: dict[str, Pipeline] = {}
    groups = train_df["community_id"].fillna("unknown").astype(str)

    _append_unavailable_optional_models(output_dir)

    for model_name, regressor in models.items():
        _append_model_experiment(
            output_dir,
            {
                "event": "model_started",
                "model_name": model_name,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "params": _model_params(regressor),
            },
        )
        pipeline = _build_pipeline(regressor)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                started_at = time.perf_counter()
                cv_mae_mean, cv_mae_std, cv_mae_scores = _cross_validate_model(
                    pipeline,
                    x_train,
                    y_train,
                    groups,
                    config.cv_folds,
                )
                pipeline.fit(x_train, y_train)
                train_seconds = time.perf_counter() - started_at
            predict_started_at = time.perf_counter()
            predictions = pipeline.predict(x_test)
            predict_seconds = time.perf_counter() - predict_started_at
            metric = _evaluate_predictions(model_name, y_test, predictions, cv_mae_mean, cv_mae_std)
            metrics.append(metric)
            predictions_by_model[model_name] = predictions
            pipelines_by_model[model_name] = pipeline
            fitted_regressor = pipeline.named_steps["regressor"]
            selected_alpha = getattr(fitted_regressor, "alpha_", None)
            _append_model_experiment(
                output_dir,
                {
                    "event": "model_completed",
                    "model_name": model_name,
                    "cv_mae_scores": cv_mae_scores,
                    "cv_mae_mean": cv_mae_mean,
                    "cv_mae_std": cv_mae_std,
                    "mae": metric.mae,
                    "rmse": metric.rmse,
                    "r2": metric.r2,
                    "mape": metric.mape,
                    "train_seconds": train_seconds,
                    "predict_seconds": predict_seconds,
                    "selected_alpha": float(selected_alpha) if selected_alpha is not None else None,
                },
            )
        except Exception as exc:
            _append_model_experiment(
                output_dir,
                {
                    "event": "model_failed",
                    "model_name": model_name,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            raise

    _append_blend_metric(metrics, predictions_by_model, y_test, output_dir)
    _append_two_stage_residual_metric(
        metrics,
        predictions_by_model,
        pipelines_by_model,
        x_train,
        y_train,
        x_test,
        y_test,
        groups,
        config,
        output_dir,
    )
    best_metric = min(metrics, key=lambda item: item.mae)
    prediction_df = test_df[["house_id", TARGET_COLUMN]].copy()
    prediction_df["predicted_price_wan"] = predictions_by_model[best_metric.model_name]
    prediction_df["residual_wan"] = prediction_df[TARGET_COLUMN] - prediction_df["predicted_price_wan"]
    if best_metric.model_name == "two_stage_residual":
        prediction_df["stage1_predicted_price_wan"] = predictions_by_model["two_stage_stage1"]
        prediction_df["stage2_residual_prediction_wan"] = predictions_by_model["two_stage_stage2"]
    best_pipeline_name = best_metric.model_name if best_metric.model_name in pipelines_by_model else _best_single_model_name(metrics)
    return metrics, prediction_df, pipelines_by_model[best_pipeline_name], predictions_by_model


def _append_blend_metric(
    metrics: list[ModelMetric],
    predictions_by_model: dict[str, np.ndarray],
    y_test: pd.Series,
    output_dir: Path,
) -> None:
    candidate_metrics = [metric for metric in metrics if metric.model_name != "mean_baseline"]
    top_metrics = sorted(candidate_metrics, key=lambda item: item.cv_mae_mean)[:3]
    if len(top_metrics) < 2:
        return
    blend_predictions = np.mean(
        [predictions_by_model[metric.model_name] for metric in top_metrics],
        axis=0,
    )
    blend_cv_mae_mean = float(np.mean([metric.cv_mae_mean for metric in top_metrics]))
    blend_cv_mae_std = float(np.mean([metric.cv_mae_std for metric in top_metrics]))
    blend_metric = _evaluate_predictions(
        "blend_top3",
        y_test,
        blend_predictions,
        blend_cv_mae_mean,
        blend_cv_mae_std,
    )
    metrics.append(blend_metric)
    predictions_by_model["blend_top3"] = blend_predictions
    _append_model_experiment(
        output_dir,
        {
            "event": "model_completed",
            "model_name": "blend_top3",
            "base_models": [metric.model_name for metric in top_metrics],
            "cv_mae_mean": blend_cv_mae_mean,
            "cv_mae_std": blend_cv_mae_std,
            "mae": blend_metric.mae,
            "rmse": blend_metric.rmse,
            "r2": blend_metric.r2,
            "mape": blend_metric.mape,
        },
    )


def _append_unavailable_optional_models(output_dir: Path) -> None:
    for model_name, error_message in OPTIONAL_MODEL_IMPORT_ERRORS.items():
        _append_model_experiment(
            output_dir,
            {
                "event": "model_unavailable",
                "model_name": model_name,
                "error_type": "ImportError",
                "error_message": error_message,
            },
        )


def _append_two_stage_residual_metric(
    metrics: list[ModelMetric],
    predictions_by_model: dict[str, np.ndarray],
    pipelines_by_model: dict[str, Pipeline],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    groups: pd.Series,
    config: ResearchRunConfig,
    output_dir: Path,
) -> None:
    base_model_name = _best_stage1_model_name(metrics)
    base_pipeline = pipelines_by_model[base_model_name]
    residual_regressor = _build_residual_regressor(config.random_state)
    residual_pipeline = _build_pipeline(residual_regressor)

    _append_model_experiment(
        output_dir,
        {
            "event": "model_started",
            "model_name": "two_stage_residual",
            "stage1_model": base_model_name,
            "stage2_model": residual_regressor.__class__.__name__,
            "residual_protocol": "out_of_fold",
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "params": _model_params(residual_regressor),
        },
    )

    try:
        started_at = time.perf_counter()
        cv_mae_scores = _cross_validate_two_stage(
            base_pipeline,
            residual_regressor,
            x_train,
            y_train,
            groups,
            config.cv_folds,
            config.random_state,
        )
        stage1_train_predictions = _build_oof_stage1_predictions(
            base_pipeline,
            x_train,
            y_train,
            groups,
            config.cv_folds,
            config.random_state,
        )
        train_residual = y_train - stage1_train_predictions
        residual_pipeline.fit(x_train, train_residual)
        train_seconds = time.perf_counter() - started_at
        predict_started_at = time.perf_counter()
        stage1_test_predictions = base_pipeline.predict(x_test)
        stage2_test_predictions = residual_pipeline.predict(x_test)
        predictions = stage1_test_predictions + stage2_test_predictions
        predict_seconds = time.perf_counter() - predict_started_at

        cv_mae_mean = float(np.mean(cv_mae_scores))
        cv_mae_std = float(np.std(cv_mae_scores))
        metric = _evaluate_predictions(
            "two_stage_residual",
            y_test,
            predictions,
            cv_mae_mean,
            cv_mae_std,
        )
        metrics.append(metric)
        predictions_by_model["two_stage_residual"] = predictions
        predictions_by_model["two_stage_stage1"] = stage1_test_predictions
        predictions_by_model["two_stage_stage2"] = stage2_test_predictions
        _append_model_experiment(
            output_dir,
            {
                "event": "model_completed",
                "model_name": "two_stage_residual",
                "stage1_model": base_model_name,
                "stage2_model": residual_regressor.__class__.__name__,
                "residual_protocol": "out_of_fold",
                "cv_mae_scores": cv_mae_scores,
                "cv_mae_mean": cv_mae_mean,
                "cv_mae_std": cv_mae_std,
                "mae": metric.mae,
                "rmse": metric.rmse,
                "r2": metric.r2,
                "mape": metric.mape,
                "train_seconds": train_seconds,
                "predict_seconds": predict_seconds,
            },
        )
    except Exception as exc:
        _append_model_experiment(
            output_dir,
            {
                "event": "model_failed",
                "model_name": "two_stage_residual",
                "stage1_model": base_model_name,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def _best_stage1_model_name(metrics: list[ModelMetric]) -> str:
    candidate_metrics = [
        metric
        for metric in metrics
        if metric.model_name not in {"mean_baseline", "blend_top3", "two_stage_residual"}
    ]
    return min(candidate_metrics, key=lambda item: item.cv_mae_mean).model_name


def _build_residual_regressor(random_state: int) -> RegressorMixin:
    return HistGradientBoostingRegressor(
        max_iter=160,
        learning_rate=0.045,
        l2_regularization=0.1,
        random_state=random_state,
    )


def _cross_validate_two_stage(
    base_pipeline: Pipeline,
    residual_regressor: RegressorMixin,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    cv_folds: int,
    random_state: int,
) -> list[float]:
    scores: list[float] = []
    for train_index, valid_index in _build_cv_splits(groups, cv_folds, random_state):
        fold_base_pipeline = cast(Pipeline, clone(base_pipeline))
        fold_residual_pipeline = _build_pipeline(cast(RegressorMixin, clone(residual_regressor)))
        x_fold_train = x_train.iloc[train_index]
        y_fold_train = y_train.iloc[train_index]
        x_fold_valid = x_train.iloc[valid_index]
        y_fold_valid = y_train.iloc[valid_index]
        fold_groups = groups.iloc[train_index]

        fold_stage1_predictions = _build_oof_stage1_predictions(
            fold_base_pipeline,
            x_fold_train,
            y_fold_train,
            fold_groups,
            cv_folds,
            random_state,
        )
        fold_train_residual = y_fold_train - fold_stage1_predictions
        fold_residual_pipeline.fit(x_fold_train, fold_train_residual)
        fold_base_pipeline.fit(x_fold_train, y_fold_train)
        fold_predictions = fold_base_pipeline.predict(x_fold_valid) + fold_residual_pipeline.predict(x_fold_valid)
        scores.append(float(mean_absolute_error(y_fold_valid, fold_predictions)))
    return scores


def _build_oof_stage1_predictions(
    base_pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    cv_folds: int,
    random_state: int,
) -> np.ndarray:
    predictions = np.zeros(len(x_train), dtype=float)
    for train_index, valid_index in _build_cv_splits(groups, cv_folds, random_state):
        fold_base_pipeline = cast(Pipeline, clone(base_pipeline))
        fold_base_pipeline.fit(x_train.iloc[train_index], y_train.iloc[train_index])
        predictions[valid_index] = fold_base_pipeline.predict(x_train.iloc[valid_index])
    return predictions


def _build_cv_splits(groups: pd.Series, cv_folds: int, random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_group_count = int(groups.nunique())
    fold_count = min(cv_folds, max(2, unique_group_count))
    row_index = np.arange(len(groups))
    if unique_group_count >= fold_count:
        splitter = GroupKFold(n_splits=fold_count)
        return [(train_idx, valid_idx) for train_idx, valid_idx in splitter.split(row_index, groups=groups)]
    splitter = KFold(n_splits=fold_count, shuffle=True, random_state=random_state)
    return [(train_idx, valid_idx) for train_idx, valid_idx in splitter.split(row_index)]


def _best_single_model_name(metrics: list[ModelMetric]) -> str:
    single_metrics = [metric for metric in metrics if metric.model_name not in {"blend_top3", "two_stage_residual"}]
    return min(single_metrics, key=lambda item: item.mae).model_name


def _build_model_zoo(random_state: int) -> dict[str, RegressorMixin]:
    models: dict[str, RegressorMixin] = {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "ridge": RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0, 1000.0), cv=3),
        "lasso": LassoCV(cv=3, n_alphas=100, random_state=random_state, max_iter=50000, tol=0.01),
        "random_forest": RandomForestRegressor(
            n_estimators=180,
            max_depth=14,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=240,
            learning_rate=0.055,
            l2_regularization=0.05,
            random_state=random_state,
        ),
    }
    if XGBRegressor is not None:
        models["xgboost"] = XGBRegressor(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.045,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    if LGBMRegressor is not None:
        models["lightgbm"] = LGBMRegressor(
            n_estimators=260,
            max_depth=-1,
            learning_rate=0.045,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=0.1,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    if CatBoostRegressor is not None:
        models["catboost"] = CatBoostRegressor(
            iterations=260,
            depth=6,
            learning_rate=0.045,
            l2_leaf_reg=3.0,
            loss_function="RMSE",
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        )

    stacking_estimators: list[tuple[str, RegressorMixin]] = [
        ("random_forest", cast(RegressorMixin, clone(models["random_forest"]))),
        ("gradient_boosting", cast(RegressorMixin, clone(models["gradient_boosting"]))),
        ("hist_gradient_boosting", cast(RegressorMixin, clone(models["hist_gradient_boosting"]))),
    ]
    for optional_name in ("xgboost", "lightgbm", "catboost"):
        optional_model = models.get(optional_name)
        if optional_model is not None:
            stacking_estimators.append((optional_name, cast(RegressorMixin, clone(optional_model))))
    models["stacking_ridge"] = StackingRegressor(
        estimators=stacking_estimators,
        final_estimator=Ridge(alpha=1.0, random_state=random_state),
        cv=3,
        n_jobs=None,
    )
    return models


def _model_params(regressor: RegressorMixin) -> dict[str, JsonValue]:
    raw_params = regressor.get_params(deep=False)
    params: dict[str, JsonValue] = {}
    for key, value in raw_params.items():
        if isinstance(value, str | int | float | bool) or value is None:
            params[key] = value
        else:
            params[key] = str(value)
    return params


def _build_pipeline(regressor: RegressorMixin) -> Pipeline:
    return _build_pipeline_for_features(regressor, NUMERIC_FEATURES, CATEGORICAL_FEATURES)


def _build_pipeline_for_features(
    regressor: RegressorMixin,
    numeric_features: tuple[str, ...],
    categorical_features: tuple[str, ...],
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(numeric_features)),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(categorical_features)),
        ],
        sparse_threshold=0.0,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


def _cross_validate_model(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    cv_folds: int,
) -> tuple[float, float, list[float]]:
    unique_group_count = int(groups.nunique())
    fold_count = min(cv_folds, max(2, unique_group_count))
    if unique_group_count >= fold_count:
        splitter = GroupKFold(n_splits=fold_count)
        scores = cross_val_score(
            pipeline,
            x_train,
            y_train,
            cv=splitter,
            groups=groups,
            scoring="neg_mean_absolute_error",
            n_jobs=None,
        )
    else:
        splitter = KFold(n_splits=fold_count, shuffle=True, random_state=42)
        scores = cross_val_score(
            pipeline,
            x_train,
            y_train,
            cv=splitter,
            scoring="neg_mean_absolute_error",
            n_jobs=None,
        )
    positive_scores = -scores
    cv_mae_scores = [float(score) for score in positive_scores]
    return float(np.mean(positive_scores)), float(np.std(positive_scores)), cv_mae_scores


def _evaluate_predictions(
    model_name: str,
    y_true: pd.Series,
    predictions: np.ndarray,
    cv_mae_mean: float,
    cv_mae_std: float,
) -> ModelMetric:
    mse = mean_squared_error(y_true, predictions)
    mae = mean_absolute_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)
    mape = float(np.mean(np.abs((y_true.to_numpy() - predictions) / np.clip(y_true.to_numpy(), 1e-6, None))))
    try:
        return ModelMetric(
            model_name=model_name,
            mae=float(mae),
            rmse=float(np.sqrt(mse)),
            r2=float(r2),
            mape=mape,
            cv_mae_mean=cv_mae_mean,
            cv_mae_std=cv_mae_std,
        )
    except ValidationError as exc:
        raise ParseError(f"模型指标校验失败: model_name={model_name}, error={exc}") from exc


def _write_metrics(output_dir: Path, metrics: list[ModelMetric]) -> None:
    rows = [metric.model_dump() for metric in metrics]
    metrics_df = pd.DataFrame(rows).sort_values(by="mae")
    metrics_df.to_csv(output_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")


def _write_feature_generation_log(output_dir: Path) -> None:
    records: list[dict[str, JsonValue]] = [
        {
            "feature_name": "poi_balance_score",
            "formula": "entropy(normalized poi counts) / log(poi_type_count)",
            "source_fields": [f"poi_{poi_type}_count" for poi_type in POI_TYPES],
            "depends_on_target": False,
            "stage": "stage3",
        },
        {
            "feature_name": "transit_medical_ratio",
            "formula": "(poi_bus_count + 2 * poi_subway_count) / (poi_hospital_count + 1)",
            "source_fields": ["poi_bus_count", "poi_subway_count", "poi_hospital_count"],
            "depends_on_target": False,
            "stage": "stage3",
        },
        {
            "feature_name": "education_commerce_ratio",
            "formula": "poi_school_count / (poi_restaurant_count + poi_shop_count + 1)",
            "source_fields": ["poi_school_count", "poi_restaurant_count", "poi_shop_count"],
            "depends_on_target": False,
            "stage": "stage3",
        },
        {
            "feature_name": "area_layout_efficiency",
            "formula": "area_sqm / median(area_sqm by house_layout)",
            "source_fields": ["area_sqm", "house_layout"],
            "depends_on_target": False,
            "stage": "stage3",
        },
        {
            "feature_name": "room_area_interaction",
            "formula": "room_count * area_per_room",
            "source_fields": ["room_count", "area_per_room"],
            "depends_on_target": False,
            "stage": "stage3",
        },
        {
            "feature_name": "bath_room_ratio",
            "formula": "bath_count / max(room_count, 1)",
            "source_fields": ["bath_count", "room_count"],
            "depends_on_target": False,
            "stage": "stage3",
        },
        {
            "feature_name": "region_accessibility_interaction",
            "formula": "accessibility_index / median(accessibility_index by region_slug)",
            "source_fields": ["accessibility_index", "region_slug"],
            "depends_on_target": False,
            "stage": "stage3",
        },
        {
            "feature_name": "floor_area_interaction",
            "formula": "floor_level_score * log1p(area_sqm)",
            "source_fields": ["floor_level_score", "area_sqm"],
            "depends_on_target": False,
            "stage": "stage3",
        },
    ]
    with (output_dir / "feature_generation_log.jsonl").open("w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_feature_ablation_metrics(output_dir: Path, metrics: list[ModelMetric]) -> None:
    two_stage_metric = next((metric for metric in metrics if metric.model_name == "two_stage_residual"), None)
    if two_stage_metric is None:
        return
    rows = [
        {
            "feature_set": "stage3_enhanced_two_stage",
            "model_name": "two_stage_residual",
            "mae": two_stage_metric.mae,
            "rmse": two_stage_metric.rmse,
            "r2": two_stage_metric.r2,
            "mape": two_stage_metric.mape,
            "delta_mae_vs_base": 0.0,
            "source_run": "current_run",
        },
    ]
    pd.DataFrame(rows).to_csv(output_dir / "feature_ablation_metrics.csv", index=False, encoding="utf-8-sig")


def _write_feature_set_ablation_metrics(
    output_dir: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ResearchRunConfig,
) -> None:
    feature_sets: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
        "raw_features": (
            ("area_sqm", "room_count", "hall_count", "bath_count", "total_floors"),
            ("region_slug", "decoration_type", "orientation", "floor_level", "has_elevator", "is_unique_housing"),
        ),
        "base_engineered_features": (BASE_NUMERIC_FEATURES, CATEGORICAL_FEATURES),
        "stage3_enhanced_features": (NUMERIC_FEATURES, CATEGORICAL_FEATURES),
    }
    rows: list[dict[str, JsonValue]] = []
    groups = train_df["community_id"].fillna("unknown").astype(str)
    y_train = train_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    for feature_set_name, (numeric_features, categorical_features) in feature_sets.items():
        x_train = train_df[list(numeric_features) + list(categorical_features)]
        x_test = test_df[list(numeric_features) + list(categorical_features)]
        hist_pipeline = _build_pipeline_for_features(
            HistGradientBoostingRegressor(
                max_iter=240,
                learning_rate=0.055,
                l2_regularization=0.05,
                random_state=config.random_state,
            ),
            numeric_features,
            categorical_features,
        )
        hist_pipeline.fit(x_train, y_train)
        hist_predictions = hist_pipeline.predict(x_test)
        hist_metric = _evaluate_predictions(
            "hist_gradient_boosting",
            y_test,
            hist_predictions,
            0.0,
            0.0,
        )
        rows.append({"feature_set": feature_set_name, **hist_metric.model_dump()})

        residual_pipeline = _build_pipeline_for_features(
            _build_residual_regressor(config.random_state),
            numeric_features,
            categorical_features,
        )
        stage1_oof = _build_oof_stage1_predictions(
            hist_pipeline,
            x_train,
            y_train,
            groups,
            config.cv_folds,
            config.random_state,
        )
        residual_pipeline.fit(x_train, y_train - stage1_oof)
        stage1_pipeline = cast(Pipeline, clone(hist_pipeline))
        stage1_pipeline.fit(x_train, y_train)
        two_stage_predictions = stage1_pipeline.predict(x_test) + residual_pipeline.predict(x_test)
        two_stage_metric = _evaluate_predictions(
            "two_stage_residual_hist_base",
            y_test,
            two_stage_predictions,
            0.0,
            0.0,
        )
        rows.append({"feature_set": feature_set_name, **two_stage_metric.model_dump()})

    pd.DataFrame(rows).to_csv(output_dir / "feature_set_ablation_metrics.csv", index=False, encoding="utf-8-sig")


def _write_model_prediction_errors(
    output_dir: Path,
    test_df: pd.DataFrame,
    predictions_by_model: dict[str, np.ndarray],
) -> None:
    rows: list[dict[str, JsonValue]] = []
    y_true = test_df[TARGET_COLUMN].to_numpy()
    for model_name, predictions in predictions_by_model.items():
        if model_name in {"two_stage_stage1", "two_stage_stage2"}:
            continue
        absolute_errors = np.abs(y_true - predictions)
        for house_id, actual_value, predicted_value, absolute_error in zip(
            test_df["house_id"].astype(str),
            y_true,
            predictions,
            absolute_errors,
            strict=True,
        ):
            rows.append(
                {
                    "model_name": model_name,
                    "house_id": house_id,
                    "actual_price_wan": float(actual_value),
                    "predicted_price_wan": float(predicted_value),
                    "absolute_error_wan": float(absolute_error),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "model_prediction_errors.csv", index=False, encoding="utf-8-sig")


def _write_significance_tests(
    output_dir: Path,
    test_df: pd.DataFrame,
    predictions_by_model: dict[str, np.ndarray],
) -> None:
    y_true = test_df[TARGET_COLUMN].to_numpy()
    target_models = [
        model_name
        for model_name in ("hist_gradient_boosting", "catboost", "stacking_ridge")
        if model_name in predictions_by_model
    ]
    if "two_stage_residual" not in predictions_by_model:
        pd.DataFrame([]).to_csv(output_dir / "significance_tests.csv", index=False, encoding="utf-8-sig")
        return

    rows: list[dict[str, JsonValue]] = []
    two_stage_errors = np.abs(y_true - predictions_by_model["two_stage_residual"])
    for model_name in target_models:
        baseline_errors = np.abs(y_true - predictions_by_model[model_name])
        paired_delta = baseline_errors - two_stage_errors
        ttest_result = stats.ttest_rel(baseline_errors, two_stage_errors)
        wilcoxon_result = stats.wilcoxon(baseline_errors, two_stage_errors, zero_method="zsplit")
        ci_low, ci_high = _bootstrap_mean_ci(paired_delta)
        rows.append(
            {
                "comparison": f"{model_name}_minus_two_stage_residual",
                "mean_mae_delta_wan": float(np.mean(paired_delta)),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "paired_t_p_value": float(ttest_result.pvalue),
                "wilcoxon_p_value": float(wilcoxon_result.pvalue),
                "sample_count": int(len(paired_delta)),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "significance_tests.csv", index=False, encoding="utf-8-sig")


def _bootstrap_mean_ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    means = [
        float(np.mean(rng.choice(values, size=len(values), replace=True)))
        for _ in range(1000)
    ]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _write_error_stratification(output_dir: Path, test_df: pd.DataFrame, predictions_df: pd.DataFrame) -> None:
    analysis_df = test_df.merge(
        predictions_df[["house_id", "residual_wan"]],
        on="house_id",
        how="inner",
    ).copy()
    analysis_df["absolute_error_wan"] = analysis_df["residual_wan"].abs()
    analysis_df["area_bin"] = pd.qcut(analysis_df["area_sqm"], q=4, duplicates="drop").astype(str)
    analysis_df["price_bin"] = pd.qcut(analysis_df[TARGET_COLUMN], q=4, duplicates="drop").astype(str)

    rows: list[dict[str, JsonValue]] = []
    for column_name in ("region_slug", "house_layout", "area_bin", "price_bin"):
        grouped = analysis_df.groupby(column_name, dropna=False)
        for group_value, group_df in grouped:
            rows.append(
                {
                    "stratify_column": column_name,
                    "stratify_value": str(group_value),
                    "sample_count": int(len(group_df)),
                    "mae": float(group_df["absolute_error_wan"].mean()),
                    "mape": float(
                        np.mean(
                            group_df["absolute_error_wan"].to_numpy()
                            / np.clip(group_df[TARGET_COLUMN].to_numpy(), 1e-6, None)
                        )
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "error_stratification.csv", index=False, encoding="utf-8-sig")


def _write_temporal_validation_profile(
    output_dir: Path,
    feature_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    extracted_at = pd.to_datetime(feature_df["extracted_at"], errors="coerce")
    rows = [
        {
            "scope": "all",
            "sample_count": int(len(feature_df)),
            "min_extracted_at": str(extracted_at.min()),
            "max_extracted_at": str(extracted_at.max()),
            "unique_extracted_day_count": int(extracted_at.dt.date.nunique()),
        },
        {
            "scope": "train",
            "sample_count": int(len(train_df)),
            "min_extracted_at": str(pd.to_datetime(train_df["extracted_at"], errors="coerce").min()),
            "max_extracted_at": str(pd.to_datetime(train_df["extracted_at"], errors="coerce").max()),
            "unique_extracted_day_count": int(pd.to_datetime(train_df["extracted_at"], errors="coerce").dt.date.nunique()),
        },
        {
            "scope": "test",
            "sample_count": int(len(test_df)),
            "min_extracted_at": str(pd.to_datetime(test_df["extracted_at"], errors="coerce").min()),
            "max_extracted_at": str(pd.to_datetime(test_df["extracted_at"], errors="coerce").max()),
            "unique_extracted_day_count": int(pd.to_datetime(test_df["extracted_at"], errors="coerce").dt.date.nunique()),
        },
    ]
    pd.DataFrame(rows).to_csv(output_dir / "temporal_validation_profile.csv", index=False, encoding="utf-8-sig")


def _write_feature_importance(
    output_dir: Path,
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int,
) -> None:
    x_test = test_df[list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)]
    y_test = test_df[TARGET_COLUMN]
    result = permutation_importance(
        pipeline,
        x_test,
        y_test,
        n_repeats=5,
        random_state=random_state,
        scoring="neg_mean_absolute_error",
    )
    importance_df = pd.DataFrame(
        {
            "feature": list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values(by="importance_mean", ascending=False)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")


def _write_shap_analysis(
    output_dir: Path,
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int,
) -> None:
    status_path = output_dir / "shap_status.json"
    try:
        shap_module = __import__("shap")
    except ImportError as exc:
        status_path.write_text(
            json.dumps(
                {"status": "unavailable", "reason": f"SHAP依赖未安装: {exc}"},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return

    x_train = train_df[list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)]
    x_test = test_df[list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)]
    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["regressor"]
    sample_count = min(len(x_test), 300)
    if sample_count == 0:
        status_path.write_text(
            json.dumps({"status": "skipped", "reason": "测试集为空"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return
    x_sample = x_test.sample(n=sample_count, random_state=random_state)
    preprocessor.fit(x_train)
    transformed_sample = preprocessor.transform(x_sample)
    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    explained_model = regressor
    explained_model_name = regressor.__class__.__name__
    if isinstance(regressor, StackingRegressor):
        candidate_estimators = getattr(regressor, "named_estimators_", {})
        explained_model = candidate_estimators.get("hist_gradient_boosting")
        explained_model_name = "stacking_ridge.hist_gradient_boosting"
        if explained_model is None:
            status_path.write_text(
                json.dumps(
                    {"status": "skipped", "reason": "StackingRegressor缺少可解释的hist_gradient_boosting基模型"},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return

    try:
        explainer = shap_module.TreeExplainer(explained_model)
        shap_values = explainer.shap_values(transformed_sample)
    except Exception as exc:
        status_path.write_text(
            json.dumps(
                {"status": "skipped", "reason": f"当前解释目标不支持TreeExplainer: {exc}"},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return

    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 0]
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.mean(np.abs(shap_array), axis=0),
        }
    ).sort_values(by="mean_abs_shap", ascending=False)
    importance_df.to_csv(output_dir / "shap_importance.csv", index=False, encoding="utf-8-sig")

    shap_module.summary_plot(shap_array, transformed_sample, feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_beeswarm.png", dpi=180)
    plt.close()
    status_path.write_text(
        json.dumps(
            {"status": "completed", "sample_count": sample_count, "explained_model": explained_model_name},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_plots(output_dir: Path, feature_df: pd.DataFrame, predictions_df: pd.DataFrame) -> None:
    numeric_df = feature_df[[TARGET_COLUMN] + list(NUMERIC_FEATURES)].copy()
    plt.figure(figsize=(14, 10))
    sns.heatmap(numeric_df.corr(numeric_only=True), cmap="coolwarm", center=0)
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(predictions_df["residual_wan"], bins=40, kde=True)
    plt.xlabel("Residual Wan")
    plt.ylabel("Count")
    plt.subplot(1, 2, 2)
    sns.scatterplot(
        x=predictions_df[TARGET_COLUMN],
        y=predictions_df["predicted_price_wan"],
        s=18,
    )
    plt.xlabel("Actual Wan")
    plt.ylabel("Predicted Wan")
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_error.png", dpi=180)
    plt.close()


def _write_report(
    output_dir: Path,
    config: ResearchRunConfig,
    audit: DatasetAudit,
    metrics: list[ModelMetric],
) -> None:
    metrics_sorted = sorted(metrics, key=lambda item: item.mae)
    best_metric = metrics_sorted[0]
    baseline_metric = next(metric for metric in metrics if metric.model_name == "mean_baseline")
    mae_gain = (baseline_metric.mae - best_metric.mae) / baseline_metric.mae if baseline_metric.mae > 0 else 0.0
    two_stage_lines = _build_two_stage_report_lines(metrics)
    stage3_lines = _build_stage3_report_lines(metrics)
    lines = [
        "南昌房价算法设计说明",
        "",
        f"run_id: {config.run_id}",
        f"研究任务: 全房型二手房总价回归，目标字段为{TARGET_COLUMN}",
        f"可用样本数: {audit.usable_house_count}",
        f"关键字段可用率: {audit.key_field_usable_rate:.4f}",
        f"小区关联成功率: {audit.community_join_rate:.4f}",
        "",
        "模型结论:",
        f"最优模型: {best_metric.model_name}",
        f"MAE: {best_metric.mae:.4f}",
        f"RMSE: {best_metric.rmse:.4f}",
        f"R2: {best_metric.r2:.4f}",
        f"MAPE: {best_metric.mape:.4f}",
        f"相对均值基线MAE提升: {mae_gain:.4f}",
        "",
        "工程设计:",
        "1. 使用时间靠后样本作为留出集，模拟未来挂牌样本预测。",
        "2. 交叉验证阶段优先使用community_id分组，降低同小区泄漏。",
        "3. 研究特征全部在产物目录生成，不覆盖原始采集数据。",
        "",
        "创新点:",
        "1. 从小区POI统计构造accessibility_index，刻画生活配套可达性。",
        "2. 构造house_layout、area_per_room、layout_density刻画户型与面积交互。",
        "3. 引入二阶段残差校正，验证全局规律与局部偏差修正的组合收益。",
        "4. 通过置换重要度输出可解释特征排序，辅助总结价格规律。",
        "",
        "二阶段残差校正:",
        *two_stage_lines,
        "",
        "第三阶段增强特征:",
        *stage3_lines,
        "",
        "风险说明:",
        "挂牌价不是成交价，存在平台报价偏差。",
        "同一小区内样本相似度较高，已通过分组验证降低但不能完全消除。",
        "当前时间跨度较短，长期趋势结论需要后续多月数据补充验证。",
    ]
    (output_dir / "algorithm_design_report.txt").write_text("\n".join(lines), encoding="utf-8")


def _build_two_stage_report_lines(metrics: list[ModelMetric]) -> list[str]:
    two_stage_metric = next((metric for metric in metrics if metric.model_name == "two_stage_residual"), None)
    if two_stage_metric is None:
        return ["未运行二阶段残差校正模型。"]

    comparator_metric = min(
        [metric for metric in metrics if metric.model_name not in {"mean_baseline", "blend_top3", "two_stage_residual"}],
        key=lambda item: item.mae,
    )
    blend_metric = next((metric for metric in metrics if metric.model_name == "blend_top3"), None)
    comparator_mae_delta = comparator_metric.mae - two_stage_metric.mae
    lines = [
        f"二阶段模型MAE: {two_stage_metric.mae:.4f}",
        f"相对最优横向模型({comparator_metric.model_name}) MAE变化: {comparator_mae_delta:.4f}",
    ]
    if blend_metric is not None:
        blend_mae_delta = blend_metric.mae - two_stage_metric.mae
        lines.append(f"相对融合模型(blend_top3) MAE变化: {blend_mae_delta:.4f}")
    if comparator_mae_delta > 0:
        lines.append("结论: 二阶段残差校正在主指标MAE上带来增益，可作为候选主模型。")
    else:
        lines.append("结论: 二阶段残差校正在主指标MAE上未超过最优横向模型，暂作为研究对照保留。")
    return lines


def _build_stage3_report_lines(metrics: list[ModelMetric]) -> list[str]:
    two_stage_metric = next((metric for metric in metrics if metric.model_name == "two_stage_residual"), None)
    if two_stage_metric is None:
        return ["未生成第三阶段增强特征下的二阶段模型结果。"]
    lines = [
        "新增特征: poi_balance_score, transit_medical_ratio, education_commerce_ratio, area_layout_efficiency, room_area_interaction, bath_room_ratio, region_accessibility_interaction, floor_area_interaction",
        f"当前OOF二阶段MAE: {two_stage_metric.mae:.4f}",
        "raw/base/stage3特征组对比见feature_set_ablation_metrics.csv。",
    ]
    lines.append("结论: 第三阶段新特征需结合独立特征组消融判断，不再与旧协议历史run直接比较。")
    return lines

