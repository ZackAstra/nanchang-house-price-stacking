"""
任务 2.3：LLM 语义特征 vs 规则语义特征增量效果对比

核心逻辑：
1. 加载 LLM 提取的语义特征 CSV
2. 复现规则匹配的语义特征
3. 在以下基线上分别叠加两种语义特征：
   - 原始基线（无对数、无清洗）
   - 组合基线（对数目标 + 严格清洗）
4. quick 模式只跑 stacking_ridge，对比 6 种配置的 MAE
5. 全量模式跑完整 model_zoo

实验配置（7 种）：
- original: 原始特征
- rule_semantic: 原始特征 + 规则语义
- llm_semantic: 原始特征 + LLM 语义
- combined_original: 对数变换 + 严格清洗
- combined_rule_semantic: 对数变换 + 严格清洗 + 规则语义
- combined_llm_semantic: 对数变换 + 严格清洗 + LLM 语义（最终希望）
- combined_rule_llm_semantic: 对数变换 + 严格清洗 + 规则语义 + LLM 语义

评审修复协议（review2）：
1. 全量特征表先按时间划分训练/测试集，所有实验共用同一固定测试集；
2. 标签噪声检测只在训练集上运行，分位数边界仅由训练集估计；
3. 严格清洗只删除训练集样本，测试集保持不动。
"""

from __future__ import annotations

import argparse
import json
import re
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.research.models import JsonValue, ModelMetric, ResearchRunConfig, ResearchResult
from src.research.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    _apply_sample_limit,
    _build_features,
    _build_model_zoo,
    _build_pipeline_for_features,
    _clean_and_join,
    _cross_validate_model,
    _evaluate_predictions,
    _load_communities,
    _load_houses,
    _split_by_time,
    _apply_reference_group_medians,
    _compute_region_ai_medians,
    _write_metrics,
)

# 新旧双基线
OLD_BASELINE_MAE = 20.911922387801575
NEW_BASELINE_MAE = 20.1429  # combined__stacking_ridge 全量验证结果

# 规则语义关键词（复用 title_semantic_feature_runner）
RULE_SEMANTIC_KEYWORDS: dict[str, list[str]] = {
    "subway": ["地铁", "地铁口", "地铁站", "地铁房", "轨道交通"],
    "bus_hub": ["公交", "公交站", "交通便利", "出行方便"],
    "school_district": ["学区", "学位", "名校", "重点学校", "学区房"],
    "school_nearby": ["学校", "幼儿园", "小学", "中学", "附中"],
    "river_view": ["江景", "湖景", "河景", "水景", "一线江景"],
    "park_nearby": ["公园", "绿化", "生态"],
    "high_floor_view": ["视野", "无遮挡", "景观房", "观景"],
    "luxury_decor": ["精装", "豪装", "豪华装修", "精装修", "品牌装修", "装修好", "精装好房"],
    "simple_decor": ["简装", "简单装修", "普通装修"],
    "rough": ["毛坯", "清水"],
    "urgent_sale": ["急售", "急卖", "急！急！急！", "急售房源", "特价", "房东急售"],
    "below_market": ["低于市场价", "低于市价", "捡漏", "抄底", "白菜价"],
    "sincere": ["诚心", "诚意", "诚心出售", "诚意出", "房东诚心", "诚心卖"],
    "quality_community": ["品质", "高档", "高端", "豪华小区", "豪宅"],
    "new_community": ["次新", "新小区", "新房", "次新房", "小区新", "次新小区"],
    "old_community": ["老小区", "老旧", "老房", "房龄老"],
    "low_density": ["低密度", "低密", "密度低", "居住密度低"],
    "good_lighting": ["采光好", "采光", "南向采光", "正南朝向", "朝南", "南北通透", "朝向好"],
    "elevator": ["电梯", "电梯房", "带电梯", "有电梯"],
    "shopping": ["商圈", "商场", "超市", "购物方便", "购物", "配套完善", "配套成熟"],
    "direct_seller": ["房东直卖", "业主直卖", "直卖"],
    "price_negotiable": ["价格可议", "可议价", "价格美丽", "合适可谈", "可谈", "价格可商量"],
    "garage": ["车库", "车位", "车位充足", "带车位", "有车位"],
    "good_floor": ["高楼层", "中间好楼层", "中间楼层", "好楼层", "楼层好", "高层"],
    "duplex": ["复式", "跃层", "loft"],
    "garden": ["花园", "露台", "晒台", "院子", "天台"],
    "furnished": ["送家具", "带家具", "家具齐全", "拎包入住"],
    "double_bath": ["双卫", "双卫生间", "双卫格局", "两个卫生间"],
    "board_building": ["板楼", "板式建筑"],
}

RULE_SEMANTIC_FEATURES = tuple(sorted(RULE_SEMANTIC_KEYWORDS.keys()))

# LLM 语义标签（复用 llm_semantic_extractor，去掉 bus_hub 因为 LLM 没有）
LLM_SEMANTIC_LABELS = [
    "subway", "school_district", "school_nearby", "river_view", "park_nearby",
    "high_floor_view", "luxury_decor", "simple_decor", "rough", "urgent_sale",
    "below_market", "sincere", "quality_community", "new_community", "old_community",
    "low_density", "good_lighting", "elevator", "shopping", "direct_seller",
    "price_negotiable", "garage", "good_floor", "duplex", "garden",
    "furnished", "double_bath", "board_building",
]


def run_llm_semantic_vs_rule_experiment(
    config: ResearchRunConfig,
    llm_features_path: Path,
    only_experiments: list[str] | None = None,
) -> ResearchResult:
    """执行 LLM 语义 vs 规则语义增量效果对比实验。

    only_experiments: 仅运行指定名称的实验组（用于补充格子的增量重跑），None 为全部。
    """
    output_dir = config.output_root / config.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_run_log(
        output_dir,
        "run_started",
        {
            "run_id": config.run_id,
            "experiment": "llm_semantic_vs_rule",
            "llm_features_path": str(llm_features_path),
        },
    )

    # 1. 数据加载
    houses_df = _load_houses(config.houses_path)
    communities_df = _load_communities(config.communities_path)
    merged_df, audit = _clean_and_join(houses_df, communities_df)
    feature_df = _build_features(merged_df)

    # 2. 加载 LLM 语义特征并合并
    llm_df = _load_llm_semantic_features(llm_features_path)
    feature_df = feature_df.merge(llm_df, on="house_id", how="left")
    for col in LLM_SEMANTIC_LABELS:
        feature_df[col] = feature_df[col].fillna(0).astype(int)

    # 3. 生成规则语义特征（从 merged_df 的 title），避免列名冲突加后缀
    rule_df = _extract_rule_semantic_features(merged_df)
    # 对重叠列重命名，防止 merge 时产生 _x/_y 后缀
    overlap_cols = set(RULE_SEMANTIC_FEATURES) & set(LLM_SEMANTIC_LABELS)
    rule_rename = {c: f"{c}_rule" for c in overlap_cols}
    rule_df_renamed = rule_df.rename(columns=rule_rename)
    feature_df = feature_df.merge(rule_df_renamed, on="house_id", how="left")
    for col in RULE_SEMANTIC_FEATURES:
        col_name = f"{col}_rule" if col in overlap_cols else col
        feature_df[col_name] = feature_df[col_name].fillna(0).astype(int)

    # 4. 抽样限制
    feature_df = _apply_sample_limit(feature_df, config.sample_limit, config.random_state)

    # 5. 新增对数特征
    feature_df["log_total_price_wan"] = np.log1p(feature_df[TARGET_COLUMN].clip(lower=1e-6))
    feature_df["log_area_sqm"] = np.log1p(feature_df["area_sqm"].clip(lower=1e-6))

    # 5b. 先按时间划分固定训练/测试集（所有实验共用同一测试集）
    train_full_df, fixed_test_df = _split_by_time(feature_df, config.test_size)
    train_full_df, fixed_test_df = _apply_reference_group_medians(train_full_df, fixed_test_df, _compute_region_ai_medians(config.communities_path))

    # 5c. 噪声检测仅作用于训练集（分位数边界仅由训练集估计）
    noise_report = _detect_label_noise(train_full_df)
    train_strict_mask = ~noise_report["is_noise_strict"]
    strict_removed = int((~train_strict_mask).sum())

    _write_run_log(
        output_dir,
        "train_label_cleaning",
        {
            "train_full_samples": int(len(train_full_df)),
            "strict_removed_samples": strict_removed,
            "strict_removal_rate": float(strict_removed / len(train_full_df)),
            "fixed_test_samples": int(len(fixed_test_df)),
        },
    )

    # 6. 写入语义特征分布对比（全量描述性统计，无泄漏问题）
    _write_semantic_comparison(feature_df, output_dir)

    # 7. 定义实验配置（keep_mask 基于训练集索引）；--only 过滤增量重跑
    is_quick = config.sample_limit is not None
    experiments = _build_experiment_configs(train_strict_mask, is_quick)
    if only_experiments:
        experiments = [e for e in experiments if str(e["name"]) in set(only_experiments)]
        if not experiments:
            raise ValueError(f"--only 未匹配到任何实验组: {only_experiments}")

    all_metrics: list[ModelMetric] = []
    summary_rows: list[dict[str, JsonValue]] = []

    for exp_config in experiments:
        exp_name = str(exp_config["name"])
        target_col = str(exp_config["target_col"])
        keep_mask = cast(pd.Series, exp_config["keep_mask"])
        use_log_target = bool(exp_config["use_log_target"])
        use_log_area = bool(exp_config.get("use_log_area", False))
        semantic_type = str(exp_config["semantic_type"])  # "none" / "rule" / "llm" / "rule_llm"

        # 清洗只作用于训练集，测试集固定不动
        train_df = train_full_df.loc[keep_mask].copy()
        test_df = fixed_test_df

        _write_run_log(
            output_dir,
            "experiment_started",
            {
                "experiment_name": exp_name,
                "target_col": target_col,
                "use_log_target": use_log_target,
                "use_log_area": use_log_area,
                "semantic_type": semantic_type,
                "train_samples": int(len(train_df)),
                "test_samples": int(len(test_df)),
            },
        )

        metrics, _ = _run_variant(
            train_df=train_df,
            test_df=test_df,
            config=config,
            target_col=target_col,
            use_log_target=use_log_target,
            use_log_area=use_log_area,
            semantic_type=semantic_type,
            output_dir=output_dir,
            experiment_name=exp_name,
            quick_mode=is_quick,
        )

        all_metrics.extend(metrics)

        best_metric = min(metrics, key=lambda m: m.mae)
        baseline = NEW_BASELINE_MAE if "combined" in exp_name else OLD_BASELINE_MAE
        summary_rows.append(
            {
                "experiment_name": exp_name,
                "semantic_type": semantic_type,
                "target_col": target_col,
                "use_log_target": use_log_target,
                "use_log_area": use_log_area,
                "kept_samples": int(len(train_df)),
                "test_samples": int(len(test_df)),
                "best_model": best_metric.model_name.split("__")[-1],
                "mae": best_metric.mae,
                "rmse": best_metric.rmse,
                "r2": best_metric.r2,
                "mape": best_metric.mape,
                "delta_vs_new_baseline": best_metric.mae - NEW_BASELINE_MAE,
                "delta_vs_old_baseline": best_metric.mae - OLD_BASELINE_MAE,
            }
        )

    # 8. 写入产物
    _write_metrics(output_dir, all_metrics)
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "experiment_summary.csv", index=False, encoding="utf-8-sig"
    )
    _write_stacking_comparison(output_dir, all_metrics)

    best_overall = min(all_metrics, key=lambda m: m.mae)
    _write_run_log(
        output_dir,
        "run_completed",
        {
            "best_experiment_model": best_overall.model_name,
            "best_mae": best_overall.mae,
            "new_baseline_mae": NEW_BASELINE_MAE,
            "old_baseline_mae": OLD_BASELINE_MAE,
        },
    )

    print("=" * 60)
    print("LLM Semantic vs Rule Semantic Experiment Complete")
    print(f"Run ID: {config.run_id}")
    print(f"Output: {output_dir}")
    print(f"Best model: {best_overall.model_name}")
    print(f"Best MAE: {best_overall.mae:.4f}")
    print(f"New baseline MAE: {NEW_BASELINE_MAE:.4f}")
    print(f"Old baseline MAE: {OLD_BASELINE_MAE:.4f}")
    print("=" * 60)

    return ResearchResult(
        run_id=config.run_id,
        output_dir=output_dir,
        audit=audit,
        metrics=all_metrics,
        best_model_name=best_overall.model_name,
    )


def _load_llm_semantic_features(path: Path) -> pd.DataFrame:
    """加载 LLM 语义特征 CSV。"""
    if not path.exists():
        raise FileNotFoundError(f"LLM 语义特征文件不存在: {path}")
    df = pd.read_csv(path)
    # 确保列名正确
    expected_cols = ["house_id"] + LLM_SEMANTIC_LABELS
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"LLM 语义特征 CSV 缺少列: {col}")
    return df[["house_id"] + LLM_SEMANTIC_LABELS]


def _extract_rule_semantic_features(merged_df: pd.DataFrame) -> pd.DataFrame:
    """从房源标题中提取规则语义特征。"""
    df = merged_df[["house_id", "title"]].copy()
    df["title"] = df["title"].fillna("").astype(str)

    for feature_name, keywords in RULE_SEMANTIC_KEYWORDS.items():
        df[feature_name] = df["title"].apply(
            lambda t: int(any(kw in t for kw in keywords))
        )

    return df[["house_id"] + list(RULE_SEMANTIC_FEATURES)]


def _detect_label_noise(feature_df: pd.DataFrame) -> pd.DataFrame:
    """检测标签噪声（复用 combined_breakthrough_runner 逻辑）。"""
    df = feature_df.copy()
    df["unit_price"] = df[TARGET_COLUMN] / df["area_sqm"].clip(lower=1.0)

    group_keys = df["region_slug"].astype(str) + "__" + df["house_layout"].astype(str)
    df["unit_price_group"] = group_keys
    unit_price_low = df.groupby("unit_price_group")["unit_price"].transform(lambda x: x.quantile(0.01))
    unit_price_high = df.groupby("unit_price_group")["unit_price"].transform(lambda x: x.quantile(0.99))
    group_counts = df.groupby("unit_price_group")["unit_price"].transform("count")
    global_price_low = df["unit_price"].quantile(0.01)
    global_price_high = df["unit_price"].quantile(0.99)
    df["unit_price_low"] = np.where(group_counts >= 10, unit_price_low, global_price_low)
    df["unit_price_high"] = np.where(group_counts >= 10, unit_price_high, global_price_high)
    df["is_unit_price_outlier"] = (df["unit_price"] < df["unit_price_low"]) | (df["unit_price"] > df["unit_price_high"])

    total_price_low = df[TARGET_COLUMN].quantile(0.005)
    total_price_high = df[TARGET_COLUMN].quantile(0.995)
    df["is_total_price_outlier"] = (df[TARGET_COLUMN] < total_price_low) | (df[TARGET_COLUMN] > total_price_high)

    area_low = df["area_sqm"].quantile(0.005)
    area_high = df["area_sqm"].quantile(0.995)
    df["is_area_outlier"] = (df["area_sqm"] < area_low) | (df["area_sqm"] > area_high)

    df["is_noise_strict"] = df["is_unit_price_outlier"] | df["is_total_price_outlier"] | df["is_area_outlier"]

    return df


def _build_experiment_configs(
    strict_mask: pd.Series, is_quick: bool
) -> list[dict[str, JsonValue]]:
    """构建 7 种实验配置（strict_mask 基于训练集索引）。"""
    configs = [
        # 原始基线组
        {
            "name": "original",
            "target_col": TARGET_COLUMN,
            "keep_mask": pd.Series(True, index=strict_mask.index),
            "use_log_target": False,
            "semantic_type": "none",
        },
        {
            "name": "rule_semantic",
            "target_col": TARGET_COLUMN,
            "keep_mask": pd.Series(True, index=strict_mask.index),
            "use_log_target": False,
            "semantic_type": "rule",
        },
        {
            "name": "llm_semantic",
            "target_col": TARGET_COLUMN,
            "keep_mask": pd.Series(True, index=strict_mask.index),
            "use_log_target": False,
            "semantic_type": "llm",
        },
        # 组合基线组
        {
            "name": "combined_original",
            "target_col": "log_total_price_wan",
            "keep_mask": strict_mask,
            "use_log_target": True,
            "semantic_type": "none",
        },
        {
            "name": "combined_rule_semantic",
            "target_col": "log_total_price_wan",
            "keep_mask": strict_mask,
            "use_log_target": True,
            "semantic_type": "rule",
        },
        {
            "name": "combined_llm_semantic",
            "target_col": "log_total_price_wan",
            "keep_mask": strict_mask,
            "use_log_target": True,
            "semantic_type": "llm",
        },
        {
            "name": "combined_rule_llm_semantic",
            "target_col": "log_total_price_wan",
            "keep_mask": strict_mask,
            "use_log_target": True,
            "semantic_type": "rule_llm",
        },
        # log×语义（不清洗）补充格子：清洗已被证伪（review3_combined_breakthrough），
        # 需要 log-only 基线上的语义增量来定位最终模型；
        # log_original 为 log 基线对照锚点（配对检验用）
        {
            "name": "log_original",
            "target_col": "log_total_price_wan",
            "keep_mask": pd.Series(True, index=strict_mask.index),
            "use_log_target": True,
            "semantic_type": "none",
        },
        {
            "name": "log_rule_semantic",
            "target_col": "log_total_price_wan",
            "keep_mask": pd.Series(True, index=strict_mask.index),
            "use_log_target": True,
            "semantic_type": "rule",
        },
        {
            "name": "log_llm_semantic",
            "target_col": "log_total_price_wan",
            "keep_mask": pd.Series(True, index=strict_mask.index),
            "use_log_target": True,
            "semantic_type": "llm",
        },
        {
            "name": "log_rule_llm_semantic",
            "target_col": "log_total_price_wan",
            "keep_mask": pd.Series(True, index=strict_mask.index),
            "use_log_target": True,
            "semantic_type": "rule_llm",
        },
        # log 面积 × 规则语义补全格：2×2 设计（log_area × rule_semantic）的最后一格，
        # 用于回答"最终模型是否应同时纳入 log 面积"（log_transform 实验中
        # log_target_and_area 单独有 −0.128 的边际收益，需检验其与语义是否叠加）
        {
            "name": "log_area_rule_semantic",
            "target_col": "log_total_price_wan",
            "keep_mask": pd.Series(True, index=strict_mask.index),
            "use_log_target": True,
            "use_log_area": True,
            "semantic_type": "rule",
        },
    ]

    # quick 模式只跑包含 "original" 和 "semantic" 的代表性配置
    if is_quick:
        # quick 模式跑所有配置（因为每个都只跑 stacking_ridge，很快）
        pass

    return configs


def _get_features_for_semantic_type(semantic_type: str) -> tuple[list[str], list[str]]:
    """根据语义类型获取特征列表。"""
    base_numeric = list(NUMERIC_FEATURES)
    base_categorical = list(CATEGORICAL_FEATURES)

    # 重叠列加了 _rule 后缀，规则列与 LLM 列名已去重
    overlap_cols = set(RULE_SEMANTIC_FEATURES) & set(LLM_SEMANTIC_LABELS)
    rule_cols = [f"{c}_rule" if c in overlap_cols else c for c in RULE_SEMANTIC_FEATURES]

    if semantic_type == "rule":
        numeric = base_numeric + rule_cols
    elif semantic_type == "llm":
        numeric = base_numeric + list(LLM_SEMANTIC_LABELS)
    elif semantic_type == "rule_llm":
        numeric = base_numeric + rule_cols + list(LLM_SEMANTIC_LABELS)
    else:
        numeric = base_numeric

    return numeric, base_categorical


def _run_variant(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ResearchRunConfig,
    target_col: str,
    use_log_target: bool,
    semantic_type: str,
    output_dir: Path,
    experiment_name: str,
    quick_mode: bool = False,
    use_log_area: bool = False,
) -> tuple[list[ModelMetric], dict[str, np.ndarray]]:
    """执行单个实验变体。"""
    numeric_features, categorical_features = _get_features_for_semantic_type(semantic_type)
    if use_log_area:
        # 与 log_transform_experiment_runner 的 log_target_and_area 同构：
        # 仅将 area_sqm 列替换为 log_area_sqm，派生面积特征保持原构造
        numeric_features = ["log_area_sqm" if f == "area_sqm" else f for f in numeric_features]

    # 确保所有特征列都存在
    all_features = numeric_features + categorical_features
    missing = [c for c in all_features if c not in train_df.columns]
    if missing:
        raise ValueError(f"Missing columns in train_df: {missing}")

    x_train = train_df[all_features]
    y_train = train_df[target_col]
    x_test = test_df[all_features]
    y_test_original = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    model_zoo = _build_model_zoo(config.random_state)
    if quick_mode:
        model_zoo = {k: v for k, v in model_zoo.items() if k == "stacking_ridge"}

    metrics: list[ModelMetric] = []
    predictions_by_model: dict[str, np.ndarray] = {}
    groups = train_df["community_id"].fillna("unknown").astype(str)

    num_tuple = tuple(numeric_features)
    cat_tuple = tuple(categorical_features)

    for model_name, regressor in model_zoo.items():
        prefixed_name = f"{experiment_name}__{model_name}"
        pipeline = _build_pipeline_for_features(regressor, num_tuple, cat_tuple)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                started_at = time.perf_counter()
                cv_mae_mean, cv_mae_std, cv_mae_scores = _cross_validate_model(
                    pipeline, x_train, y_train, groups, config.cv_folds
                )
                pipeline.fit(x_train, y_train)
                train_seconds = time.perf_counter() - started_at

            predict_started_at = time.perf_counter()
            predictions_transformed = pipeline.predict(x_test)
            predict_seconds = time.perf_counter() - predict_started_at

            if use_log_target:
                predictions_original = np.expm1(predictions_transformed)
            else:
                predictions_original = predictions_transformed

            predictions_original = np.clip(predictions_original, a_min=1.0, a_max=None)

            metric = _evaluate_predictions(
                prefixed_name,
                pd.Series(y_test_original),
                predictions_original,
                cv_mae_mean,
                cv_mae_std,
            )
            metrics.append(metric)
            predictions_by_model[prefixed_name] = predictions_original

            # 逐样本预测落盘（仅 stacking_ridge，供配对显著性检验）
            if model_name == "stacking_ridge":
                pred_df = pd.DataFrame(
                    {
                        "house_id": test_df["house_id"].to_numpy(),
                        "actual_price_wan": y_test_original,
                        "predicted_price_wan": predictions_original,
                        "absolute_error_wan": np.abs(y_test_original - predictions_original),
                    }
                )
                pred_df.to_csv(
                    output_dir / f"predictions_{experiment_name}__stacking_ridge.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

            _append_experiment_log(
                output_dir,
                {
                    "event": "model_completed",
                    "experiment": experiment_name,
                    "model_name": model_name,
                    "prefixed_name": prefixed_name,
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
            _append_experiment_log(
                output_dir,
                {
                    "event": "model_failed",
                    "experiment": experiment_name,
                    "model_name": model_name,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            raise

    return metrics, predictions_by_model


def _write_semantic_comparison(feature_df: pd.DataFrame, output_dir: Path) -> None:
    """写入规则语义和 LLM 语义的分布对比。"""
    rows: list[dict[str, JsonValue]] = []
    overlap_cols = set(RULE_SEMANTIC_FEATURES) & set(LLM_SEMANTIC_LABELS)

    for feature_name in RULE_SEMANTIC_FEATURES:
        rule_col_name = f"{feature_name}_rule" if feature_name in overlap_cols else feature_name
        rule_col = feature_df[rule_col_name]
        if feature_name in LLM_SEMANTIC_LABELS:
            llm_col = feature_df[feature_name]
        else:
            llm_col = pd.Series(0, index=feature_df.index)
        rows.append(
            {
                "feature_name": feature_name,
                "rule_positive_count": int(rule_col.sum()),
                "rule_positive_rate": float(rule_col.mean()),
                "llm_positive_count": int(llm_col.sum()),
                "llm_positive_rate": float(llm_col.mean()),
                "agreement": float((rule_col == llm_col).mean()),
                "both_one": int(((rule_col == 1) & (llm_col == 1)).sum()),
                "rule_only": int(((rule_col == 1) & (llm_col == 0)).sum()),
                "llm_only": int(((rule_col == 0) & (llm_col == 1)).sum()),
                "both_zero": int(((rule_col == 0) & (llm_col == 0)).sum()),
            }
        )

    pd.DataFrame(rows).to_csv(
        output_dir / "semantic_feature_comparison.csv", index=False, encoding="utf-8-sig"
    )


def _write_stacking_comparison(output_dir: Path, all_metrics: list[ModelMetric]) -> None:
    """提取并写入 stacking_ridge 的对比表。"""
    rows: list[dict[str, JsonValue]] = []
    for metric in all_metrics:
        if "__stacking_ridge" not in metric.model_name:
            continue
        parts = metric.model_name.split("__")
        experiment_name = parts[0]
        rows.append(
            {
                "experiment": experiment_name,
                "model_name": metric.model_name,
                "mae": metric.mae,
                "rmse": metric.rmse,
                "r2": metric.r2,
                "mape": metric.mape,
                "cv_mae_mean": metric.cv_mae_mean,
                "cv_mae_std": metric.cv_mae_std,
                "delta_vs_new_baseline": metric.mae - NEW_BASELINE_MAE,
                "delta_vs_old_baseline": metric.mae - OLD_BASELINE_MAE,
            }
        )
    if rows:
        pd.DataFrame(rows).sort_values(by="mae").to_csv(
            output_dir / "stacking_comparison.csv", index=False, encoding="utf-8-sig"
        )


def _write_run_log(output_dir: Path, event: str, payload: dict[str, JsonValue]) -> None:
    log_record: dict[str, JsonValue] = {
        "event": event,
        "created_at": datetime.now().isoformat(),
        "payload": payload,
    }
    with (output_dir / "run_log.jsonl").open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(log_record, ensure_ascii=False) + "\n")


def _append_experiment_log(output_dir: Path, record: dict[str, JsonValue]) -> None:
    log_record: dict[str, JsonValue] = {
        "created_at": datetime.now().isoformat(),
        **record,
    }
    with (output_dir / "experiment_log.jsonl").open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(log_record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM 语义 vs 规则语义增量效果对比")
    parser.add_argument("--houses", required=True, help="房源 JSONL 路径")
    parser.add_argument("--communities", required=True, help="小区 JSONL 路径")
    parser.add_argument("--llm-features", required=True, help="LLM 语义特征 CSV 路径")
    parser.add_argument("--output-dir", required=True, help="研究产物根目录")
    parser.add_argument("--run-id", help="研究运行 ID")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--sample-limit", type=int, help="抽样上限（quick 模式）")
    parser.add_argument("--only", help="仅运行指定实验组（逗号分隔），用于增量补跑")
    args = parser.parse_args()

    run_id = args.run_id or f"llm_semantic_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    config = ResearchRunConfig(
        houses_path=Path(args.houses),
        communities_path=Path(args.communities),
        output_root=Path(args.output_dir),
        run_id=run_id,
        random_state=args.random_state,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        sample_limit=args.sample_limit,
    )
    result = run_llm_semantic_vs_rule_experiment(
        config=config,
        llm_features_path=Path(args.llm_features),
        only_experiments=[s.strip() for s in args.only.split(",")] if args.only else None,
    )


if __name__ == "__main__":
    main()
