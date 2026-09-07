"""
任务 2.1：LLM/规则语义特征工程——从房源标题提取高阶语义

核心逻辑：
1. 复用 pipeline 的数据加载、清洗、特征工程
2. 用关键词词典从 title 中提取语义标签（规则匹配，无需 LLM API）
3. 将语义标签转为 OneHot 特征，加入特征工程管线
4. 对比实验：原始特征 vs 原始+语义特征 vs 仅语义特征
5. 所有评估统一在万元原始尺度上计算 MAE

语义标签设计（基于标题样本观察）：
- 交通: subway, bus_hub
- 教育: school_district, school_nearby
- 景观: river_view, park_nearby, high_floor_view
- 装修: luxury_decor, simple_decor, rough
- 品质: quality_community, new_community, old_community
- 急售: urgent_sale, below_market, sincere
- 楼层/采光: good_lighting, good_floor, elevator
- 配套: shopping, garage
- 其他: direct_seller, price_negotiable, low_density, duplex, garden, furnished, double_bath
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
    _write_metrics,
)

CURRENT_BEST_FULL_MAE = 20.911922387801575

# 语义关键词词典
TITLE_SEMANTIC_KEYWORDS: dict[str, list[str]] = {
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

SEMANTIC_FEATURES = tuple(sorted(TITLE_SEMANTIC_KEYWORDS.keys()))


def run_title_semantic_experiment(config: ResearchRunConfig) -> ResearchResult:
    """执行标题语义特征工程实验。"""
    output_dir = config.output_root / config.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_run_log(output_dir, "run_started", {"run_id": config.run_id, "experiment": "title_semantic"})

    # 1. 数据加载与清洗
    houses_df = _load_houses(config.houses_path)
    communities_df = _load_communities(config.communities_path)
    merged_df, audit = _clean_and_join(houses_df, communities_df)
    feature_df = _build_features(merged_df)
    feature_df = _apply_sample_limit(feature_df, config.sample_limit, config.random_state)

    # 2. 提取语义特征
    # 需要从原始 merged_df 中获取 title（因为 _build_features 会丢弃 title）
    # 所以我们需要在 merged_df 上做语义提取，然后 join 到 feature_df
    semantic_df = _extract_semantic_features(merged_df)
    # 确保 house_id 对齐
    feature_df = feature_df.merge(semantic_df, on="house_id", how="left")
    for col in SEMANTIC_FEATURES:
        feature_df[col] = feature_df[col].fillna(0).astype(int)

    # 3. 写入语义特征分布
    semantic_summary = _build_semantic_summary(feature_df)
    semantic_summary.to_csv(output_dir / "semantic_feature_distribution.csv", index=False, encoding="utf-8-sig")

    # 4. 划分训练/测试集
    train_df, test_df = _split_by_time(feature_df, config.test_size)

    # 5. 定义实验配置
    experiments = _build_experiment_configs()

    # quick 模式下只跑 original 和 semantic_enhanced
    is_quick_mode = config.sample_limit is not None
    if is_quick_mode:
        experiments = [e for e in experiments if str(e["name"]) in {"original", "semantic_enhanced"}]

    all_metrics: list[ModelMetric] = []
    summary_rows: list[dict[str, JsonValue]] = []

    for exp_config in experiments:
        exp_name = str(exp_config["name"])
        numeric_features = cast(tuple[str, ...], exp_config["numeric_features"])
        categorical_features = cast(tuple[str, ...], exp_config["categorical_features"])

        _write_run_log(
            output_dir,
            "experiment_started",
            {
                "experiment_name": exp_name,
                "numeric_feature_count": len(numeric_features),
                "categorical_feature_count": len(categorical_features),
                "semantic_feature_count": len(SEMANTIC_FEATURES) if "semantic" in exp_name else 0,
            },
        )

        metrics, predictions_by_model = _run_semantic_variant(
            train_df=train_df,
            test_df=test_df,
            config=config,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            output_dir=output_dir,
            experiment_name=exp_name,
            quick_mode=is_quick_mode,
        )

        all_metrics.extend(metrics)

        best_metric = min(metrics, key=lambda m: m.mae)
        summary_rows.append(
            {
                "experiment_name": exp_name,
                "best_model_name": best_metric.model_name.split("__")[-1],
                "best_mae": best_metric.mae,
                "best_rmse": best_metric.rmse,
                "best_r2": best_metric.r2,
                "best_mape": best_metric.mape,
                "delta_mae_vs_baseline": best_metric.mae - CURRENT_BEST_FULL_MAE,
            }
        )

    # 6. 写入产物
    _write_metrics(output_dir, all_metrics)
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "semantic_experiment_summary.csv", index=False, encoding="utf-8-sig"
    )
    _write_stacking_comparison(output_dir, all_metrics)

    best_overall = min(all_metrics, key=lambda m: m.mae)
    _write_run_log(
        output_dir,
        "run_completed",
        {
            "best_experiment_model": best_overall.model_name,
            "best_mae": best_overall.mae,
            "baseline_mae": CURRENT_BEST_FULL_MAE,
            "delta_vs_baseline": best_overall.mae - CURRENT_BEST_FULL_MAE,
        },
    )

    return ResearchResult(
        run_id=config.run_id,
        output_dir=output_dir,
        audit=audit,
        metrics=all_metrics,
        best_model_name=best_overall.model_name,
    )


def _extract_semantic_features(merged_df: pd.DataFrame) -> pd.DataFrame:
    """从房源标题中提取语义特征。"""
    df = merged_df[["house_id", "title"]].copy()
    df["title"] = df["title"].fillna("").astype(str)

    for feature_name, keywords in TITLE_SEMANTIC_KEYWORDS.items():
        df[feature_name] = df["title"].apply(
            lambda t: int(any(kw in t for kw in keywords))
        )

    # 返回 house_id + 语义特征
    return df[["house_id"] + list(SEMANTIC_FEATURES)]


def _build_semantic_summary(feature_df: pd.DataFrame) -> pd.DataFrame:
    """构建语义特征的分布摘要。"""
    rows: list[dict[str, JsonValue]] = []
    for feature_name in SEMANTIC_FEATURES:
        col = feature_df[feature_name]
        rows.append(
            {
                "feature_name": feature_name,
                "positive_count": int(col.sum()),
                "positive_rate": float(col.mean()),
                "keywords": "|".join(TITLE_SEMANTIC_KEYWORDS[feature_name]),
            }
        )
    return pd.DataFrame(rows).sort_values(by="positive_count", ascending=False)


def _build_experiment_configs() -> list[dict[str, JsonValue | tuple[str, ...]]]:
    """构建实验配置。"""
    original_numeric = NUMERIC_FEATURES
    semantic_numeric = NUMERIC_FEATURES + SEMANTIC_FEATURES

    return [
        {
            "name": "original",
            "numeric_features": original_numeric,
            "categorical_features": CATEGORICAL_FEATURES,
        },
        {
            "name": "semantic_enhanced",
            "numeric_features": semantic_numeric,
            "categorical_features": CATEGORICAL_FEATURES,
        },
        {
            "name": "semantic_only",
            "numeric_features": SEMANTIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
        },
    ]


def _run_semantic_variant(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ResearchRunConfig,
    numeric_features: tuple[str, ...],
    categorical_features: tuple[str, ...],
    output_dir: Path,
    experiment_name: str,
    quick_mode: bool = False,
) -> tuple[list[ModelMetric], dict[str, np.ndarray]]:
    """执行单个实验变体的训练和评估。"""
    x_train = train_df[list(numeric_features) + list(categorical_features)]
    y_train = train_df[TARGET_COLUMN]
    x_test = test_df[list(numeric_features) + list(categorical_features)]
    y_test = test_df[TARGET_COLUMN]

    model_zoo = _build_model_zoo(config.random_state)
    if quick_mode:
        model_zoo = {k: v for k, v in model_zoo.items() if k == "stacking_ridge"}

    metrics: list[ModelMetric] = []
    predictions_by_model: dict[str, np.ndarray] = {}
    groups = train_df["community_id"].fillna("unknown").astype(str)

    for model_name, regressor in model_zoo.items():
        prefixed_name = f"{experiment_name}__{model_name}"
        pipeline = _build_pipeline_for_features(regressor, numeric_features, categorical_features)
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
            predictions = pipeline.predict(x_test)
            predict_seconds = time.perf_counter() - predict_started_at

            predictions = np.clip(predictions, a_min=1.0, a_max=None)

            metric = _evaluate_predictions(
                prefixed_name,
                y_test,
                predictions,
                cv_mae_mean,
                cv_mae_std,
            )
            metrics.append(metric)
            predictions_by_model[prefixed_name] = predictions

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
                "delta_vs_baseline": metric.mae - CURRENT_BEST_FULL_MAE,
            }
        )
    if rows:
        comparison_df = pd.DataFrame(rows).sort_values(by="mae")
        comparison_df.to_csv(
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
    parser = argparse.ArgumentParser(description="标题语义特征工程实验")
    parser.add_argument("--houses", required=True, help="房源JSONL路径")
    parser.add_argument("--communities", required=True, help="小区JSONL路径")
    parser.add_argument("--output-dir", required=True, help="研究产物根目录")
    parser.add_argument("--run-id", help="研究运行ID，不传则自动生成")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    parser.add_argument("--test-size", type=float, default=0.2, help="留出集比例")
    parser.add_argument("--cv-folds", type=int, default=3, help="交叉验证折数")
    parser.add_argument("--sample-limit", type=int, help="抽样上限，用于快速验证")
    args = parser.parse_args()

    run_id = args.run_id or f"title_semantic_{datetime.now().strftime('%Y%m%d%H%M%S')}"
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
    result = run_title_semantic_experiment(config)
    print("=" * 60)
    print("Title Semantic Feature Experiment Complete")
    print(f"Run ID: {result.run_id}")
    print(f"Output: {result.output_dir}")
    print(f"Best model: {result.best_model_name}")
    best_metric = min(result.metrics, key=lambda m: m.mae)
    print(f"Best MAE: {best_metric.mae:.4f}")
    print(f"Baseline MAE: {CURRENT_BEST_FULL_MAE:.4f}")
    print(f"Delta: {best_metric.mae - CURRENT_BEST_FULL_MAE:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
