"""
任务 1.2：标签噪声检测与清洗

核心逻辑：
1. 复用 pipeline 的数据加载、清洗、特征工程
2. 在特征工程后的数据上，用三种独立方法检测噪声样本：
   - 单价异常：按 region_slug + house_layout 分组，标记超出组内 1%-99% 分位的样本
   - 总价异常：全局 total_price_wan 超出 0.5%-99.5% 分位（比当前 1%-99% 更严格）
   - 面积异常：全局 area_sqm 超出 0.5%-99.5% 分位
3. 组合两种清洗策略：
   - strict：单价正常 AND 总价正常 AND 面积正常
   - moderate：单价正常 OR 总价正常（至少一项正常）
4. 先按时间划分固定训练/测试集，噪声检测与清洗只作用于训练集，
   所有策略共用同一固定测试集，训练 stacking_ridge
5. 与未清洗的原始数据做横向对比（统一在原始万元尺度上评估 MAE）

评审修复协议（review2）：
1. 全量特征表先按时间划分训练/测试集，所有实验共用同一固定测试集；
2. 标签噪声检测只在训练集上运行，分位数边界仅由训练集估计；
3. 清洗策略只删除训练集样本，测试集保持不动。
"""

from __future__ import annotations

import argparse
import json
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

CURRENT_BEST_FULL_MAE = 20.911922387801575


def run_label_noise_experiment(config: ResearchRunConfig) -> ResearchResult:
    """执行标签噪声检测与清洗实验。"""
    output_dir = config.output_root / config.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_run_log(output_dir, "run_started", {"run_id": config.run_id, "experiment": "label_noise"})

    # 1. 数据加载与清洗
    houses_df = _load_houses(config.houses_path)
    communities_df = _load_communities(config.communities_path)
    merged_df, audit = _clean_and_join(houses_df, communities_df)
    feature_df = _build_features(merged_df)
    feature_df = _apply_sample_limit(feature_df, config.sample_limit, config.random_state)

    # 2. 先按时间划分固定训练/测试集（所有策略共用同一测试集）
    train_full_df, fixed_test_df = _split_by_time(feature_df, config.test_size)
    train_full_df, fixed_test_df = _apply_reference_group_medians(train_full_df, fixed_test_df, _compute_region_ai_medians(config.communities_path))

    # 3. 噪声检测仅作用于训练集（分位数边界仅由训练集估计）
    noise_report = _detect_label_noise(train_full_df)
    noise_report.to_csv(output_dir / "noise_detection_report.csv", index=False, encoding="utf-8-sig")

    # 4. 构建清洗策略配置（keep_mask 基于训练集索引）
    cleaning_configs = _build_cleaning_configs(noise_report)

    all_metrics: list[ModelMetric] = []
    summary_rows: list[dict[str, JsonValue]] = []

    # quick 模式下只跑最关键的 3 种策略和 stacking_ridge 单模型
    is_quick_mode = config.sample_limit is not None
    if is_quick_mode:
        cleaning_configs = [c for c in cleaning_configs if str(c["name"]) in {
            "original_no_cleaning", "strict_cleaning", "unit_price_only_cleaning"
        }]

    for clean_config in cleaning_configs:
        clean_name = str(clean_config["name"])
        keep_mask = cast(pd.Series, clean_config["keep_mask"])

        # 清洗只作用于训练集，测试集固定不动
        train_df = train_full_df.loc[keep_mask].copy()
        test_df = fixed_test_df

        _write_run_log(
            output_dir,
            "cleaning_started",
            {
                "clean_name": clean_name,
                "train_full_samples": int(len(train_full_df)),
                "train_samples": int(len(train_df)),
                "removed_samples": int(len(train_full_df) - len(train_df)),
                "removal_rate": float(1.0 - len(train_df) / len(train_full_df)),
                "test_samples": int(len(test_df)),
                "quick_mode": is_quick_mode,
            },
        )

        metrics, predictions_by_model = _run_cleaned_variant(
            train_df=train_df,
            test_df=test_df,
            config=config,
            output_dir=output_dir,
            experiment_name=clean_name,
            quick_mode=is_quick_mode,
        )

        all_metrics.extend(metrics)

        best_metric = min(metrics, key=lambda m: m.mae)
        summary_rows.append(
            {
                "experiment_name": clean_name,
                "train_full_samples": int(len(train_full_df)),
                "kept_samples": int(len(train_df)),
                "removed_samples": int(len(train_full_df) - len(train_df)),
                "removal_rate": float(1.0 - len(train_df) / len(train_full_df)),
                "test_samples": int(len(test_df)),
                "best_model_name": best_metric.model_name.split("__")[-1],
                "best_mae": best_metric.mae,
                "best_rmse": best_metric.rmse,
                "best_r2": best_metric.r2,
                "best_mape": best_metric.mape,
                "delta_mae_vs_baseline": best_metric.mae - CURRENT_BEST_FULL_MAE,
            }
        )

    # 5. 写入产物
    _write_metrics(output_dir, all_metrics)
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "label_noise_summary.csv", index=False, encoding="utf-8-sig"
    )
    _write_stacking_comparison(output_dir, all_metrics)
    _write_noise_sample_list(output_dir, noise_report, cleaning_configs)

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


def _detect_label_noise(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    检测标签噪声样本。

    返回 DataFrame，包含每行的噪声标记。
    """
    df = feature_df.copy()
    df["unit_price"] = df[TARGET_COLUMN] / df["area_sqm"].clip(lower=1.0)

    # 1. 单价异常检测（按 region_slug + house_layout 分组）
    group_keys = df["region_slug"].astype(str) + "__" + df["house_layout"].astype(str)
    df["unit_price_group"] = group_keys

    unit_price_low = df.groupby("unit_price_group")["unit_price"].transform(lambda x: x.quantile(0.01))
    unit_price_high = df.groupby("unit_price_group")["unit_price"].transform(lambda x: x.quantile(0.99))
    # 对于样本量太小的组（<10），使用全局分位
    group_counts = df.groupby("unit_price_group")["unit_price"].transform("count")
    global_price_low = df["unit_price"].quantile(0.01)
    global_price_high = df["unit_price"].quantile(0.99)

    df["unit_price_low"] = np.where(group_counts >= 10, unit_price_low, global_price_low)
    df["unit_price_high"] = np.where(group_counts >= 10, unit_price_high, global_price_high)
    df["is_unit_price_outlier"] = (df["unit_price"] < df["unit_price_low"]) | (df["unit_price"] > df["unit_price_high"])

    # 2. 总价异常检测（全局 0.5%-99.5% 分位，比当前 1%-99% 更严格）
    total_price_low = df[TARGET_COLUMN].quantile(0.005)
    total_price_high = df[TARGET_COLUMN].quantile(0.995)
    df["is_total_price_outlier"] = (df[TARGET_COLUMN] < total_price_low) | (df[TARGET_COLUMN] > total_price_high)

    # 3. 面积异常检测（全局 0.5%-99.5% 分位）
    area_low = df["area_sqm"].quantile(0.005)
    area_high = df["area_sqm"].quantile(0.995)
    df["is_area_outlier"] = (df["area_sqm"] < area_low) | (df["area_sqm"] > area_high)

    # 4. 组合标记
    df["is_noise_strict"] = df["is_unit_price_outlier"] | df["is_total_price_outlier"] | df["is_area_outlier"]
    df["is_noise_moderate"] = df["is_unit_price_outlier"] & df["is_total_price_outlier"]

    # 选择输出列
    report_df = df[
        [
            "house_id",
            "community_id",
            "region_slug",
            "house_layout",
            TARGET_COLUMN,
            "area_sqm",
            "unit_price",
            "is_unit_price_outlier",
            "is_total_price_outlier",
            "is_area_outlier",
            "is_noise_strict",
            "is_noise_moderate",
        ]
    ].copy()

    return report_df


def _build_cleaning_configs(
    noise_report: pd.DataFrame,
) -> list[dict[str, JsonValue | pd.Series]]:
    """构建清洗策略配置。"""
    original_mask = pd.Series(True, index=noise_report.index)
    strict_mask = ~noise_report["is_noise_strict"]
    moderate_mask = ~noise_report["is_noise_moderate"]
    # 仅单价清洗
    unit_price_only_mask = ~noise_report["is_unit_price_outlier"]
    # 仅总价清洗
    total_price_only_mask = ~noise_report["is_total_price_outlier"]

    return [
        {
            "name": "original_no_cleaning",
            "keep_mask": original_mask,
        },
        {
            "name": "strict_cleaning",
            "keep_mask": strict_mask,
        },
        {
            "name": "moderate_cleaning",
            "keep_mask": moderate_mask,
        },
        {
            "name": "unit_price_only_cleaning",
            "keep_mask": unit_price_only_mask,
        },
        {
            "name": "total_price_only_cleaning",
            "keep_mask": total_price_only_mask,
        },
    ]


def _run_cleaned_variant(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ResearchRunConfig,
    output_dir: Path,
    experiment_name: str,
    quick_mode: bool = False,
) -> tuple[list[ModelMetric], dict[str, np.ndarray]]:
    """在清洗后的数据上训练评估。"""
    x_train = train_df[list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)]
    y_train = train_df[TARGET_COLUMN]
    x_test = test_df[list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)]
    y_test = test_df[TARGET_COLUMN]

    model_zoo = _build_model_zoo(config.random_state)
    # quick 模式下只跑 stacking_ridge
    if quick_mode:
        model_zoo = {k: v for k, v in model_zoo.items() if k == "stacking_ridge"}

    metrics: list[ModelMetric] = []
    predictions_by_model: dict[str, np.ndarray] = {}
    groups = train_df["community_id"].fillna("unknown").astype(str)

    for model_name, regressor in model_zoo.items():
        prefixed_name = f"{experiment_name}__{model_name}"
        pipeline = _build_pipeline_for_features(regressor, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
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


def _write_noise_sample_list(
    output_dir: Path,
    noise_report: pd.DataFrame,
    cleaning_configs: list[dict[str, JsonValue | pd.Series]],
) -> None:
    """写入噪声样本清单及清洗策略覆盖情况。"""
    # 标记每个清洗策略移除了哪些样本
    for clean_config in cleaning_configs:
        clean_name = str(clean_config["name"])
        keep_mask = cast(pd.Series, clean_config["keep_mask"])
        noise_report[f"kept_by_{clean_name}"] = keep_mask.reindex(noise_report.index).fillna(False)

    noise_report.to_csv(output_dir / "noise_sample_detail.csv", index=False, encoding="utf-8-sig")

    # 统计摘要
    summary_rows: list[dict[str, JsonValue]] = []
    for clean_config in cleaning_configs:
        clean_name = str(clean_config["name"])
        keep_mask = cast(pd.Series, clean_config["keep_mask"])
        removed_count = int((~keep_mask).sum())
        removal_rate = float((~keep_mask).mean())
        summary_rows.append(
            {
                "clean_name": clean_name,
                "total_samples": int(len(noise_report)),
                "removed_samples": removed_count,
                "removal_rate": removal_rate,
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "cleaning_strategy_summary.csv", index=False, encoding="utf-8-sig"
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
    parser = argparse.ArgumentParser(description="标签噪声检测与清洗实验")
    parser.add_argument("--houses", required=True, help="房源JSONL路径")
    parser.add_argument("--communities", required=True, help="小区JSONL路径")
    parser.add_argument("--output-dir", required=True, help="研究产物根目录")
    parser.add_argument("--run-id", help="研究运行ID，不传则自动生成")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    parser.add_argument("--test-size", type=float, default=0.2, help="留出集比例")
    parser.add_argument("--cv-folds", type=int, default=3, help="交叉验证折数")
    parser.add_argument("--sample-limit", type=int, help="抽样上限，用于快速验证")
    args = parser.parse_args()

    run_id = args.run_id or f"label_noise_{datetime.now().strftime('%Y%m%d%H%M%S')}"
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
    result = run_label_noise_experiment(config)
    print("=" * 60)
    print("标签噪声检测与清洗实验完成")
    print(f"运行ID: {result.run_id}")
    print(f"输出目录: {result.output_dir}")
    print(f"最优模型: {result.best_model_name}")
    best_metric = min(result.metrics, key=lambda m: m.mae)
    print(f"最优MAE: {best_metric.mae:.4f}")
    print(f"基准MAE: {CURRENT_BEST_FULL_MAE:.4f}")
    print(f"差值: {best_metric.mae - CURRENT_BEST_FULL_MAE:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
