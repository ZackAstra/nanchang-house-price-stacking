"""
任务 1.3：信息上限近似诊断

核心逻辑：
1. 房价理论方差分析：按 region_slug + house_layout + area_sqm(分箱) 分组，
   计算组内价格变异系数(CV)。CV 越高，说明特征缺口越大。
2. 社区均价基准：用 community_id 的挂牌均价做简单预测，其 MAE 作为"完美社区信息"的近似下限。
3. 特征边际递减分析：依次增加特征组（raw → base → stage3），观察 R² 的边际增益。
4. 当前最优模型 vs 理论上限差距：report 中明确给出 "当前 R² / 估计上限 R²" 的比例。

不引入复杂理论，轻量快速完成。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

from src.research.models import JsonValue, ResearchRunConfig
from src.research.pipeline import (
    BASE_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    _apply_sample_limit,
    _build_features,
    _build_pipeline_for_features,
    _clean_and_join,
    _load_communities,
    _load_houses,
    _split_by_time,
)

# 更新自 review5_log_transform / review5_combined_breakthrough 的 log 组 stacking_ridge（方案B中位数修复后重跑）
CURRENT_BEST_FULL_MAE = 20.552331586754455
CURRENT_BEST_FULL_R2 = 0.30586988768649703


def run_information_ceiling_diagnosis(config: ResearchRunConfig) -> Path:
    """执行信息上限近似诊断。"""
    output_dir = config.output_root / config.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 数据加载
    houses_df = _load_houses(config.houses_path)
    communities_df = _load_communities(config.communities_path)
    merged_df, audit = _clean_and_join(houses_df, communities_df)
    feature_df = _build_features(merged_df)
    feature_df = _apply_sample_limit(feature_df, config.sample_limit, config.random_state)
    train_df, test_df = _split_by_time(feature_df, config.test_size)

    report: dict[str, JsonValue] = {
        "run_id": config.run_id,
        "sample_count": int(len(feature_df)),
        "train_count": int(len(train_df)),
        "test_count": int(len(test_df)),
        "current_best_mae": CURRENT_BEST_FULL_MAE,
        "current_best_r2": CURRENT_BEST_FULL_R2,
    }

    # 2. 房价理论方差分析
    variance_analysis = _analyze_price_variance(feature_df)
    variance_analysis.to_csv(output_dir / "price_variance_analysis.csv", index=False, encoding="utf-8-sig")
    report["variance_analysis"] = _variance_summary(variance_analysis)

    # 3. 社区均价基准
    community_baseline = _community_mean_baseline(train_df, test_df)
    report["community_mean_baseline"] = community_baseline

    # 4. 特征边际递减分析
    marginal_analysis = _feature_marginal_analysis(train_df, test_df, config.random_state)
    marginal_analysis.to_csv(output_dir / "feature_marginal_analysis.csv", index=False, encoding="utf-8-sig")
    report["feature_marginal_analysis"] = marginal_analysis.to_dict(orient="records")

    # 5. 信息上限综合估计
    ceiling_estimate = _estimate_information_ceiling(report)
    report["estimated_r2_ceiling"] = ceiling_estimate

    # 6. 写入报告
    report_path = output_dir / "ceiling_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 同时写入文本摘要
    _write_text_summary(output_dir, report)

    print("=" * 60)
    print("Information Ceiling Diagnosis Complete")
    print(f"Run ID: {config.run_id}")
    print(f"Output: {output_dir}")
    print(f"Estimated R2 ceiling: {ceiling_estimate:.4f}")
    print(f"Current R2: {CURRENT_BEST_FULL_R2:.4f}")
    print(f"Gap: {ceiling_estimate - CURRENT_BEST_FULL_R2:.4f}")
    print("=" * 60)

    return output_dir


def _analyze_price_variance(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    按关键特征分组，分析组内价格变异系数。
    
    分组维度：region_slug + house_layout + area_sqm(分4箱)
    """
    df = feature_df.copy()
    df["area_bin"] = pd.qcut(df["area_sqm"], q=4, duplicates="drop").astype(str)
    df["group_key"] = df["region_slug"].astype(str) + "__" + df["house_layout"].astype(str) + "__" + df["area_bin"]

    rows: list[dict[str, JsonValue]] = []
    for group_key, group_df in df.groupby("group_key", dropna=False):
        if len(group_df) < 3:
            continue
        prices = group_df[TARGET_COLUMN].to_numpy(dtype=float)
        mean_price = float(np.mean(prices))
        std_price = float(np.std(prices))
        cv = std_price / mean_price if mean_price > 0 else 0.0
        rows.append(
            {
                "group_key": str(group_key),
                "sample_count": int(len(group_df)),
                "mean_price_wan": mean_price,
                "std_price_wan": std_price,
                "cv": cv,
                "min_price_wan": float(np.min(prices)),
                "max_price_wan": float(np.max(prices)),
                "median_price_wan": float(np.median(prices)),
            }
        )

    result_df = pd.DataFrame(rows).sort_values(by="sample_count", ascending=False)
    return result_df


def _variance_summary(variance_df: pd.DataFrame) -> dict[str, JsonValue]:
    """汇总变异系数分析结果。"""
    if len(variance_df) == 0:
        return {"group_count": 0, "message": "无有效分组"}

    cv_values = variance_df["cv"].to_numpy(dtype=float)
    return {
        "group_count": int(len(variance_df)),
        "total_samples_in_groups": int(variance_df["sample_count"].sum()),
        "mean_cv": float(np.mean(cv_values)),
        "median_cv": float(np.median(cv_values)),
        "max_cv": float(np.max(cv_values)),
        "min_cv": float(np.min(cv_values)),
        "groups_with_cv_above_0_15": int(np.sum(cv_values > 0.15)),
        "groups_with_cv_above_0_10": int(np.sum(cv_values > 0.10)),
        "groups_with_cv_above_0_20": int(np.sum(cv_values > 0.20)),
        "pct_groups_cv_above_0_15": float(np.mean(cv_values > 0.15)),
    }


def _community_mean_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, JsonValue]:
    """
    用社区均价做简单预测，评估"完美社区信息"的近似下限。
    
    逻辑：
    - 训练集中计算每个 community_id 的均价
    - 测试集中，用对应 community_id 的均价预测；若社区未在训练集中出现，用全局均值
    """
    community_means = train_df.groupby("community_id")[TARGET_COLUMN].mean().to_dict()
    global_mean = float(train_df[TARGET_COLUMN].mean())

    predictions: list[float] = []
    for _, row in test_df.iterrows():
        community_id = row["community_id"]
        pred = community_means.get(community_id, global_mean)
        predictions.append(pred)

    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=float)
    preds = np.array(predictions, dtype=float)

    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    return {
        "mae": mae,
        "r2": r2,
        "global_mean": global_mean,
        "community_with_mean_count": len(community_means),
        "test_community_covered_count": int(
            sum(1 for cid in test_df["community_id"] if cid in community_means)
        ),
        "test_community_coverage_rate": float(
            sum(1 for cid in test_df["community_id"] if cid in community_means) / max(len(test_df), 1)
        ),
    }


def _feature_marginal_analysis(
    train_df: pd.DataFrame, test_df: pd.DataFrame, random_state: int
) -> pd.DataFrame:
    """
    特征边际递减分析：依次增加特征组，观察 R² 的边际增益。
    
    特征组：
    - raw: area_sqm, room_count, hall_count, bath_count, total_floors + region_slug, decoration_type, orientation, floor_level, has_elevator, is_unique_housing
    - base: raw + poi counts + accessibility_index + area_per_room + layout_density + floor_level_score
    - stage3: base + poi_balance_score, transit_medical_ratio, ...
    """
    feature_sets = [
        ("raw", (
            ("area_sqm", "room_count", "hall_count", "bath_count", "total_floors"),
            ("region_slug", "decoration_type", "orientation", "floor_level", "has_elevator", "is_unique_housing"),
        )),
        ("base", (BASE_NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
        ("stage3", (NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
    ]

    rows: list[dict[str, JsonValue]] = []
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    for set_name, (numeric_features, categorical_features) in feature_sets:
        x_train = train_df[list(numeric_features) + list(categorical_features)]
        x_test = test_df[list(numeric_features) + list(categorical_features)]

        pipeline = _build_pipeline_for_features(
            HistGradientBoostingRegressor(
                max_iter=240,
                learning_rate=0.055,
                l2_regularization=0.05,
                random_state=random_state,
            ),
            numeric_features,
            categorical_features,
        )
        pipeline.fit(x_train, train_df[TARGET_COLUMN])
        predictions = pipeline.predict(x_test)
        mae = float(mean_absolute_error(y_test, predictions))
        r2 = float(r2_score(y_test, predictions))

        rows.append(
            {
                "feature_set": set_name,
                "numeric_feature_count": len(numeric_features),
                "categorical_feature_count": len(categorical_features),
                "mae": mae,
                "r2": r2,
                "delta_r2_vs_raw": 0.0 if set_name == "raw" else None,
                "delta_mae_vs_raw": 0.0 if set_name == "raw" else None,
            }
        )

    result_df = pd.DataFrame(rows)
    raw_r2 = float(result_df.loc[result_df["feature_set"] == "raw", "r2"].iloc[0])
    raw_mae = float(result_df.loc[result_df["feature_set"] == "raw", "mae"].iloc[0])
    result_df["delta_r2_vs_raw"] = result_df["r2"] - raw_r2
    result_df["delta_mae_vs_raw"] = result_df["mae"] - raw_mae
    return result_df


def _estimate_information_ceiling(report: dict[str, JsonValue]) -> float:
    """
    综合估计信息上限 R²。

    方法（保守估计）：
    1. 基于组内 CV：CV 中位数 = 0.21 → 即使完美分组，R² 上限 ≈ 1 - CV² ≈ 0.96
       （但这是一个宽松的上界，实际不可能达到）
    2. 基于特征边际递减外推：
       - raw → base: ΔR² = +0.35
       - base → stage3: ΔR² = +0.13
       - stage3 → stacking_ridge: ΔR² = +0.11
       - 假设未来每轮特征工程增量为上一轮的一半
       - 再假设 stacking 集成方法的增益也会边际递减
    3. 取两种方法的较保守值
    """
    current_r2 = CURRENT_BEST_FULL_R2
    variance_summary = cast(dict, report.get("variance_analysis", {}))
    median_cv = float(variance_summary.get("median_cv", 0.0))

    # 方法1: 基于组内 CV 的理论上限（非常宽松）
    cv_based_ceiling = max(0.0, 1.0 - median_cv ** 2)

    # 方法2: 基于特征边际递减的外推
    marginal = cast(list, report.get("feature_marginal_analysis", []))
    stage3_r2 = next((r["r2"] for r in marginal if r["feature_set"] == "stage3"), current_r2)
    base_r2 = next((r["r2"] for r in marginal if r["feature_set"] == "base"), 0.0)

    # 最近一轮特征工程增量 (base → stage3)
    recent_feature_delta = stage3_r2 - base_r2
    # 集成方法增益 (stage3 hist_gb → stacking_ridge)
    ensemble_delta = current_r2 - stage3_r2

    # 外推：假设还能添加 1-2 轮新特征，每轮增量递减
    # 未来特征增益 = recent_delta * (0.5 + 0.25) = recent_delta * 0.75
    future_feature_gain = recent_feature_delta * 0.75
    # 未来集成增益 = ensemble_delta * 0.3（更保守）
    future_ensemble_gain = ensemble_delta * 0.3
    extrapolated_r2 = current_r2 + future_feature_gain + future_ensemble_gain

    # 综合：取 CV 上限和外推上限的较低值，并加上惩罚项
    # 如果大量分组 CV > 0.15，说明信息缺口大，上限应更保守
    pct_high_cv = float(variance_summary.get("pct_groups_cv_above_0_15", 0.0))
    cv_penalty = pct_high_cv * 0.05  # 73% 高 CV 分组 → 约 0.037 惩罚

    ceiling = min(cv_based_ceiling, extrapolated_r2) - cv_penalty

    # 确保 ceiling 在合理范围内
    ceiling = max(ceiling, current_r2 + 0.02)  # 至少还有 2% 提升空间
    ceiling = min(ceiling, 0.70)  # 房价预测实际极限约 0.6-0.7

    return float(ceiling)


def _write_text_summary(output_dir: Path, report: dict[str, JsonValue]) -> None:
    """写入人类可读的文本摘要。"""
    variance = cast(dict, report.get("variance_analysis", {}))
    community = cast(dict, report.get("community_mean_baseline", {}))
    marginal = cast(list, report.get("feature_marginal_analysis", []))
    ceiling = float(report.get("estimated_r2_ceiling", 0.0))

    lines = [
        "信息上限近似诊断报告",
        "=" * 60,
        f"运行ID: {report.get('run_id')}",
        f"样本量: {report.get('sample_count')} (训练 {report.get('train_count')}, 测试 {report.get('test_count')})",
        "",
        "当前最优模型基准:",
        f"  MAE: {CURRENT_BEST_FULL_MAE:.4f} 万元",
        f"  R²:  {CURRENT_BEST_FULL_R2:.4f}",
        "",
        "房价理论方差分析 (region + layout + area_bin 分组):",
        f"  有效分组数: {variance.get('group_count', 'N/A')}",
        f"  组内价格变异系数(CV)中位数: {variance.get('median_cv', 'N/A'):.4f}",
        f"  组内 CV > 0.15 的分组占比: {variance.get('pct_groups_cv_above_0_15', 'N/A'):.2%}",
        "  解读: CV 越高，说明即使完美知道区域+户型+面积，仍有大量不可解释方差",
        "",
        "社区均价基准 (用社区均价预测测试集):",
        f"  MAE: {community.get('mae', 'N/A'):.4f} 万元",
        f"  R²:  {community.get('r2', 'N/A'):.4f}",
        f"  测试集社区覆盖率: {community.get('test_community_coverage_rate', 'N/A'):.2%}",
        "  解读: 这是'完美社区信息'的近似下限，当前模型与它的差距反映信息缺口",
        "",
        "特征边际递减分析:",
    ]
    for row in marginal:
        lines.append(
            f"  {row['feature_set']}: MAE={row['mae']:.4f}, R²={row['r2']:.4f}, "
            f"ΔR²(vs raw)={row['delta_r2_vs_raw']:.4f}"
        )
    lines.extend([
        "",
        "信息上限综合估计:",
        f"  估计 R² 上限: {ceiling:.4f}",
        f"  当前 R²:     {CURRENT_BEST_FULL_R2:.4f}",
        f"  可提升空间:  {ceiling - CURRENT_BEST_FULL_R2:.4f} ({(ceiling - CURRENT_BEST_FULL_R2) / max(1 - CURRENT_BEST_FULL_R2, 0.001) * 100:.1f}% 剩余空间)",
        "",
        "诊断结论:",
    ])

    if ceiling - CURRENT_BEST_FULL_R2 < 0.05:
        lines.append("  当前模型已接近信息上限，继续优化算法收益有限。")
        lines.append("  建议：转向数据采集（新增月份、扩大区域、采集成交价替代挂牌价）。")
    elif ceiling - CURRENT_BEST_FULL_R2 < 0.15:
        lines.append("  当前模型有一定提升空间，但边际递减明显。")
        lines.append("  建议：尝试特征增强（LLM语义提取等），若无效则转向数据采集。")
    else:
        lines.append("  当前模型远未触及信息上限，特征/数据层面有显著优化空间。")
        lines.append("  建议：优先投入特征工程和数据质量提升。")

    lines.append("=" * 60)
    (output_dir / "ceiling_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="信息上限近似诊断")
    parser.add_argument("--houses", required=True, help="房源JSONL路径")
    parser.add_argument("--communities", required=True, help="小区JSONL路径")
    parser.add_argument("--output-dir", required=True, help="研究产物根目录")
    parser.add_argument("--run-id", help="研究运行ID，不传则自动生成")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    parser.add_argument("--test-size", type=float, default=0.2, help="留出集比例")
    parser.add_argument("--sample-limit", type=int, help="抽样上限")
    args = parser.parse_args()

    run_id = args.run_id or f"ceiling_diagnosis_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    config = ResearchRunConfig(
        houses_path=Path(args.houses),
        communities_path=Path(args.communities),
        output_root=Path(args.output_dir),
        run_id=run_id,
        random_state=args.random_state,
        test_size=args.test_size,
        cv_folds=3,
        sample_limit=args.sample_limit,
    )
    run_information_ceiling_diagnosis(config)


if __name__ == "__main__":
    main()
