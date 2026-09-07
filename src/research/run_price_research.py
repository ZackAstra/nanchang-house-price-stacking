from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.research.models import ResearchRunConfig
from src.research.pipeline import run_price_research


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="南昌房价规律算法研究入口")
    parser.add_argument("--houses", required=True, help="房源JSONL路径")
    parser.add_argument("--communities", required=True, help="小区JSONL路径")
    parser.add_argument("--output-dir", required=True, help="研究产物根目录")
    parser.add_argument("--run-id", help="研究运行ID，不传则自动生成")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    parser.add_argument("--test-size", type=float, default=0.2, help="留出集比例")
    parser.add_argument("--cv-folds", type=int, default=5, help="交叉验证折数")
    parser.add_argument("--split-mode", choices=["time", "random"], default="time", help="划分方式（random 为稳健性对照）")
    parser.add_argument("--sample-limit", type=int, help="抽样上限，用于快速验证")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or f"price_research_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    config = ResearchRunConfig(
        houses_path=Path(args.houses),
        communities_path=Path(args.communities),
        output_root=Path(args.output_dir),
        run_id=run_id,
        random_state=args.random_state,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        sample_limit=args.sample_limit,
        split_mode=args.split_mode,
    )
    result = run_price_research(config)
    best_metric = min(result.metrics, key=lambda item: item.mae)

    print("=" * 60)
    print("南昌房价算法研究完成")
    print(f"运行ID: {result.run_id}")
    print(f"输出目录: {result.output_dir}")
    print(f"可用样本数: {result.audit.usable_house_count}")
    print(f"小区关联成功率: {result.audit.community_join_rate:.4f}")
    print(f"最优模型: {result.best_model_name}")
    print(f"MAE: {best_metric.mae:.4f}")
    print(f"RMSE: {best_metric.rmse:.4f}")
    print(f"R2: {best_metric.r2:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

