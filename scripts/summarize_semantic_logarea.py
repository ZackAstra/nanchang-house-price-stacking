# -*- coding: utf-8 -*-
"""汇总 log-area × 规则语义 与 log × 规则语义 的逐种子配对对比（论文 Table 17 log-area 行）。

输入（随仓库发布）：
    data/runs/semantic_logarea_seed{s}/experiment_summary.csv       行 log_area_rule_semantic
    data/runs/semantic_deepseek/experiment_summary.csv              s=42 的 log_rule_semantic
    data/runs/semantic_deepseek_seed_{s}/experiment_summary.csv     s=7/84/168 的 log_rule_semantic
    data/runs/semantic_logarea_seed{s}/predictions_log_area_rule_semantic__stacking_ridge.csv
    data/runs/semantic_deepseek/predictions/predictions_log_rule_semantic__stacking_ridge.csv   (s=42)
    data/runs/semantic_deepseek_seed_{s}/predictions_log_rule_semantic__stacking_ridge.csv      (s=7/84/168)
输出：data/expected_outputs/semantic_logarea_summary.csv
    （seed,log_area_rule_mae,log_area_rule_r2,log_rule_mae,log_rule_r2,delta_mae,paired_t_p）

规则：MAE/R²/Δ 保留 4 位小数；配对 t 检验作用于两条流水线的逐样本 absolute_error_wan；
      p 值以 %.2e 输出（与已发布产物逐字符一致）。

用法：uv run python scripts/summarize_semantic_logarea.py
"""
import os

import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "data", "runs")
OUT_PATH = os.path.join(ROOT, "data", "expected_outputs", "semantic_logarea_summary.csv")

SEEDS = [42, 7, 84, 168]


def _logarea_dir(seed: int) -> str:
    return os.path.join(RUNS, "semantic_logarea_seed" + str(seed))


def _logrule_dir(seed: int) -> str:
    if seed == 42:
        return os.path.join(RUNS, "semantic_deepseek")
    return os.path.join(RUNS, "semantic_deepseek_seed_" + str(seed))


def _logrule_pred_path(seed: int) -> str:
    # 发布版把 seed 42 的逐实验预测放在 predictions/ 子目录；重跑 runner 会写在目录根下。
    candidates = []
    if seed == 42:
        candidates.append(os.path.join(RUNS, "semantic_deepseek", "predictions",
                                       "predictions_log_rule_semantic__stacking_ridge.csv"))
    candidates.append(os.path.join(_logrule_dir(seed), "predictions_log_rule_semantic__stacking_ridge.csv"))
    for path in candidates:
        if os.path.exists(path):
            return path
    raise SystemExit("missing predictions for seed " + str(seed) + "; tried: " + ", ".join(candidates))


def _mae_r2(summary_path: str, experiment_name: str) -> tuple[float, float]:
    df = pd.read_csv(summary_path, encoding="utf-8-sig")
    row = df[df["experiment_name"] == experiment_name]
    if row.empty:
        raise SystemExit("missing experiment " + experiment_name + " in " + summary_path)
    return float(row.iloc[0]["mae"]), float(row.iloc[0]["r2"])


def main() -> int:
    header = "seed,log_area_rule_mae,log_area_rule_r2,log_rule_mae,log_rule_r2,delta_mae,paired_t_p"
    lines = [header]
    for seed in SEEDS:
        area_mae, area_r2 = _mae_r2(os.path.join(_logarea_dir(seed), "experiment_summary.csv"),
                                    "log_area_rule_semantic")
        rule_mae, rule_r2 = _mae_r2(os.path.join(_logrule_dir(seed), "experiment_summary.csv"),
                                    "log_rule_semantic")
        pa = pd.read_csv(os.path.join(_logarea_dir(seed), "predictions_log_area_rule_semantic__stacking_ridge.csv"),
                         encoding="utf-8-sig")
        pb = pd.read_csv(_logrule_pred_path(seed), encoding="utf-8-sig")
        t_p = stats.ttest_rel(pa["absolute_error_wan"].values, pb["absolute_error_wan"].values).pvalue
        delta = area_mae - rule_mae
        lines.append(",".join([
            str(seed), str(round(area_mae, 4)), str(round(area_r2, 4)),
            str(round(rule_mae, 4)), str(round(rule_r2, 4)),
            str(round(delta, 4)), "%.2e" % t_p,
        ]))
        print("seed", seed, "delta=%.4f" % delta, "p=%.2e" % t_p)

    crlf = chr(13) + chr(10)
    with open(OUT_PATH, "wb") as handle:
        handle.write(("\ufeff" + crlf.join(lines) + crlf).encode("utf-8"))
    print("written:", OUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
