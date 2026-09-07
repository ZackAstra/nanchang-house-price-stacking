# -*- coding: utf-8 -*-
"""评审修复任务2：训练/推理算力成本表 + 新 Table 12 全模型指标表（含 ridge/lasso 新旧对照）。

输入:  data/expected_outputs/model_experiments.jsonl（主实验完整输出，未随附）
       data/expected_outputs/model_metrics.csv
输出:  data/expected_outputs/compute_cost.csv
       data/expected_outputs/compute_cost.md
       data/expected_outputs/new_table12_metrics.csv
       data/expected_outputs/new_table12_metrics.md
"""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "data", "expected_outputs")
OUT_DIR = os.path.join(ROOT, "data", "expected_outputs")
N_TEST = 1383

# 旧 Table 12 中的 ridge/lasso MAE（评审修复前）
OLD_MAE = {"ridge": 24.5763, "lasso": 27.6530}


def load_completed_events(path):
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("event") == "model_completed":
                events.append(rec)
    return events


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    events = load_completed_events(os.path.join(BASE, "model_experiments.jsonl"))
    metrics = pd.read_csv(os.path.join(BASE, "model_metrics.csv"), encoding="utf-8-sig")

    # ---- 算力成本表 ----
    rows = []
    for e in events:
        train_s = e.get("train_seconds")
        pred_s = e.get("predict_seconds")
        rows.append(
            {
                "model_name": e["model_name"],
                "train_seconds": train_s,
                "predict_seconds_total": pred_s,
                "latency_ms_per_sample": (pred_s / N_TEST * 1000) if pred_s is not None else None,
                "mae": e.get("mae"),
                "cv_mae_mean": e.get("cv_mae_mean"),
                "selected_alpha": e.get("selected_alpha"),
            }
        )
    cost = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)

    # StackingRidge vs CatBoost 成本-收益对比
    sr = cost[cost["model_name"] == "stacking_ridge"].iloc[0]
    cb = cost[cost["model_name"] == "catboost"].iloc[0]
    train_mult = sr["train_seconds"] / cb["train_seconds"]
    lat_mult = sr["latency_ms_per_sample"] / cb["latency_ms_per_sample"]
    mae_gain = cb["mae"] - sr["mae"]
    mae_gain_pct = mae_gain / cb["mae"] * 100
    cost["train_cost_x_catboost"] = cost["train_seconds"] / cb["train_seconds"]

    cost.to_csv(os.path.join(OUT_DIR, "compute_cost.csv"), index=False, encoding="utf-8-sig")

    lines = [
        "# 训练 / 推理算力成本对比（测试集 n = %d）" % N_TEST,
        "",
        "| Model | Train (s) | Predict total (s) | Latency (ms/sample) | Train cost ×CatBoost | MAE (万元) |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in cost.iterrows():
        train = f"{r['train_seconds']:.3f}" if pd.notna(r["train_seconds"]) else "N/A"
        pred = f"{r['predict_seconds_total']:.4f}" if pd.notna(r["predict_seconds_total"]) else "N/A"
        lat = f"{r['latency_ms_per_sample']:.4f}" if pd.notna(r["latency_ms_per_sample"]) else "N/A"
        mult = f"{r['train_cost_x_catboost']:.1f}×" if pd.notna(r["train_cost_x_catboost"]) else "N/A"
        lines.append(f"| {r['model_name']} | {train} | {pred} | {lat} | {mult} | {r['mae']:.4f} |")
    lines += [
        "",
        f"**成本-收益结论**：StackingRidge 训练耗时 {sr['train_seconds']:.1f}s，为 CatBoost（{cb['train_seconds']:.1f}s）的 "
        f"{train_mult:.1f} 倍；单样本推理延迟 {sr['latency_ms_per_sample']:.4f}ms vs CatBoost {cb['latency_ms_per_sample']:.4f}ms"
        f"（{lat_mult:.1f} 倍）。换来 MAE 从 {cb['mae']:.4f} 降至 {sr['mae']:.4f} 万元，"
        f"绝对改进 {mae_gain:.4f} 万元（相对改进 {mae_gain_pct:.2f}%）。",
        "",
        "注：blend_top3 为对既有模型预测的事后加权融合，无独立 train/predict 计时，记为 N/A。",
        "",
    ]
    with open(os.path.join(OUT_DIR, "compute_cost.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ---- 新 Table 12：全模型指标 + ridge/lasso 新旧对照 ----
    t12 = metrics.rename(
        columns={"mae": "MAE", "rmse": "RMSE", "r2": "R2", "mape": "MAPE", "cv_mae_mean": "CV_MAE"}
    )[["model_name", "MAE", "RMSE", "R2", "MAPE", "CV_MAE"]].copy()
    t12["old_MAE"] = t12["model_name"].map(OLD_MAE)
    t12["delta_vs_old"] = t12["MAE"] - t12["old_MAE"]
    alpha_map = {e["model_name"]: e.get("selected_alpha") for e in events}
    t12["selected_alpha"] = t12["model_name"].map(alpha_map)
    t12.to_csv(os.path.join(OUT_DIR, "new_table12_metrics.csv"), index=False, encoding="utf-8-sig")

    lines = [
        "# 新 Table 12：全模型测试集性能指标（重跑后）",
        "",
        "| Model | MAE (万元) | RMSE | R² | MAPE | CV MAE | 旧 MAE | Δ vs 旧 | selected_alpha |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in t12.iterrows():
        old = f"{r['old_MAE']:.4f}" if pd.notna(r["old_MAE"]) else "—"
        delta = f"{r['delta_vs_old']:+.4f}" if pd.notna(r["delta_vs_old"]) else "—"
        alpha = f"{r['selected_alpha']:.6g}" if pd.notna(r["selected_alpha"]) else "—"
        lines.append(
            f"| {r['model_name']} | {r['MAE']:.4f} | {r['RMSE']:.4f} | {r['R2']:.4f} "
            f"| {r['MAPE']:.4f} | {r['CV_MAE']:.4f} | {old} | {delta} | {alpha} |"
        )
    lines.append("")
    with open(os.path.join(OUT_DIR, "new_table12_metrics.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 控制台摘要
    print(cost[["model_name", "train_seconds", "latency_ms_per_sample", "mae"]].to_string(index=False))
    print(f"\nStackingRidge vs CatBoost: 训练倍数 {train_mult:.1f}x, 延迟倍数 {lat_mult:.1f}x, "
          f"MAE 改进 {mae_gain:.4f} 万元 ({mae_gain_pct:.2f}%)")
    for m, old in OLD_MAE.items():
        new = float(t12.loc[t12["model_name"] == m, "MAE"].iloc[0])
        print(f"{m}: 旧 MAE {old} -> 新 MAE {new:.4f} (Δ {new - old:+.4f}), selected_alpha={alpha_map.get(m)}")
    print(f"\n输出: {OUT_DIR}/compute_cost.csv|md, new_table12_metrics.csv|md")


if __name__ == "__main__":
    main()
