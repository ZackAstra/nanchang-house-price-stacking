# 南昌二手房价格预测：特征工程与 Stacking 集成

本仓库是论文《Comparative Study of Second-Hand Housing Price Prediction Based on Machine Learning Regression Algorithms: Feature Engineering and Stacking Ensemble Modeling》的配套代码与数据示例。

[English version: README.md](README.md)

## 项目简介

本仓库复现论文的完整实验链路：

- **数据采集**：面向主流房产挂牌平台（安居客·南昌）的四级穿透爬虫——房源链接 → 房源详情 → 小区 POI 配套（`main.py`、`src/fetchers/`、`src/parsers/`、`src/storage/`、`config/regions.json`）；
- **特征工程**：四维特征体系——物理属性、POI 设施、地理可达性（Accessibility Index）、户型结构（Layout Density），以及一套 29 标签的标题语义体系（规则词典标注）（`src/research/pipeline.py`）；
- **模型对比**：8 个基线回归模型、Ridge 元学习器的 Stacking 集成、OOF-TSRM 两阶段残差模型；固定时间敏感留出测试集 + 社区分组 3 折交叉验证（`src/research/run_price_research.py`）；
- **误差诊断实验**（论文 §4.4）：log 变换、标签清洗剂量-效应、多项式交互、分位数回归、信息上限诊断（`src/research/*_runner.py`）；
- **语义特征实验**（论文 §4.5）：规则标注 vs LLM 标注，含 prompt×温度稳健性对照（`scripts/build_llm_batch_jsonl.py`、`scripts/run_qwen_prompt_bridge.py`、`scripts/evaluate_llm_3x2_batch.py`）；
- **分析脚本**：Holm 校正显著性矩阵、多种子汇总、算力成本、描述统计、方差分解、构造指标敏感性（`scripts/`）；
- **图表**：论文全部图由 `scripts/generate_figures_review3.py` 与 `scripts/generate_shap_beeswarm_review3.py` 重绘至 `docs/figures/`。

## 主要结果（可与 `data/expected_outputs/` 对照核验）

- 最终模型（log 目标 + 规则语义特征 + StackingRidge）：外推留出测试集上 **MAE = 20.17（万元）、R² = 0.3214**；随机同分布划分下 R² = 0.68；
- StackingRidge 在 5 个随机种子中 4 次第一，显著优于最强单模型 CatBoost（Holm 校正 p = 4.1 × 10⁻⁷）。

## 项目结构

```
├── main.py / config/            # 爬虫入口与区域配置
├── src/
│   ├── fetchers/ parsers/ storage/ models/   # 爬虫组件
│   └── research/                # 特征工程、模型、实验 runner
├── scripts/                     # 分析与绘图脚本（论文 §4.7–4.9）
├── tests/                       # 爬虫集成测试
├── data/
│   ├── samples/                 # 200 条分层抽样示例（房源/社区/LLM 标注/LLM 原始响应）
│   ├── expected_outputs/        # 聚合指标（复现核对基准）
│   ├── reference/               # 区域 AI 中位数（聚合统计）
│   └── full/                    # 完整数据放置处（见 data/full/README.md）
├── llm_prompts/                 # 3 个 LLM 标注 prompt 全文（baseline / few-shot / strict）
└── docs/figures/                # 论文图（PNG + SVG）
```

## 环境准备（uv）

```bash
uv venv
uv pip install -r requirements.txt
# 仅爬虫需要：uv run playwright install chromium
```

## 快速开始（基于随附示例数据的冒烟运行）

```bash
uv run python -m src.research.run_price_research \
  --houses data/samples/houses_sample.jsonl \
  --communities data/samples/communities_sample.jsonl \
  --output-dir data/runs --run-id smoke --sample-limit 200 --cv-folds 3
```

200 条示例上的指标仅用于验证流程可运行。复现论文数字请先获取完整数据（见下文），放入 `data/full/` 后去掉 `--sample-limit` 运行同一命令（保留 `--cv-folds 3`，与论文的社区分组 3 折交叉验证一致）。

## 数据获取

与论文 Data Availability 声明一致：

- **示例数据**：`data/samples/` 内含 200 条分层抽样（区域×价格段），含 LLM 语义标注与 10 条 LLM 原始响应示例；
- **聚合结果**：论文全部头条指标见 `data/expected_outputs/`，供核验；
- **完整数据**（7,140 条房源 + 1,062 个社区 + 全量标注）：请联系作者（zhaoxingchen@hotmail.com）合理获取，需遵守原平台服务条款。数据采集自公开挂牌页面，仅供学术研究使用。

## 复现论文各表格

论文每一张表的聚合数字已随附在 `data/expected_outputs/`。如需从完整数据（放入 `data/full/` 后）自行重算：

1. 按论文的 CV 协议（社区分组 3 折）运行主实验：

   ```bash
   uv run python -m src.research.run_price_research \
     --houses data/full/nanchang_houses.jsonl \
     --communities data/full/nanchang_communities.jsonl \
     --output-dir data/runs --run-id full --cv-folds 3
   ```

2. 将分析脚本读取的三个逐样本中间产物复制到 `data/expected_outputs/`：

   ```bash
   cp data/runs/full/cleaned_features.csv data/expected_outputs/
   cp data/runs/full/model_prediction_errors.csv data/expected_outputs/
   cp data/runs/full/model_experiments.jsonl data/expected_outputs/
   ```

3. 运行分析脚本（每个脚本头部均注明输入与输出），例如 `uv run python scripts/descriptive_stats.py`。

文件与论文表格对照：

| 论文表格 / 小节 | `data/expected_outputs/` 下对应文件 |
|---|---|
| Table 3 描述性统计 | `descriptive_stats_numeric.csv`、`descriptive_stats_categorical.csv` |
| Table 13 模型对比 | `model_metrics.csv` |
| Table 14 Voting vs Stacking | `voting_vs_stacking_metrics.csv` |
| Table 15 多种子稳定性 | `multi_seed_summary.csv` |
| Table 16 误差诊断 | `combined_experiment_summary.csv`（log / 清洗 / 组合行） |
| Table 17 语义特征 | `stacking_comparison.csv`（语义各行）+ `semantic_logarea_summary.csv`（log-area 行） |
| Table 19 显著性矩阵 | `significance_matrix.csv` |
| Table 21 算力成本 | `compute_cost.csv` |
| Table 22 指标敏感性 | `sensitivity_results.csv` |
| §4.4.4 方差分解 | `variance_decomposition.csv` |
| §4.5.3 语义一致性 | `semantic_agreement.csv` |

说明：

- `stacking_comparison.csv` 存放的是**语义特征实验（Table 17）**，而非 Voting-vs-Stacking 对比；后者在 `voting_vs_stacking_metrics.csv`。
- Table 16 的其余行（moderate / 单价 / 总价清洗剂量、多项式交互、分位数回归）可用对应 runner 重算（`label_noise_experiment_runner.py`、`log_transform_experiment_runner.py`、`polynomial_feature_runner.py`、`quantile_regression_runner.py`），但未随附其结果文件。
- 头条结果在 Python 3.11.10 + `requirements.txt` 锁定版本下产出；代码兼容 Python ≥ 3.11。

## LLM 标注说明

语义标签由大模型从房源标题提取。三种 prompt 设计（baseline / few-shot / structured-strict）全文公开于 `llm_prompts/`；标注使用 SiliconFlow 异步批量推理（Batch API）。标注示例与原始响应示例见 `data/samples/`。

## 许可与声明

代码以 MIT 许可证发布（见 `LICENSE`）。数据来自公开挂牌平台，仅供学术研究，须遵守原平台服务条款。
