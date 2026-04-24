# 电商用户分群 × 复购预测｜端到端分析 Demo

> **一句话定位**：基于 Online Retail II 公开数据集（~82 万条交易 / 5,939 客户 / 2 年跨度），完成"数据清洗 → RFM 用户分群 → 复购预测建模 → 业务建议"的完整 CRM 分析链路。

[📊 **在线交互看板 (GitHub Pages)**](./reports/dashboard.html) · [📓 Notebooks](./notebooks) · [💡 业务建议](./reports/business_recommendations.md)

## 项目背景

零售/电商平台每天产生大量交易数据，但真正被转化为**运营决策**的只是一小部分。本项目以英国某 B2B 礼品零售商 2009-12 ~ 2011-12 的交易流水为原料，模拟一个典型的 CRM 分析师工作流：**识别高价值用户 → 预测复购倾向 → 输出可执行的运营动作**。

## 核心结果

| 维度 | 数字 |
|---|---|
| 清洗后交易数据 | **824,293 行** · 44,870 订单 · 5,939 客户（原始 107 万行剔除 23% 缺失 customer_id 的行；退货作为独立标记行保留，金额为负） |
| 时间跨度 | 2009-12-01 ~ 2011-12-09（~24 个月） |
| 净 GMV | **£17.7M**（扣除 £1.1M 退货后） |
| 复购率（订单 ≥ 2） | **72.4%** |
| **RFM 分群** | 16 个 R×F 象限 → **12 类运营分群**；冠军 + 忠诚 = 27.3% 客户，贡献 **73.9% GMV**（接近 80/20） |
| **复购预测** | **21 维行为特征** · Logistic / RF / XGBoost(tuned) ROC-AUC ≈ **0.77-0.82** · 3 folds 时间滚动验证 |
| 业务 Lift | 3 个月预测窗口下 · Top 10% 覆盖 22-27% 真实复购者（**lift 2.2-2.7x**），Top 20% 覆盖 43-47%（lift ≈ 2.3x） |

## 技术栈

- **数据处理**：Pandas / NumPy
- **建模**：Scikit-learn（Logistic / Random Forest）· XGBoost
- **可视化**：Plotly（交互图 + 静态 Dashboard）
- **协作**：Jupyter Lab · Claude Code（AI-assisted coding）

## 目录导航

```
.
├── notebooks/
│   ├── 01_eda.ipynb                    # 数据概览 · 月度 GMV · 客户/品类结构 · 退货分析
│   ├── 02_rfm_segmentation.ipynb       # RFM 打分 · 12 类运营分群（由 16 个 R×F 映射）· K-Means 交叉验证
│   └── 03_repurchase_prediction.ipynb  # 特征工程 · 三模型对比 · Lift 分析
├── reports/
│   ├── dashboard.html                  # Plotly 交互看板（GitHub Pages 入口）
│   └── business_recommendations.md     # 3 条运营建议（数据证据 + 执行动作）
├── src/
│   ├── preprocessing.py                # 数据清洗 + 退货处理
│   ├── rfm.py                          # RFM 打分 + 16 象限 → 12 类运营分群映射
│   └── modeling.py                     # 特征工程 + 三模型训练 + Top-K 评估
├── data/README.md                      # 数据下载说明（原始数据不入库）
└── requirements.txt
```

## 快速开始

```bash
pip install -r requirements.txt
# 1. 按 data/README.md 下载 Online Retail II 原始 CSV → data/online_retail_II.csv
python src/preprocessing.py    # 输出 data/processed/transactions.parquet
jupyter lab                    # 按 01 → 02 → 03 顺序跑 notebook
python build_dashboard.py      # 重新生成 reports/dashboard.html（可选）
```

## 一个诚实的说明

- 英国 2009-2011 年 B2B 礼品批发商的历史数据，**不反映当前中国零售环境**；选它的原因：公开、干净、体量与时间跨度兼备，适合端到端 demo
- RFM 的 M 维用"净成交额"（含退货扣减），贴近实战
- 模型 AUC ≈ 0.81 在 6 个月 horizon（正样本率 52%）下接近天花板；**真正更有说服力的是 3 个月 horizon + 3 折时间滚动验证**，正样本率降到 32-35%（更贴近生产），**Logistic Regression 与 XGBoost 基本持平**（信号以线性为主），Top 10% lift 2.2-2.7x
- v3 做了一次诚实的特征瘦身：v2 的 35 维里有 15 个 Top-SKU one-hot，ablation 显示其对 AUC 贡献 <0.005，已移除；同时修掉了 avg_basket（原本按行计算，实为"平均行金额"而非订单 AOV）、avg_gap_days（单订单客户填 0 造成信号反转）、return_count（行级计数而非订单级）三处语义 bug
- 业务建议中的"预期收益"为**基于模型信号估算的上界**，不等同于未做 A/B 测试的实盘结果
- 进一步优化方向：引入用户画像 / 商品品类聚合、成本敏感阈值调优、对接真实 A/B 测试验证线上 lift

---

# E-commerce User Segmentation × Repurchase Prediction

> **TL;DR**: End-to-end CRM analytics pipeline on the Online Retail II dataset (~824K transactions, 5,939 customers, 24-month span), covering data cleaning → RFM segmentation → repurchase prediction → actionable business recommendations.

[📊 **Interactive Dashboard**](./reports/dashboard.html) · [📓 Notebooks](./notebooks) · [💡 Recommendations](./reports/business_recommendations.md)

## Key Findings

| Metric | Value |
|---|---|
| Cleaned transactions | **824,293 rows** · 44,870 invoices · 5,939 customers (dropped 23% rows missing customer_id from raw 1.07M) |
| Period | 2009-12-01 ~ 2011-12-09 (~24 months) |
| Net GMV | **£17.7M** (after £1.1M returns) |
| Repeat rate (≥ 2 invoices) | **72.4%** |
| **RFM segmentation** | 16 R×F quadrants → **12 operational segments**; Champions + Loyal = 27.3% of customers → **73.9% of GMV** |
| **Repurchase prediction** | **21 behavioral features** · Logistic / RF / XGBoost (tuned) — ROC-AUC ≈ **0.77-0.82** · 3-fold time-rolling validation |
| Business lift | On 3-month horizon, top-10% scored customers capture 22-27% of true repurchasers (**lift 2.2-2.7x**) |

## Stack

Pandas · Scikit-learn · XGBoost · Plotly · Jupyter Lab · Claude Code (AI-assisted development)

## Honest Caveats

- 2009-2011 UK B2B gift wholesaler — does not reflect current market behavior; chosen for being public, clean, and end-to-end demonstrable
- RFM's M uses **net amount** (returns subtracted), closer to real-world CRM practice
- On the 6-month horizon (52% positive rate), AUC ≈ 0.81 is near ceiling. **The more defensible story is the 3-fold time-rolling validation on 3-month horizons**: positive rate drops to 32-35% (closer to production); **Logistic Regression and XGBoost perform on par** (signal is largely linear), top-decile lift 2.2-2.7x
- v3 feature diet: v2's 35-dim vector included 15 Top-SKU one-hots that added <0.005 AUC in ablation — dropped. Also fixed three semantic bugs: `avg_basket` (row-level mean masquerading as order AOV), `avg_gap_days` (single-order customers filled with 0 → signal reversal), `return_count` (line-level count instead of invoice-level)
- Business-recommendation projections are **model-signal upper bounds**, not A/B-tested lift numbers
- Next steps: demographic / category-level features, cost-sensitive threshold tuning, real A/B tests to validate online lift

---

**Author**: Sicheng (Sean) Ni · Master of Commerce (Data Analysis for Business), University of Sydney · [github.com/nsc328-del](https://github.com/nsc328-del)
