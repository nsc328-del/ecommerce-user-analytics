"""
生成 reports/dashboard.html — 高信息密度交互看板（14 张图 + KPI + 业务建议）。
"""
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "src")

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from sklearn.metrics import roc_curve, precision_recall_curve

from rfm import compute_rfm
from modeling import build_features, train_and_evaluate, top_k_capture_rate, time_rolling_validation

OUT = Path("reports/dashboard.html")
OUT.parent.mkdir(parents=True, exist_ok=True)

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
           "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC"]
BG = "rgba(0,0,0,0)"
FONT = "-apple-system, 'Helvetica Neue', 'Segoe UI', 'PingFang SC', sans-serif"

pio.templates["sean"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family=FONT, size=12, color="#2E3440"),
        colorway=PALETTE,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        xaxis=dict(gridcolor="#EEE", zerolinecolor="#EEE"),
        yaxis=dict(gridcolor="#EEE", zerolinecolor="#EEE"),
        margin=dict(l=50, r=30, t=60, b=50),
        hoverlabel=dict(font_family=FONT, font_size=12),
    )
)
pio.templates.default = "sean"

# ---------------------------------------------------------------- Load data
df = pd.read_parquet("data/processed/transactions.parquet")
clean = df[~df["is_return"]].copy()
rfm = compute_rfm(df)

split_ts = pd.Timestamp("2011-06-01")
features, feat_cols = build_features(df, split_ts)
results = train_and_evaluate(features, feat_cols)
best = max(results, key=lambda r: r.auc)

# 时间滚动验证（3 folds × 3 个月 horizon）
rolling = time_rolling_validation(
    df,
    split_points=[pd.Timestamp("2011-01-01"), pd.Timestamp("2011-04-01"), pd.Timestamp("2011-07-01")],
    label_horizon_days=90,
)

# ============================ FIGURES =====================================

# ---- 1. 月度 GMV 双轴
monthly = (
    clean.assign(month=clean["invoice_date"].dt.to_period("M").dt.to_timestamp())
    .groupby("month", as_index=False)
    .agg(gmv=("amount", "sum"),
         customers=("customer_id", "nunique"),
         orders=("invoice", "nunique"),
         aov=("amount", "sum"))
)
monthly["aov"] = monthly["gmv"] / monthly["orders"]
fig1 = make_subplots(specs=[[{"secondary_y": True}]])
fig1.add_trace(go.Bar(x=monthly["month"], y=monthly["gmv"], name="GMV",
                      marker_color="#4C78A8", marker_line_width=0,
                      hovertemplate="%{x|%Y-%m}<br>GMV: £%{y:,.0f}<extra></extra>"))
fig1.add_trace(go.Scatter(x=monthly["month"], y=monthly["customers"], name="活跃客户",
                          line=dict(color="#F58518", width=3), mode="lines+markers",
                          hovertemplate="%{x|%Y-%m}<br>Customers: %{y}<extra></extra>"),
               secondary_y=True)
fig1.update_layout(title="① 月度 GMV & 活跃客户数", height=360, hovermode="x unified",
                   legend=dict(orientation="h", y=1.08, x=0))
fig1.update_yaxes(title_text="GMV (£)", secondary_y=False)
fig1.update_yaxes(title_text="活跃客户", secondary_y=True, showgrid=False)

# ---- 2. 日历热力图（每日 GMV）
daily = clean.groupby(clean["invoice_date"].dt.date).agg(gmv=("amount", "sum")).reset_index()
daily["invoice_date"] = pd.to_datetime(daily["invoice_date"])
daily["weekday"] = daily["invoice_date"].dt.day_name()
daily["week"] = daily["invoice_date"].dt.isocalendar().week + (daily["invoice_date"].dt.year - 2009) * 53
daily["year_week"] = daily["invoice_date"].dt.strftime("%Y-W%V")
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
pivot = daily.pivot_table(index="weekday", columns="year_week", values="gmv", aggfunc="sum").reindex(weekday_order)
fig2 = go.Figure(go.Heatmap(
    z=pivot.values, x=pivot.columns, y=pivot.index,
    colorscale="Blues", showscale=True,
    colorbar=dict(title="GMV (£)", thickness=12),
    hovertemplate="%{x} · %{y}<br>GMV: £%{z:,.0f}<extra></extra>",
))
fig2.update_layout(title="② 每日交易热力图 — 周度 × 星期", height=290,
                   xaxis=dict(showticklabels=False, title=""), yaxis_title="")

# ---- 3. 国家 Choropleth
country_gmv = (clean.groupby("country")
               .agg(gmv=("amount", "sum"), customers=("customer_id", "nunique"))
               .reset_index())
# Map Eire→Ireland, Channel Islands etc 保守处理
country_gmv["country_iso"] = country_gmv["country"].replace({
    "EIRE": "Ireland", "RSA": "South Africa",
    "European Community": None, "Unspecified": None,
    "Channel Islands": None, "Hong Kong": "Hong Kong",
})
fig3 = px.choropleth(
    country_gmv.dropna(subset=["country_iso"]),
    locations="country_iso", locationmode="country names",
    color="gmv", color_continuous_scale="Blues",
    hover_data={"country": True, "gmv": ":,.0f", "customers": True, "country_iso": False},
    title="③ 国家 GMV 分布",
)
fig3.update_layout(height=430, geo=dict(showframe=False, bgcolor=BG, showcoastlines=True,
                                        coastlinecolor="#CCC", projection_type="natural earth"),
                   margin=dict(l=0, r=0, t=50, b=0))

# ---- 4. Top 10 国家（排除 UK 看对比）
top_countries = country_gmv.sort_values("gmv", ascending=False).head(10).sort_values("gmv")
fig4 = go.Figure(go.Bar(x=top_countries["gmv"], y=top_countries["country"], orientation="h",
                        marker=dict(color=top_countries["gmv"], colorscale="Blues", showscale=False),
                        hovertemplate="%{y}<br>GMV: £%{x:,.0f}<extra></extra>"))
fig4.update_layout(title="④ TOP 10 国家 GMV", height=380, xaxis_title="GMV (£)", yaxis_title="")

# ---- 5. Cohort 留存热力图（真正牛逼的电商分析经典图）
clean["order_month"] = clean["invoice_date"].dt.to_period("M")
clean["cohort_month"] = clean.groupby("customer_id")["order_month"].transform("min")
clean["period"] = (clean["order_month"] - clean["cohort_month"]).apply(lambda x: x.n)
cohort_counts = clean.groupby(["cohort_month", "period"])["customer_id"].nunique().unstack(fill_value=0)
cohort_sizes = cohort_counts.iloc[:, 0]
retention = cohort_counts.divide(cohort_sizes, axis=0) * 100
retention = retention.loc[:, retention.columns <= 12]  # 只看 12 个月
retention.index = retention.index.astype(str)
# 留存率文字
txt = retention.round(0).astype("Int64").astype(str).values
txt = np.where(retention.values > 0, txt + "%", "")
fig5 = go.Figure(go.Heatmap(
    z=retention.values, x=[f"M{i}" for i in retention.columns], y=retention.index,
    text=txt, texttemplate="%{text}", textfont=dict(size=10),
    colorscale="Blues", zmin=0, zmax=100,
    colorbar=dict(title="留存率 %", thickness=12),
    hovertemplate="Cohort %{y}<br>%{x}<br>留存: %{z:.1f}%<extra></extra>",
))
fig5.update_layout(title="⑤ Cohort 月度留存热力图 — 按首购月份追踪",
                   height=520, xaxis_title="首购后第 N 月", yaxis_title="首购月份",
                   yaxis=dict(autorange="reversed"))

# ---- 6. RFM 3D 散点
rfm_viz = rfm.sample(min(3000, len(rfm)), random_state=42)
fig6 = px.scatter_3d(
    rfm_viz, x="recency", y="frequency", z="monetary",
    color="segment", size="monetary", size_max=18, opacity=0.75,
    log_z=True, title="⑥ RFM 三维散点 — 按分群着色",
    labels={"recency": "R (天)", "frequency": "F (订单数)", "monetary": "M (£, log)"},
)
fig6.update_layout(height=560, scene=dict(bgcolor=BG))

# ---- 7. 分群气泡图
seg_stats = (rfm.groupby("segment")
             .agg(customers=("customer_id", "count"),
                  avg_recency=("recency", "mean"),
                  avg_frequency=("frequency", "mean"),
                  avg_monetary=("monetary", "mean"),
                  total_monetary=("monetary", "sum"))
             .reset_index())
fig7 = px.scatter(seg_stats, x="avg_recency", y="avg_frequency",
                  size="total_monetary", color="segment", text="segment", size_max=70,
                  title="⑦ RFM 分群气泡 — 大小=总 GMV · 位置=R/F")
fig7.update_traces(textposition="top center", textfont_size=10)
fig7.update_layout(height=520, xaxis_title="平均 Recency (天)",
                   yaxis_title="平均 Frequency (订单数)", showlegend=False)

# ---- 8. Sunburst 国家 → 分群
rfm_country = rfm.merge(clean[["customer_id", "country"]].drop_duplicates(subset="customer_id"),
                         on="customer_id", how="left")
top_ctry = rfm_country["country"].value_counts().head(6).index.tolist()
rfm_country["country_g"] = rfm_country["country"].where(rfm_country["country"].isin(top_ctry), "Others")
sb = (rfm_country.groupby(["country_g", "segment"])
      .agg(customers=("customer_id", "count"), gmv=("monetary", "sum"))
      .reset_index())
fig8 = px.sunburst(sb, path=["country_g", "segment"], values="gmv",
                   color="gmv", color_continuous_scale="Blues",
                   title="⑧ 国家 × 分群 Sunburst — GMV 聚合")
fig8.update_layout(height=480, margin=dict(l=0, r=0, t=50, b=0))

# ---- 9. Treemap 分群 GMV
tm = seg_stats.copy()
tm["gmv_label"] = tm["total_monetary"].apply(lambda v: f"£{v/1000:.0f}K")
fig9 = px.treemap(tm, path=[px.Constant("全部客户"), "segment"], values="total_monetary",
                  color="avg_monetary", color_continuous_scale="Blues",
                  title="⑨ 分群 GMV Treemap — 颜色=人均 ARPU")
fig9.update_traces(textinfo="label+value+percent parent",
                   texttemplate="<b>%{label}</b><br>£%{value:,.0f}<br>%{percentParent}")
fig9.update_layout(height=480, margin=dict(l=0, r=0, t=50, b=0))

# ---- 10. Radar 多分群画像对比
radar_segs = ["冠军用户", "忠诚用户", "流失预警", "已流失高价值", "沉睡用户"]
radar_df = rfm[rfm["segment"].isin(radar_segs)].groupby("segment").agg(
    recency=("recency", "mean"),
    frequency=("frequency", "mean"),
    monetary=("monetary", "mean"),
    R_score=("R_score", "mean"),
    F_score=("F_score", "mean"),
    M_score=("M_score", "mean"),
)
# 标准化到 0-100
for col in ["recency", "frequency", "monetary"]:
    v = radar_df[col]
    if col == "recency":
        radar_df[col + "_norm"] = 100 * (1 - (v - v.min()) / (v.max() - v.min() + 1e-9))
    else:
        radar_df[col + "_norm"] = 100 * (v - v.min()) / (v.max() - v.min() + 1e-9)

axes = ["Recency (近度)", "Frequency (频次)", "Monetary (金额)", "R Score", "F Score", "M Score"]
fig10 = go.Figure()
for i, seg in enumerate(radar_segs):
    if seg not in radar_df.index:
        continue
    r = radar_df.loc[seg]
    vals = [r["recency_norm"], r["frequency_norm"], r["monetary_norm"],
            r["R_score"] / 4 * 100, r["F_score"] / 4 * 100, r["M_score"] / 4 * 100]
    fig10.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=axes + [axes[0]],
                                    name=seg, fill="toself",
                                    line=dict(color=PALETTE[i], width=2),
                                    opacity=0.5))
fig10.update_layout(title="⑩ 分群雷达画像 — 多维能力圈（归一化 0-100）",
                    height=500, polar=dict(radialaxis=dict(range=[0, 100], showline=False)))

# ---- 11. Sankey 首购月份 → 活跃/流失
clean["status"] = np.where(
    (clean["invoice_date"].max() - clean.groupby("customer_id")["invoice_date"].transform("max")).dt.days <= 90,
    "近 90 天活跃", "90+ 天未回访")
sankey_df = (clean.drop_duplicates("customer_id")
             .groupby(["cohort_month", "status"])["customer_id"].count()
             .reset_index())
sankey_df["cohort_month"] = sankey_df["cohort_month"].astype(str)
cohort_list = sorted(sankey_df["cohort_month"].unique())
status_list = ["近 90 天活跃", "90+ 天未回访"]
labels = cohort_list + status_list
src = [cohort_list.index(r["cohort_month"]) for _, r in sankey_df.iterrows()]
tgt = [len(cohort_list) + status_list.index(r["status"]) for _, r in sankey_df.iterrows()]
val = sankey_df["customer_id"].tolist()
fig11 = go.Figure(go.Sankey(
    node=dict(label=labels, pad=15, thickness=18,
              color=["#4C78A8"] * len(cohort_list) + ["#54A24B", "#E45756"]),
    link=dict(source=src, target=tgt, value=val,
              color=["rgba(84, 162, 75, 0.35)" if labels[t] == "近 90 天活跃" else "rgba(228, 87, 86, 0.35)" for t in tgt]),
))
fig11.update_layout(title="⑪ 客户生命周期 Sankey — 首购月份 → 当前状态", height=520)

# ---- 12. 模型对比柱（ROC + Top10/Top20）
model_summary = pd.DataFrame([
    {"模型": r.name, "ROC-AUC": round(r.auc, 3),
     "Top 10% 覆盖": top_k_capture_rate(r.y_test, r.y_proba, 0.1),
     "Top 20% 覆盖": top_k_capture_rate(r.y_test, r.y_proba, 0.2)}
    for r in results
])
fig12 = go.Figure()
fig12.add_trace(go.Bar(x=model_summary["模型"], y=model_summary["ROC-AUC"],
                       name="ROC-AUC", marker_color="#4C78A8",
                       text=[f"{v:.3f}" for v in model_summary["ROC-AUC"]], textposition="outside"))
fig12.add_trace(go.Bar(x=model_summary["模型"], y=model_summary["Top 10% 覆盖"],
                       name="Top 10% 覆盖率", marker_color="#F58518",
                       text=[f"{v:.1%}" for v in model_summary["Top 10% 覆盖"]], textposition="outside"))
fig12.add_trace(go.Bar(x=model_summary["模型"], y=model_summary["Top 20% 覆盖"],
                       name="Top 20% 覆盖率", marker_color="#54A24B",
                       text=[f"{v:.1%}" for v in model_summary["Top 20% 覆盖"]], textposition="outside"))
fig12.update_layout(title="⑫ 三模型对比 — 离线指标", barmode="group", height=380, yaxis_tickformat=".2f")

# ---- 13. ROC + Precision-Recall 组合
fig13 = make_subplots(rows=1, cols=2, subplot_titles=("ROC Curve", "Precision-Recall"))
for i, r in enumerate(results):
    fpr, tpr, _ = roc_curve(r.y_test, r.y_proba)
    fig13.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{r.name} (AUC={r.auc:.3f})",
                               line=dict(width=2.5, color=PALETTE[i])), row=1, col=1)
    prec, rec, _ = precision_recall_curve(r.y_test, r.y_proba)
    fig13.add_trace(go.Scatter(x=rec, y=prec, name=r.name, showlegend=False,
                               line=dict(width=2.5, color=PALETTE[i])), row=1, col=2)
fig13.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"),
                           name="Random", showlegend=False), row=1, col=1)
fig13.update_xaxes(title_text="FPR", row=1, col=1)
fig13.update_yaxes(title_text="TPR", row=1, col=1)
fig13.update_xaxes(title_text="Recall", row=1, col=2)
fig13.update_yaxes(title_text="Precision", row=1, col=2)
fig13.update_layout(title="⑬ ROC & Precision-Recall 双视图", height=420)

# ---- 14. Gain / Lift 曲线
def gain_lift(y_true, y_proba, bins=20):
    order = np.argsort(y_proba)[::-1]
    y_sorted = np.array(y_true)[order]
    cum = np.cumsum(y_sorted) / y_sorted.sum()
    pct = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    # downsample
    idx = np.linspace(0, len(pct) - 1, 100).astype(int)
    return pct[idx], cum[idx]

fig14 = make_subplots(rows=1, cols=2, subplot_titles=("累计 Gain", "Lift"))
for i, r in enumerate(results):
    pct, gain = gain_lift(r.y_test, r.y_proba)
    fig14.add_trace(go.Scatter(x=pct, y=gain, name=r.name, line=dict(width=2.5, color=PALETTE[i])), row=1, col=1)
    lift = gain / pct
    fig14.add_trace(go.Scatter(x=pct, y=lift, name=r.name, showlegend=False,
                               line=dict(width=2.5, color=PALETTE[i])), row=1, col=2)
fig14.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"),
                           name="Random", showlegend=False), row=1, col=1)
fig14.add_hline(y=1, line_dash="dash", line_color="gray", row=1, col=2)
fig14.update_xaxes(title_text="按打分降序的客户占比", row=1, col=1, tickformat=".0%")
fig14.update_yaxes(title_text="累计复购者覆盖", row=1, col=1, tickformat=".0%")
fig14.update_xaxes(title_text="按打分降序的客户占比", row=1, col=2, tickformat=".0%")
fig14.update_yaxes(title_text="Lift", row=1, col=2)
fig14.update_layout(title="⑭ Gain & Lift — 模型业务价值", height=420)

# ---- 15. 特征重要度 Top 20（horizontal + gradient）
rf = next(r for r in results if r.name == "RandomForest").model
imp = (pd.DataFrame({"feature": feat_cols, "importance": rf.feature_importances_})
       .sort_values("importance", ascending=False).head(20)
       .sort_values("importance"))
fig15 = go.Figure(go.Bar(x=imp["importance"], y=imp["feature"], orientation="h",
                          marker=dict(color=imp["importance"], colorscale="Blues", showscale=False),
                          text=[f"{v:.1%}" for v in imp["importance"]], textposition="outside"))
fig15.update_layout(title="⑮ Random Forest · Top 20 特征重要度", height=520, xaxis_title="重要度", yaxis_title="")

# ---- 16. 时间滚动验证 — 3 folds × 3 模型 AUC
fig16 = px.bar(
    rolling, x="fold_split", y="auc", color="model", barmode="group",
    text=rolling["auc"].apply(lambda v: f"{v:.3f}"),
    title="⑯ 时间滚动验证 — 3 个 rolling folds × 3 个月预测窗口",
    labels={"fold_split": "Fold 观察期截止日", "auc": "ROC-AUC", "model": ""},
)
fig16.update_traces(textposition="outside")
fig16.update_layout(height=400, yaxis_range=[0.7, 0.85],
                    legend=dict(orientation="h", y=1.12, x=0))

# ============================ HTML ========================================

def div(fig):
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"displaylogo": False})


total_gmv = clean["amount"].sum()
repeat_rate = (clean.groupby("customer_id")["invoice"].nunique() >= 2).mean()
pareto_seg = seg_stats[seg_stats["segment"].isin(["冠军用户", "忠诚用户"])]
pareto_gmv = pareto_seg["total_monetary"].sum() / seg_stats["total_monetary"].sum()
pareto_cust = pareto_seg["customers"].sum() / seg_stats["customers"].sum()


kpi_cards = [
    ("净 GMV",              f"£{total_gmv/1e6:.1f}M",    "2009-12 ~ 2011-12"),
    ("唯一客户",            f"{df['customer_id'].nunique():,}", "有效 customer_id"),
    ("订单总数",            f"{clean['invoice'].nunique():,}", f"单均 £{total_gmv/clean['invoice'].nunique():.0f}"),
    ("复购率 (≥2 单)",      f"{repeat_rate:.1%}",        "全样本"),
    ("80/20 验证",          f"{pareto_gmv:.0%} GMV",     f"来自 TOP {pareto_cust:.0%} 客户"),
    ("退货金额",            f"£{-df.loc[df['is_return'], 'amount'].sum()/1000:.0f}K", f"{df['is_return'].mean():.1%} 退货率"),
    ("最佳模型 AUC",        f"{best.auc:.3f}",            f"{best.name}"),
    ("Top 20% 覆盖率",      f"{top_k_capture_rate(best.y_test, best.y_proba, 0.2):.0%}", "真实复购者"),
]

kpi_html = "".join([
    f'<div class="kpi"><div class="label">{lbl}</div>'
    f'<div class="value">{val}</div><div class="sub">{sub}</div></div>'
    for lbl, val, sub in kpi_cards
])


recos = [
    ("🔥", "已流失高价值客户挽回",
     "45 人 · 人均历史消费 £3,087 · 总 GMV £138,926",
     "KA 人工挽回 + 专属折扣；按单次订单 ≈ £400 估算，挽回 30% ≈ £5-6K 直接 GMV"),
    ("⚠️", "流失预警分层召回",
     "214 人 · Recency 90-180 天 · 3 个月 Top 20% lift ≈ 2.3x",
     "按模型打分分 3 档推送邮件；高分给 10% 折扣，低分纳内容营销"),
    ("💎", "VIP 体系守住 27% 的 74% 贡献者",
     "冠军 + 忠诚 共 1,593 人 · 贡献 73.9% GMV",
     "VIP 权益 + Recency 告警阈值；流失这群人比拉新贵一个数量级"),
]
reco_html = "".join([
    f'<div class="reco"><div class="reco-icon">{icon}</div>'
    f'<div class="reco-body"><div class="reco-title">{title}</div>'
    f'<div class="reco-meta">{meta}</div>'
    f'<div class="reco-action">{action}</div></div></div>'
    for icon, title, meta, action in recos
])

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>E-commerce CRM Analytics · Sicheng Ni</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{
    --bg: #F7F8FA; --card: #FFFFFF; --ink: #1F2937; --muted: #6B7280;
    --accent: #4C78A8; --line: #E5E7EB;
  }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: {FONT}; background: var(--bg); color: var(--ink);
          max-width: 1280px; margin: 0 auto; padding: 28px 24px; }}
  header {{ margin-bottom: 20px; }}
  header h1 {{ font-size: 28px; margin: 0 0 4px; letter-spacing: -0.3px; }}
  header .meta {{ color: var(--muted); font-size: 13px; }}
  header .meta a {{ color: var(--accent); text-decoration: none; }}
  .kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 18px 0 24px; }}
  .kpi {{ background: var(--card); padding: 14px 16px; border-radius: 10px;
          border: 1px solid var(--line); }}
  .kpi .label {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }}
  .kpi .value {{ font-size: 22px; font-weight: 600; margin: 4px 0 2px; color: var(--ink); }}
  .kpi .sub {{ font-size: 11px; color: var(--muted); }}
  h2.section {{ font-size: 15px; margin: 28px 0 12px; color: var(--muted);
                text-transform: uppercase; letter-spacing: 1px; font-weight: 600;
                border-top: 1px solid var(--line); padding-top: 18px; }}
  .grid {{ display: grid; gap: 14px; }}
  .grid-2 {{ grid-template-columns: 1fr 1fr; }}
  .grid-3 {{ grid-template-columns: 1fr 1fr 1fr; }}
  .chart {{ background: var(--card); padding: 10px 14px; border-radius: 10px;
           border: 1px solid var(--line); }}
  .chart.full {{ grid-column: 1 / -1; }}
  .recos {{ display: grid; gap: 12px; margin-top: 14px; }}
  .reco {{ display: flex; gap: 14px; background: var(--card); padding: 16px 18px;
           border-radius: 10px; border: 1px solid var(--line); }}
  .reco-icon {{ font-size: 22px; line-height: 1.4; }}
  .reco-title {{ font-weight: 600; font-size: 15px; }}
  .reco-meta {{ color: var(--muted); font-size: 12px; margin: 4px 0; }}
  .reco-action {{ font-size: 13px; color: var(--ink); }}
  footer {{ color: var(--muted); font-size: 12px; text-align: center;
           margin-top: 32px; padding-top: 18px; border-top: 1px solid var(--line); }}
  @media (max-width: 980px) {{
    .kpis {{ grid-template-columns: repeat(2, 1fr); }}
    .grid-2, .grid-3 {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<header>
  <h1>E-commerce CRM Analytics Dashboard</h1>
  <div class="meta">
    Online Retail II (UCI) · 2009-12 ~ 2011-12 · 824K txns · 5.9K customers<br>
    <b>Sicheng (Sean) Ni</b> · MCom Data Analysis, University of Sydney ·
    <a href="https://github.com/nsc328-del" target="_blank">GitHub</a> ·
    Built with Pandas / scikit-learn / XGBoost / Plotly · Generated {datetime.now():%Y-%m-%d}
  </div>
</header>

<div class="kpis">{kpi_html}</div>

<h2 class="section">Ⅰ · 业务脉搏</h2>
<div class="grid grid-2">
  <div class="chart full">{div(fig1)}</div>
  <div class="chart full">{div(fig2)}</div>
  <div class="chart">{div(fig3)}</div>
  <div class="chart">{div(fig4)}</div>
</div>

<h2 class="section">Ⅱ · 客户分层 & 生命周期</h2>
<div class="grid grid-2">
  <div class="chart full">{div(fig5)}</div>
  <div class="chart">{div(fig6)}</div>
  <div class="chart">{div(fig7)}</div>
  <div class="chart">{div(fig8)}</div>
  <div class="chart">{div(fig9)}</div>
  <div class="chart full">{div(fig10)}</div>
  <div class="chart full">{div(fig11)}</div>
</div>

<h2 class="section">Ⅲ · 复购预测模型</h2>
<div class="grid grid-2">
  <div class="chart full">{div(fig16)}</div>
  <div class="chart full">{div(fig12)}</div>
  <div class="chart full">{div(fig13)}</div>
  <div class="chart full">{div(fig14)}</div>
  <div class="chart full">{div(fig15)}</div>
</div>

<h2 class="section">Ⅳ · 运营建议（基于分群 + 预测结果）</h2>
<div class="recos">{reco_html}</div>

<footer>
  © {datetime.now():%Y} Sicheng Ni · 项目仓库：
  <a href="https://github.com/nsc328-del/ecommerce-user-analytics">nsc328-del/ecommerce-user-analytics</a>
</footer>
</body>
</html>
"""

OUT.write_text(html, encoding="utf-8")
print(f"✓ {OUT} · {OUT.stat().st_size/1024:.0f} KB · 16 charts + 8 KPI + 3 recos")
