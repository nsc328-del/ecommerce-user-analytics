"""
RFM 打分与分群（基于 Online Retail II 清洗后数据）。

- Recency:    距 snapshot 的天数（越小越好）
- Frequency:  不同 Invoice 数量（一个订单算一次）
- Monetary:   净消费额 = 所有 amount 之和（退货扣减）
"""
import pandas as pd

SEGMENT_MAP = {
    (4, 4): "冠军用户",
    (4, 3): "忠诚用户",
    (4, 2): "潜力用户",
    (4, 1): "新客",
    (3, 4): "忠诚用户",
    (3, 3): "稳定用户",
    (3, 2): "一般用户",
    (3, 1): "新客",
    (2, 4): "流失预警",
    (2, 3): "需要挽回",
    (2, 2): "一般用户",
    (2, 1): "低频新客",
    (1, 4): "已流失高价值",
    (1, 3): "已流失",
    (1, 2): "已流失",
    (1, 1): "沉睡用户",
}


def compute_rfm(df: pd.DataFrame, snapshot_ts: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Args:
        df: 清洗后的交易表，含 customer_id / invoice / invoice_date / amount
        snapshot_ts: 计算 Recency 的参考时点；默认取数据中最大日期 + 1 天
    """
    if snapshot_ts is None:
        snapshot_ts = df["invoice_date"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("customer_id")
        .agg(
            recency=("invoice_date", lambda s: (snapshot_ts - s.max()).days),
            frequency=("invoice", "nunique"),
            monetary=("amount", "sum"),
        )
        .reset_index()
    )
    rfm = rfm[rfm["monetary"] > 0]  # 过滤净退款用户

    rfm["R_score"] = pd.qcut(rfm["recency"], 4, labels=[4, 3, 2, 1]).astype(int)
    rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["segment"] = rfm.apply(lambda r: SEGMENT_MAP.get((r["R_score"], r["F_score"]), "其他"), axis=1)
    return rfm
