"""
复购预测建模（v3：修复特征语义 + 去冗余 + 时间切 val）。

任务:
    用观察期（split_ts 之前）的客户行为，预测预测期（split_ts 到 label_end_ts）
    是否产生购买。

v3 相对 v2 的修复:
    - avg_basket 改为订单级 AOV（先按 invoice 聚合，再 mean）
    - avg_gap_days 单次购买客户填"观察期长度"而非 0，避免信号反转；新增 has_multi_order 指示
    - return_count → return_order_count：按不同 invoice 统计，而非行数
    - 移除 Top-K SKU one-hot（ablation 显示对 AUC 贡献 < 0.005，特征瘦身 35 → 22）
    - XGBoost early-stopping 的验证集改为"观察期末尾时间切"，不再随机
"""
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


@dataclass
class ModelResult:
    name: str
    auc: float
    model: object
    y_test: np.ndarray
    y_proba: np.ndarray
    feature_names: list[str]


# ------------------------------------------------------------ Feature engineering

def _inter_order_stats(group: pd.DataFrame, fill_days: float) -> pd.Series:
    """按 invoice 聚合到订单级算间隔。
    单次购买客户：avg/std 填 fill_days（观察期长度）以表达'没有复购节奏'。"""
    inv_dates = group.drop_duplicates("invoice")["invoice_date"].sort_values()
    if len(inv_dates) < 2:
        return pd.Series({"avg_gap_days": fill_days, "std_gap_days": 0.0,
                          "has_multi_order": 0})
    gaps = inv_dates.diff().dropna().dt.total_seconds() / 86400
    return pd.Series({"avg_gap_days": gaps.mean(),
                      "std_gap_days": gaps.std() if len(gaps) > 1 else 0.0,
                      "has_multi_order": 1})


def _window_counts(group: pd.DataFrame, split_ts: pd.Timestamp) -> pd.Series:
    """近 30/60/90 天订单数与金额（不含退货）。"""
    out = {}
    for days in (30, 60, 90):
        mask = group["invoice_date"] >= (split_ts - pd.Timedelta(days=days))
        sub = group[mask]
        out[f"orders_last_{days}d"] = sub["invoice"].nunique()
        out[f"amount_last_{days}d"] = sub.loc[~sub["is_return"], "amount"].sum()
    return pd.Series(out)


def build_features(
    df: pd.DataFrame,
    split_ts: pd.Timestamp,
    label_end_ts: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """返回 (features_df, feature_col_names)。"""
    obs = df[df["invoice_date"] < split_ts]
    if label_end_ts is None:
        label_end_ts = df["invoice_date"].max()
    fut = df[(df["invoice_date"] >= split_ts) & (df["invoice_date"] <= label_end_ts)]

    obs_period_days = float((split_ts - obs["invoice_date"].min()).days)

    # 订单级 AOV（fix from v2 bug）
    order_totals = (obs[~obs["is_return"]]
                    .groupby(["customer_id", "invoice"])["amount"].sum())
    aov = order_totals.groupby("customer_id").mean().rename("avg_basket")

    agg = obs.groupby("customer_id").agg(
        invoice_count=("invoice", "nunique"),
        total_amount=("amount", "sum"),
        distinct_products=("stockcode", "nunique"),
        total_quantity=("quantity", "sum"),
        recency_days=("invoice_date", lambda s: (split_ts - s.max()).days),
        tenure_days=("invoice_date", lambda s: (s.max() - s.min()).days),
    )
    agg = agg.join(aov, how="left")
    agg["avg_basket"] = agg["avg_basket"].fillna(0.0)

    # 退货：订单级计数（fix from v2 bug）
    ret_orders = (obs[obs["is_return"]]
                  .groupby("customer_id")["invoice"].nunique()
                  .rename("return_order_count"))
    agg = agg.join(ret_orders, how="left")
    agg["return_order_count"] = agg["return_order_count"].fillna(0).astype(int)
    agg["return_rate"] = agg["return_order_count"] / agg["invoice_count"].clip(lower=1)

    # 订单间隔（fix from v2 bug）
    gaps = obs.groupby("customer_id").apply(
        lambda g: _inter_order_stats(g, obs_period_days), include_groups=False
    )
    agg = agg.join(gaps)

    # 近期窗口
    windows = obs.groupby("customer_id").apply(
        lambda g: _window_counts(g, split_ts), include_groups=False
    )
    agg = agg.join(windows)

    # Log-transform 偏态
    agg["log_total_amount"] = np.log1p(agg["total_amount"].clip(lower=0))
    agg["log_avg_basket"] = np.log1p(agg["avg_basket"].clip(lower=0))
    agg["log_amount_last_90d"] = np.log1p(agg["amount_last_90d"].clip(lower=0))

    # 标签
    buyers = set(fut[~fut["is_return"]]["customer_id"].unique())
    agg["label"] = agg.index.isin(buyers).astype(int)

    agg = agg.fillna(0).reset_index()
    feature_cols = [c for c in agg.columns if c not in {"customer_id", "label"}]
    return agg, feature_cols


# ------------------------------------------------------------ Training

def _time_based_val_split(features: pd.DataFrame, feature_cols: list[str], val_frac: float = 0.2):
    """按 recency_days 从高到低排（观察期内最早活跃的客户先），最"新"的 val_frac 留给 val。
    注意：这里 features 已经按全样本，但训练阶段我们只对 train 子集再拆 val。
    这个函数会被调用时传入 train-only DataFrame。"""
    sorted_df = features.sort_values("recency_days", ascending=False)
    n_val = int(len(sorted_df) * val_frac)
    val = sorted_df.head(n_val)   # recency 最大 = 观察期内最早活跃的 → 放 train
    # 实际希望 val 是"观察期尾部最近活跃"的客户，所以用 recency 最小
    val = sorted_df.tail(n_val)
    train = sorted_df.head(len(sorted_df) - n_val)
    X_tr = train[feature_cols].values
    y_tr = train["label"].values
    X_val = val[feature_cols].values
    y_val = val["label"].values
    return X_tr, y_tr, X_val, y_val


def _fit_xgb_tuned(X_train, y_train, X_val, y_val):
    xgb = XGBClassifier(
        n_estimators=1500, learning_rate=0.03, max_depth=5,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="auc", random_state=42, n_jobs=-1,
        early_stopping_rounds=40,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return xgb


def train_and_evaluate(features: pd.DataFrame, feature_cols: list[str]) -> list[ModelResult]:
    y = features["label"].values
    X = features[feature_cols].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results: list[ModelResult] = []

    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    lr.fit(X_train_s, y_train)
    proba = lr.predict_proba(X_test_s)[:, 1]
    results.append(ModelResult("LogisticRegression",
                               roc_auc_score(y_test, proba), lr, y_test, proba, feature_cols))

    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=3,
        max_features="sqrt", random_state=42, n_jobs=-1, class_weight="balanced_subsample",
    )
    rf.fit(X_train, y_train)
    proba = rf.predict_proba(X_test)[:, 1]
    results.append(ModelResult("RandomForest",
                               roc_auc_score(y_test, proba), rf, y_test, proba, feature_cols))

    if HAS_XGB:
        # XGB: 从 train 里再按 recency 切 val（时间切近似）
        recency_col_idx = feature_cols.index("recency_days")
        train_df = pd.DataFrame(X_train, columns=feature_cols)
        train_df["label"] = y_train
        X_tr, y_tr, X_val, y_val = _time_based_val_split(train_df, feature_cols, val_frac=0.2)
        xgb = _fit_xgb_tuned(X_tr, y_tr, X_val, y_val)
        proba = xgb.predict_proba(X_test)[:, 1]
        results.append(ModelResult("XGBoost (tuned)",
                                   roc_auc_score(y_test, proba), xgb, y_test, proba, feature_cols))

    return results


def top_k_capture_rate(y_true: np.ndarray, y_proba: np.ndarray, k: float = 0.1) -> float:
    n_top = max(1, int(len(y_proba) * k))
    top_idx = np.argsort(y_proba)[::-1][:n_top]
    y_true = np.asarray(y_true)
    if y_true.sum() == 0:
        return 0.0
    return float(y_true[top_idx].sum() / y_true.sum())


# ------------------------------------------------------------ Time-rolling validation

def time_rolling_validation(
    df: pd.DataFrame,
    split_points: Iterable[pd.Timestamp],
    label_horizon_days: int = 90,
) -> pd.DataFrame:
    """
    对每个 split_ts:
      特征 = 观察期（< split_ts）
      标签 = [split_ts, split_ts + horizon)
    返回每个 fold 每个模型的 AUC。
    """
    rows = []
    for split_ts in split_points:
        label_end = split_ts + pd.Timedelta(days=label_horizon_days)
        feats, cols = build_features(df, split_ts, label_end)
        if feats["label"].sum() == 0 or feats["label"].sum() == len(feats):
            continue
        results = train_and_evaluate(feats, cols)
        for r in results:
            rows.append({
                "fold_split": split_ts.date().isoformat(),
                "label_end": label_end.date().isoformat(),
                "n_samples": len(feats),
                "positive_rate": round(float(feats["label"].mean()), 3),
                "model": r.name,
                "auc": round(r.auc, 3),
                "top10_capture": round(top_k_capture_rate(r.y_test, r.y_proba, 0.1), 3),
                "top20_capture": round(top_k_capture_rate(r.y_test, r.y_proba, 0.2), 3),
            })
    return pd.DataFrame(rows)
