"""
清洗 Online Retail II 数据 → 输出 Parquet 子集。

原始字段:
    Invoice, StockCode, Description, Quantity, InvoiceDate,
    Price, Customer ID, Country

清洗规则:
    - 丢弃 Customer ID 缺失的行（占 ~23%，无法做用户维度分析）
    - 拆分退货: Invoice 以 'C' 开头 或 Quantity < 0 → is_return=True
    - 计算 amount = Quantity × Price（退货为负）
    - 过滤 Price <= 0 的异常行（手续费/调整单）

用法:
    python src/preprocessing.py
"""
from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/online_retail_II.csv")
OUT_PATH = Path("data/processed/transactions.parquet")


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, low_memory=False)
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
    # → invoice, stockcode, description, quantity, invoicedate, price, customer_id, country

    df = df.rename(columns={"invoicedate": "invoice_date"})
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["is_return"] = df["invoice"].astype(str).str.startswith("C") | (df["quantity"] < 0)
    df["amount"] = df["quantity"] * df["price"]

    df = df.dropna(subset=["customer_id"])
    df = df[df["price"] > 0]
    df["customer_id"] = df["customer_id"].astype(int)
    df["description"] = df["description"].fillna("").str.strip()

    return df.sort_values(["customer_id", "invoice_date"]).reset_index(drop=True)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = load_and_clean()
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(df):,} rows · {df['customer_id'].nunique():,} customers → {OUT_PATH}")
    print(f"Date range: {df['invoice_date'].min()} ~ {df['invoice_date'].max()}")
    print(f"Returns: {df['is_return'].sum():,} rows · total refund {df.loc[df['is_return'], 'amount'].sum():,.0f}")


if __name__ == "__main__":
    main()
