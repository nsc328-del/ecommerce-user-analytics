# 数据说明

本目录**不存放原始数据**（已在 `.gitignore` 中排除）。

## 数据源

**Online Retail II**（UCI Machine Learning Repository）
- 官方页面：https://archive.ics.uci.edu/dataset/502/online+retail+ii
- Kaggle 镜像：https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
- 文件：`online_retail_II.csv`（~45MB 压缩 / ~90MB 解压）
- 规模：1,067,371 行 · 2009-12 ~ 2011-12（~2 年）
- 业务：英国某礼品零售商 B2B 批发订单

## 原始字段

| 字段 | 含义 |
|---|---|
| `Invoice` | 订单号（以 `C` 开头表示退货） |
| `StockCode` | 商品 SKU |
| `Description` | 商品描述 |
| `Quantity` | 件数（负数表示退货） |
| `InvoiceDate` | 下单时间 |
| `Price` | 单价（英镑） |
| `Customer ID` | 客户 ID（~23% 缺失） |
| `Country` | 客户所在国家 |

## 清洗后字段（`data/processed/transactions.parquet`）

在 `Invoice / StockCode / Quantity / InvoiceDate / Price / Customer ID / Country / Description` 基础上：
- 列名统一为 snake_case：`invoice_date`, `customer_id` 等
- 新增 `amount = quantity × price`
- 新增 `is_return`（Invoice 以 C 开头或 quantity < 0）
- 丢弃 customer_id 缺失行（~243,007 行）与 price ≤ 0 行

## 使用步骤

1. 从上述任一链接下载 `online_retail_II.csv`
2. 放到 `data/online_retail_II.csv`
3. 运行 `python src/preprocessing.py` → 输出 `data/processed/transactions.parquet`

> 原始 CSV 与 processed 目录均已 gitignore，不会推送到 GitHub。
