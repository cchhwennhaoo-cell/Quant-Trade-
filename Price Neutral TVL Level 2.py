import pandas as pd
import numpy as np

# ========= 1. 读取 CSV =========
tvl_path = "ethereum_tvl_2023-01-01_2026-01-01.csv"
price_path = "kline_ETHUSDT_D_20230101_20260101_spot.csv"

tvl_df = pd.read_csv(tvl_path)
price_df = pd.read_csv(price_path)

# ========= 2. 统一日期 =========
tvl_df['date'] = pd.to_datetime(tvl_df['date']).dt.date
price_df['date'] = pd.to_datetime(price_df['datetime']).dt.date

# ========= 3. 精简价格列 =========
price_df = price_df[['date', 'close']].rename(columns={'close': 'eth_price'})

# ========= 4. 按日期合并 =========
df = pd.merge(tvl_df, price_df, on='date', how='inner')

# ========= 5. 计算 Price Neutral TVL =========
df['price_neutral_tvl'] = df['tvl_usd'] / df['eth_price']

# ========= 6. 保留小数点后 2 位 =========
df['price_neutral_tvl_2dec'] = df['price_neutral_tvl'].round(2)

# ========= 7. 计算变化率 =========
df['pntvl_change'] = df['price_neutral_tvl_2dec'].pct_change()
df['eth_return'] = df['eth_price'].pct_change()

# ========= 8. 显示结果（控制打印格式） =========
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)

print(df.head(20))
print("\n--- Tail ---\n")
print(df.tail(20))
