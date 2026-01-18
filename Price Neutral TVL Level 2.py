import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# ========= 设置浏览器渲染 =========
pio.renderers.default = "browser"

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

# ========= 6. 四舍五入到 4 位有效数字并去掉小数点 =========
def round_to_4sig_int(x):
    if x == 0:
        return 0
    e = int(np.floor(np.log10(abs(x))))
    scaled = x / (10**e)
    rounded = round(scaled, 3)
    return int(rounded * (10**e))

df['price_neutral_tvl_4sig'] = df['price_neutral_tvl'].apply(round_to_4sig_int)

# ========= 7. 计算变化率 =========
df['pntvl_change'] = df['price_neutral_tvl_4sig'].pct_change()
df['eth_return'] = df['eth_price'].pct_change()

# ========= 8. 绘制 Plotly 图 =========
fig = go.Figure()

# Δ Price Neutral TVL
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['pntvl_change'],
    mode='lines+markers',
    name='Δ Price Neutral TVL',
    line=dict(color='blue')
))

# Δ ETH Price
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['eth_return'],
    mode='lines+markers',
    name='Δ ETH Price',
    line=dict(color='red')
))

# 布局
fig.update_layout(
    title='Δ Price Neutral TVL vs Δ ETH Price',
    xaxis_title='Date',
    yaxis_title='Daily Change (%)',
    legend=dict(x=0, y=1),
    template='plotly_white'
)

# ========= 9. 在浏览器打开 =========
fig.show()
