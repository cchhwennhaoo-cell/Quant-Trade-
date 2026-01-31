import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

# =========================================================
# 1. 读取数据
# =========================================================
tvl_path = "ethereum_tvl_2023-01-01_2025-01-01.csv"
price_path = "kline_ETHUSDT_D_20230101_20250101.csv"

tvl_df = pd.read_csv(tvl_path)
price_df = pd.read_csv(price_path)

tvl_df['date'] = pd.to_datetime(tvl_df['date']).dt.date
price_df['date'] = pd.to_datetime(price_df['datetime']).dt.date

price_df = price_df[['date', 'close']].rename(columns={'close': 'eth_price'})
df = pd.merge(tvl_df, price_df, on='date', how='inner')

# =========================================================
# 2. 构造指标
# =========================================================
df['price_neutral_tvl'] = df['tvl_usd'] / df['eth_price']
df['price_neutral_tvl_2dec'] = df['price_neutral_tvl'].round(2)
df['eth_return'] = df['eth_price'].pct_change()
df['pntvl_change'] = df['price_neutral_tvl_2dec'].pct_change()

# =========================================================
# 3. 背离强度
# =========================================================
df['divergence_strength'] = df['eth_return'] - df['pntvl_change']

# =========================================================
# 4. 滑动窗口 Z-score
# =========================================================
window = 75
df['div_mean'] = df['divergence_strength'].rolling(window).mean()
df['div_std'] = df['divergence_strength'].rolling(window).std()
df['divergence_z'] = (df['divergence_strength'] - df['div_mean']) / df['div_std']

# =========================================================
# 5. 信号生成
# =========================================================
z_threshold = 1.1
df['signal'] = 0
df.loc[
    (df['eth_return'] < 0) & (df['pntvl_change'] > 0) & (df['divergence_z'] < -z_threshold),
    'signal'
] = 1
df.loc[
    (df['eth_return'] > 0) & (df['pntvl_change'] < 0) & (df['divergence_z'] > z_threshold),
    'signal'
] = -1

# =========================================================
# 6. T+1 执行
# =========================================================
df['position'] = df['signal'].shift(1).fillna(0)

# =========================================================
# 7. 策略收益（加入手续费 & 滑点）
# =========================================================
fee_rate = 0.0005
slippage_rate = 0.0002
cost_rate = fee_rate + slippage_rate

df['prev_position'] = df['position'].shift(1).fillna(0)
df['turnover'] = (df['position'] - df['prev_position']).abs()
df['cost_return'] = df['turnover'] * cost_rate
df['strategy_return'] = df['position'] * df['eth_return'] - df['cost_return']
df['strategy_return'] = df['strategy_return'].fillna(0)
df['equity_curve'] = (1 + df['strategy_return']).cumprod()

# =========================================================
# 8. 回撤 & 绩效指标
# =========================================================
df['equity_peak'] = df['equity_curve'].cummax()
df['drawdown'] = df['equity_curve'] / df['equity_peak'] - 1
max_drawdown = df['drawdown'].min()
total_days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
annual_return = df['equity_curve'].iloc[-1] ** (365 / total_days) - 1
sharpe_ratio = (df['strategy_return'].mean() / df['strategy_return'].std()) * np.sqrt(365)
calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

# =========================================================
# 9. 交易统计
# =========================================================
df['trade'] = (df['turnover'] > 0)
trade_count = df['trade'].sum()
df['trade_id'] = df['trade'].cumsum()
trade_returns = df[df['trade']].groupby('trade_id')['strategy_return'].sum()
win_rate = (trade_returns > 0).mean()

# =========================================================
# 10. 打印结果
# =========================================================
print("\n========== Strategy Performance Level 4 ==========")
print(f"Annual Return    : {annual_return:.2%}")
print(f"Sharpe Ratio     : {sharpe_ratio:.2f}")
print(f"Calmar Ratio     : {calmar_ratio:.2f}")
print(f"Max Drawdown     : {max_drawdown:.2%}")
print(f"Win Rate         : {win_rate:.2%}")
print(f"Trade Count      : {trade_count}")
print("==========================================\n")

# =========================================================
# 11. 资金曲线可视化（保留原图）
# =========================================================
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=df['date'],
    y=df['equity_curve'],
    mode='lines',
    name='Equity Curve'
))
fig1.update_layout(
    title='Equity Curve (After Cost)',
    xaxis_title='Date',
    yaxis_title='Net Value',
    width=1200,
    height=600
)
fig1.show()

# =========================================================
# 12. ETH收盘价 + 交易信号图（新图，反转信号直接标开仓）
# =========================================================
df['trade_type'] = np.nan
for i in range(1, len(df)):
    prev_pos = df.loc[i-1, 'position']
    curr_pos = df.loc[i, 'position']

    if curr_pos != prev_pos:  # 仓位变化
        if curr_pos > 0:
            df.loc[i, 'trade_type'] = 'Buy'      # 做多开仓（或反转到多）
        elif curr_pos < 0:
            df.loc[i, 'trade_type'] = 'Sell'     # 做空开仓（或反转到空）
        # curr_pos == 0 可以忽略，因为你的策略反转信号几乎不会出现完全平仓

fig2 = go.Figure()

# ETH 收盘价
fig2.add_trace(go.Scatter(
    x=df['date'],
    y=df['eth_price'],
    mode='lines',
    name='ETH Close Price',
    line=dict(color='blue')
))

# 做多开仓点
buy_points = df[df['trade_type'] == 'Buy']
fig2.add_trace(go.Scatter(
    x=buy_points['date'],
    y=buy_points['eth_price'],
    mode='markers',
    marker=dict(symbol='triangle-up', color='green', size=12),
    name='Buy (Long)'
))

# 做空开仓点
sell_points = df[df['trade_type'] == 'Sell']
fig2.add_trace(go.Scatter(
    x=sell_points['date'],
    y=sell_points['eth_price'],
    mode='markers',
    marker=dict(symbol='triangle-down', color='red', size=12),
    name='Sell (Short)'
))

fig2.update_layout(
    title='ETH Close Price with Trade Signals (Reversal Marked as Open)',
    xaxis_title='Date',
    yaxis_title='ETH Price',
    width=1200,
    height=600
)
fig2.show()
