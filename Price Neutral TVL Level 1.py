import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# ========= 浏览器显示 =========
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

# ========= 4. 合并 TVL 与价格 =========
df = pd.merge(tvl_df, price_df, on='date', how='inner')

# ========= 5. 计算价格中性 TVL =========
df['price_neutral_tvl'] = df['tvl_usd'] / df['eth_price']

# 四舍五入 4 位有效数字
def round_to_4sig_int(x):
    if x == 0: return 0
    e = int(np.floor(np.log10(abs(x))))
    scaled = x / (10**e)
    rounded = round(scaled, 3)
    return int(rounded * (10**e))

df['price_neutral_tvl_4sig'] = df['price_neutral_tvl'].apply(round_to_4sig_int)

# ========= 6. 计算每日变化率 =========
df['pntvl_change'] = df['price_neutral_tvl_4sig'].pct_change()
df['eth_return'] = df['eth_price'].pct_change()

# ========= 7. 生成交易信号 =========
def trading_signal(row):
    if row['pntvl_change'] > row['eth_return']:
        return 1
    elif row['pntvl_change'] < row['eth_return']:
        return -1
    else:
        return 0

df['signal'] = df.apply(trading_signal, axis=1)

# ========= 8. 生成 T+1 持仓 =========
df['position'] = df['signal'].shift(1).fillna(0)

# ========= 9. 策略每日收益 =========
df['strategy_return'] = df['position'] * df['eth_return']

# ========= 10. 累积资金曲线 =========
df['equity_curve'] = (1 + df['strategy_return']).cumprod()

# ========= 11. 回测指标 =========
trading_days = 365
total_years = len(df) / trading_days

# CAGR（复利年化收益率）
cagr = df['equity_curve'].iloc[-1]**(1/total_years) - 1

# 年化回报率（算术平均）
annual_return = df['strategy_return'].mean() * trading_days

# 最大回撤
cum_max = df['equity_curve'].cummax()
drawdown = df['equity_curve'] / cum_max - 1
max_drawdown = drawdown.min()

# 夏普比率
sharpe_ratio = df['strategy_return'].mean() / df['strategy_return'].std() * np.sqrt(trading_days)

# Calmar 比率
calmar_ratio = cagr / abs(max_drawdown)

# 胜率
winning_rate = (df['strategy_return'] > 0).sum() / (df['strategy_return'] != 0).sum()

# ========= 12. 计算总交易笔数（方向变化计算） =========
df['position_shift'] = df['position'].shift(1).fillna(0)
df['trade_flag'] = (df['position'] != df['position_shift']).astype(int)
total_trades = df['trade_flag'].sum()

# ========= 13. 输出指标 =========
print("===== 策略回测指标 (T+1 开仓 + 持仓方向变化计数) =====")
print(f"CAGR (复利年化收益率): {cagr:.2%}")
print(f"年化回报率 (算术平均): {annual_return:.2%}")
print(f"最大回撤: {max_drawdown:.2%}")
print(f"夏普比率: {sharpe_ratio:.2f}")
print(f"Calmar比率: {calmar_ratio:.2f}")
print(f"胜率: {winning_rate:.2%}")
print(f"总交易笔数（每次方向变化算一次）: {total_trades}")

# ========= 14. 可视化资金曲线 + ETH 收盘价 + 交易信号 =========
fig = go.Figure()

# 策略资金曲线
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['equity_curve'],
    mode='lines',
    name='Equity Curve',
    line=dict(color='blue', width=2)
))

# ETH 收盘价（红色线）
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['eth_price'],
    mode='lines',
    name='ETH Close Price',
    line=dict(color='red', width=2),
    yaxis='y2'
))

# 做多点（signal=1）蓝色小点
long_signals = df[df['signal'] == 1]
fig.add_trace(go.Scatter(
    x=long_signals['date'],
    y=long_signals['eth_price'],
    mode='markers',
    marker=dict(color='blue', size=6),
    name='Long Signal',
    yaxis='y2'
))

# 做空点（signal=-1）黑色小点
short_signals = df[df['signal'] == -1]
fig.add_trace(go.Scatter(
    x=short_signals['date'],
    y=short_signals['eth_price'],
    mode='markers',
    marker=dict(color='black', size=6),
    name='Short Signal',
    yaxis='y2'
))

# 双 Y 轴布局
fig.update_layout(
    title='策略资金曲线 + ETH 收盘价 + 交易信号',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Equity Curve', side='left'),
    yaxis2=dict(title='ETH Close Price', overlaying='y', side='right'),
    template='plotly_white'
)

fig.show()

# ========= 15. 查看前几行 =========
print(df[['date','price_neutral_tvl_4sig','pntvl_change','eth_return','signal','position','strategy_return','equity_curve']].head())
