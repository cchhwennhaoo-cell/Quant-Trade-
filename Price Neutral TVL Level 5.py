import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

# =========================================================
# 1. 读取数据
# =========================================================
tvl_path = "ethereum_tvl_2022-01-01_2025-01-01.csv"
price_path = "kline_ETHUSDT_D_20220101_20250101.csv"

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
# 4. Z-score
# =========================================================
window = 75
df['div_mean'] = df['divergence_strength'].rolling(window).mean()
df['div_std'] = df['divergence_strength'].rolling(window).std()
df['divergence_z'] = (df['divergence_strength'] - df['div_mean']) / df['div_std']

# =========================================================
# 5. 信号生成（保持你原逻辑）
# =========================================================
z_threshold = 1.1
df['signal'] = 0

df.loc[
    (df['eth_return'] < 0) &
    (df['pntvl_change'] > 0) &
    (df['divergence_z'] < -z_threshold),
    'signal'
] = 1

df.loc[
    (df['eth_return'] > 0) &
    (df['pntvl_change'] < 0) &
    (df['divergence_z'] > z_threshold),
    'signal'
] = -1

# =========================================================
# 6. T+1 执行 + 仓位比例
# =========================================================
max_position = 0.3          # 最大 30% 仓位（关键）
df['position'] = df['signal'].shift(1).fillna(0) * max_position

# =========================================================
# 7. 资金级回测（核心）
# =========================================================
initial_capital = 100000.0
fee_rate = 0.0005
slippage_rate = 0.0002
cost_rate = fee_rate + slippage_rate

df['capital'] = initial_capital
df['prev_capital'] = df['capital'].shift(1)

df['prev_position'] = df['position'].shift(1).fillna(0)
df['turnover'] = (df['position'] - df['prev_position']).abs()

# 每日盈亏
df['gross_pnl'] = df['prev_capital'] * df['position'] * df['eth_return']
df['cost'] = df['prev_capital'] * df['turnover'] * cost_rate
df['daily_pnl'] = df['gross_pnl'] - df['cost']

# 更新资金
df['capital'] = initial_capital + df['daily_pnl'].cumsum()
df['capital'] = df['capital'].fillna(initial_capital)

# 资金曲线
df['equity_curve'] = df['capital'] / initial_capital

# =========================================================
# 8. 回撤 & 风控（非常重要）
# =========================================================
df['equity_peak'] = df['capital'].cummax()
df['drawdown'] = df['capital'] / df['equity_peak'] - 1

# 回撤超过 15% → 仓位减半
df.loc[df['drawdown'] < -0.15, 'position'] *= 0.5

# =========================================================
# 9. 绩效指标
# =========================================================
total_days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
annual_return = df['equity_curve'].iloc[-1] ** (365 / total_days) - 1

daily_return = df['capital'].pct_change().dropna()
sharpe_ratio = daily_return.mean() / daily_return.std() * np.sqrt(365)

max_drawdown = df['drawdown'].min()
calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

# =========================================================
# 10. 交易统计
# =========================================================
df['trade'] = df['turnover'] > 0
trade_count = df['trade'].sum()

df['trade_id'] = df['trade'].cumsum()
trade_returns = df[df['trade']].groupby('trade_id')['daily_pnl'].sum()
win_rate = (trade_returns > 0).mean()

# =========================================================
# 11. 打印结果
# =========================================================
print("\n========== Capital-Based Backtest ==========")
print(f"Initial Capital : {initial_capital:,.0f}")
print(f"Final Capital   : {df['capital'].iloc[-1]:,.0f}")
print(f"Annual Return   : {annual_return:.2%}")
print(f"Sharpe Ratio    : {sharpe_ratio:.2f}")
print(f"Calmar Ratio    : {calmar_ratio:.2f}")
print(f"Max Drawdown    : {max_drawdown:.2%}")
print(f"Win Rate        : {win_rate:.2%}")
print(f"Trade Count     : {trade_count}")
print("===========================================\n")

# =========================================================
# 12. 资金曲线
# =========================================================
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['equity_curve'],
    mode='lines',
    name='Equity Curve'
))
fig.update_layout(
    title='Equity Curve (Capital-Based Backtest)',
    xaxis_title='Date',
    yaxis_title='Net Value',
    width=1200,
    height=600
)
fig.show()
