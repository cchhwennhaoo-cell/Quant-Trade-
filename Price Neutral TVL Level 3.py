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
# 3. 背离强度（原始）
# =========================================================
df['divergence_strength'] = df['eth_return'] - df['pntvl_change']

# =========================================================
# 4. ⭐ 滑动窗口 Z-score（核心升级）
# =========================================================
window = 75  # 推荐 30 / 60 / 90 你可以回测比较

df['div_mean'] = df['divergence_strength'].rolling(window).mean()
df['div_std'] = df['divergence_strength'].rolling(window).std()

df['divergence_z'] = (
    df['divergence_strength'] - df['div_mean']
) / df['div_std']

# =========================================================
# 5. ⭐ 信号生成（基于 Z-score）
# =========================================================
z_threshold = 1.1  # ⭐ 推荐从 1.2 开始

df['signal'] = 0

# 做多：ETH 跌 + PNTVL 涨 + 背离极端（负）
df.loc[
    (df['eth_return'] < 0) &
    (df['pntvl_change'] > 0) &
    (df['divergence_z'] < -z_threshold),
    'signal'
] = 1

# 做空：ETH 涨 + PNTVL 跌 + 背离极端（正）
df.loc[
    (df['eth_return'] > 0) &
    (df['pntvl_change'] < 0) &
    (df['divergence_z'] > z_threshold),
    'signal'
] = -1

# =========================================================
# 6. ⭐ T+1 执行
# =========================================================
df['position'] = df['signal'].shift(1).fillna(0)

# =========================================================
# 7. 策略收益 & 资金曲线
# =========================================================
df['strategy_return'] = df['position'] * df['eth_return']
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

sharpe_ratio = (
    df['strategy_return'].mean() /
    df['strategy_return'].std()
) * np.sqrt(365)

calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

# =========================================================
# 9. 交易统计
# =========================================================
df['prev_position'] = df['position'].shift(1)
df['trade'] = (df['prev_position'] != df['position']) & (df['prev_position'] != 0)

trade_count = df['trade'].sum()

df['trade_id'] = df['trade'].cumsum()
trade_returns = df[df['trade']].groupby('trade_id')['strategy_return'].sum()
win_rate = (trade_returns > 0).mean()

# =========================================================
# 10. 打印结果
# =========================================================
pd.set_option('display.max_rows', 40)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1200)
pd.set_option('display.float_format', '{:.4f}'.format)

print(df[['date','eth_price','price_neutral_tvl_2dec',
          'eth_return','pntvl_change',
          'divergence_strength','divergence_z',
          'signal','position']])

print("\n========== Strategy Performance Level 3 ==========")
print(f"Annual Return    : {annual_return:.2%}")
print(f"Sharpe Ratio     : {sharpe_ratio:.2f}")
print(f"Calmar Ratio     : {calmar_ratio:.2f}")
print(f"Max Drawdown     : {max_drawdown:.2%}")
print(f"Win Rate         : {win_rate:.2%}")
print(f"Trade Count      : {trade_count}")
print("==========================================\n")

# =========================================================
# 11. 资金曲线可视化
# =========================================================
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['equity_curve'],
    mode='lines',
    name='Equity Curve'
))

fig.update_layout(
    title='Equity Curve (Rolling Z-score Divergence Strategy)',
    xaxis_title='Date',
    yaxis_title='Net Value',
    width=1200,
    height=600
)

fig.show()
