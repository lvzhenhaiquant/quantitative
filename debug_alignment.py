"""
调试日期对齐问题 - 模拟回测中的日期对齐过程
"""

import sys
sys.path.append('/home/zhenhai1/quantitative')

import yaml
import pandas as pd
from utils.data_loader import DataLoader

print("=" * 80)
print("调试日期对齐")
print("=" * 80)

# 1. 加载配置
with open('configs/backtest_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

benchmark_code = config['backtest']['benchmark']['code']

# 2. 初始化数据加载器
data_loader = DataLoader()

# 3. 加载基准数据（完整的每日数据）
start_date = '2020-12-11'
end_date = '2021-02-28'

benchmark_data = data_loader.load_benchmark_prices(
    benchmark_code,
    start_date,
    end_date
)

print(f"\n原始基准数据（每日）: {len(benchmark_data)} 天")
print(f"起始: {benchmark_data.iloc[0]:.2f}")
print(f"结束: {benchmark_data.iloc[-1]:.2f}")
print(f"趋势: {'上涨' if benchmark_data.iloc[-1] > benchmark_data.iloc[0] else '下跌'}")

# 4. 模拟每周调仓 - 只取每周第一天的数据
print("\n" + "=" * 80)
print("模拟weekly调仓 - 取每周第一个交易日")
print("=" * 80)

all_dates = benchmark_data.index
rebalance_dates = []

# 使用"前后两天连不上"的逻辑（与backtester.py相同）
for i in range(len(all_dates)):
    if i == 0:
        rebalance_dates.append(all_dates[i])
    else:
        gap = (all_dates[i] - all_dates[i-1]).days
        if gap > 1:
            rebalance_dates.append(all_dates[i])

print(f"\n调仓日期数量: {len(rebalance_dates)}")
print("\n前10个调仓日:")
for d in rebalance_dates[:10]:
    print(f"  {d}")

# 5. 对齐基准数据（只保留调仓日的数据）
print("\n" + "=" * 80)
print("对齐基准数据到调仓日")
print("=" * 80)

# 模拟 backtester.py 中的对齐逻辑
portfolio_dates = pd.to_datetime(rebalance_dates)
benchmark_aligned = benchmark_data.loc[portfolio_dates]

print(f"\n对齐后的基准数据: {len(benchmark_aligned)} 个调仓日")
print("\n前10个数据点:")
print(benchmark_aligned.head(10))

# 6. 归一化
print("\n" + "=" * 80)
print("归一化后的基准")
print("=" * 80)

benchmark_nav = benchmark_aligned.values / benchmark_aligned.values[0]

print(f"\n归一化起点: {benchmark_nav[0]:.6f}")
print(f"归一化终点: {benchmark_nav[-1]:.6f}")
print(f"累计收益率: {(benchmark_nav[-1] - 1) * 100:.2f}%")

# 7. 检查是否单调递减
is_decreasing = all(benchmark_nav[i] >= benchmark_nav[i+1] for i in range(len(benchmark_nav)-1))
print(f"\n是否单调递减: {is_decreasing}")

if is_decreasing:
    print("\n⚠️ 警告: 基准数据确实一直向下！")
else:
    print("\n✓ 正常: 基准数据有波动")

# 8. 打印所有归一化后的值
print("\n" + "=" * 80)
print("所有归一化后的基准值:")
print("=" * 80)
for i, (date, nav) in enumerate(zip(rebalance_dates, benchmark_nav)):
    print(f"{i+1:2d}. {date}  NAV={nav:.6f}  变化={((nav/benchmark_nav[i-1]-1)*100 if i > 0 else 0):.2f}%")
