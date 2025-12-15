"""
调试基准数据 - 查看回测中实际使用的基准数据
"""

import sys
sys.path.append('/home/zhenhai1/quantitative')

import yaml
import pandas as pd
from utils.data_loader import DataLoader

print("=" * 80)
print("调试基准数据")
print("=" * 80)

# 1. 加载配置
with open('configs/backtest_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

benchmark_code = config['backtest']['benchmark']['code']
benchmark_name = config['backtest']['benchmark']['name']

print(f"\n配置中的基准指数: {benchmark_code} - {benchmark_name}")

# 2. 初始化数据加载器
data_loader = DataLoader()

# 3. 加载基准数据（从2020-12-11开始，50个交易日）
start_date = '2020-12-11'
end_date = '2021-02-28'  # 大约50个交易日

print(f"\n加载基准数据: {start_date} ~ {end_date}")

benchmark_data = data_loader.load_benchmark_prices(
    benchmark_code,
    start_date,
    end_date
)

print(f"\n基准数据总共有 {len(benchmark_data)} 个交易日")
print("\n前50个交易日的基准价格:")
print(benchmark_data.head(50))

# 4. 计算归一化值
print("\n" + "=" * 80)
print("归一化后的基准（前50天）:")
print("=" * 80)

benchmark_nav = benchmark_data / benchmark_data.iloc[0]
print(benchmark_nav.head(50))

# 5. 计算累计收益
print("\n" + "=" * 80)
print("统计信息:")
print("=" * 80)
print(f"第1天价格: {benchmark_data.iloc[0]:.2f}")
print(f"第50天价格: {benchmark_data.iloc[49]:.2f}")
print(f"累计收益率: {(benchmark_data.iloc[49] / benchmark_data.iloc[0] - 1) * 100:.2f}%")
print(f"归一化起点: {benchmark_nav.iloc[0]:.6f}")
print(f"归一化终点: {benchmark_nav.iloc[49]:.6f}")
