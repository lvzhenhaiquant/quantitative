import sys
sys.path.append('/home/zhenhai1/quantitative')

from utils.data_loader import DataLoader
import pandas as pd
import numpy as np

print("=" * 80)
print("手动计算20日动量因子")
print("=" * 80)

# Step 1: 准备数据
data_loader = DataLoader()
stock_list = data_loader.get_stock_list_by_date('csi1000', '2024-01-01')
test_stocks = stock_list[:10]  # 取10只股票测试

print(f"\n测试股票数量: {len(test_stocks)}")
print(f"测试股票: {test_stocks[:5]}...\n")

# Step 2: 加载价格数据(需要多加载20天)
prices = data_loader.load_stock_prices(
    stock_list=test_stocks,
    start_date='2024-01-01',
    end_date='2024-01-31',
    fields=['$close'],
    lookback_days=30  # 多加载30天
)

print(f"价格数据形状: {prices.shape}")
print(f"价格数据前5行:\n{prices.head()}\n")

# Step 3: 计算动量因子
momentum_dict = {}

for stock in test_stocks:
    try:
        # 提取该股票的价格序列
        stock_prices = prices.loc[stock]['$close']

        # 确保有足够数据
        if len(stock_prices) < 20:
            print(f"⚠ {stock}: 数据不足,跳过")
            continue

        # 计算动量: (当前价格 / 20天前价格) - 1
        current_price = stock_prices.iloc[-1]      # 最新价格
        past_price = stock_prices.iloc[-20]        # 20天前价格

        momentum = (current_price / past_price) - 1

        momentum_dict[stock] = momentum

        print(f"✓ {stock}: 当前={current_price:.2f}, 20天前={past_price:.2f}, 动量={momentum:.4f}")

    except Exception as e:
        print(f"✗ {stock}: 计算失败 - {e}")
        continue

# Step 4: 转换为DataFrame
momentum_df = pd.DataFrame.from_dict(
    momentum_dict,
    orient='index',
    columns=['momentum']
)

print("\n" + "=" * 80)
print("因子计算结果")
print("=" * 80)
print(momentum_df)

print("\n" + "=" * 80)
print("因子统计")
print("=" * 80)
print(momentum_df.describe())

print("\n✓ 手动计算因子完成!")