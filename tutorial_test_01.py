import sys
sys.path.append('/home/zhenhai1/quantitative')

from utils.data_loader import DataLoader

# 创建数据加载器
print("=" * 80)
print("第1步: 创建DataLoader")
print("=" * 80)
data_loader = DataLoader()
print("✓ DataLoader创建成功\n")

# 测试1: 加载中证1000股票池
print("=" * 80)
print("第2步: 加载中证1000股票池")
print("=" * 80)
stock_list = data_loader.get_stock_list_by_date('csi1000', '2024-01-01')
print(f"✓ 成功加载 {len(stock_list)} 只股票")
print(f"前5只股票: {stock_list[:5]}\n")

# 测试2: 加载3只股票的价格数据
print("=" * 80)
print("第3步: 加载股票价格数据")
print("=" * 80)
test_stocks = stock_list[:3]
print(f"测试股票: {test_stocks}")

prices = data_loader.load_stock_prices(
    stock_list=test_stocks,
    start_date='2024-01-01',
    end_date='2024-01-31',
    fields=['$close']
)
print(f"✓ 成功加载价格数据")
print(f"数据形状: {prices.shape}")
print(f"\n前5行数据:\n{prices.head()}\n")

# 测试3: 加载基准指数
print("=" * 80)
print("第4步: 加载基准指数(沪深300)")
print("=" * 80)
benchmark = data_loader.load_benchmark_prices(
    benchmark='sh000300',
    start_date='2024-01-01',
    end_date='2024-01-31'
)
print(f"✓ 成功加载基准数据")
print(f"数据长度: {len(benchmark)}")
print(f"\n前5行数据:\n{benchmark.head()}\n")

print("=" * 80)
print("所有测试通过! DataLoader工作正常")
print("=" * 80)