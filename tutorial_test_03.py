import sys
sys.path.append('/home/zhenhai1/quantitative')

from utils.data_loader import DataLoader
from factor_production.market_factors.beta_factor import BetaFactor
import yaml

print("=" * 80)
print("使用框架计算Beta因子")
print("=" * 80)

# Step 1: 加载配置
print("\n[1] 加载因子配置...")
with open('configs/factor_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

beta_params = config['beta_factor']['params']
print(f"因子参数: {beta_params}")

# Step 2: 创建数据加载器
print("\n[2] 创建数据加载器...")
data_loader = DataLoader()
print("✓ DataLoader创建成功")

# Step 3: 获取股票池(只取10只股票做测试)
print("\n[3] 获取测试股票...")
stock_list = data_loader.get_stock_list_by_date('csi1000', '2024-01-01')
test_stocks = stock_list[:10]
print(f"测试股票数量: {len(test_stocks)}")

# Step 4: 创建Beta因子实例
print("\n[4] 创建Beta因子...")
beta_factor = BetaFactor(beta_params)
print(f"✓ {beta_factor}")

# Step 5: 计算因子
print("\n[5] 开始计算因子...")
print("(这一步会自动完成: 数据加载 → 预处理 → 计算 → 后处理)")
print("-" * 80)

factor_values = beta_factor.calculate(
    stock_list=test_stocks,
    start_date='2024-01-01',
    end_date='2024-12-01',
    data_loader=data_loader
)

print("-" * 80)
print("\n[6] 因子计算完成!")

# Step 6: 查看结果
print("\n" + "=" * 80)
print("因子值")
print("=" * 80)
print(factor_values)

print("\n" + "=" * 80)
print("因子统计")
print("=" * 80)
print(factor_values.describe())

# Step 7: 保存结果
print("\n[7] 保存结果...")
output_path = "factor_production/cache/tutorial_beta_test.csv"
beta_factor.save(factor_values, output_path)

print("\n" + "=" * 80)
print("✓ 教程完成!")
print("=" * 80)
print(f"结果已保存到: {output_path}")