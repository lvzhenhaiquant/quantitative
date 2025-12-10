"""
测试 FactorScheduler - 每日滚动计算因子

这个脚本用来测试FactorScheduler的功能:
1. 每日计算History_Sigma因子
2. 保存到因子库
3. 加载并验证数据
"""

import sys
sys.path.append('/home/zhenhai1/quantitative')

import yaml
from utils.data_loader import DataLoader
from factor_production.factor_scheduler import FactorScheduler
from factor_production.market_factors.beta_factor import HistorySigmaFactor

print("=" * 80)
print("测试 FactorScheduler - 每日滚动计算因子")
print("=" * 80)

# ============================================================================
# Step 1: 加载配置
# ============================================================================
print("\n[Step 1] 加载配置文件...")

# 加载调度器配置
with open('configs/scheduler_config.yaml', 'r', encoding='utf-8') as f:
    scheduler_config = yaml.safe_load(f)

# 加载因子配置
with open('configs/factor_config.yaml', 'r', encoding='utf-8') as f:
    factor_config = yaml.safe_load(f)

print("✓ 配置加载完成")

# ============================================================================
# Step 2: 初始化组件
# ============================================================================
print("\n[Step 2] 初始化组件...")

# 创建数据加载器
data_loader = DataLoader()
print("✓ DataLoader初始化完成")

# 创建调度器
scheduler = FactorScheduler(data_loader, scheduler_config['scheduler'])
print(f"✓ {scheduler}")

# 创建History_Sigma因子
history_sigma = HistorySigmaFactor(factor_config['history_sigma_factor']['params'])
print(f"✓ {history_sigma}")

# ============================================================================
# Step 3: 每日滚动计算因子（测试：2个月数据）
# ============================================================================
print("\n[Step 3] 开始每日滚动计算因子...")
print("注意: 这是测试，只计算2个月的数据")
print("如果要计算5年数据，修改下面的日期即可")
print("-" * 80)

# 测试参数
test_params = {
    'instrument_name': 'csi1000',
    'start_date': '2024-10-01',  # 测试: 2个月
    'end_date': '2024-12-01',
}

# 如果要计算5年数据，使用下面的参数:
# production_params = {
#     'instrument_name': 'csi1000',
#     'start_date': '2019-12-10',
#     'end_date': '2024-12-10',
# }

# 开始计算
save_path = scheduler.daily_calculate(
    factor=history_sigma,
    **test_params
)

if save_path:
    print("\n" + "=" * 80)
    print(f"✓ 因子计算完成!")
    print(f"✓ 保存路径: {save_path}")
    print("=" * 80)
else:
    print("\n✗ 因子计算失败!")
    sys.exit(1)

# ============================================================================
# Step 4: 加载并验证数据
# ============================================================================
print("\n[Step 4] 加载并验证因子数据...")

# 加载数据
factor_data = scheduler.load_factor_data(
    factor_name='history_sigma',
    start_date=test_params['start_date'],
    end_date=test_params['end_date'],
    freq='daily'
)

if factor_data is not None:
    print("\n数据预览:")
    print("-" * 80)
    print(factor_data.head(10))

    print("\n数据信息:")
    print("-" * 80)
    print(f"总记录数: {len(factor_data)}")
    print(f"数据形状: {factor_data.shape}")
    print(f"列名: {list(factor_data.columns)}")
    print(f"日期范围: {factor_data['date'].min()} ~ {factor_data['date'].max()}")
    print(f"日期数: {factor_data['date'].nunique()}")

    # 检查某一天的数据
    sample_date = factor_data['date'].iloc[0]
    sample_data = factor_data[factor_data['date'] == sample_date]
    print(f"\n示例: {sample_date} 的数据")
    print("-" * 80)
    print(f"股票数: {len(sample_data)}")
    print(sample_data.head())

    print("\n" + "=" * 80)
    print("✓ 测试全部完成!")
    print("=" * 80)

    print("\n下一步:")
    print("1. 如果测试成功，可以修改日期为5年范围重新运行")
    print("2. 然后开发回测模块，支持周频/月频回测")

else:
    print("✗ 加载数据失败!")