"""
测试回测系统 - 低波动策略

策略逻辑:
1. 使用History_Sigma因子(残差波动率)
2. 每周第一个交易日调仓
3. 选择波动率最低的50只股票(低波动策略)
4. 等权重配置
"""

import sys
sys.path.append('/home/zhenhai1/quantitative')

import yaml
from utils.data_loader import DataLoader
from backtest.backtester import Backtester

print("=" * 80)
print("测试回测系统 - 低波动策略")
print("=" * 80)

# ============================================================================
# Step 1: 加载配置
# ============================================================================
print("\n[Step 1] 加载配置...")

with open('configs/backtest_config.yaml', 'r', encoding='utf-8') as f:
    backtest_config = yaml.safe_load(f)

print("✓ 回测配置加载完成")

# ============================================================================
# Step 2: 初始化组件
# ============================================================================
print("\n[Step 2] 初始化组件...")

# 创建数据加载器
data_loader = DataLoader()
print("✓ DataLoader初始化完成")

# 创建回测引擎
backtester = Backtester(backtest_config['backtest'], data_loader)
print("✓ Backtester初始化完成")

# ============================================================================
# Step 3: 运行回测
# ============================================================================
print("\n[Step 3] 运行回测...")
print("=" * 80)
print("策略说明:")
print("- 因子: History_Sigma (残差波动率)")
print("- 选股: 低波动50只 (ascending=True)")
print("- 调仓: 每周第一个交易日 (weekly)")
print("- 权重: 等权 (equal)")
print("- 基准: 沪深300")
print("=" * 80)

# 运行回测
results = backtester.run(factor_name='history_sigma')

# ============================================================================
# Step 4: 查看结果
# ============================================================================
if results:
    print("\n[Step 4] 查看详细结果...")

    # 净值曲线
    print("\n净值曲线 (前10天):")
    print(results['portfolio_values'].head(10))

    # 交易记录
    if len(results['trades']) > 0:
        print(f"\n交易记录 ({len(results['trades'])}笔交易, 显示前5笔):")
        print(results['trades'].head())

    # 持仓记录
    if len(results['positions']) > 0:
        print(f"\n持仓记录 ({len(results['positions'])}条, 显示前5条):")
        print(results['positions'].head())

    print("\n" + "=" * 80)
    print("✓ 回测完成!")
    print("=" * 80)

    print("\n接下来可以:")
    print("1. 查看保存的结果文件: backtest/results/")
    print("2. 绘制净值曲线")
    print("3. 分析交易明细")
    print("4. 对比不同参数的回测结果")

else:
    print("\n✗ 回测失败!")