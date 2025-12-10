# 第二课: 手把手实操教程

## 一、最简单的例子: 测试DataLoader

让我们从最基础的开始,先测试数据加载是否正常。

### 步骤1: 创建一个简单的测试文件

创建文件: `tutorial_test_01.py`

```python
"""
第一个测试: 验证数据加载
"""
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
```

### 步骤2: 运行测试

```bash
cd /home/zhenhai1/quantitative
source venv/bin/activate
python tutorial_test_01.py
```

### 步骤3: 理解输出

你会看到:
```
第1步: 创建DataLoader
✓ DataLoader创建成功

第2步: 加载中证1000股票池
✓ 成功加载 1000 只股票
前5只股票: ['SZ300171', 'SZ000829', ...]

第3步: 加载股票价格数据
测试股票: ['SZ300171', 'SZ000829', 'SZ002421']
✓ 成功加载价格数据
数据形状: (66, 1)

前5行数据:
                       $close
instrument datetime
SZ000829   2024-01-02    9.47
           2024-01-03    9.37
...
```

**理解要点**:
- DataLoader自动初始化了QLib
- 股票代码格式: `SZ000001`(深圳) 或 `SH600000`(上海)
- 价格数据是MultiIndex: (股票代码, 日期)

## 二、第二个例子: 手动计算简单的因子

不使用框架,手动计算一个简单因子,理解计算过程。

### 步骤1: 创建文件 `tutorial_test_02.py`

```python
"""
第二个测试: 手动计算简单的动量因子
公式: momentum = (当前价格 / 20天前价格) - 1
"""
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
```

### 步骤2: 运行并理解

```bash
python tutorial_test_02.py
```

**你会学到**:
1. 如何从MultiIndex DataFrame中提取单只股票数据
2. 如何计算简单的因子值
3. 如何处理数据不足的情况
4. 如何将结果转换为DataFrame

## 三、第三个例子: 使用框架计算Beta因子

现在使用框架来计算Beta因子,体验框架的便利性。

### 步骤1: 创建文件 `tutorial_test_03.py`

```python
"""
第三个测试: 使用框架计算Beta因子
"""
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
```

### 步骤2: 运行

```bash
python tutorial_test_03.py
```

### 步骤3: 理解框架的优势

对比手动计算(tutorial_test_02.py)和使用框架(tutorial_test_03.py):

**手动计算**:
- ❌ 需要自己处理数据加载
- ❌ 需要自己写循环计算每只股票
- ❌ 需要自己处理异常
- ❌ 需要自己保存结果

**使用框架**:
- ✅ 只需调用`calculate()`一个方法
- ✅ 框架自动处理所有流程
- ✅ 框架自动记录日志
- ✅ 代码简洁,易维护

## 四、查看日志和结果

### 4.1 查看计算日志

```bash
# 查看今天的日志
tail -50 logs/factor.BetaFactor_20251209.log
```

你会看到详细的计算过程:
```
2025-12-09 15:30:00 - factor.BetaFactor - INFO - 初始化因子: beta
2025-12-09 15:30:01 - factor.BetaFactor - INFO - 开始计算因子 beta
2025-12-09 15:30:02 - factor.BetaFactor - INFO - 加载数据...
2025-12-09 15:30:03 - factor.BetaFactor - INFO - 开始计算Beta因子...
2025-12-09 15:30:04 - factor.BetaFactor - INFO - Beta计算完成: 成功10只, 跳过0只
```

### 4.2 查看因子结果

```bash
cat factor_production/cache/tutorial_beta_test.csv
```

结果格式:
```
,beta
SH600151,0.873636
SH600252,1.155802
...
```

## 五、练习题

完成以下练习,巩固理解:

### 练习1: 修改测试股票数量
修改`tutorial_test_03.py`,把测试股票从10只改为20只,观察:
- 计算时间有什么变化?
- 日志输出有什么变化?

### 练习2: 修改因子参数
修改`configs/factor_config.yaml`中的参数:
```yaml
lookback_days: 126    # 从251改为126(半年)
half_life: 21         # 从63改为21(1月)
```

重新运行`tutorial_test_03.py`,观察:
- 因子值有什么变化?
- 为什么会有这些变化?

### 练习3: 查看详细日志
使用日志工具查看完整计算过程:
```bash
source venv/bin/activate
python utils/logger.py today factor.BetaFactor
```

## 六、总结

通过这一课,你应该学会了:

✅ **基础使用**: 如何使用DataLoader加载数据
✅ **手动计算**: 理解因子计算的基本流程
✅ **框架使用**: 如何使用框架简化因子计算
✅ **日志查看**: 如何查看计算日志和结果
✅ **参数调整**: 如何修改配置文件

## 下一课预告

**第三课: 编写你自己的因子**
- 如何继承BaseFactor
- 如何实现_calculate_core()
- 如何加载自定义数据
- 如何调试因子计算

准备好了吗? 如果你已经理解了第二课的内容,我们可以继续第三课!