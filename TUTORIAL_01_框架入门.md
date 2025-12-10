# 第一课: 因子框架入门教程

## 一、整体架构理解

这个框架是一个**分层架构**,从下到上分为4层:

```
┌─────────────────────────────────────┐
│  4. 用户层 (你的使用代码)              │  ← 你在这里写代码
├─────────────────────────────────────┤
│  3. 因子生产层 (factor_production)   │  ← 因子计算逻辑
│     - BaseFactor (基类)              │
│     - BetaFactor (具体因子)          │
├─────────────────────────────────────┤
│  2. 工具层 (utils)                   │  ← 数据加载、日志等工具
│     - DataLoader                     │
│     - Logger                         │
├─────────────────────────────────────┤
│  1. 数据层                           │  ← QLib数据存储
│     - qlib_data/cn_data             │
│     - data/download_data            │
└─────────────────────────────────────┘
```

## 二、项目目录结构

```
quantitative/
├── configs/                    # 配置文件
│   ├── data_config.yaml       # 数据路径配置
│   └── factor_config.yaml     # 因子参数配置
│
├── utils/                      # 工具层
│   ├── data_loader.py         # 数据加载器(封装QLib)
│   └── logger.py              # 日志系统
│
├── factor_production/          # 因子生产层
│   ├── base_factor.py         # 因子基类(模板)
│   ├── market_factors/        # 市场类因子
│   │   └── beta_factor.py     # Beta和History_Sigma因子
│   └── cache/                 # 因子计算结果缓存
│
├── data/                       # 原始数据
│   └── download_data/         # 下载的CSV数据
│
├── qlib_data/                  # QLib二进制数据
│   └── cn_data/               # 中国市场数据
│
├── logs/                       # 日志输出目录
│
└── test_beta_factor.py        # 测试脚本(示例)
```

## 三、核心组件详解

### 3.1 DataLoader (数据加载器)

**作用**: 封装QLib的数据访问,让你不用直接和QLib打交道

**位置**: `utils/data_loader.py`

**核心方法**:
```python
class DataLoader:
    # 1. 加载股票价格
    load_stock_prices(stock_list, start_date, end_date)

    # 2. 加载基准指数价格
    load_benchmark_prices(benchmark, start_date, end_date)

    # 3. 加载市值数据
    load_market_cap(stock_list, start_date, end_date)

    # 4. 加载股票池
    load_instruments(instrument_name='csi1000')

    # 5. 获取指定日期的股票列表
    get_stock_list_by_date(instrument_name, target_date)

    # 6. 查询股票行业分类
    get_industry(stock_code, level=1)
```

### 3.2 BaseFactor (因子基类)

**作用**: 定义因子计算的标准流程(模板方法模式)

**位置**: `factor_production/base_factor.py`

**核心流程**:
```python
class BaseFactor:
    def calculate():
        # Step 1: 加载数据
        data = self._load_data(...)

        # Step 2: 数据预处理
        data = self._preprocess(data)

        # Step 3: 核心计算 (子类实现)
        factor_values = self._calculate_core(data)

        # Step 4: 数据后处理
        factor_values = self._postprocess(factor_values)

        return factor_values
```

**你需要做的**: 继承BaseFactor,只实现`_calculate_core()`方法

### 3.3 BetaFactor (Beta因子示例)

**作用**: 计算个股相对市场的敏感度

**位置**: `factor_production/market_factors/beta_factor.py`

**核心逻辑**:
```python
class BetaFactor(BaseFactor):
    def _calculate_core(self, data):
        # 1. 计算对数收益率
        stock_returns = log(P_t / P_{t-1})
        benchmark_returns = log(P_t / P_{t-1})

        # 2. 生成半衰权重(近期数据权重更大)
        weights = half_decay_weights(half_life=63, length=553)

        # 3. 加权线性回归
        for each stock:
            beta = weighted_regression(y=stock_returns, x=benchmark_returns, weights=weights)

        return beta_values
```

## 四、数据流向

```
用户请求
   ↓
因子.calculate(stock_list, start_date, end_date, data_loader)
   ↓
BaseFactor._load_data() → DataLoader.load_stock_prices()
   ↓                      → DataLoader.load_benchmark_prices()
   ↓                      ↓
   ↓                    QLib.D.features() 读取 qlib_data/
   ↓                      ↓
   ↓              返回 DataFrame(价格数据)
   ↓
BaseFactor._preprocess(data)  # 数据预处理(去极值、缺失值)
   ↓
BetaFactor._calculate_core(data)  # 核心计算
   ↓
   ├─ 计算收益率
   ├─ 生成权重
   └─ 回归计算Beta
   ↓
BaseFactor._postprocess(factor_values)  # 后处理(标准化、中性化)
   ↓
返回: DataFrame(股票代码 → 因子值)
```

## 五、配置文件说明

### 5.1 data_config.yaml (数据配置)

```yaml
data:
  qlib_data_path: "/home/zhenhai1/quantitative/qlib_data/cn_data"  # QLib数据路径
  shenwan_path: "..."      # 申万行业分类数据
  instruments:             # 股票池配置
    csi1000: "..."         # 中证1000成分股
    csi300: "..."          # 沪深300成分股
```

**作用**: 告诉DataLoader去哪里找数据

### 5.2 factor_config.yaml (因子配置)

```yaml
beta_factor:
  name: "beta"
  params:
    lookback_days: 251     # 回看天数(1年交易日)
    half_life: 63          # 半衰期(1季度)
    benchmark: "sh000300"  # 基准指数(沪深300)
    min_valid_days: 50     # 最少有效数据点
```

**作用**: 定义因子计算参数

## 六、关键概念

### 6.1 半衰权重 (Half-Decay Weights)

**为什么需要**: 我们认为近期数据比远期数据更重要

**原理**:
```
权重 = 0.5^((T-t)/half_life)

其中:
- T = 最新日期
- t = 当前数据点日期
- half_life = 半衰期(63天)
```

**示例**:
```python
# 假设有5个数据点,半衰期=2
weights = [0.25, 0.35, 0.50, 0.71, 1.00]  # 越近权重越大

# 归一化后
weights = [0.08, 0.12, 0.18, 0.25, 0.35]  # 和为1
```

### 6.2 对数收益率

**公式**: `return = log(P_t / P_{t-1})`

**为什么用对数**:
- 对数收益率可以直接相加
- 更符合正态分布
- 避免价格绝对值影响

### 6.3 加权线性回归

**公式**: `R_stock = alpha + beta * R_market + epsilon`

**目标**: 找到最优的beta,使得加权残差平方和最小

**实现**: 使用sklearn的LinearRegression,传入sample_weight参数

## 七、下一步学习

完成第一课后,你应该理解:
- ✅ 项目的4层架构
- ✅ 各个目录的作用
- ✅ DataLoader是什么
- ✅ BaseFactor的工作流程
- ✅ 数据如何从QLib流向因子计算

**下一课预告**: 第二课将教你如何一步步使用这个框架

准备好了吗? 我们可以继续第二课!
