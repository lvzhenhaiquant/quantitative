# 量化因子回测系统

基于 QLib 的量化因子研究和回测框架，支持多种因子计算和策略回测。

## 项目特点

- **简洁的API**: 一行代码完成回测
- **避免未来函数**: 因子计算、权重优化均使用历史数据
- **完整的交易成本**: 手续费、印花税、滑点
- **多种权重方法**: 等权、最大夏普、最小方差
- **模块化设计**: 因子计算与回测解耦

## 快速开始

### 1. 环境准备

```bash
cd /home/zhenhai1/quantitative
source venv/bin/activate
```

### 2. 计算因子

```bash
python run_backtest.py --calc volatility
```

### 3. 运行回测

```bash
# 命令行方式
python run_backtest.py -f volatility -d min -w equal -n 30

# Python方式
from backtest import backtest
result = backtest('volatility', direction='min')
```

## 项目结构

```
quantitative/
├── factor_production/         # 因子生产模块
│   ├── data/                  # 数据加载层 (QLib → Polars)
│   │   └── data_manager.py
│   ├── engine/                # 因子计算引擎
│   │   └── factor_engine.py
│   └── factors/               # 因子计算函数
│       ├── volatility.py      # 历史波动率
│       ├── idio_vol.py        # 特质波动率
│       ├── downside_vol.py    # 下行波动率
│       └── turnover_*.py      # 换手率因子
├── backtest/                  # 回测模块
│   ├── engine.py              # 简化版回测引擎 ★
│   ├── backtester.py          # 完整回测逻辑
│   └── weight_optimizer.py    # 权重优化器
├── utils/                     # 工具类
│   ├── data_loader.py         # QLib数据加载器
│   └── logger.py              # 日志工具
├── data/                      # 数据下载和转换
│   ├── DownLoadData.py        # Tushare/Baostock下载
│   └── ToQlib.py              # 转换为QLib格式
├── configs/                   # 配置文件
├── docs/                      # 文档
│   └── 使用指南.md
├── run_backtest.py            # 命令行入口 ★
└── qlib_data -> ...           # QLib数据 (软链接)
```

## 可用因子

| 因子 | 说明 | 推荐方向 |
|------|------|----------|
| `volatility` | 历史波动率 | min |
| `idio_vol` | 特质波动率 | min |
| `downside_vol` | 下行波动率 | min |
| `turnover_mean` | 平均换手率 | min |
| `turnover_bias` | 换手率偏离度 | - |
| `turnover_vol` | 换手率波动率 | min |

## 使用方法

### 命令行

```bash
# 查看帮助
python run_backtest.py --help

# 计算因子
python run_backtest.py --calc volatility
python run_backtest.py --calc turnover_mean

# 运行回测
python run_backtest.py -f volatility -d min              # 低波动等权
python run_backtest.py -f volatility -d min -w max_sharpe # 低波动最大夏普
python run_backtest.py -f turnover_mean -d min -n 50     # 低换手50只

# 列出可用因子
python run_backtest.py --list
```

### Python API

```python
from backtest import BacktestEngine

engine = BacktestEngine()

# 运行回测
result = engine.run(
    factor='volatility',      # 因子名称
    direction='min',          # 'min' 或 'max'
    weight='equal',           # 'equal' / 'max_sharpe' / 'min_vol'
    n_stocks=30               # 选股数量
)

# 查看结果
print(f"年化收益: {result['metrics']['annual_return']:.2%}")
print(f"夏普比率: {result['metrics']['sharpe_ratio']:.2f}")
print(f"最大回撤: {result['metrics']['max_drawdown']:.2%}")
```

### 策略对比

```python
engine = BacktestEngine()

# 对比不同配置
df = engine.compare(
    factor='volatility',
    directions=['min', 'max'],
    weights=['equal', 'max_sharpe'],
    n_stocks=30
)
print(df)
```

## 参数说明

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `factor` | 因子名称 | volatility, turnover_mean, ... |
| `direction` | 选股方向 | `min` (选最小), `max` (选最大) |
| `weight` | 权重方法 | `equal`, `max_sharpe`, `min_vol` |
| `n_stocks` | 选股数量 | 默认 30 |
| `benchmark` | 基准指数 | 默认 sh000852 (中证1000) |

## 默认配置

- 初始资金: 1000万
- 交易成本: 万3手续费 + 千1印花税 + 千1滑点
- 调仓频率: 周频
- 基准指数: 中证1000

## 文档

详细使用文档请参考: [docs/使用指南.md](docs/使用指南.md)

## License

MIT License
