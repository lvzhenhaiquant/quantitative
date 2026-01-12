# 量化因子回测系统

基于 Polars 的量化因子研究和回测框架，支持因子计算、中性化和策略回测。

## 项目特点

- **高性能**: 基于 Polars + Numba，支持多核并行计算
- **独立运行**: 不依赖 QLib，直接读取 Parquet 文件
- **避免未来函数**: 因子计算、权重优化、成分股筛选均使用历史数据
- **完整的交易成本**: 手续费、印花税、滑点
- **股票过滤**: 自动排除 ST、停牌、涨跌停股票
- **因子中性化**: 支持行业+市值中性化（申万2级+流通市值）
- **多种调仓频率**: 日频、周频、月频
- **多种权重方法**: 等权、最大夏普、最小方差

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
python run_backtest.py -f volatility -d min -r monthly

# Python方式
from backtest import BacktestEngine
engine = BacktestEngine()
result = engine.run('volatility', direction='min')
```

## 项目结构

```
quantitative/
├── factor_production/         # 因子生产模块
│   ├── data/
│   │   └── data_manager.py    # 数据管理器 (Parquet)
│   ├── factors/               # 因子实现
│   ├── neutralize.py          # 因子中性化
│   ├── combiner/              # 多因子合成
│   └── cache/                 # 因子缓存 (Parquet)
│
├── backtest/                  # 回测模块
│   ├── engine.py              # 简化版回测引擎
│   ├── backtester.py          # 完整回测逻辑
│   ├── weight_optimizer.py    # 权重优化器
│   └── filters/               # 股票过滤 (ST/停牌/涨跌停)
│
├── data/                      # 数据目录 (Parquet)
│   ├── daily/                 # 日线行情
│   ├── basic/                 # 估值/换手率
│   ├── adj/                   # 复权因子
│   ├── index_daily/           # 指数行情
│   └── index_weight/          # 成分股 (JSON)
│
├── configs/                   # 配置文件
├── docs/                      # 文档
└── run_backtest.py            # 命令行入口
```

## 使用方法

### 命令行

```bash
# 查看帮助
python run_backtest.py --help

# 计算因子
python run_backtest.py --calc volatility

# 运行回测
python run_backtest.py -f volatility -d min              # 月度调仓
python run_backtest.py -f volatility -d min -r weekly    # 周度调仓
python run_backtest.py -f volatility -d min -N           # 开启中性化
python run_backtest.py -f volatility -d min -w max_sharpe # 最大夏普权重

# 指定股票池
python run_backtest.py -f volatility -d min -u csi500    # 中证500

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
    n_stocks=30,              # 选股数量
    rebalance='monthly',      # 'daily' / 'weekly' / 'monthly'
    neutralize=True           # 开启中性化 (可选)
)

# 查看结果
print(f"年化收益: {result['metrics']['annual_return']:.2%}")
print(f"夏普比率: {result['metrics']['sharpe_ratio']:.2f}")
print(f"最大回撤: {result['metrics']['max_drawdown']:.2%}")
```

## 参数说明

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `-f, --factor` | 因子名称 | 使用 `--list` 查看可用因子 |
| `-d, --direction` | 选股方向 | `min` (选最小), `max` (选最大) |
| `-w, --weight` | 权重方法 | `equal`, `max_sharpe`, `min_vol` |
| `-n` | 选股数量 | 默认 30 |
| `-r, --rebalance` | 调仓频率 | `daily`, `weekly`, `monthly` |
| `-u, --universe` | 股票池 | `csi300`, `csi500`, `csi1000`, `csiall` |
| `-N, --neutralize` | 因子中性化 | 申万2级行业 + 流通市值 |

## 默认配置

- 初始资金: 1000万
- 交易成本: 万3手续费 + 千1印花税 + 千1滑点
- 调仓频率: 月频
- 基准指数: 中证1000 (sh000852)
- 股票过滤: 排除 ST、停牌、涨跌停
- 后复权: 默认开启

## 文档

详细使用文档请参考: [docs/使用指南.md](docs/使用指南.md)

## License

MIT License