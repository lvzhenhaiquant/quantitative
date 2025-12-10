# 量化交易回测框架

基于QLib的低波动率策略回测系统，实现了History_Sigma因子（残差波动率）的计算和回测。

## 项目特点

- ✅ 避免未来函数：因子计算使用前一日数据
- ✅ 完整的交易成本模拟：手续费、印花税、滑点
- ✅ 灵活的调仓频率：支持日频、周频、月频
- ✅ 模块化设计：因子计算与回测解耦

## 项目结构

```
quantitative/
├── data/                      # 数据下载和转换
│   ├── DownLoadData.py       # Tushare/Baostock数据下载
│   └── ToQlib.py             # 转换为QLib格式
├── utils/                     # 工具类
│   ├── data_loader.py        # 数据加载器
│   └── logger.py             # 日志工具
├── factor_production/         # 因子生产
│   ├── base_factor.py        # 因子基类
│   ├── factor_scheduler.py   # 因子调度器
│   └── market_factors/
│       └── beta_factor.py    # History_Sigma因子
├── backtest/                  # 回测模块
│   └── backtester.py         # 回测引擎
├── configs/                   # 配置文件
├── test_scheduler.py          # 测试因子计算
└── test_backtest.py           # 测试回测
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行测试

```bash
# 计算因子
python test_scheduler.py

# 运行回测
python test_backtest.py
```

## 策略说明

**History_Sigma 低波动率策略**:
- 选择残差波动率最低的50只股票
- 等权重配置
- 每周调仓

## License

MIT License
