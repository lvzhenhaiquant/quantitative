#!/usr/bin/env python
"""
量化回测系统 - 命令行入口

使用方法:
    # 1. 计算因子
    python run_backtest.py --calc volatility

    # 2. 运行回测 (完整参数)
    python run_backtest.py --factor volatility --direction min --weight equal --n 30

    # 3. 快速回测 (使用默认参数)
    python run_backtest.py -f volatility -d min

    # 4. 列出可用因子
    python run_backtest.py --list

可用因子:
    volatility     - 历史波动率 (推荐 direction=min)
    idio_vol       - 特质波动率 (推荐 direction=min)
    downside_vol   - 下行波动率 (推荐 direction=min)
    turnover_mean  - 平均换手率 (推荐 direction=min)
    turnover_bias  - 换手率偏离度
    turnover_vol   - 换手率波动率 (推荐 direction=min)

权重方法:
    equal      - 等权重 (默认)
    max_sharpe - 最大夏普比率
    min_vol    - 最小方差

详细文档: docs/使用指南.md
"""
import sys
sys.path.insert(0, '/home/zhenhai1/quantitative')

import argparse


# 因子配置: (计算函数, 字段列表, 额外参数)
FACTOR_MAP = {
    'volatility': ('calc_volatility', ['$close'], {'window': 20}),
    'idio_vol': ('calc_idio_vol', ['$close'], {'window': 20}),
    'downside_vol': ('calc_downside_vol', ['$close'], {'window': 20}),
    'turnover_mean': ('calc_turnover_mean', ['$turnover_rate_f'], {'window': 20}),
    'turnover_bias': ('calc_turnover_bias', ['$turnover_rate_f'], {'window': 20}),
    'turnover_vol': ('calc_turnover_vol', ['$turnover_rate_f'], {'window': 20}),
}


def calc_factor(factor_name: str):
    """计算因子"""
    from factor_production import DataManager, FactorEngine
    from factor_production import factors as factor_funcs

    if factor_name not in FACTOR_MAP:
        print(f"未知因子: {factor_name}")
        print(f"可用因子: {list(FACTOR_MAP.keys())}")
        return

    func_name, fields, kwargs = FACTOR_MAP[factor_name]
    func = getattr(factor_funcs, func_name)

    # 初始化
    dm = DataManager()
    engine = FactorEngine(dm, cache_dir='/home/zhenhai1/quantitative/factor_production/cache')

    # 计算因子
    engine.run(
        factor_func=func,
        stocks='csi1000',
        start='2020-01-01',
        end='2025-12-17',
        fields=fields,
        **kwargs
    )

    print(f"\n因子 {factor_name} 计算完成!")


def run_backtest(factor: str, direction: str, weight: str, n_stocks: int):
    """运行回测"""
    from backtest import BacktestEngine

    engine = BacktestEngine()
    result = engine.run(
        factor=factor,
        direction=direction,
        weight=weight,
        n_stocks=n_stocks
    )

    return result


def list_factors():
    """列出可用因子"""
    from backtest import BacktestEngine
    engine = BacktestEngine()
    engine.list_factors()


def main():
    parser = argparse.ArgumentParser(
        description='量化回测系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 计算因子
  python run_backtest.py --calc volatility
  python run_backtest.py --calc turnover_mean

  # 运行回测
  python run_backtest.py -f volatility -d min                    # 低波动等权
  python run_backtest.py -f volatility -d min -w max_sharpe      # 低波动最大夏普
  python run_backtest.py -f turnover_mean -d min -n 50           # 低换手50只

  # 查看可用因子
  python run_backtest.py --list

可用因子: volatility, idio_vol, downside_vol, turnover_mean, turnover_bias, turnover_vol
详细文档: docs/使用指南.md
        """
    )

    # 计算因子
    parser.add_argument('--calc', type=str, metavar='FACTOR',
                       help='计算指定因子 (volatility/turnover_mean/...)')

    # 回测参数
    parser.add_argument('--factor', '-f', type=str, metavar='NAME',
                       help='回测因子名称')
    parser.add_argument('--direction', '-d', type=str, default='min',
                       choices=['min', 'max'],
                       help='选股方向: min=选最小 / max=选最大 (默认: min)')
    parser.add_argument('--weight', '-w', type=str, default='equal',
                       choices=['equal', 'max_sharpe', 'min_vol'],
                       help='权重方法: equal/max_sharpe/min_vol (默认: equal)')
    parser.add_argument('--n', type=int, default=30, metavar='NUM',
                       help='选股数量 (默认: 30)')

    # 其他
    parser.add_argument('--list', action='store_true',
                       help='列出所有已计算的因子')

    args = parser.parse_args()

    if args.calc:
        calc_factor(args.calc)
    elif args.list:
        list_factors()
    elif args.factor:
        run_backtest(args.factor, args.direction, args.weight, args.n)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
