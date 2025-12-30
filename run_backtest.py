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
    history_sigma  - 半衰加权特质波动率 (推荐 direction=min)
    turnover_mean  - 平均换手率 (推荐 direction=min)
    turnover_bias  - 换手率偏离度
    turnover_vol   - 换手率波动率 (推荐 direction=min)
    peg_ratio      - PEG估值因子 (推荐 direction=min)

权重方法:
    equal      - 等权重 (默认)
    max_sharpe - 最大夏普比率
    min_vol    - 最小方差

详细文档: docs/使用指南.md
"""
import sys
sys.path.insert(0, '/home/zhenhai1/quantitative')

import argparse


# 因子配置
# 格式: {
#   'factor_name': {
#       'func': '计算函数名',
#       'fields': ['QLib字段'],
#       'financial_fields': ['财报字段'],  # 可选，需要财报数据时使用
#       'kwargs': {}
#   }
# }
FACTOR_MAP = {
    # 波动率因子
    'volatility': {
        'func': 'calc_volatility',
        'fields': ['$close'],
        'kwargs': {'window': 20}
    },
    'idio_vol': {
        'func': 'calc_idio_vol',
        'fields': ['$close'],
        'kwargs': {'window': 20}
    },
    'downside_vol': {
        'func': 'calc_downside_vol',
        'fields': ['$close'],
        'kwargs': {'window': 20}
    },
    # 换手率因子
    'turnover_mean': {
        'func': 'calc_turnover_mean',
        'fields': ['$turnover_rate_f'],
        'kwargs': {'window': 20}
    },
    'turnover_bias': {
        'func': 'calc_turnover_bias',
        'fields': ['$turnover_rate_f'],
        'kwargs': {'window': 20}
    },
    'turnover_vol': {
        'func': 'calc_turnover_vol',
        'fields': ['$turnover_rate_f'],
        'kwargs': {'window': 20}
    },
    'turnover_ratio': {
        'func': 'calc_turnover_ratio',
        'fields': ['$turnover_rate_f'],
        'kwargs': {'short_window': 10, 'long_window': 120}
    },
    'turnover_ma5': {
        'func': 'calc_turnover_ma5',
        'fields': ['$turnover_rate_f'],
        'kwargs': {'window': 5}
    },
    # 成交额因子
    'amount_std': {
        'func': 'calc_amount_std',
        'fields': ['$amount'],
        'kwargs': {'window': 20}
    },
    'amount_ma': {
        'func': 'calc_amount_ma',
        'fields': ['$amount'],
        'kwargs': {'window': 6}
    },
    # 估值因子 (需要财报数据)
    'peg_ratio': {
        'func': 'calc_peg_ratio',
        'fields': ['$pe_ttm'],
        'financial_fields': ['netprofit_yoy'],
        'kwargs': {}
    },
    # 历史波动率因子 (需要基准数据)
    'history_sigma': {
        'func': 'calc_history_sigma',
        'fields': ['$close'],
        'benchmark': 'sh000300',  # 沪深300作为基准
        'kwargs': {'window': 250, 'half_life': 63}
    },
    # 收益率方差因子
    'return_var': {
        'func': 'calc_return_var',
        'fields': ['$close'],
        'kwargs': {'window': 121}
    },
    # 营业外收支占比因子 (纯财报因子)
    'non_oper_ratio': {
        'func': 'calc_non_oper_ratio',
        'fields': [],  # 不需要日频价格数据
        'financial_fields': ['non_oper_income', 'non_oper_exp', 'total_profit'],
        'kwargs': {}
    },
    # 资产负债率同比变化因子 (独立计算，直接读取原始年报)
    'debt_ratio_yoy': {
        'func': 'calc_debt_ratio_yoy',
        'fields': [],
        'standalone': True,  # 独立计算，不需要 load_financial
        'kwargs': {'start': '2020-01-01', 'end': '2025-12-17'}
    },
}


def calc_factor(factor_name: str, universe: str = 'csi1000'):
    """计算因子"""
    from factor_production import DataManager, FactorEngine
    from factor_production import factors as factor_funcs

    if factor_name not in FACTOR_MAP:
        print(f"未知因子: {factor_name}")
        print(f"可用因子: {list(FACTOR_MAP.keys())}")
        return

    config = FACTOR_MAP[factor_name]
    func_name = config['func']
    fields = config['fields']
    kwargs = config.get('kwargs', {})
    financial_fields = config.get('financial_fields', [])
    benchmark_code = config.get('benchmark', None)
    standalone = config.get('standalone', False)

    func = getattr(factor_funcs, func_name)

    # 初始化
    dm = DataManager()
    engine = FactorEngine(dm, cache_dir='/home/zhenhai1/quantitative/factor_production/cache')

    # 配置参数
    stocks = universe
    start = '2020-01-01'
    from datetime import datetime
    end = datetime.now().strftime('%Y-%m-%d')

    # 独立计算的因子（直接调用函数，不需要额外数据加载）
    if standalone:
        print(f"\n独立计算因子: {factor_name}")
        result = func(**kwargs)

        factor_col = factor_name
        if factor_col in result.columns:
            factor_values = result[factor_col].drop_nulls().drop_nans()
            print(f"\n因子统计:")
            print(f"  有效值: {len(factor_values)}")
            if len(factor_values) > 0:
                print(f"  均值: {factor_values.mean():.4f}")
                print(f"  标准差: {factor_values.std():.4f}")
                print(f"  最小值: {factor_values.min():.4f}")
                print(f"  最大值: {factor_values.max():.4f}")

            save_path = engine._save(result, factor_col, start, end)
            print(f"\n已保存到: {save_path}")

        print(f"\n因子 {factor_name} 计算完成!")
        return

    # 检查是否需要基准数据
    elif benchmark_code:
        print(f"\n此因子需要基准数据: {benchmark_code}")

        # 加载个股数据
        df_price = dm.load(stocks, start, end, fields)

        # 加载基准数据
        df_benchmark = dm.load_benchmark(benchmark_code, start, end)

        if df_benchmark.is_empty():
            print(f"\n错误: 基准数据为空 ({benchmark_code})")
            return

        print(f"基准数据: {len(df_benchmark)} 行")

        # 计算因子
        result = func(df_price, df_benchmark, **kwargs)

        # 保存结果
        factor_col = factor_name
        if factor_col in result.columns:
            # 统计 (同时过滤 null 和 NaN)
            factor_values = result[factor_col].drop_nulls().drop_nans()
            print(f"\n因子统计:")
            print(f"  有效值: {len(factor_values)}")
            if len(factor_values) > 0:
                print(f"  均值: {factor_values.mean():.4f}")
                print(f"  标准差: {factor_values.std():.4f}")
                print(f"  最小值: {factor_values.min():.4f}")
                print(f"  最大值: {factor_values.max():.4f}")

            # 保存
            save_path = engine._save(result, factor_col, start, end)
            print(f"\n已保存到: {save_path}")

    # 检查是否需要财报数据
    elif financial_fields:
        print(f"\n此因子需要财报数据: {financial_fields}")

        # 加载财报数据
        df_financial = dm.load_financial(stocks, start, end, financial_fields)

        if df_financial.is_empty():
            print("\n错误: 财报数据为空，请先下载财报数据:")
            print("  python factor_production/data/download_financial.py --fields " + ",".join(financial_fields))
            return

        # 根据是否需要日频价格数据选择调用方式
        if fields:
            # 需要日频价格数据 (如 peg_ratio 需要 $pe_ttm)
            df_price = dm.load(stocks, start, end, fields)
            result = func(df_price, df_financial, **kwargs)
        else:
            # 纯财报因子，不需要日频价格数据
            result = func(df_financial, **kwargs)

        # 保存结果
        factor_col = factor_name
        if factor_col in result.columns:
            # 统计 (同时过滤 null 和 NaN)
            factor_values = result[factor_col].drop_nulls().drop_nans()
            print(f"\n因子统计:")
            print(f"  有效值: {len(factor_values)}")
            if len(factor_values) > 0:
                print(f"  均值: {factor_values.mean():.4f}")
                print(f"  标准差: {factor_values.std():.4f}")
                print(f"  最小值: {factor_values.min():.4f}")
                print(f"  最大值: {factor_values.max():.4f}")

            # 保存
            save_path = engine._save(result, factor_col, start, end)
            print(f"\n已保存到: {save_path}")
    else:
        # 普通因子：使用 FactorEngine.run()
        engine.run(
            factor_func=func,
            stocks=stocks,
            start=start,
            end=end,
            fields=fields,
            **kwargs
        )

    print(f"\n因子 {factor_name} 计算完成!")


def run_backtest(factor: str, direction: str, weight: str, n_stocks: int,
                 universe: str = 'csi1000', rebalance: str = 'monthly',
                 neutralize: bool = False):
    """运行回测"""
    from backtest import BacktestEngine

    engine = BacktestEngine()
    result = engine.run(
        factor=factor,
        direction=direction,
        weight=weight,
        n_stocks=n_stocks,
        universe=universe,
        rebalance=rebalance,
        neutralize=neutralize
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
  python run_backtest.py -f history_sigma -d min -N              # 开启中性化

  # 查看可用因子
  python run_backtest.py --list

可用因子: volatility, idio_vol, history_sigma, turnover_mean, turnover_ratio, amount_std, return_var
详细文档: docs/使用指南.md
        """
    )

    # 计算因子
    parser.add_argument('--calc', type=str, metavar='FACTOR',
                       help='计算指定因子 (volatility/turnover_mean/...)')
    parser.add_argument('--universe', '-u', type=str, default='csi1000',
                       help='股票池: csi1000/csi500/csi300/csiall (默认: csi1000)')

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
    parser.add_argument('--rebalance', '-r', type=str, default='monthly',
                       choices=['daily', 'weekly', 'monthly'],
                       help='调仓频率: daily/weekly/monthly (默认: monthly)')
    parser.add_argument('--neutralize', '-N', action='store_true',
                       help='开启因子中性化 (申万2级行业 + 流通市值)')

    # 其他
    parser.add_argument('--list', action='store_true',
                       help='列出所有已计算的因子')

    args = parser.parse_args()

    if args.calc:
        calc_factor(args.calc, universe=args.universe)
    elif args.list:
        list_factors()
    elif args.factor:
        run_backtest(args.factor, args.direction, args.weight, args.n, args.universe, args.rebalance, args.neutralize)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
