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

    # 5. IC加权多因子合成
    python run_backtest.py --combine turnover_ratio,history_sigma,return_var --date 2025-12-31
    python run_backtest.py --combine turnover_ratio,history_sigma,return_var -n 30  # 选股

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
sys.path.insert(0, '/home/yunbo/project/quantitative')

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
# 公开因子
FACTOR_MAP = {
    'amount_ma': {
        'func': 'calc_amount_ma',
        'fields': ['$amount'],
        'kwargs': {'window': 6}
    },
}

# 本地因子配置 (从 local_factors.py 导入，不上传到 git)
try:
    from local_factors import LOCAL_FACTORS
    FACTOR_MAP.update(LOCAL_FACTORS)
except ImportError:
    pass  # 本地因子配置不存在时跳过


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
    engine = FactorEngine(dm, cache_dir='/home/yunbo/project/quantitative/factor_production/cache')

    # 配置参数
    stocks = universe
    start = config.get('start', '2020-01-01')  # 支持自定义开始日期
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


def combine_factors(factors: list, date: str, n_stocks: int = 30, ic_window: int = 63,
                    ret_period: int = 21, universe: str = None, run_backtest_flag: bool = False):
    """
    IC加权多因子合成

    Args:
        factors: 因子名称列表
        date: 目标日期
        n_stocks: 选股数量
        ic_window: IC计算窗口（默认63天/3个月）
        ret_period: IC收益率周期（默认21天，与月度调仓匹配）
        universe: 股票池（申万行业）
        run_backtest_flag: 是否运行回测
    """
    from factor_production.combiner import FactorCombiner

    print(f"\n{'='*60}")
    print(f"IC加权多因子合成")
    print(f"{'='*60}")
    print(f"因子: {factors}")
    print(f"日期: {date}")
    print(f"IC窗口: {ic_window} 天")
    print(f"收益率周期: {ret_period} 天")
    print(f"选股数量: {n_stocks}")
    print(f"{'='*60}\n")

    combiner = FactorCombiner(
        factors=factors,
        ic_window=ic_window,
        ret_period=ret_period
    )

    # 加载股票池
    stocks = None
    if universe:
        from pathlib import Path
        from datetime import datetime
        instruments_file = Path(f'/home/yunbo/project/quantitative/qlib_data/cn_data/instruments/{universe}.txt')
        if instruments_file.exists():
            target_date = datetime.strptime(date, '%Y-%m-%d').date()
            stocks = []
            with open(instruments_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        stock = parts[0].upper()
                        start = datetime.strptime(parts[1], '%Y-%m-%d').date()
                        end = datetime.strptime(parts[2], '%Y-%m-%d').date()
                        if start <= target_date <= end:
                            stocks.append(stock)
            print(f"股票池 {universe}: {len(stocks)} 只\n")

    # 显示权重
    combiner.summary(date)

    # 选股
    if n_stocks > 0:
        selected = combiner.select_stocks(date, n_stocks=n_stocks, stocks=stocks)
        print(f"\n综合因子选股 (前{n_stocks}只):")
        print("-" * 40)
        for i, s in enumerate(selected, 1):
            print(f"  {i:2d}. {s}")
        print("-" * 40)

        # 保存到文件
        output_file = f'/home/yunbo/project/mean_recursion/holdings_combined_{date.replace("-", "")}.txt'
        with open(output_file, 'w') as f:
            f.write('\n'.join(selected))
        print(f"\n已保存到: {output_file}")

    # 如果需要回测，使用向量化方式生成合成因子并运行回测
    if run_backtest_flag:
        print(f"\n{'='*60}")
        print("生成合成因子并运行回测（向量化方式）...")
        print(f"{'='*60}\n")

        import polars as pl
        from pathlib import Path

        cache_dir = Path('/home/yunbo/project/quantitative/factor_production/cache')

        # 加载所有因子数据
        factor_dfs = {}
        for factor_name in factors:
            factor_files = list(cache_dir.glob(f"{factor_name}_*.parquet"))
            if not factor_files:
                print(f"错误: 找不到因子 {factor_name} 的缓存文件")
                return combiner
            df = pl.read_parquet(sorted(factor_files)[-1])
            df = df.select(['stock', 'date', factor_name])
            factor_dfs[factor_name] = df
            print(f"  加载 {factor_name}: {len(df)} 行")

        # 合并所有因子
        print("\n合并因子数据...")
        merged = factor_dfs[factors[0]]
        for factor_name in factors[1:]:
            merged = merged.join(factor_dfs[factor_name], on=['stock', 'date'], how='outer_coalesce')

        print(f"  合并后: {len(merged)} 行")

        # 按日期分组计算排名（百分位）
        print("计算因子排名...")
        for factor_name in factors:
            merged = merged.with_columns(
                (pl.col(factor_name).rank().over('date') / pl.col(factor_name).count().over('date'))
                .alias(f'{factor_name}_rank')
            )

        # 使用最新权重（简化：使用固定权重而不是滚动权重）
        # 因为历史滚动权重计算太慢，这里使用最新的权重
        weights = combiner.get_weights(date)
        print(f"\n使用权重: {weights}")

        # 加权合成
        print("加权合成...")
        weighted_sum_expr = pl.lit(0.0)
        for factor_name in factors:
            w = weights.get(factor_name, 0)
            weighted_sum_expr = weighted_sum_expr + pl.col(f'{factor_name}_rank').fill_null(0.5) * w

        merged = merged.with_columns(weighted_sum_expr.alias('combined_factor'))
        merged = merged.select(['stock', 'date', 'combined_factor'])
        merged = merged.drop_nulls(subset=['combined_factor'])
        merged = merged.sort(['date', 'stock'])

        # 保存合成因子
        dates = merged['date'].unique().sort().to_list()
        start_date = str(dates[0])[:10].replace('-', '')
        end_date = str(dates[-1])[:10].replace('-', '')
        save_path = cache_dir / f"combined_factor_{start_date}_{end_date}.parquet"
        merged.write_parquet(save_path)
        print(f"\n合成因子已保存: {save_path}")
        print(f"  记录数: {len(merged)}")
        print(f"  日期范围: {dates[0]} ~ {dates[-1]}")

        # 运行回测
        print(f"\n{'='*60}")
        print("运行回测...")
        print(f"{'='*60}\n")

        from backtest import BacktestEngine
        engine = BacktestEngine()
        result = engine.run(
            factor='combined_factor',
            direction='max',  # 合成因子越大越好
            weight='equal',
            n_stocks=n_stocks,
            universe='csi1000',
            rebalance='monthly'
        )

    return combiner


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

  # IC加权多因子合成
  python run_backtest.py --combine turnover_ratio,history_sigma,return_var --date 2025-12-31
  python run_backtest.py --combine turnover_ratio,history_sigma -n 30 -u shenwan_select

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

    # IC加权多因子合成
    parser.add_argument('--combine', '-C', type=str, metavar='FACTORS',
                       help='IC加权合成多因子，逗号分隔 (如: turnover_ratio,history_sigma,return_var)')
    parser.add_argument('--date', type=str, default=None,
                       help='目标日期 (默认: 最新交易日)')
    parser.add_argument('--ic-window', type=int, default=63,
                       help='IC计算窗口天数 (默认: 63，约3个月)')
    parser.add_argument('--ret-period', type=int, default=21,
                       help='IC收益率周期天数 (默认: 21，与月度调仓匹配)')
    parser.add_argument('--backtest', '-B', action='store_true',
                       help='合成因子后运行回测')

    # 其他
    parser.add_argument('--list', action='store_true',
                       help='列出所有已计算的因子')

    args = parser.parse_args()

    if args.calc:
        calc_factor(args.calc, universe=args.universe)
    elif args.combine:
        # IC加权多因子合成
        factors = [f.strip() for f in args.combine.split(',')]
        date = args.date
        if not date:
            from datetime import datetime
            date = datetime.now().strftime('%Y-%m-%d')
        combine_factors(factors, date, n_stocks=args.n, ic_window=args.ic_window,
                       ret_period=args.ret_period,
                       universe=args.universe if args.universe != 'csi1000' else None,
                       run_backtest_flag=args.backtest)
    elif args.list:
        list_factors()
    elif args.factor:
        run_backtest(args.factor, args.direction, args.weight, args.n, args.universe, args.rebalance, args.neutralize)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
