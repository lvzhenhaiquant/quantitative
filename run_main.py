#!/usr/bin/env python
"""
量化回测系统 - 直接填参数运行版

使用方法：
1. 在下方【运行参数配置】区域修改要执行的操作和对应的参数
2. 直接运行该脚本即可（python run_main.py）
"""
import sys
from datetime import datetime
import os
# 请确保这个路径是你实际的量化项目根目录
sys.path.insert(0, '/home/yunbo/project/quantitative')


# 因子配置
# 格式: {
#   'factor_name': {
#       'func': '计算函数名',
#       'fields': ['QLib字段'],
#       'financial_fields': ['财报字段'],  # 可选，需要财报数据时使用
#       'kwargs': {}
#   }
# }


def calc_factor(factor_name: str, universe: str = 'csi1000', start_date: str = None, end_date: str = None):
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

    # 配置日期参数
    if start_date is None:
        start_date = '2020-01-01'  # 默认开始日期
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')  # 默认结束日期为今天

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

            save_path = engine._save(result, factor_col, start_date, end_date)
            print(f"\n已保存到: {save_path}")

        print(f"\n因子 {factor_name} 计算完成!")
        return

    # 检查是否需要基准数据
    elif benchmark_code:
        print(f"\n此因子需要基准数据: {benchmark_code}")

        # 加载个股数据
        df_price = dm.load(universe, start_date, end_date, fields)

        # 加载基准数据
        df_benchmark = dm.load_benchmark(benchmark_code, start_date, end_date)

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
            save_path = engine._save(result, factor_col, start_date, end_date)
            print(f"\n已保存到: {save_path}")

    # 检查是否需要财报数据
    elif financial_fields:
        print(f"\n此因子需要财报数据: {financial_fields}")

        # 加载财报数据
        df_financial = dm.load_financial(universe, start_date, end_date, financial_fields)

        if df_financial.is_empty():
            print("\n错误: 财报数据为空，请先下载财报数据:")
            print("  python factor_production/data/download_financial.py --fields " + ",".join(financial_fields))
            return

        # 根据是否需要日频价格数据选择调用方式
        if fields:
            # 需要日频价格数据 (如 peg_ratio 需要 $pe_ttm)
            df_price = dm.load(universe, start_date, end_date, fields)
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
            save_path = engine._save(result, factor_col, start_date, end_date)
            print(f"\n已保存到: {save_path}")
    else:
        # 普通因子：使用 FactorEngine.run()
        engine.run(
            factor_func=func,
            stocks=universe,
            start=start_date,
            end=end_date,
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
    
    # 打印回测结果
    if result is not None:
        print("\n回测完成，结果概要：")
        print(f"年化收益率: {result.get('annual_return', 'N/A'):.4%}")
        print(f"最大回撤: {result.get('max_drawdown', 'N/A'):.4%}")
        print(f"夏普比率: {result.get('sharpe_ratio', 'N/A'):.4f}")
        print(f"信息比率: {result.get('info_ratio', 'N/A'):.4f}")
    else:
        print("\n回测失败，未返回结果")
    
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
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
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


if __name__ == '__main__':
    # ======================== 运行参数配置（重点修改这里）========================
    # 公开因子
    FACTOR_MAP = {
        'amount_ma': {
            'func': 'calc_amount_ma',
            'fields': ['$amount'],
            'kwargs': {'window': 6}
        },
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
        'small_cap': {
            'func': 'calc_small_cap',
            'fields': ['$circ_mv'],
            'kwargs': {'window': 0}
        },
        'drl_test_factor': {
            'func': 'calc_drl_test_factor',
            'fields': ['$open', '$high', '$low', '$close'],
            'kwargs': {'window': 60}
        },
    }
    
    # 操作模式选择
    # 可选值: 'calc_factor' (计算因子), 'run_backtest' (运行回测), 
    #         'list_factors' (列出因子), 'combine_factors' (多因子合成), 'all' (计算因子后运行回测)
    OPERATION = 'all'  
    
    # --- 计算因子的参数（OPERATION='calc_factor'或'all'时生效）---
    CALC_FACTOR_NAME = 'drl_test_factor'  # 要计算的因子名称
    CALC_UNIVERSE = 'csi1000'          # 股票池: csi1000/csi500/csi300/csiall
    START_DATE = '2020-01-01'          # 开始日期
    END_DATE = '2026-01-18'            # 结束日期
    
    # --- 运行回测的参数（OPERATION='run_backtest'或'all'时生效）---
    BACKTEST_FACTOR = CALC_FACTOR_NAME   # 回测的因子名称
    BACKTEST_DIRECTION = 'min'           # 选股方向：min（选最小）/ max（选最大）
    BACKTEST_WEIGHT = 'equal'            # 权重方法：equal / max_sharpe / min_vol
    BACKTEST_N_STOCKS = 30               # 选股数量
    BACKTEST_UNIVERSE = 'csi1000'        # 股票池
    BACKTEST_REBALANCE = 'weekly'       # 调仓频率: daily/weekly/monthly
    BACKTEST_NEUTRALIZE = True           # 是否开启因子中性化 (申万2级行业 + 流通市值)
    
    # --- 多因子合成的参数（OPERATION='combine_factors'时生效）---
    COMBINE_FACTORS = ['turnover_ratio', 'history_sigma', 'return_var']  # 要合成的因子列表
    COMBINE_DATE = datetime.now().strftime('%Y-%m-%d')  # 目标日期
    COMBINE_N_STOCKS = 30                              # 选股数量
    COMBINE_IC_WINDOW = 63                             # IC计算窗口（天）
    COMBINE_RET_PERIOD = 21                            # IC收益率周期（天）
    COMBINE_UNIVERSE = None                            # 股票池（申万行业）
    COMBINE_RUN_BACKTEST = False                       # 是否运行回测
    
    # ============================================================================
    
    # 根据操作模式执行相应功能
    if OPERATION == 'calc_factor' or OPERATION == 'all':
        print(f"开始计算因子：{CALC_FACTOR_NAME}")
        calc_factor(CALC_FACTOR_NAME, CALC_UNIVERSE, START_DATE, END_DATE)
    
    if OPERATION == 'run_backtest' or OPERATION == 'all':
        print(f"\n开始回测 - 因子：{BACKTEST_FACTOR} | 方向：{BACKTEST_DIRECTION} | 权重：{BACKTEST_WEIGHT} | 选股数：{BACKTEST_N_STOCKS}")
        print(f"股票池：{BACKTEST_UNIVERSE} | 调仓频率：{BACKTEST_REBALANCE} | 中性化：{BACKTEST_NEUTRALIZE}")
        run_backtest(BACKTEST_FACTOR, BACKTEST_DIRECTION, BACKTEST_WEIGHT, BACKTEST_N_STOCKS, 
                     BACKTEST_UNIVERSE, BACKTEST_REBALANCE, BACKTEST_NEUTRALIZE)
    
    # if OPERATION == 'list_factors':
    #     print("列出所有可用因子：")
    #     list_factors()
    
    # if OPERATION == 'combine_factors':
    #     combine_factors(COMBINE_FACTORS, COMBINE_DATE, COMBINE_N_STOCKS, COMBINE_IC_WINDOW,
    #                    COMBINE_RET_PERIOD, COMBINE_UNIVERSE, COMBINE_RUN_BACKTEST)
    
    print("\n操作完成！")