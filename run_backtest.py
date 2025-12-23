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
    # 估值因子 (需要财报数据)
    'peg_ratio': {
        'func': 'calc_peg_ratio',
        'fields': ['$pe_ttm'],
        'financial_fields': ['netprofit_yoy'],
        'kwargs': {}
    },
}


def calc_factor(factor_name: str):
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

    func = getattr(factor_funcs, func_name)

    # 初始化
    dm = DataManager()
    engine = FactorEngine(dm, cache_dir='/home/zhenhai1/quantitative/factor_production/cache')

    # 配置参数
    stocks = 'csi1000'
    start = '2020-01-01'
    end = '2025-12-17'

    # 检查是否需要财报数据
    if financial_fields:
        print(f"\n此因子需要财报数据: {financial_fields}")

        # 加载日频数据
        df_price = dm.load(stocks, start, end, fields)

        # 加载财报数据
        df_financial = dm.load_financial(stocks, start, end, financial_fields)

        if df_financial.is_empty():
            print("\n错误: 财报数据为空，请先下载财报数据:")
            print("  python factor_production/data/download_financial.py")
            return

        # 计算因子 (传入两个 DataFrame)
        result = func(df_price, df_financial, **kwargs)

        # 保存结果
        factor_col = factor_name
        if factor_col in result.columns:
            # 统计
            factor_values = result[factor_col].drop_nulls()
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

可用因子: volatility, idio_vol, downside_vol, turnover_mean, turnover_bias, turnover_vol, peg_ratio
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
