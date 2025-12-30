"""
换手率比值因子
"""
import polars as pl


def calc_turnover_ratio(df: pl.DataFrame,
                        col: str = 'turnover_rate_f',
                        short_window: int = 10,
                        long_window: int = 120) -> pl.DataFrame:
    """
    计算换手率比值 (短期/长期)

    公式: MA(turnover, 10) / MA(turnover, 120)

    说明:
    - 比值 > 1: 短期换手率高于长期平均，交易活跃度上升
    - 比值 < 1: 短期换手率低于长期平均，交易活跃度下降
    - 可用于捕捉流动性变化、资金关注度变化

    策略逻辑:
    - 低换手率比值可能意味着股票被忽视，有反转机会
    - 高换手率比值可能意味着过度关注，短期见顶

    Args:
        df: 输入数据，需含 ['date', 'stock', col] 列
        col: 换手率字段名
        short_window: 短期窗口 (默认10日)
        long_window: 长期窗口 (默认120日)

    Returns:
        添加 'turnover_ratio' 列的 DataFrame
    """
    df = df.with_columns([
        pl.col(col)
          .rolling_mean(window_size=short_window, min_periods=short_window // 2)
          .over('stock')
          .alias('_ma_short'),
        pl.col(col)
          .rolling_mean(window_size=long_window, min_periods=long_window // 2)
          .over('stock')
          .alias('_ma_long')
    ])

    # 计算比值，避免除零
    df = df.with_columns(
        pl.when(pl.col('_ma_long') > 0)
          .then(pl.col('_ma_short') / pl.col('_ma_long'))
          .otherwise(None)
          .alias('turnover_ratio')
    )

    # 清理临时列
    return df.drop(['_ma_short', '_ma_long'])
