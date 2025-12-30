"""
成交额均值因子
"""
import polars as pl


def calc_amount_ma(df: pl.DataFrame,
                   col: str = 'amount',
                   window: int = 6) -> pl.DataFrame:
    """
    计算成交额移动平均值

    公式: MA(amount, 6)

    说明:
    - 衡量近期成交活跃程度
    - 高成交额意味着流动性好、关注度高
    - 低成交额可能意味着被忽视

    Args:
        df: 输入数据，需含 ['date', 'stock', col] 列
        col: 成交额字段名
        window: 滚动窗口 (默认6日)

    Returns:
        添加 'amount_ma' 列的 DataFrame
    """
    return df.with_columns(
        pl.col(col)
          .rolling_mean(window_size=window, min_periods=window // 2)
          .over('stock')
          .alias('amount_ma')
    )
