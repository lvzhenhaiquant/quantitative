"""
换手率波动率因子
"""
import polars as pl


def calc_turnover_vol(df: pl.DataFrame,
                      col: str = 'turnover_rate_f',
                      window: int = 20) -> pl.DataFrame:
    """
    计算换手率波动率

    公式: std(turnover, window)

    说明:
    - 衡量换手率的稳定性
    - 高波动表示交易行为不稳定
    - 低波动表示交易行为稳定

    Args:
        df: 输入数据，需含 ['date', 'stock', col] 列
        col: 换手率字段名
        window: 滚动窗口

    Returns:
        添加 'turnover_vol' 列的 DataFrame
    """
    return df.with_columns(
        pl.col(col)
          .rolling_std(window_size=window, min_periods=window // 2)
          .over('stock')
          .alias('turnover_vol')
    )
