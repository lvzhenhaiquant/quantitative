"""
换手率偏离度因子
"""
import polars as pl


def calc_turnover_bias(df: pl.DataFrame,
                       col: str = 'turnover_rate_f',
                       window: int = 20) -> pl.DataFrame:
    """
    计算换手率偏离度

    公式: (turnover - mean) / std

    说明:
    - 正值表示换手率高于平均水平
    - 负值表示换手率低于平均水平
    - 用于捕捉异常交易活跃度

    Args:
        df: 输入数据，需含 ['date', 'stock', col] 列
        col: 换手率字段名
        window: 滚动窗口

    Returns:
        添加 'turnover_bias' 列的 DataFrame
    """
    return df.with_columns(
        (
            (pl.col(col) - pl.col(col).rolling_mean(window_size=window, min_periods=window // 2))
            / pl.col(col).rolling_std(window_size=window, min_periods=window // 2)
        )
        .over('stock')
        .alias('turnover_bias')
    )
