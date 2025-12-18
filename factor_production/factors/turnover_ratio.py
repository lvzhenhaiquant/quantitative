"""
换手率变化比因子
"""
import polars as pl


def calc_turnover_ratio(df: pl.DataFrame,
                        col: str = 'turnover_rate_f',
                        short_window: int = 5,
                        long_window: int = 20) -> pl.DataFrame:
    """
    计算换手率变化比

    公式: mean(turnover, short) / mean(turnover, long)

    说明:
    - >1 表示近期换手率上升
    - <1 表示近期换手率下降
    - 用于捕捉交易活跃度的变化趋势

    Args:
        df: 输入数据，需含 ['date', 'stock', col] 列
        col: 换手率字段名
        short_window: 短期窗口
        long_window: 长期窗口

    Returns:
        添加 'turnover_ratio' 列的 DataFrame
    """
    return df.with_columns(
        (
            pl.col(col).rolling_mean(window_size=short_window, min_periods=short_window // 2)
            / pl.col(col).rolling_mean(window_size=long_window, min_periods=long_window // 2)
        )
        .over('stock')
        .alias('turnover_ratio')
    )
