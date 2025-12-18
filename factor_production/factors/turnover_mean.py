"""
N日平均换手率因子
"""
import polars as pl


def calc_turnover_mean(df: pl.DataFrame,
                       col: str = 'turnover_rate_f',
                       window: int = 20) -> pl.DataFrame:
    """
    计算N日平均换手率

    公式: mean(turnover, window)

    说明:
    - 低换手率 → 筹码稳定，机构持有
    - 高换手率 → 投机性强，散户参与多
    - A股通常是负向因子（低换手率未来收益更高）

    Args:
        df: 输入数据，需含 ['date', 'stock', col] 列
        col: 换手率字段名
        window: 滚动窗口

    Returns:
        添加 'turnover_mean' 列的 DataFrame
    """
    return df.with_columns(
        pl.col(col)
          .rolling_mean(window_size=window, min_periods=window // 2)
          .over('stock')
          .alias('turnover_mean')
    )
