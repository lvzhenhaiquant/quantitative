"""
历史波动率因子
"""
import polars as pl


def calc_volatility(df: pl.DataFrame,
                    col: str = 'close',
                    window: int = 20) -> pl.DataFrame:
    """
    计算历史波动率 (History Sigma)

    公式: std(returns, window) * sqrt(252)

    说明:
    - 年化波动率
    - 衡量股票价格的波动程度
    - 低波动股票通常有更高的风险调整后收益

    Args:
        df: 输入数据，需含 ['date', 'stock', col] 列
        col: 价格字段名
        window: 滚动窗口

    Returns:
        添加 'volatility' 列的 DataFrame
    """
    return df.with_columns(
        (pl.col(col).pct_change())
          .rolling_std(window_size=window, min_periods=window // 2)
          .over('stock')
          .mul(252 ** 0.5)
          .alias('volatility')
    )
