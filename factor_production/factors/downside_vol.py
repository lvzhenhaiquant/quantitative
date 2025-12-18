"""
下行波动率因子
"""
import polars as pl


def calc_downside_vol(df: pl.DataFrame,
                      col: str = 'close',
                      window: int = 20) -> pl.DataFrame:
    """
    计算下行波动率 (Downside Volatility)

    公式: std(min(returns, 0), window) * sqrt(252)

    说明:
    - 只考虑负收益的波动率
    - 更准确地衡量下行风险
    - 投资者通常更关心下行风险而非上行波动

    Args:
        df: 输入数据，需含 ['date', 'stock', col] 列
        col: 价格字段名
        window: 滚动窗口

    Returns:
        添加 'downside_vol' 列的 DataFrame
    """
    return df.with_columns(
        pl.col(col)
          .pct_change()
          .clip(upper_bound=0)  # 只保留负收益
          .rolling_std(window_size=window, min_periods=window // 2)
          .over('stock')
          .mul(252 ** 0.5)
          .alias('downside_vol')
    )
