"""
特质波动率因子
"""
import polars as pl


def calc_idio_vol(df: pl.DataFrame,
                  col: str = 'close',
                  market_col: str = 'market_return',
                  window: int = 20) -> pl.DataFrame:
    """
    计算特质波动率 (Idiosyncratic Volatility)

    公式: std(stock_return - beta * market_return, window) * sqrt(252)

    说明:
    - 去除市场因素后的波动率
    - 高特质波动率通常预示低未来收益
    - 简化版本：直接用残差波动率近似

    注意：此为简化版本，假设beta=1
    完整版本需要先计算beta再计算残差

    Args:
        df: 输入数据，需含 ['date', 'stock', col] 列
            如果有market_return列则使用，否则用全市场均值近似
        col: 价格字段名
        market_col: 市场收益率字段名
        window: 滚动窗口

    Returns:
        添加 'idio_vol' 列的 DataFrame
    """
    # 计算个股收益率
    df = df.with_columns(
        pl.col(col).pct_change().over('stock').alias('_stock_return')
    )

    # 如果没有市场收益率，用全市场均值近似
    if market_col not in df.columns:
        df = df.with_columns(
            pl.col('_stock_return').mean().over('date').alias('_market_return')
        )
        market_col = '_market_return'

    # 计算残差收益率 (简化版本，假设beta=1)
    df = df.with_columns(
        (pl.col('_stock_return') - pl.col(market_col)).alias('_resid_return')
    )

    # 计算特质波动率
    df = df.with_columns(
        pl.col('_resid_return')
          .rolling_std(window_size=window, min_periods=window // 2)
          .over('stock')
          .mul(252 ** 0.5)
          .alias('idio_vol')
    )

    # 清理临时列
    return df.drop(['_stock_return', '_resid_return'] +
                   (['_market_return'] if '_market_return' in df.columns else []))
