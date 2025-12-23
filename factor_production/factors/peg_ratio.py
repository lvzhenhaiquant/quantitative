"""
PEG Ratio 因子

PEG = PE_TTM / 盈利增长率
用于衡量股票估值是否合理，考虑了成长性
"""
import polars as pl


def calc_peg_ratio(df_price: pl.DataFrame,
                   df_financial: pl.DataFrame = None,
                   pe_col: str = 'pe_ttm',
                   growth_col: str = 'netprofit_yoy') -> pl.DataFrame:
    """
    计算 PEG Ratio

    公式: PEG = PE_TTM / 净利润同比增长率(%)

    说明:
    - PEG < 1: 可能被低估（成长性高于估值）
    - PEG > 1: 可能被高估（估值高于成长性）
    - PEG ≈ 1: 估值合理

    过滤条件:
    - PE_TTM <= 0 时，PEG 无意义（亏损股）
    - 增长率 <= 5% 时，PEG 无意义（低/负增长）

    Args:
        df_price: 日频价格数据，含 ['date', 'stock', pe_col]
        df_financial: 财报数据（已前向填充），含 ['date', 'stock', growth_col]
                      如果为 None，则尝试从 df_price 中获取
        pe_col: PE_TTM 字段名
        growth_col: 盈利增长率字段名

    Returns:
        添加 'peg_ratio' 列的 DataFrame
    """
    # 如果没有传入财报数据，假设数据已在 df_price 中
    if df_financial is None:
        df = df_price
    else:
        # 合并价格数据和财报数据
        df = df_price.join(
            df_financial.select(['stock', 'date', growth_col]),
            on=['stock', 'date'],
            how='left'
        )

    # 计算 PEG
    # 只在 PE > 0 且增长率 > 5% 时计算，否则为 null
    df = df.with_columns(
        pl.when(
            (pl.col(pe_col) > 0) &
            (pl.col(growth_col) > 5)  # 增长率 > 5%
        )
        .then(pl.col(pe_col) / pl.col(growth_col))
        .otherwise(None)
        .alias('peg_ratio')
    )

    return df
