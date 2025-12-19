import polars as pl

def calc_small_cap(df: pl.DataFrame, col: str = "total_mv",window: int = 20) -> pl.DataFrame:
    """
    计算小市值因子，输入需为 long-form polars.DataFrame，包含列: date, stock, <col>
    返回包含列: date, stock, small_cap
    """
    required = {"date", "stock", col}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"输入数据须包含列: {required}")

# 先计算 -log(total_mv)
    df = df.with_columns(
        pl.when(pl.col(col) > 0).then(-pl.col(col).log()).otherwise(None).alias("raw")
    )
    # 截面秩并归一到 [-1,1]
    df = df.with_columns([
        pl.col("raw").rank().over("date").alias("_r"),
        pl.count().over("date").alias("_n"),
    ]).with_columns(
        pl.when(pl.col("_n")>1)
        .then(((pl.col("_r")-1)/(pl.col("_n")-1))*2-1)
        .otherwise(0)
        .alias("small_cap")
    ).select(["date","stock","small_cap"])
    return df