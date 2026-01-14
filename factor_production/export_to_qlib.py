"""
将因子 Parquet 导出为 QLib bin 格式
"""
import numpy as np
import polars as pl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import struct


def read_qlib_calendar(qlib_path: str) -> list:
    """读取 QLib 交易日历"""
    cal_file = Path(qlib_path) / 'calendars' / 'day.txt'
    with open(cal_file, 'r') as f:
        dates = [line.strip() for line in f if line.strip()]
    return dates


def date_to_qlib_index(date_str: str, calendar: list) -> int:
    """将日期转换为 QLib 索引"""
    # QLib 用从起始日期开始的天数索引
    try:
        return calendar.index(date_str)
    except ValueError:
        return -1


def write_qlib_bin(data: np.ndarray, start_idx: int, filepath: Path):
    """
    写入 QLib bin 格式

    QLib bin 格式:
    - 前 4 字节: float32, 起始索引 (start_index)
    - 后面: float32 数组, 每个值对应一天
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        # 写入起始索引
        f.write(struct.pack('<f', float(start_idx)))
        # 写入数据
        data.astype(np.float32).tofile(f)


def export_factor_to_qlib(
    factor_parquet: str,
    factor_name: str,
    qlib_path: str = '/home/yunbo/project/quantitative/qlib_data/cn_data'
):
    """
    将因子 Parquet 导出为 QLib bin 格式

    Args:
        factor_parquet: 因子 Parquet 文件路径
        factor_name: 因子名称
        qlib_path: QLib 数据根目录
    """
    print(f"导出因子: {factor_name}")
    print(f"源文件: {factor_parquet}")

    # 读取因子数据
    df = pl.read_parquet(factor_parquet)
    print(f"因子数据: {len(df)} 行")

    # 确保有必要的列
    if 'stock' not in df.columns or 'date' not in df.columns:
        raise ValueError("因子数据需要 'stock' 和 'date' 列")

    # 读取 QLib 日历
    calendar = read_qlib_calendar(qlib_path)
    print(f"QLib 日历: {len(calendar)} 天 ({calendar[0]} ~ {calendar[-1]})")

    # 转换日期格式
    if df['date'].dtype == pl.Date:
        df = df.with_columns(
            pl.col('date').dt.strftime('%Y-%m-%d').alias('date_str')
        )
    else:
        df = df.with_columns(
            pl.col('date').alias('date_str')
        )

    # 按股票分组
    stocks = df['stock'].unique().to_list()
    print(f"股票数: {len(stocks)}")

    features_path = Path(qlib_path) / 'features'
    bin_filename = f"{factor_name}.day.bin"

    def process_stock(stock: str):
        # 转换股票代码: SH600000 -> sh600000
        stock_lower = stock.lower()
        stock_dir = features_path / stock_lower

        if not stock_dir.exists():
            return None

        # 获取该股票的因子数据
        stock_df = df.filter(pl.col('stock') == stock).sort('date_str')

        if len(stock_df) == 0:
            return None

        # 找到日期范围在日历中的索引
        dates = stock_df['date_str'].to_list()
        values = stock_df[factor_name].to_list()

        # 构建完整的数据数组
        first_date = dates[0]
        last_date = dates[-1]

        start_idx = date_to_qlib_index(first_date, calendar)
        end_idx = date_to_qlib_index(last_date, calendar)

        if start_idx < 0 or end_idx < 0:
            return None

        # 创建数组并填充
        length = end_idx - start_idx + 1
        data = np.full(length, np.nan, dtype=np.float32)

        for date, val in zip(dates, values):
            idx = date_to_qlib_index(date, calendar)
            if idx >= 0:
                data[idx - start_idx] = val if val is not None else np.nan

        # 写入 bin 文件
        bin_path = stock_dir / bin_filename
        write_qlib_bin(data, start_idx, bin_path)

        return stock

    # 并行处理
    print(f"写入 QLib bin 文件...")
    with ThreadPoolExecutor(max_workers=32) as executor:
        results = list(tqdm(executor.map(process_stock, stocks), total=len(stocks)))

    success = sum(1 for r in results if r is not None)
    print(f"完成: {success}/{len(stocks)} 只股票")
    print(f"文件位置: {features_path}/<stock>/{bin_filename}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("用法: python export_to_qlib.py <factor_parquet> <factor_name>")
        print("示例: python export_to_qlib.py factor_production/cache/volatility_20200101_20251217.parquet volatility")
        sys.exit(1)

    factor_parquet = sys.argv[1]
    factor_name = sys.argv[2]

    export_factor_to_qlib(factor_parquet, factor_name)
