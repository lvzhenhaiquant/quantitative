"""
处理财报数据 - 从 QLib 读取并前向填充到日频

QLib 的 .6mon.bin 文件已包含财报数据（按日历存储，大部分是 nan）
本脚本直接读取并前向填充，无需从 Tushare 下载

使用方法:
    python download_financial.py
    python download_financial.py --fields netprofit_yoy,or_yoy

输出:
    cache/financial/netprofit_yoy.parquet  (日频，前向填充)
"""
import os
import sys
import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

# 添加项目根目录
sys.path.insert(0, '/home/zhenhai1/quantitative')

# QLib 数据目录
QLIB_DIR = Path('/home/zhenhai1/quantitative/qlib_data/cn_data')
FEATURES_DIR = QLIB_DIR / 'features'
CALENDAR_FILE = QLIB_DIR / 'calendars' / 'day.txt'

# 输出目录
CACHE_DIR = Path('/home/zhenhai1/quantitative/factor_production/cache/financial')


def load_calendar() -> List[str]:
    """加载交易日历"""
    with open(CALENDAR_FILE) as f:
        return [line.strip() for line in f.readlines()]


def process_field(field: str, start: str, end: str) -> pl.DataFrame:
    """
    处理单个财报字段

    Args:
        field: 字段名，如 'netprofit_yoy'
        start: 开始日期
        end: 结束日期

    Returns:
        日频 DataFrame [stock, date, field]
    """
    # 加载日历
    dates = load_calendar()

    # 获取所有股票目录
    stock_dirs = [d for d in FEATURES_DIR.iterdir() if d.is_dir()]

    all_data = []

    for stock_dir in tqdm(stock_dirs, desc=f"处理 {field}", ncols=80):
        stock_code = stock_dir.name.lower()  # sh600000
        bin_file = stock_dir / f"{field}.6mon.bin"

        if not bin_file.exists():
            continue

        # 读取 bin 文件
        data = np.fromfile(bin_file, dtype='<f4')

        if len(data) == 0:
            continue

        # 对齐日期（从后往前对齐）
        aligned_dates = dates[-len(data):]

        # 创建 DataFrame
        df = pd.DataFrame({
            'date': aligned_dates,
            field: data
        })

        # 前向填充
        df[field] = df[field].ffill()

        # 添加股票代码
        df['stock'] = stock_code

        # 筛选日期范围
        df = df[(df['date'] >= start) & (df['date'] <= end)]

        # 删除空值
        df = df.dropna(subset=[field])

        if len(df) > 0:
            all_data.append(df)

    if not all_data:
        return pl.DataFrame()

    # 合并所有股票
    result = pd.concat(all_data, ignore_index=True)

    # 转换为 Polars
    result_pl = pl.from_pandas(result)
    result_pl = result_pl.with_columns(
        pl.col('date').str.to_date()
    )

    # 重新排列列顺序
    result_pl = result_pl.select(['stock', 'date', field])

    return result_pl


def save_data(df: pl.DataFrame, field: str):
    """保存数据"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    filepath = CACHE_DIR / f"{field}.parquet"
    df.write_parquet(filepath)

    size_mb = os.path.getsize(filepath) / 1024 / 1024
    print(f"已保存: {filepath} ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description='处理财报数据（从 QLib 读取并前向填充）')
    parser.add_argument('--start', type=str, default='2020-01-01',
                        help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-12-31',
                        help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--fields', type=str, default='netprofit_yoy',
                        help='财务字段，逗号分隔')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  财报数据处理器（从 QLib 读取）")
    print(f"  日期范围: {args.start} ~ {args.end}")
    print(f"  字段: {args.fields}")
    print("=" * 60)

    # 检查目录
    if not FEATURES_DIR.exists():
        print(f"错误: QLib 数据目录不存在: {FEATURES_DIR}")
        return

    # 解析字段
    fields = [f.strip() for f in args.fields.split(',')]

    # 处理每个字段
    for field in fields:
        print(f"\n处理字段: {field}")

        df = process_field(field, args.start, args.end)

        if len(df) > 0:
            print(f"  记录数: {len(df)}")
            print(f"  股票数: {df['stock'].n_unique()}")
            save_data(df, field)
        else:
            print(f"  警告: 无数据")

    print("\n" + "=" * 60)
    print("  完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
