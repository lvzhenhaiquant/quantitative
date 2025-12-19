"""
下载股票过滤数据 (三进程版)

从Tushare下载ST、停牌、涨跌停数据，保存为JSON格式
三个任务分别在独立进程中运行，互不干扰

使用方法:
    python download_filter_data.py --start 20200101 --end 20251231

数据格式:
    st_stocks.json: {date: [stock_list]}
    suspend_stocks.json: {date: [stock_list]}
    limit_stocks.json: {date: {limit_up: [...], limit_down: [...]}}
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Optional
import multiprocessing as mp

import tushare as ts
import pandas as pd
from tqdm import tqdm

# 添加项目根目录
sys.path.insert(0, '/home/zhenhai1/quantitative')

# Tushare Token
TUSHARE_TOKEN = 'a79f284e5d10967dacb6531a3c755a701ca79341ff0c60d59f1fcbf1'

# 数据保存目录
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def init_tushare(retry=3):
    """初始化 Tushare API（带重试）"""
    for i in range(retry):
        try:
            ts.set_token(TUSHARE_TOKEN)
            return ts.pro_api()
        except Exception as e:
            time.sleep(0.5 * (i + 1))
    ts.set_token(TUSHARE_TOKEN)
    return ts.pro_api()


def get_trade_dates(start_date: str, end_date: str) -> List[str]:
    """获取交易日历"""
    pro = init_tushare()
    df = pro.trade_cal(
        exchange='SSE',
        start_date=start_date,
        end_date=end_date,
        is_open='1'
    )
    dates = df['cal_date'].tolist()
    return sorted(dates)


def load_json(filename: str) -> Dict:
    """加载已有的JSON文件"""
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_json(data: Dict, filename: str):
    """保存为JSON文件"""
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    size_kb = os.path.getsize(filepath) / 1024
    return len(data), size_kb


# ============================================================
#  进程1: 停牌数据
# ============================================================
def download_suspend_process(trade_dates: List[str], incremental: bool = False):
    """停牌数据下载（独立进程）"""
    pro = init_tushare()

    # 增量模式：加载已有数据，只下载缺失日期
    if incremental:
        suspend_data = load_json('suspend_stocks.json')
        existing_dates = set(suspend_data.keys())
        trade_dates = [d for d in trade_dates if d not in existing_dates]
        print(f"[进程1] 增量模式: 已有 {len(existing_dates)} 天, 待下载 {len(trade_dates)} 天")
    else:
        suspend_data = {}

    if not trade_dates:
        print(f"[进程1] 无需下载")
        return

    for date in tqdm(trade_dates, desc="[进程1] 停牌", ncols=80, colour='green', position=0):
        try:
            df = pro.suspend_d(suspend_type='S', trade_date=date)
            if df is not None and len(df) > 0:
                suspend_data[date] = df['ts_code'].tolist()
            time.sleep(0.1)  # API限速
        except Exception as e:
            time.sleep(0.5)
            continue

    count, size = save_json(suspend_data, 'suspend_stocks.json')
    print(f"\n[进程1] 停牌数据完成: {count} 天, {size:.1f} KB")


# ============================================================
#  进程2: 涨跌停数据
# ============================================================
def download_limit_process(trade_dates: List[str], incremental: bool = False):
    """涨跌停数据下载（独立进程）"""
    pro = init_tushare()

    # 增量模式：加载已有数据，只下载缺失日期
    if incremental:
        limit_data = load_json('limit_stocks.json')
        existing_dates = set(limit_data.keys())
        trade_dates = [d for d in trade_dates if d not in existing_dates]
        print(f"[进程2] 增量模式: 已有 {len(existing_dates)} 天, 待下载 {len(trade_dates)} 天")
    else:
        limit_data = {}

    if not trade_dates:
        print(f"[进程2] 无需下载")
        return

    for date in tqdm(trade_dates, desc="[进程2] 涨跌停", ncols=80, colour='yellow', position=1):
        try:
            # 涨停
            df_up = pro.limit_list_d(trade_date=date, limit_type='U')
            limit_up = df_up['ts_code'].tolist() if df_up is not None and len(df_up) > 0 else []

            time.sleep(0.1)

            # 跌停
            df_down = pro.limit_list_d(trade_date=date, limit_type='D')
            limit_down = df_down['ts_code'].tolist() if df_down is not None and len(df_down) > 0 else []

            if limit_up or limit_down:
                limit_data[date] = {'limit_up': limit_up, 'limit_down': limit_down}

            time.sleep(0.1)
        except Exception as e:
            time.sleep(0.5)
            continue

    count, size = save_json(limit_data, 'limit_stocks.json')
    print(f"\n[进程2] 涨跌停数据完成: {count} 天, {size:.1f} KB")


# ============================================================
#  进程3: ST数据
# ============================================================
def download_st_process(trade_dates: List[str], incremental: bool = False):
    """ST数据下载（独立进程）

    注：ST数据是从 namechange 接口整体获取的，增量模式下也会重新生成全量数据
    """
    pro = init_tushare()
    st_data = {}

    print("[进程3] ST数据: 获取股票名称变更记录...")

    try:
        df = pro.namechange()

        if df is not None and len(df) > 0:
            st_records = df[df['name'].str.contains('ST', case=False, na=False)]
            print(f"[进程3] 找到 {len(st_records)} 条ST记录")

            for _, row in tqdm(st_records.iterrows(),
                              total=len(st_records),
                              desc="[进程3] ST整理",
                              ncols=80, colour='red', position=2):
                start = row['start_date']
                end = row['end_date'] if pd.notna(row['end_date']) else '20991231'
                code = row['ts_code']

                for date in trade_dates:
                    if start <= date <= end:
                        if date not in st_data:
                            st_data[date] = []
                        if code not in st_data[date]:
                            st_data[date].append(code)

    except Exception as e:
        print(f"[进程3] ST数据获取失败: {e}")

    count, size = save_json(st_data, 'st_stocks.json')
    print(f"\n[进程3] ST数据完成: {count} 天, {size:.1f} KB")


# ============================================================
#  主函数
# ============================================================
def download_all(start_date: str, end_date: str,
                 skip_st: bool = False,
                 skip_suspend: bool = False,
                 skip_limit: bool = False,
                 incremental: bool = False):
    """
    三进程并行下载

    Args:
        incremental: 增量模式，只下载缺失的日期
    """
    print("\n" + "=" * 60)
    print("  过滤数据下载器 (三进程并行)")
    print(f"  日期范围: {start_date} ~ {end_date}")
    print(f"  模式: {'增量' if incremental else '全量'}")
    print(f"  CPU核心数: {mp.cpu_count()}")
    print("=" * 60)

    start_time = time.time()

    # 获取交易日
    print("\n获取交易日历...")
    trade_dates = get_trade_dates(start_date, end_date)
    print(f"交易日数量: {len(trade_dates)}\n")

    # 创建三个进程
    processes = []

    if not skip_suspend:
        p1 = mp.Process(target=download_suspend_process, args=(trade_dates, incremental))
        p1.name = "停牌"
        processes.append(p1)

    if not skip_limit:
        p2 = mp.Process(target=download_limit_process, args=(trade_dates, incremental))
        p2.name = "涨跌停"
        processes.append(p2)

    if not skip_st:
        p3 = mp.Process(target=download_st_process, args=(trade_dates, incremental))
        p3.name = "ST"
        processes.append(p3)

    print(f"启动 {len(processes)} 个进程: {[p.name for p in processes]}\n")

    # 启动（间隔1秒避免 Tushare token 文件冲突）
    for i, p in enumerate(processes):
        p.start()
        if i < len(processes) - 1:
            time.sleep(1)

    # 等待完成
    for p in processes:
        p.join()

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print(f"  全部完成! 总耗时: {elapsed:.1f} 秒")
    print("=" * 60)

    # 汇总
    print("\n文件汇总:")
    for filename in ['suspend_stocks.json', 'limit_stocks.json', 'st_stocks.json']:
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath) as f:
                data = json.load(f)
            size_kb = os.path.getsize(filepath) / 1024
            print(f"  {filename}: {len(data)} 天, {size_kb:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description='下载股票过滤数据 (三进程并行)')
    parser.add_argument('--start', type=str, default='20200101',
                        help='开始日期 (YYYYMMDD)')
    parser.add_argument('--end', type=str, default='20251231',
                        help='结束日期 (YYYYMMDD)')
    parser.add_argument('--incremental', '-i', action='store_true',
                        help='增量模式: 只下载缺失的日期')
    parser.add_argument('--skip-st', action='store_true',
                        help='跳过ST数据')
    parser.add_argument('--skip-suspend', action='store_true',
                        help='跳过停牌数据')
    parser.add_argument('--skip-limit', action='store_true',
                        help='跳过涨跌停数据')

    args = parser.parse_args()

    download_all(
        args.start,
        args.end,
        skip_st=args.skip_st,
        skip_suspend=args.skip_suspend,
        skip_limit=args.skip_limit,
        incremental=args.incremental
    )


if __name__ == '__main__':
    main()
