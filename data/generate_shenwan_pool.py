"""
从申万二级行业生成股票池

使用方法:
    python generate_shenwan_pool.py

输出:
    1. shenwan_select.json  - 用于 quantitative 项目
    2. shenwan_select.txt   - 用于 Qlib
"""
import pandas as pd
import json
import os

# ============================================================
# 配置：修改这里选择你想要的行业
# ============================================================
INDUSTRY_CODES = [
    '801055',  # 工业金属
    '801056',  # 能源金属
    '801081',  # 半导体
    '801102',  # 通信设备
    '801156',  # 医疗服务
    '801151',  # 化学制药
    '801737',  # 电池
    '801093',  # 汽车零部件
    '801741',  # 航天装备Ⅱ
    '801745',  # 军工电子Ⅱ
    '801034',  # 化学制品
]

# 数据路径
SHENWAN_CONSTITUENT_PATH = '/home/zhenhai1/quantitative/data/download_data/shenwan_constituent_stock/'
SHENWAN_CLASSIFY_PATH = '/home/zhenhai1/quantitative/data/download_data/shenwan/download_shenwan_classify_df_L2.csv'

# 输出路径
OUTPUT_JSON = '/home/zhenhai1/quantitative/data/download_data/index_weight/shenwan_select.json'
OUTPUT_TXT = '/home/zhenhai1/quantitative/qlib_data/cn_data/instruments/shenwan_select.txt'


def get_industry_names(codes):
    """获取行业代码对应的名称"""
    df = pd.read_csv(SHENWAN_CLASSIFY_PATH)
    codes_full = [f'{c}.SI' for c in codes]
    result = df[df['index_code'].isin(codes_full)][['index_code', 'industry_name']]
    return dict(zip(result['index_code'], result['industry_name']))


def get_constituent_stocks(codes):
    """获取行业成分股"""
    all_stocks = []

    for code in codes:
        file_path = f'{SHENWAN_CONSTITUENT_PATH}SI{code}.parquet'
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            # 只取当前成分 (is_new='Y' 或 out_date 为空)
            current = df[(df['is_new'] == 'Y') | (df['out_date'].isna())]
            stocks = current['ts_code'].tolist()
            print(f'  {code}: {len(stocks)} 只')
            all_stocks.extend(stocks)
        else:
            print(f'  {code}: 文件不存在')

    # 去重
    all_stocks = list(set(all_stocks))
    return all_stocks


def convert_code_to_qlib(ts_code):
    """转换代码格式: 000001.SZ -> SZ000001"""
    code, market = ts_code.split('.')
    return f'{market}{code}'


def convert_code_to_qlib_lower(ts_code):
    """转换代码格式: 000001.SZ -> sz000001"""
    code, market = ts_code.split('.')
    return f'{market.lower()}{code}'


def generate_json(stocks, output_path):
    """生成 JSON 格式股票池（用于 quantitative 项目）"""
    qlib_codes = sorted([convert_code_to_qlib(s) for s in stocks])

    # 生成 2018-2025 的月度日期
    dates = pd.date_range('2018-01-01', '2025-12-31', freq='MS').strftime('%Y-%m-%d').tolist()
    pool_data = {date: qlib_codes for date in dates}

    with open(output_path, 'w') as f:
        json.dump(pool_data, f)

    print(f'JSON 已保存到: {output_path}')


def generate_txt(stocks, output_path):
    """生成 TXT 格式股票池（用于 Qlib）"""
    qlib_codes = sorted([convert_code_to_qlib_lower(s) for s in stocks])

    with open(output_path, 'w') as f:
        for code in qlib_codes:
            f.write(f'{code}\t2018-01-01\t2099-12-31\n')

    print(f'TXT 已保存到: {output_path}')


def main():
    print('=' * 60)
    print('申万二级行业股票池生成器')
    print('=' * 60)

    # 显示选择的行业
    print('\n选择的行业:')
    industry_names = get_industry_names(INDUSTRY_CODES)
    for code in INDUSTRY_CODES:
        name = industry_names.get(f'{code}.SI', '未知')
        print(f'  {code}: {name}')

    # 获取成分股
    print('\n获取成分股:')
    stocks = get_constituent_stocks(INDUSTRY_CODES)
    print(f'\n总计: {len(stocks)} 只股票')

    # 生成文件
    print('\n生成股票池文件:')
    generate_json(stocks, OUTPUT_JSON)
    generate_txt(stocks, OUTPUT_TXT)

    print('\n完成!')


if __name__ == '__main__':
    main()