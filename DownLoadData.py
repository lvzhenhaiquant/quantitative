
from multiprocessing import Pool
import os
import shutil
from asyncio import as_completed
from collections import defaultdict
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import requests
import tushare as ts
import pandas as pd
from pandas.core.dtypes.common import is_object_dtype
from tqdm import tqdm  # 看进度
from typing import List
import json

from urllib3.exceptions import NameResolutionError, MaxRetryError, ConnectionError


# 获取中证1000基本面数据

class DownloadDataFromTushare_Baostock:
    def __init__(self, token):
        self.token = token
        self.save_dir_download = './download_data/orgin_download_data'

        self.save_dir_basic = './download_data/basic'
        self.save_dir_adj = './download_data/adj'
        self.save_dir_daily = './download_data/daily'
        self.save_dir_fina_indicator = './download_data/fina_indicator'
        self.save_dir_income = './download_data/income'
        self.save_dir_balancesheet = './download_data/balancesheet'
        self.save_dir_basic_60mins = './download_data/basic_60mins'
        self.save_dir_basic_5mins = './download_data/basic_5mins'
        self.save_dir_shenwan = './download_data/shenwan'
        self.save_dir_shenwan_daily = './download_data/shenwan_daily'
        self.save_dir_download_index = './download_data/origin_download_index'
        self.save_dir_index_daily = './download_data/index_daily'
        self.save_dir_updown_limit = './download_data/updown_limit'
        self.save_dir_moneyflow = './download_data/moneyflow'
        self.save_dir_index_weight = './download_data/index_weight'

        self.index_mapping = {
            "上证50": "000016.SH",
            "沪深300": "000300.SH",
            "科创50": "000688.SH",
            "中证1000": "000852.SH",
            "中证500": "000905.SH",
            "中证800": "000906.SH",
            "中证全指": "000985.SH",
            "创业板指": "399006.SZ",
        }
        self.code_to_csi = {
            "000016.SH": "csi50",  # 上证50
            "000300.SH": "csi300",  # 沪深300
            "000688.SH": "csi50kechuang",  # 科创50
            "000852.SH": "csi1000",  # 中证1000
            "000905.SH": "csi500",  # 中证500
            "000906.SH": "csi800",  # 中证800
            "000985.SH": "csiall",  # 中证全指
            "399006.SZ": "csigg",  # 创业板指
        }

        self.compute_change_position_flag = False  # 是否计算 _change_position_time 如月度调仓需要往前多获取一个月
        self.add_comlums_shenewn_flag = False  # 是否在下载数据中增加万申L1,L2,L3列，容易造成数据冗余
        self.add_adj_comlums_flag = False  # 是否在basic中合并adj数据，容易造成内存不够

        self.MAX_RETRY = 10  # 最大重试次数
        self.MAX_WORKERS = 196  # 最大线程数

        if self.compute_change_position_flag:
            self.more_month = 1
        else:
            self.more_month = 0

        try:
            ts.set_token(self.token)
            self.pro = ts.pro_api()
        except Exception as e:
            self.pro = None
            print(f"Tushare初始化失败：{e}")
        

    def download_tushare_basic(self, start_date_str, end_date_str):
        if self.pro is None:
            print(f"Tushare初始化失败")
            return
        obtain_date_str = (pd.to_datetime(start_date_str, format='%Y%m%d') - pd.DateOffset(months=self.more_month)).strftime('%Y%m%d')  # '20220901'

        file_name_adj = os.path.join(self.save_dir_download, f'download_adj_df_{start_date_str}_{end_date_str}.parquet')
        file_name_basic = os.path.join(self.save_dir_download,f'download_basic_df_{start_date_str}_{end_date_str}.parquet')
        file_name_daily = os.path.join(self.save_dir_download,f'download_daily_df_{start_date_str}_{end_date_str}.parquet')
        adjust_file_name = os.path.join(self.save_dir_download,f'all_returns_adjusted_{start_date_str}_{end_date_str}.parquet')

        # 验证文件是否存在
        adj_exists = os.path.exists(file_name_adj)
        basic_exists = os.path.exists(file_name_basic)
        daily_exists = os.path.exists(file_name_daily)
        adjusted_exists = os.path.exists(adjust_file_name)
        if adj_exists and basic_exists and daily_exists and adjusted_exists:
            print("[download_tushare_basic]以下四个文件均已存在，无需重复处理，程序退出!")
            return

        # 日期子集跳出
        file_name_adjust_prefix = 'all_returns_adjusted_'
        file_name_adjust_exist, temp1 = self._utils_read_matched_csv_by_prefix(self.save_dir_download,file_name_adjust_prefix)
        temp1 = pd.DataFrame()
        if file_name_adjust_exist:
            print(f"{file_name_adjust_exist}文件已存在，请使用增量下载")
            return

        # 1 分批获取指数权重（避免API调用超限）
        index_weight_df = self._get_index_weight_df('000852.SH',start_date_str, end_date_str)
        if len(index_weight_df) == 0:
            print("获取日线数据为空，退出!")
            return

        # 2 获取所有需查询的股票
        all_stocks = index_weight_df['con_code'].dropna().unique().tolist()
        if len(all_stocks) == 0:
            print("无有效成分股数据，程序终止")
            return

        # 3 获取日线行情
        def get_basic_df():  # 为了多线程处理，改成函数形式
            if os.path.exists(file_name_basic):
                basic_df = pd.read_parquet(file_name_basic, engine='pyarrow')
                if len(basic_df) == 0:
                    print("本地日线数据为空，退出!")
            else:
                basic_df = self._get_daily_basic(all_stocks, obtain_date_str, end_date_str)
                basic_df.to_parquet(file_name_basic, index=False, engine='pyarrow')
            return basic_df

        def get_adj_df():  # 为了多线程处理，改成函数形式
            # 4 获取复权因子
            if os.path.exists(file_name_adj):
                adj_df = pd.read_parquet(file_name_adj, engine='pyarrow')
                if len(adj_df) == 0:
                    print("本地复权因子数据为空，退出!")
            else:
                adj_df = self._get_adj_factor(all_stocks, obtain_date_str, end_date_str)
                adj_df.to_parquet(file_name_adj, index=False, engine='pyarrow')
            return adj_df

        def get_daily_df():  # 为了多线程处理，改成函数形式
            # 4 获取复权因子
            if os.path.exists(file_name_daily):
                daily_df = pd.read_parquet(file_name_daily, engine='pyarrow')
                if len(daily_df) == 0:
                    print("本地日线行情数据为空，退出!")
            else:
                daily_df = self._get_daily(all_stocks, obtain_date_str, end_date_str)
                daily_df.to_parquet(file_name_daily, index=False, engine='pyarrow')
            return daily_df

        with ThreadPoolExecutor(max_workers=3) as executor:
            # 提交任务
            basic_df = executor.submit(get_basic_df)
            adj_df = executor.submit(get_adj_df)
            daily_df = executor.submit(get_daily_df)
            # 获取结果
            basic_df = basic_df.result()
            adj_df = adj_df.result()
            daily_df = daily_df.result()

        if len(basic_df) == 0:
            print("远程日线数据为空，退出!")
            return
        if len(adj_df) == 0:
            print("远程复权因子数据为空，退出!")
            return
        if len(daily_df) == 0:
            print("远程日线行情数据为空，退出!")
            return

        basic_df = basic_df[(pd.to_datetime(basic_df['trade_date']) >= pd.to_datetime(start_date_str)) & (pd.to_datetime(basic_df['trade_date']) <= pd.to_datetime(end_date_str))].reset_index(
            drop=True)  # 重置索引

        # #  5 复权处理+合并数据
        adjust_file_name_df = basic_df
        if self.add_adj_comlums_flag:
            adjust_file_name_df = self._add_adj_comlums(adj_df, basic_df)

        # 6 月度调仓处理
        if self.compute_change_position_flag:
            adjust_file_name_df = self._change_position_time(index_weight_df, adjust_file_name_df, "month", 1,start_date_str, end_date_str)

        if self.add_comlums_shenewn_flag:
            adjust_file_name_df = self.add_wanshen_classify(adjust_file_name_df)

        # 7 保存
        adjust_file_name_df['ts_code'] = adjust_file_name_df['ts_code'].apply(lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")
        adj_df['ts_code'] = adj_df['ts_code'].apply(lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")
        daily_df['ts_code'] = daily_df['ts_code'].apply(lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")
        adjust_file_name_df.to_parquet(adjust_file_name, index=False, engine='pyarrow')
        print(f"调整后的数据已保存至{adjust_file_name}，共{len(adjust_file_name_df)}条记录")

        daily_df.rename(columns={'vol': 'volume'}, inplace=True)
        adj_df.rename(columns={'adj_factor': 'factor'}, inplace=True)

        # 8 单独保存至单股文件夹
        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.submit(self._save_substock_data, self.save_dir_basic, basic_df, 0)
            executor.submit(self._save_substock_data, self.save_dir_daily, daily_df, 1)
            if not self.add_adj_comlums_flag:
                executor.submit(self._save_substock_data, self.save_dir_adj, adj_df, 2)



    def _add_adj_comlums(self, adj_df, basic_df):
        adj_required_cols = ['ts_code', 'trade_date', 'adj_factor']
        basic_required_cols = ['ts_code', 'trade_date', 'close']
        # 检查复权因子数据列
        missing_adj_cols = [col for col in adj_required_cols if col not in adj_df.columns]
        if missing_adj_cols:
            raise ValueError(f"复权因子数据缺失必要列：{missing_adj_cols}，必须包含{adj_required_cols}")
        # 检查行情基础数据列
        missing_basic_cols = [col for col in basic_required_cols if col not in basic_df.columns]
        if missing_basic_cols:
            raise ValueError(f"行情基础数据缺失必要列：{missing_basic_cols}，必须包含{basic_required_cols}")
        adj_df = adj_df.sort_values(['ts_code', 'trade_date'])
        adj_df['trade_date'] = pd.to_datetime(adj_df['trade_date'], errors='coerce')  # 转换日期，无效值转为NaT
        adj_df = adj_df.set_index('trade_date')
        adj_df = adj_df.groupby('ts_code')['adj_factor'].resample('D').ffill().reset_index()
        basic_df['trade_date'] = pd.to_datetime(basic_df['trade_date'], errors='coerce')

        basic_df = basic_df.merge(adj_df[['ts_code', 'trade_date', 'adj_factor']], on=['ts_code', 'trade_date'],how='left')
        basic_df['close'] = basic_df['close'].astype(float, errors='ignore')
        basic_df['adj_factor'] = basic_df['adj_factor'].astype(float, errors='ignore')
        # 计算复权收盘价（仅当close和adj_factor都不为NaN时计算）
        basic_df['adj_close'] = basic_df['close'] * basic_df['adj_factor']
        return basic_df

    def updates_tushare_basic(self, start_date_str, end_date_str):
        file_name_basic_df_prefix = 'download_basic_df_'
        file_name_adj_df_prefix = 'download_adj_df_'
        file_name_daily_df_prefix = 'download_daily_df_'
        file_name_adjusted_df_prefix = 'all_returns_adjusted_'

        file_name_basic, file_name_basic_df = self._utils_read_matched_csv_by_prefix(self.save_dir_download,file_name_basic_df_prefix)
        if len(file_name_basic_df) == 0:
            print(f"未读取到文件：{file_name_basic_df_prefix},请先下载原始数据")
            return
        file_name_adj, file_name_adj_df = self._utils_read_matched_csv_by_prefix(self.save_dir_download,file_name_adj_df_prefix)
        if len(file_name_adj_df) == 0:
            print(f"未读取到文件：{file_name_adj_df_prefix},请先下载原始数据")
            return
        file_name_daily, file_name_daily_df = self._utils_read_matched_csv_by_prefix(self.save_dir_download,file_name_daily_df_prefix)
        if len(file_name_daily_df) == 0:
            print(f"未读取到文件：{file_name_daily_df_prefix},请先下载原始数据")
            return
        file_name_adjusted, file_name_adjusted_df = self._utils_read_matched_csv_by_prefix(self.save_dir_download,file_name_adjusted_df_prefix)
        if len(file_name_adjusted_df) == 0:
            print(f"未读取到文件：{file_name_adjusted_df_prefix},请先下载原始数据")
            return
        file_name_adjusted_df = pd.DataFrame()  # 此文件为basic_df和adj_df 合并而来，设置为空防止占内存，后期赋值保存

        old_start_str, old_end_str = self._utils_extract_date_from_filename(file_name_adjusted)
        if file_name_adjusted:
            if (end_date_str == old_start_str and start_date_str < end_date_str) or (start_date_str == old_end_str and end_date_str > start_date_str):
                print("下载日期正确")
            else:
                print("[ updates_tushare_basic ]下载日期设置错误")
                return
        missing_start_end_list = self._utils_get_missing_date_ranges(old_start_str, old_end_str, start_date_str,end_date_str)
        begin_all_str = min(old_start_str, old_end_str, start_date_str, end_date_str)  # 所有日期首位相连或者重叠后的最边际日期
        end_all_str = max(old_start_str, old_end_str, start_date_str, end_date_str)
        obtain_begin_all_str = (pd.to_datetime(begin_all_str, format='%Y%m%d') - pd.DateOffset(months=self.more_month)).strftime('%Y%m%d')  # '20220901'

        # 0 分批获取指数权重
        index_weight_df = self._get_index_weight_df('000852.SH',begin_all_str, end_all_str)
        if len(index_weight_df) == 0:
            print(f"[updates_tushare_basic] index_weight_df 获取失败，退出!")
            return

        old_stock_list = file_name_basic_df['ts_code'].dropna().unique().tolist()  # 老文件股列表
        all_stocks = index_weight_df['con_code'].dropna().unique().tolist()  # 新日期内股票列表
        new_stock_list = list(set(all_stocks) - set(old_stock_list))

        # 1、从未出现过的股票，日期最小取到最大
        if len(new_stock_list) > 0:
            print("-----获取新股票所有日期数据-----：")
            with ThreadPoolExecutor(max_workers=3) as executor:
                new_basic_df = executor.submit(self._get_daily_basic, new_stock_list, obtain_begin_all_str, end_all_str)
                new_adj_df = executor.submit(self._get_adj_factor, new_stock_list, obtain_begin_all_str, end_all_str)
                new_daily_df = executor.submit(self._get_daily, new_stock_list, obtain_begin_all_str, end_all_str)
                new_basic_df = new_basic_df.result()
                new_adj_df = new_adj_df.result()
                new_daily_df = new_daily_df.result()
            # 1.1获取基础日线行情
            if len(new_basic_df) == 0:
                print("new_basic_df 远程日线数据为空")
            file_name_basic_df = pd.concat(
                [file_name_basic_df, new_basic_df],
                ignore_index=True,  # 重置索引（重要，避免索引冲突）
                axis=0)  # 纵向追加（行级追加）
            # 1.2 获取复权因子
            if len(new_adj_df) == 0:
                print("new_adj_df 远程日线数据为空")
            file_name_adj_df = pd.concat(
                [file_name_adj_df, new_adj_df],  # 先加原始数据，后加新数据
                ignore_index=True,  # 重置索引（重要，避免索引冲突）
                axis=0)  # 纵向追加（行级追加）
            # 1.3 获取日线行情
            if len(new_daily_df) == 0:
                print("new_daily_df 远程日线数据为空")
            file_name_daily_df = pd.concat(
                [file_name_daily_df, new_daily_df],  # 先加原始数据，后加新数据
                ignore_index=True,  # 重置索引（重要，避免索引冲突）
                axis=0)  # 纵向追加（行级追加）

        # 2、对于老股票，补充缺失的时间段数据
        print("-----获取旧股票新日期数据-----：")
        for missing_start_str, missing_end_str in missing_start_end_list:
            obtain_missing_date_str = (pd.to_datetime(missing_start_str, format='%Y%m%d') - pd.DateOffset(months=self.more_month)).strftime('%Y%m%d')  # '20220901'
            with ThreadPoolExecutor(max_workers=3) as executor:
                basic_df = executor.submit(self._get_daily_basic, old_stock_list, obtain_missing_date_str,missing_end_str)
                adj_df = executor.submit(self._get_adj_factor, old_stock_list, obtain_missing_date_str, missing_end_str)
                daily_df = executor.submit(self._get_daily, old_stock_list, obtain_missing_date_str, missing_end_str)
                basic_df = basic_df.result()
                adj_df = adj_df.result()
                daily_df = daily_df.result()
            # 2.1获取基础日线行情
            file_name_basic_df = pd.concat(
                [file_name_basic_df, basic_df],  # 先加原始数据，后加新数据
                ignore_index=True,  # 重置索引（重要，避免索引冲突）
                axis=0)  # 纵向追加（行级追加）
            # 2.2 获取复权因子
            file_name_adj_df = pd.concat(
                [file_name_adj_df, adj_df],  # 先加原始数据，后加新数据
                ignore_index=True,  # 重置索引（重要，避免索引冲突）
                axis=0)  # 纵向追加（行级追加）
            # 2.3 获取日线行情
            if len(daily_df) == 0:
                print("daily_df 远程日线数据为空")
            file_name_daily_df = pd.concat(
                [file_name_daily_df, daily_df],  # 先加原始数据，后加新数据
                ignore_index=True,  # 重置索引（重要，避免索引冲突）
                axis=0)  # 纵向追加（行级追加）

        # 3 排序并去重
        file_name_basic_df = file_name_basic_df.sort_values(by=['ts_code', 'trade_date']).drop_duplicates(keep="first",ignore_index=True)  # 保留第一次出现的重复行（可改为"last"保留最后一次）# 去重后重置索引
        file_name_adj_df = file_name_adj_df.sort_values(by=['ts_code', 'trade_date']).drop_duplicates(keep="first",ignore_index=True)  # 保留第一次出现的重复行（可改为"last"保留最后一次）# 去重后重置索引
        file_name_daily_df = file_name_daily_df.sort_values(by=['ts_code', 'trade_date']).drop_duplicates(keep="first",ignore_index=True)
        # 4 保存并删除
        self._to_new_csv_and_delete_old(file_name_basic, file_name_basic_df, begin_all_str, end_all_str,self.save_dir_download)
        self._to_new_csv_and_delete_old(file_name_adj, file_name_adj_df, begin_all_str, end_all_str,self.save_dir_download)
        self._to_new_csv_and_delete_old(file_name_daily_df, file_name_adj_df, begin_all_str, end_all_str,self.save_dir_download)

        file_name_basic_df['ts_code'] = file_name_basic_df['ts_code'].apply(lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")
        file_name_adj_df['ts_code'] = file_name_adj_df['ts_code'].apply(lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")
        file_name_daily_df['ts_code'] = file_name_daily_df['ts_code'].apply(lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")

        # #  5 复权处理+合并数据
        file_name_adjusted_df = file_name_basic_df
        if self.add_adj_comlums_flag:
            file_name_adjusted_df = self._add_adj_comlums(file_name_adj_df, file_name_basic_df)

        # 6 月度调仓处理,增加 列['adjusted_month', 'is_current_stock'] 到file_name_adjusted_df表
        if self.compute_change_position_flag:
            file_name_adjusted_df = self._change_position_time(index_weight_df, file_name_adjusted_df, "month", 1,start_date_str, end_date_str)

        # 7 保存并删除
        self._to_new_csv_and_delete_old(file_name_adjusted, file_name_adjusted_df, begin_all_str, end_all_str,self.save_dir_download)

        # Tushare 一些字段与qlib基础要求字段不一样
        file_name_daily_df.rename(columns={'vol': 'volume'}, inplace=True)
        file_name_adj_df.rename(columns={'adj_factor': 'factor'}, inplace=True)

        # 8 单独保存至单股文件夹
        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.submit(self._save_substock_data,self.save_dir_basic,file_name_basic_df,0)
            executor.submit(self._save_substock_data,self.save_dir_daily, file_name_daily_df,1)
            if not self.add_adj_comlums_flag:
                executor.submit(self._save_substock_data,self.save_dir_adj, file_name_adj_df,2)

    # ----↓↓↓↓-----download_tushare_basic, updates_tushare_basic 使用----↓↓↓↓-----
    def _get_daily_basic(self, stock_list, start_date_str, end_date_str):
        basic_all = []
        basic_df = pd.DataFrame()
        max_retry = self.MAX_RETRY  # 最大重试次数
        for i, stock_code in enumerate(tqdm(stock_list, desc='获取日线数据'), 1):
            retry_count = 0
            success = False
            tmp = None
            retry_delay = 1  # 每只股票重置初始重试间隔为1秒
            while retry_count < max_retry and not success:
                try:
                    # 查询该股票在日期范围内的所有数据
                    tmp = self.pro.daily_basic(ts_code=stock_code, start_date=start_date_str, end_date=end_date_str)
                    success = True  # 成功获取，退出重试循环
                    if len(tmp) > 0:
                        basic_all.append(tmp)
                    else:
                        print(f"daily_basic_ 未查询到股票{stock_code}数据，{start_date_str}---{end_date_str}")
                # 只捕获网络相关异常，非网络异常直接跳过重试
                except (NameResolutionError, MaxRetryError, ConnectionError,TimeoutError,ConnectionResetError,requests.exceptions.RequestException) as e:
                    retry_count += 1
                    if retry_count < max_retry:
                        print(f"_get_daily_basic 获取{stock_code}数据失败（网络错误）：{str(e)[:50]}... 第{retry_count}次重试，等待{retry_delay}秒")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避，间隔翻倍
                    else:
                        print(f"_get_daily_basic 获取{stock_code}数据失败：{str(e)[:50]}... 已重试{max_retry}次，跳过该股票")
                                # 非网络异常：直接终止重试，跳过该股票
                except Exception as e:
                    print(f"_get_daily_basic 获取失败（{stock_code}）：{e}，非网络错误，直接跳过")
                    break  # 跳出while重试循环

            # 合并结果
        if basic_all:
            basic_all = [df for df in basic_all if not df.empty]
            basic_df = pd.concat(basic_all, ignore_index=True)
            basic_df['trade_date'] = pd.to_datetime(basic_df['trade_date']).dt.strftime('%Y-%m-%d')
            print(f"\n共获取{len(basic_df)}条数据")
        return basic_df

    def _get_adj_factor(self, stock_list, start_date_str, end_date_str):
        adj_all = []
        adj_df = pd.DataFrame()
        max_retry = self.MAX_RETRY  # 最大重试次数
        for i, stock_code in enumerate(tqdm(stock_list, desc='获取复权因子'), 1):
            retry_count = 0
            success = False
            tmp = None
            retry_delay = 1  # 每只股票重置初始重试间隔为1秒
            while retry_count < max_retry and not success:
                try:
                    # 调用Tushare复权因子接口
                    tmp = self.pro.adj_factor(ts_code=stock_code, start_date=start_date_str, end_date=end_date_str)
                    success = True  # 成功获取，退出重试循环
                    if not tmp.empty:
                        adj_all.append(tmp)
                    else:
                        print(f"adj_factor 未查询到股票{stock_code}数据，{start_date_str}---{end_date_str}")
                # 只捕获网络相关异常，非网络异常直接跳过重试
                except (NameResolutionError, MaxRetryError, ConnectionError,TimeoutError,ConnectionResetError,requests.exceptions.RequestException) as e:
                    retry_count += 1
                    if retry_count < max_retry:
                        print(f"获取{stock_code}数据失败（网络错误）：{str(e)[:50]}... 第{retry_count}次重试，等待{retry_delay}秒")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避，间隔翻倍
                    else:
                        print(f"获取{stock_code}数据失败：{str(e)[:50]}... 已重试{max_retry}次，跳过该股票")
                except Exception as e:
                    print(f"获取复权因子失败（{stock_code}）：{e}")
                    break
        if adj_all:  # 合并复权因子（过滤无效数据）
            adj_all = [df for df in adj_all if not df.empty]
            adj_df = pd.concat(adj_all, ignore_index=True)
            adj_df = adj_df.dropna(subset=['adj_factor', 'ts_code', 'trade_date'])  # 过滤NaN
            adj_df['trade_date'] = pd.to_datetime(adj_df['trade_date']).dt.strftime('%Y-%m-%d')
            print(f"共获取{len(adj_df)}条复权因子数据")
        return adj_df

    def _get_daily(self, stock_list, start_date_str, end_date_str):
        daily_all = []
        daily_df = pd.DataFrame()
        max_retry = self.MAX_RETRY  # 最大重试次数
        for i, stock_code in enumerate(tqdm(stock_list, desc='获取日线行情'), 1):
            retry_count = 0
            success = False
            tmp = None
            retry_delay = 1  # 每只股票重置初始重试间隔为1秒
            while retry_count < max_retry and not success:
                try:
                    # 调用Tushare复权因子接口
                    tmp = self.pro.daily(ts_code=stock_code, start_date=start_date_str, end_date=end_date_str)
                    success = True  # 成功获取，退出重试循环
                    if not tmp.empty:
                        daily_all.append(tmp)
                    else:
                        print(f"daily 未查询到股票{stock_code}数据，{start_date_str}---{end_date_str}")
                # 只捕获网络相关异常，非网络异常直接跳过重试
                except (NameResolutionError, MaxRetryError, ConnectionError,TimeoutError,ConnectionResetError,requests.exceptions.RequestException) as e:
                    retry_count += 1
                    if retry_count < max_retry:
                        print(f"获取{stock_code}数据失败（网络错误）：{str(e)[:50]}... 第{retry_count}次重试，等待{retry_delay}秒")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避，间隔翻倍
                    else:
                        print(f"获取{stock_code}数据失败：{str(e)[:50]}... 已重试{max_retry}次，跳过该股票")
                except Exception as e:
                    print(f"获取日线行情失败（{stock_code}）：{e}")
                    break
        if daily_all:  # 合并复权因子（过滤无效数据）
            adj_all = [df for df in daily_all if not df.empty]
            daily_df = pd.concat(adj_all, ignore_index=True)
            daily_df = daily_df.dropna(subset=['ts_code', 'trade_date'])  # 过滤NaN
            daily_df['trade_date'] = pd.to_datetime(daily_df['trade_date']).dt.strftime('%Y-%m-%d')
            print(f"共获取{len(daily_df)}条日线行情数据")
        return daily_df

    def _change_position_time(self, index_weight_df_origin, file_name_adjusted_df_origin, period, time_num,start_date_str, end_date_str):
        index_weight_df = index_weight_df_origin.copy(deep=True)  # 深拷贝防止修改原先的值
        file_name_adjusted_df = file_name_adjusted_df_origin[['ts_code', 'trade_date']].copy(deep=True)
        if not is_object_dtype(file_name_adjusted_df_origin['trade_date']):
            file_name_adjusted_df_origin['trade_date'] = (file_name_adjusted_df_origin['trade_date'].dt.strftime('%Y%m%d'))
        file_name_adjusted_df['trade_date'] = pd.to_datetime(file_name_adjusted_df['trade_date']).dt.floor('d')  # 去掉时分秒

        # 1 获取交易日历（统一datetime格式）
        if period == 'month':
            obtain_date_str = (pd.to_datetime(start_date_str, format='%Y%m%d') - pd.DateOffset(months=time_num)).strftime('%Y%m%d')  # '20220901'
        elif period == 'day':
            obtain_date_str = (pd.to_datetime(start_date_str, format='%Y%m%d') - pd.DateOffset(days=time_num)).strftime('%Y%m%d')  # '20220901'
        else:
            print("参数period:month,day为空")
            return
        try:
            trade_cal = self.pro.trade_cal(exchange='SSE', start_date=obtain_date_str, end_date=end_date_str)
            trade_dates_str = trade_cal[trade_cal['is_open'] == 1]['cal_date'].tolist()
            trade_dates_dt = sorted([pd.to_datetime(d) for d in trade_dates_str])  # 全程用datetime
        except Exception as e:
            print(f"获取交易日历失败：{e}")
            return
        # 2 获取每月第一个交易日
        trade_dates_by_ym = defaultdict(list)
        for d in trade_dates_dt:
            trade_dates_by_ym[d.strftime('%Y%m')].append(d)
        first_td_month = {ym: min(dates) for ym, dates in trade_dates_by_ym.items()}
        first_td_month = dict(sorted(first_td_month.items()))  # 按月份排序
        first_td_month = {ym: pd.to_datetime(td.strftime('%Y%m%d')) for ym, td in first_td_month.items()}

        # 3 构建“月份-成分股”映射
        index_weight_df['month'] = index_weight_df['trade_date'].dt.strftime('%Y%m')
        date_stock_map = index_weight_df.drop_duplicates(['month', 'con_code']).groupby('month')['con_code'].unique().to_dict()
        month_stock_map = {}
        for month_key, first_td in first_td_month.items():
            first_td_str = first_td.strftime('%Y%m')
            month_stock_map[month_key] = set(date_stock_map.get(first_td_str, []))  # 确保是集合

        # 调整月份分割逻辑：以每月16-22日的第一个星期一作为分割点
        def is_split_date(date):
            day = date.day
            weekday = date.weekday()  # 0=星期一
            return (16 <= day <= 22) and (weekday == 0)  # 关键修改：判断是否在16-22日且为星期一（作为当月成分股的起始分割点）

        # 生成每个月的分割日期（16-22日的第一个星期一）
        unique_months = pd.to_datetime(file_name_adjusted_df['trade_date']).dt.to_period('M').unique()
        split_dates = {}  # 存储{月份: 分割日期}
        min_month = min(unique_months)  # 数据起始月份（如2022-10）
        prev_month = min_month - 1  # 起始月份的前一个月（如2022-09）
        split_dates[prev_month] = pd.Timestamp(f"{prev_month.year}-{prev_month.month}-22")  # 为起始月的前一个月创建虚拟分割点（确保存在）

        for month in unique_months:
            # 获取当月所有日期（含时间的30分钟数据需提取日期部分）
            month_start = month.to_timestamp()
            month_end = (month + 1).to_timestamp() - pd.Timedelta(days=1)
            # 提取当月所有不重复的日期（忽略时间部分）
            month_trade_dates = pd.to_datetime(file_name_adjusted_df[
                                                   (file_name_adjusted_df['trade_date'] >= month_start) & (file_name_adjusted_df['trade_date'] <= month_end)]['trade_date'].dt.date.unique())
            # 筛选当月16-22日的星期一（作为分割点）
            candidates = [d for d in month_trade_dates if is_split_date(d)]
            if candidates:
                split_dates[month] = min(candidates)  # 取最早的一个星期一作为分割点
            else:
                target = pd.Timestamp(f"{month.year}-{month.month}-22")  # 若无星期一，取22日或之前最近的交易日
                if target > month_end:
                    target = month_end
                valid_dates = [d for d in month_trade_dates if d <= target]  # 找22日或之前最近的交易日
                split_dates[month] = max(valid_dates) if valid_dates else month_end

        # 调整月份标记：分割点及之后的日期属于"当月成分股"（实际对应下月的成分股周期）
        def adjust_month(row):
            trade_date = row['trade_date'].date()  # 提取日期部分（忽略时间）
            original_month = row['trade_date'].to_period('M')
            current_split = split_dates.get(original_month)  # 当前月分割点（16-22日的周一）
            if current_split:
                if trade_date >= current_split.date():  # 核心逻辑：分割点后→当月，分割点前→前月
                    return original_month  # 分割点后 → 当月
                else:
                    return original_month - 1  # 分割点前 → 前月（即使是起始月）
            else:
                return original_month  # 无分割点时，全部归属于当前月

        file_name_adjusted_df['adjusted_month'] = file_name_adjusted_df.apply(adjust_month, axis=1)
        # 优化成分股标志位计算
        month_str_series = file_name_adjusted_df['adjusted_month'].dt.strftime('%Y%m')
        month_stocks_series = month_str_series.map(lambda x: date_stock_map.get(x, set()))
        file_name_adjusted_df['is_current_stock'] = [1 if ts_code in stocks else 0 for ts_code, stocks in zip(file_name_adjusted_df['ts_code'],month_stocks_series)]  # 哪些股票属于当月选股池
        file_name_adjusted_df['adjusted_month'] = file_name_adjusted_df['adjusted_month'].dt.strftime('%Y%m')
        file_name_adjusted_df['trade_date'] = file_name_adjusted_df['trade_date'].dt.strftime('%Y%m%d')
        # 直接原表增加列.  pandas 的 merge 是返回新对象，不会原地修改原 DataFrame,也不支持inplace
        file_name_adjusted_df_origin = file_name_adjusted_df_origin.merge(file_name_adjusted_df[['ts_code', 'trade_date', 'adjusted_month', 'is_current_stock']],on=['ts_code', 'trade_date'], how='left')
        print("_change_position_time 执行完毕")
        return file_name_adjusted_df_origin

    # ------↑↑↑↑----download_tushare_basic, updates_tushare_basic 使用-----↑↑↑↑-----

    def download_tushare_finance(self, start_date_str, end_date_str):
        file_name_fina = os.path.join(self.save_dir_download,f'download_fina_df_{start_date_str}_{end_date_str}.parquet')
        file_name_income = os.path.join(self.save_dir_download,f'download_income_df_{start_date_str}_{end_date_str}.parquet')
        file_name_balance = os.path.join(self.save_dir_download,f'download_balance_df_{start_date_str}_{end_date_str}.parquet')

        fine_exists = os.path.exists(file_name_fina)
        income_exists = os.path.exists(file_name_income)
        balance_exists = os.path.exists(file_name_balance)
        if fine_exists and income_exists and balance_exists:
            print("[download_tushare_finance]以下三个文件均已存在，无需重复处理，程序退出!")
            return

        file_name_fina_prefix = 'download_fina_df_'
        file_name_fina_exist, temp1 = self._utils_read_matched_csv_by_prefix(self.save_dir_download,file_name_fina_prefix)
        temp1 = pd.DataFrame()
        if file_name_fina_exist:
            print(f"{file_name_fina_exist}文件已存在，请使用增量下载")
            return

        # 分批获取指数权重
        index_weight_df = self._get_index_weight_df('000852.SH',start_date_str, end_date_str)
        # 权重数据清洗
        all_stocks = index_weight_df['con_code'].dropna().unique().tolist()
        if not all_stocks:
            print("无有效成分股数据，程序终止")
            return

        with ThreadPoolExecutor(max_workers=3) as executor:
            # 提交三个任务到线程池
            executor.submit(self._get_fina_indicator, all_stocks, start_date_str, end_date_str, file_name_fina)
            executor.submit(self._get_income_df, all_stocks, start_date_str, end_date_str, file_name_income)
            executor.submit(self._get_balance_df, all_stocks, start_date_str, end_date_str, file_name_balance)

    # ---------download_tushare_finance 使用---------
    def _get_fina_indicator(self, all_stocks, start_date_str, end_date_str, file_name_fina):
        fina_all = []
        fina_df = pd.DataFrame()
        if os.path.exists(file_name_fina):
            fina_df = pd.read_parquet(file_name_fina, engine='pyarrow')
            print("[_get_fina_indicator ]加载fina_df数据")
        else:
            for i, stock_code in enumerate(tqdm(all_stocks, desc='获取fina_df数据'), 1):
                # 查询该股票在日期范围内的所有数据
                tmp = self.pro.fina_indicator(ts_code=stock_code, start_date=start_date_str, end_date=end_date_str)
                # print(f"fina_df正在查询第{i}/{len(all_stocks)}只股票：{stock_code}，长度为：{len(tmp)}")
                if len(tmp) > 0:
                    fina_all.append(tmp)
                time.sleep(0.13)  # 您每分钟最多访问该接口500次
                # 合并结果
            if fina_all:
                fina_df = pd.concat(fina_all, ignore_index=True)
                fina_df = fina_df.sort_values(by=['ts_code', 'ann_date'])
                fina_df['ann_date'] = pd.to_datetime(fina_df['ann_date']).dt.strftime('%Y%m%d')
                print(f"\n共获取{len(fina_df)}条数据")
                fina_df['ts_code'] = fina_df['ts_code'].apply(lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")
                fina_df = fina_df.dropna(subset=['ann_date'])
                fina_df = fina_df.drop_duplicates(keep="first",ignore_index=True)  # 保留第一次出现的重复行（可改为"last"保留最后一次） # 去重后重置索引
        fina_df.to_parquet(file_name_fina, index=False, engine='pyarrow')
        self._save_substock_data(self.save_dir_fina_indicator, fina_df, 0)


    def _get_income_df(self, all_stocks, start_date_str, end_date_str, file_name_income):
        income_df = pd.DataFrame()
        income_all = []

        if os.path.exists(file_name_income):
            income_df = pd.read_parquet(file_name_income, engine='pyarrow')
            print("[_get_income_df]加载income数据...")
        else:
            for i, stock_code in enumerate(tqdm(all_stocks, desc='获取income数据'), 1):
                # 查询该股票在日期范围内的所有数据
                tmp = self.pro.income(ts_code=stock_code, start_date=start_date_str, end_date=end_date_str)
                # print(f"income_df_正在查询第{i}/{len(all_stocks)}只股票：{stock_code}，长度为：{len(tmp)}")
                if len(tmp) > 0:
                    income_all.append(tmp)
                time.sleep(0.13)  # 您每分钟最多访问该接口500次
                # 合并结果
            if income_all:
                income_df = pd.concat(income_all, ignore_index=True)
                income_df = income_df.sort_values(by=['ts_code', 'ann_date'])
                income_df['ann_date'] = pd.to_datetime(income_df['ann_date']).dt.strftime('%Y%m%d')
                print(f"\n共获取{len(income_df)}条数据")
                income_df['ts_code'] = income_df['ts_code'].apply(lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")
                income_df = income_df.dropna(subset=['ann_date'])
                income_df = income_df.drop_duplicates(keep="first",ignore_index=True)  # 保留第一次出现的重复行（可改为"last"保留最后一次） # 去重后重置索引
        income_df.to_parquet(file_name_income, index=False, engine='pyarrow')
        # 按股票代码分组并保存单个文件
        self._save_substock_data(self.save_dir_income, income_df, 1)


    def _get_balance_df(self, all_stocks, start_date_str, end_date_str, file_name_balance):
        balance_df = pd.DataFrame()
        balance_all = []
        if os.path.exists(file_name_balance):
            balance_df = pd.read_parquet(file_name_balance, engine='pyarrow')
            print("[_get_balance_df]加载balance_df数据...")
        else:
            for i, stock_code in enumerate(tqdm(all_stocks, desc='获取balance_df 数据'), 1):
                # 查询该股票在日期范围内的所有数据
                tmp = self.pro.balancesheet(ts_code=stock_code, start_date=start_date_str, end_date=end_date_str)
                # print(f"balance_df正在查询第{i}/{len(all_stocks)}只股票：{stock_code}，长度为：{len(tmp)}")
                if len(tmp) > 0:
                    balance_all.append(tmp)
                time.sleep(0.13)  # 您每分钟最多访问该接口500次
                # 合并结果
            if balance_all:
                balance_df = pd.concat(balance_all, ignore_index=True)
                balance_df = balance_df.sort_values(by=['ts_code', 'ann_date'])
                balance_df['ann_date'] = pd.to_datetime(balance_df['ann_date']).dt.strftime('%Y%m%d')
                print(f"\n共获取{len(balance_df)}条数据")
                balance_df['ts_code'] = balance_df['ts_code'].apply(lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")
                balance_df = balance_df.dropna(subset=['ann_date'])
                balance_df = balance_df.drop_duplicates(keep="first",ignore_index=True)  # 保留第一次出现的重复行（可改为"last"保留最后一次） # 去重后重置索引
        balance_df.to_parquet(file_name_balance, index=False, engine='pyarrow')
        # 按股票代码分组并保存单个文件
        self._save_substock_data(self.save_dir_balancesheet, balance_df, 2)

    # -----↑↑↑↑----download_tushare_finance 使用----↑↑↑↑-----

    # ----------------增量更新财务数据------------------
    def updates_tushare_finance(self, start_date_str, end_date_str):
        file_name_fina_df_prefix = f'download_fina_df_'
        file_name_income_df_prefix = f'download_income_df_'
        file_name_balance_df_prefix = f'download_balance_df_'

        file_name_fina, file_name_fina_df = self._utils_read_matched_csv_by_prefix(self.save_dir_download,file_name_fina_df_prefix)
        if len(file_name_fina_df) == 0:
            print(f"未读取到文件：{file_name_fina_df_prefix}")
            return

        file_name_income, file_name_income_df = self._utils_read_matched_csv_by_prefix(self.save_dir_download,file_name_income_df_prefix)
        if len(file_name_income_df) == 0:
            print(f"未读取到文件：{file_name_income_df_prefix}")
            return

        file_name_balance, file_name_balance_df = self._utils_read_matched_csv_by_prefix(self.save_dir_download,file_name_balance_df_prefix)
        if len(file_name_balance_df) == 0:
            print(f"未读取到文件：{file_name_balance_df_prefix}")
            return

        old_start_str, old_end_str = self._utils_extract_date_from_filename(file_name_fina)
        if file_name_fina:
            if (end_date_str == old_start_str and start_date_str < end_date_str) or (start_date_str == old_end_str and end_date_str > start_date_str):
                print("下载日期正确")
            else:
                print("[ updates_tushare_finance ]下载日期设置错误")
                return
        missing_start_end_list = self._utils_get_missing_date_ranges(old_start_str, old_end_str, start_date_str,end_date_str)
        begin_all_str = min(old_start_str, old_end_str, start_date_str, end_date_str)  # 所有日期首位相连或者重叠后的最边际日期
        end_all_str = max(old_start_str, old_end_str, start_date_str, end_date_str)
        #
        # 0 分批获取指数权重 日期从最小取最大，从而获取所有的新stock list
        index_weight_df = self._get_index_weight_df('000852.SH',begin_all_str, end_all_str)
        if (len(index_weight_df) > 0):
            index_weight_df['con_code'] = index_weight_df['con_code'].apply(lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")
        old_stock_list = list(set(file_name_fina_df['ts_code'].dropna().unique()) | set(file_name_income_df['ts_code'].dropna().unique()) | set(file_name_balance_df['ts_code'].dropna().unique()))  # 老文件股列表
        all_stocks = index_weight_df['con_code'].dropna().unique().tolist()  # 新日期内股票列表
        new_stock_list = list(set(all_stocks) - set(old_stock_list))
        # 1、从未出现过的股票，日期最小取到最大
        if len(new_stock_list) > 0:
            with ThreadPoolExecutor(max_workers=3) as executor:
                fina_df_temp = executor.submit(self._update_get_finace_df, new_stock_list, "fina_indicator",begin_all_str, end_all_str)
                income_df_temp = executor.submit(self._update_get_finace_df, new_stock_list, "income", begin_all_str,end_all_str)
                balance_df_temp = executor.submit(self._update_get_finace_df, new_stock_list, "balance", begin_all_str,end_all_str)
                fina_df_temp = fina_df_temp.result()
                income_df_temp = income_df_temp.result()
                balance_df_temp = balance_df_temp.result()
            # 1.1  获取财务指标数据
            # fina_df_temp = self._update_get_finace_df(new_stock_list,"fina_indicator",begin_all_str,end_all_str)
            if len(fina_df_temp) == 0:
                print("fina_df_temp 获取数据为空")
            file_name_fina_df = pd.concat([file_name_fina_df, fina_df_temp], ignore_index=True, axis=0)
            # 1.2 获取利润数据
            # income_df_temp = self._update_get_finace_df(new_stock_list,"income", begin_all_str, end_all_str)
            if len(income_df_temp) == 0:
                print("income_df_temp 获取数据为空")
            file_name_income_df = pd.concat([file_name_income_df, income_df_temp], ignore_index=True, axis=0)
            # 1.3 获取资产负债数据
            # balance_df_temp = self._update_get_finace_df(new_stock_list,"balance", begin_all_str, end_all_str)
            if len(balance_df_temp) == 0:
                print("balance_df_temp 获取数据为空")
            file_name_balance_df = pd.concat([file_name_balance_df, balance_df_temp], ignore_index=True, axis=0)
        print("new_stock_list 执行完毕")
        # 2、对于老股票，补充缺失的时间段数据
        for missing_start_str, missing_end_str in missing_start_end_list:
            with ThreadPoolExecutor(max_workers=3) as executor:
                fina_df_temp_1 = executor.submit(self._update_get_finace_df, old_stock_list, "fina_indicator",missing_start_str, missing_end_str)
                income_df_temp_1 = executor.submit(self._update_get_finace_df, old_stock_list, "income",missing_start_str, missing_end_str)
                balance_df_temp_1 = executor.submit(self._update_get_finace_df, old_stock_list, "balance",missing_start_str, missing_end_str)
                fina_df_temp_1 = fina_df_temp_1.result()
                income_df_temp_1 = income_df_temp_1.result()
                balance_df_temp_1 = balance_df_temp_1.result()
            # 2.1  获取财务指标数据
            # fina_df_temp_1 = self._update_get_finace_df(old_stock_list, "fina_indicator", missing_start_str, missing_end_str)
            if len(fina_df_temp_1) == 0:
                print("fina_df_temp_1 获取数据为空,可能无增量数据")
            file_name_fina_df = pd.concat([file_name_fina_df, fina_df_temp_1], ignore_index=True, axis=0)
            # 2.2 获取利润数据
            # income_df_temp_1 = self._update_get_finace_df(old_stock_list, "income", missing_start_str, missing_end_str)
            if len(income_df_temp_1) == 0:
                print("income_df_temp_1 获取数据为空,可能无增量数据")
            file_name_income_df = pd.concat([file_name_income_df, income_df_temp_1], ignore_index=True, axis=0)
            # 2.3 获取资产负债数据
            # balance_df_temp_1 = self._update_get_finace_df(old_stock_list, "balance", missing_start_str, missing_end_str)
            if len(balance_df_temp_1) == 0:
                print("balance_df_temp 获取数据为空,可能无增量数据")
            file_name_balance_df = pd.concat([file_name_balance_df, balance_df_temp_1], ignore_index=True, axis=0)
        print("missing_start_end_list 执行完毕")
        # 3、排序并去重
        file_name_fina_df = file_name_fina_df.sort_values(by=['ts_code', 'ann_date']).drop_duplicates(keep="first",ignore_index=True)
        file_name_income_df = file_name_income_df.sort_values(by=['ts_code', 'ann_date']).drop_duplicates(keep="first",ignore_index=True)
        file_name_balance_df = file_name_balance_df.sort_values(by=['ts_code', 'ann_date']).drop_duplicates(keep="first", ignore_index=True)

        # 4、保存新文件并删除旧文件
        self._to_new_csv_and_delete_old(file_name_fina, file_name_fina_df, begin_all_str, end_all_str,self.save_dir_download)
        self._to_new_csv_and_delete_old(file_name_income, file_name_income_df, begin_all_str, end_all_str,self.save_dir_download)
        self._to_new_csv_and_delete_old(file_name_balance, file_name_balance_df, begin_all_str, end_all_str,self.save_dir_download)

    def download_tushare_A_basic(self, start_date_str, end_date_str):
        if self.pro is None:
            print(f"Tushare初始化失败")
            return
        obtain_date_str = (pd.to_datetime(start_date_str, format='%Y%m%d') - pd.DateOffset(months=self.more_month)).strftime('%Y%m%d')  # '20220901'

        file_name_adj = os.path.join(self.save_dir_download,f'download_A_adj_df_{start_date_str}_{end_date_str}.parquet')
        file_name_basic = os.path.join(self.save_dir_download,f'download_A_basic_df_{start_date_str}_{end_date_str}.parquet')
        file_name_daily = os.path.join(self.save_dir_download,f'download_A_daily_df_{start_date_str}_{end_date_str}.parquet')
        file_name_updown_limit =os.path.join(self.save_dir_download,f'download_A_updown_limit_df_{start_date_str}_{end_date_str}.parquet')
        file_name_moneyflow =os.path.join(self.save_dir_download,f'download_A_moneyflow_df_{start_date_str}_{end_date_str}.parquet')
        adjust_file_name = os.path.join(self.save_dir_download,f'all_returns_A_adjusted_{start_date_str}_{end_date_str}.parquet')
        adjust_file_name_df = pd.DataFrame()

        adj_exists = os.path.exists(file_name_adj)
        basic_exists = os.path.exists(file_name_basic)
        daily_exists = os.path.exists(file_name_daily)
        updown_limit_exists = os.path.exists(file_name_updown_limit)
        moneyflow_exists = os.path.exists(file_name_moneyflow)
        adjusted_exists = os.path.exists(adjust_file_name)
        if adj_exists and basic_exists and daily_exists and adjusted_exists and updown_limit_exists and moneyflow_exists:
            print("[download_tushare_basic]以下6个文件均已存在，无需重复处理，程序退出!")
            return

        adjust_file_name_prefix = 'all_returns_A_adjusted_'
        adjust_file_name_exist, temp1 = self._utils_read_matched_csv_by_prefix(self.save_dir_download, adjust_file_name_prefix)
        temp1 = pd.DataFrame()
        if adjust_file_name_exist:
            print(f"{adjust_file_name_exist}文件已存在，请使用增量下载")
            return

        # 1 获取全部股票列表
        stock_basic_df = self._get_stock_basic_df()
        if len(stock_basic_df) == 0:
            print("获取日线数据为空，退出!")
            return

        # 2 获取所有需查询的股票
        all_stocks = stock_basic_df['ts_code'].dropna().unique().tolist()
        if len(all_stocks) == 0:
            print("无有效成分股数据，程序终止")
            return

        def get_basic_df():
            # 3 获取日线行情
            if os.path.exists(file_name_basic):
                basic_df = pd.read_parquet(file_name_basic, engine='pyarrow')
            else:
                basic_df = self._get_daily_basic(all_stocks, obtain_date_str, end_date_str)
                basic_df.to_parquet(file_name_basic, index=False, engine='pyarrow')
            return basic_df

        def get_adj_df():
            # 4 获取复权因子
            if os.path.exists(file_name_adj):
                adj_df = pd.read_parquet(file_name_adj, engine='pyarrow')
            else:
                adj_df = self._get_adj_factor(all_stocks, obtain_date_str, end_date_str)
                adj_df.to_parquet(file_name_adj, index=False, engine='pyarrow')
            return adj_df

        def get_daily_df():
            # 5 获取日线行情
            if os.path.exists(file_name_daily):
                daily_df = pd.read_parquet(file_name_daily, engine='pyarrow')
            else:
                daily_df = self._get_daily(all_stocks, obtain_date_str, end_date_str)
                daily_df.to_parquet(file_name_daily, index=False, engine='pyarrow')
            return daily_df
        
        def get_updown_limit_df():
            #  获取涨跌停数据
            if os.path.exists(file_name_updown_limit):
                updown_limit_df = pd.read_parquet(file_name_updown_limit, engine='pyarrow')
            else:
                updown_limit_df = self._get_updown_limit(all_stocks, obtain_date_str, end_date_str)
                updown_limit_df.to_parquet(file_name_updown_limit, index=False, engine='pyarrow')
            return updown_limit_df
        
        def get_moneyflow_df():
            #  获取资金流向数据
            if os.path.exists(file_name_moneyflow):
                moneyflow_df = pd.read_parquet(file_name_moneyflow, engine='pyarrow')
            else:
                moneyflow_df = self._get_moneyflow(all_stocks, obtain_date_str, end_date_str)
                moneyflow_df.to_parquet(file_name_moneyflow, index=False, engine='pyarrow')
            return moneyflow_df

        with ThreadPoolExecutor(max_workers=2) as executor:
            # 提交任务
            basic_df_result = executor.submit(get_basic_df)
            adj_df_result = executor.submit(get_adj_df)
            daily_df_result = executor.submit(get_daily_df)
            updown_limit_df_result = executor.submit(get_updown_limit_df)
            moneyflow_df_result = executor.submit(get_moneyflow_df)
            # 获取结果
            basic_df = basic_df_result.result()
            adj_df = adj_df_result.result()
            daily_df = daily_df_result.result()
            updown_limit_df = updown_limit_df_result.result()
            moneyflow_df = moneyflow_df_result.result()

        if len(basic_df) == 0:
            print("远程日线数据为空，退出!")
            return
        if len(adj_df) == 0:
            print("远程复权因子数据为空，退出!")
            return
        if len(daily_df) == 0:
            print("远程日线行情数据为空，退出!")
            return
        if len(updown_limit_df) == 0:
            print("远程涨跌停数据为空，退出!")
            return
        if len(moneyflow_df) == 0:
            print("远程资金流向数据为空，退出!")
            return

        #  5 复权处理+合并数据
        adjust_file_name_df = basic_df
        adjust_file_name_df = adjust_file_name_df[(pd.to_datetime(adjust_file_name_df['trade_date']) >= pd.to_datetime(start_date_str)) & (pd.to_datetime(adjust_file_name_df['trade_date']) <= pd.to_datetime(end_date_str))].reset_index(drop=True)  # 重置索引
        if self.add_adj_comlums_flag:
            adjust_file_name_df = self._add_adj_comlums(adj_df, basic_df)
        # 6 月度调仓处理
        if self.compute_change_position_flag:
            adjust_file_name_df = self._change_position_time(stock_basic_df, adjust_file_name_df, "month", 1,start_date_str, end_date_str)

        if self.add_comlums_shenewn_flag:
            adjust_file_name_df = self.add_wanshen_classify(adjust_file_name_df)
        # 7 保存
        adjust_file_name_df['ts_code'] = adjust_file_name_df['ts_code'].apply(lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")
        adj_df['ts_code'] = adj_df['ts_code'].apply(lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")
        daily_df['ts_code'] = daily_df['ts_code'].apply(lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")
        updown_limit_df['ts_code'] = updown_limit_df['ts_code'].apply(lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")
        moneyflow_df['ts_code'] = moneyflow_df['ts_code'].apply(lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")
        # adjust_file_name_df.to_parquet(adjust_file_name, index=False, engine='pyarrow')
        print(f"调整后的数据已保存至{adjust_file_name}，共{len(adjust_file_name_df)}条记录")

        daily_df.rename(columns={'vol': 'volume'}, inplace=True)
        adj_df.rename(columns={'adj_factor': 'factor'}, inplace=True)
        # 8 单独保存至单股文件夹
        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(self._save_substock_data,self.save_dir_basic,basic_df,0)
            executor.submit(self._save_substock_data,self.save_dir_daily, daily_df,1)
            if not self.add_adj_comlums_flag:
                executor.submit(self._save_substock_data,self.save_dir_adj, adj_df,2)
            executor.submit(self._save_substock_data,self.save_dir_updown_limit, updown_limit_df,3)
            executor.submit(self._save_substock_data,self.save_dir_moneyflow, moneyflow_df,4)


    def download_tushare_A_finance(self, start_date_str, end_date_str):
        file_name_fina = os.path.join(self.save_dir_download,f'download_A_fina_df_{start_date_str}_{end_date_str}.parquet')
        file_name_income = os.path.join(self.save_dir_download,f'download_A_income_df_{start_date_str}_{end_date_str}.parquet')
        file_name_balance = os.path.join(self.save_dir_download,f'download_A_balance_df_{start_date_str}_{end_date_str}.parquet')

        fine_exists = os.path.exists(file_name_fina)
        income_exists = os.path.exists(file_name_income)
        balance_exists = os.path.exists(file_name_balance)
        if fine_exists and income_exists and balance_exists:
            print("[download_tushare_A_finance]以下三个文件均已存在，无需重复处理，程序退出!")
            return

        file_name_fina_prefix = 'download_A_fina_df_'
        file_name_fina_exist, temp1 = self._utils_read_matched_csv_by_prefix(self.save_dir_download,file_name_fina_prefix)
        temp1 = pd.DataFrame()
        if file_name_fina_exist:
            print(f"{file_name_fina_exist}文件已存在，请使用增量下载")
            return

        # 获取全部股票列表
        stock_basic_df = self._get_stock_basic_df()

        # 权重数据清洗（去重+日期格式化）
        all_stocks = stock_basic_df['ts_code'].dropna().unique().tolist()
        if not all_stocks:
            print("无有效成分股数据，程序终止")
            return
        # 3.4 获取财务指标数据
        # 3.4 获取利润数据
        # 3.4 获取资产负债数据
        with ThreadPoolExecutor(max_workers=3) as executor:
            # 提交三个任务到线程池
            executor.submit(self._get_fina_indicator, all_stocks, start_date_str, end_date_str, file_name_fina)
            executor.submit(self._get_income_df, all_stocks, start_date_str, end_date_str, file_name_income)
            executor.submit(self._get_balance_df, all_stocks, start_date_str, end_date_str, file_name_balance)

    # -------↓↓↓↓---------updates_tushare_finance 使用--------↓↓↓↓----------
    def _update_get_finace_df(self, stock_list, type_fun, start_date_str, end_date_str, ):
        new_result_all = []
        new_result_df = pd.DataFrame()
        max_retry = self.MAX_RETRY  # 最大重试次数
        for i, stock_code in enumerate(tqdm(stock_list, desc=f'[_update_get_finace_df] 获取{type_fun}数据'), 1):
            retry_count = 0
            success = False
            tmp = None
            retry_delay = 1  # 每只股票重置初始重试间隔为1秒
            while retry_count < max_retry and not success:
                try:
                    if type_fun == 'fina_indicator':
                        tmp = self.pro.fina_indicator(ts_code=stock_code, start_date=start_date_str, end_date=end_date_str)
                    elif type_fun == 'income':
                        tmp = self.pro.income(ts_code=stock_code, start_date=start_date_str, end_date=end_date_str)
                    elif type_fun == 'balance':
                        tmp = self.pro.balancesheet(ts_code=stock_code, start_date=start_date_str, end_date=end_date_str)
                    else:
                        print("无type_fun,无法识别接口")
                        return new_result_df
                    # print(f"fina_df_正在查询第{i}/{len(stock_list)}只股票：{stock_code}，长度为：{len(tmp)}")
                    if len(tmp) > 0:
                        new_result_all.append(tmp)
                    success = True
                # 只捕获网络相关异常，非网络异常直接跳过重试
                except (NameResolutionError, MaxRetryError, ConnectionError,TimeoutError,ConnectionResetError,requests.exceptions.RequestException) as e:
                    retry_count += 1
                    if retry_count < max_retry:
                        print(f"_update_get_finace_df 获取{stock_code}数据失败（网络错误）：{str(e)[:50]}... 第{retry_count}次重试，等待{retry_delay}秒")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避，间隔翻倍
                    else:
                        print(f"_update_get_finace_df 获取{stock_code}数据失败：{str(e)[:50]}... 已重试{max_retry}次，跳过该股票")
                # 非网络异常：直接终止重试，跳过该股票
                except Exception as e:
                    print(f"_update_get_finace_df 获取财务数据失败（{stock_code}）：{e}，非网络错误，直接跳过")
                    break  # 跳出while重试循环
                time.sleep(0.13)  # 控制请求频率
                
        if new_result_all:
            new_result_df = pd.concat(new_result_all, ignore_index=True)
            if (len(new_result_df) > 0):
                new_result_df = new_result_df.sort_values(by=['ts_code', 'ann_date'])
                # new_result_df['ann_date'] = pd.to_datetime(new_result_df['ann_date'])
                print(f"\n共获取{len(new_result_df)}条数据")
                new_result_df['ts_code'] = new_result_df['ts_code'].apply(lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")
            print(f"fina 新增加股票数量：{len(stock_list)},原始股票+新股票增量数据：{len(new_result_df)}行")
        return new_result_df

    def _get_index_weight_df(self,index_code, start_date_str, end_date_str):
        current_start = pd.to_datetime(start_date_str)
        total_end = pd.to_datetime(end_date_str)
        weight_df = pd.DataFrame()  # 初始化DataFrame（避免后续 concat 报错）
        while current_start <= total_end:
            current_end = current_start + pd.DateOffset(months=3)  # 每3个月一批
            if current_end > total_end:
                current_end = total_end
            curr_start_str = current_start.strftime('%Y%m%d')
            curr_end_str = current_end.strftime('%Y%m%d')
            try:
                temp_df = self.pro.index_weight(index_code=index_code, start_date=curr_start_str,end_date=curr_end_str)
                weight_df = pd.concat([weight_df, temp_df], ignore_index=True)
            except Exception as e:
                print(f"获取指数权重失败（{curr_start_str}-{curr_end_str}）：{e}")
            current_start = current_end + pd.DateOffset(days=1)  # 避免重复日期
        # 权重数据清洗
        index_weight_df = weight_df.drop_duplicates(subset=['trade_date', 'con_code'])  # 去重
        index_weight_df = index_weight_df.sort_values('trade_date')
        index_weight_df['trade_date'] = pd.to_datetime(index_weight_df['trade_date'], format='%Y%m%d')  # 统一日期
        print(f"index_weight_df 获取{len(index_weight_df)}条数据")
        return index_weight_df

    # 查询当前所有正常上市交易的股票列表
    def _get_stock_basic_df(self, ):
        basic_df = pd.DataFrame()  # 初始化DataFrame
        try:
            temp_df = self.pro.stock_basic(exchange='', list_status='L',fields='ts_code,symbol,name,area,industry,list_date')
            basic_df = pd.concat([basic_df, temp_df], ignore_index=True)
        except Exception as e:
            print(f"获取全A股列表失败：{e}")

        # 权重数据清洗
        result_basic_df = basic_df.drop_duplicates(subset=['ts_code'])  # 去重
        print(f"result_basic_df 获取{len(result_basic_df)}条数据")
        return result_basic_df

    def _to_new_csv_and_delete_old(self, file_name, file_name_df, start_date_str, end_date_str,save_dir_download):
        # 提取老文件的日期（复用已有工具函数）
        old_start_str, old_end_str = self._utils_extract_date_from_filename(file_name)
        # 新日期和老日期完全一致 → 无需操作，直接返回
        if start_date_str == old_start_str and end_date_str == old_end_str:
            return

        # 根据老文件名称和日期获取新名称
        new_file_date_name = self._utils_update_file_date_name(file_name, start_date_str, end_date_str)
        new_file_path = os.path.join(save_dir_download, new_file_date_name)

        # 新老文件绝对路径是否重复（同名=路径完全一致）
        old_file_abs = os.path.abspath(file_name)
        new_file_abs = os.path.abspath(new_file_path)
        if old_file_abs == new_file_abs:
            print(f"新文件与原文件同名（路径重复），不执行任何操作！原文件：{old_file_abs}")
            return  # 直接跳出函数，终止后续所有操作
        try:
            # 1. 尝试保存新文件
            file_name_df.to_parquet(new_file_path, index=False, engine='pyarrow')
            # print(f"新文件保存请求已执行：{new_file_path}")
            # 2. 验证保存是否成功（双重保障，避免“保存无异常但文件无效”）
            # 验证1：文件是否存在
            if not os.path.exists(new_file_path):
                raise FileNotFoundError(f"新文件保存后未找到！路径：{new_file_path}")
            # 3. 保存成功且验证通过 → 删除老文件
            self._utils_delete_old_file(save_dir_download,file_name)
            # print(f"新文件保存成功并验证通过！")
        except Exception as e:
            # 4. 保存失败或验证失败 → 不删除老文件，抛出异常/打印错误
            print(f"新文件保存失败或验证未通过！老文件未删除：{file_name},错误原因：{str(e)}")

    # -------↑↑↑↑---------updates_tushare_finance 使用---------↑↑↑↑---------

    # ------↓↓↓↓-----功能函数-------↓↓↓↓----------
    def _utils_extract_date_from_filename(self, file_name):
        """
        从纯文件名中提取起始日期和终止日期（格式：xxx_起始日期_终止日期.parquet）
        :param file_name: 纯文件名（如 "all_returns_adjusted_20220101_20251114.parquet"）
        :return: start_date, end_date 返回字符串格式 "YYYYMMDD"
        """
        # 步骤1：去掉文件后缀（如 .csv、.txt 等，不区分大小写）
        name_without_ext = file_name.rsplit(".", 1)[0]  # 按最后一个 "." 分割，获取文件名主体
        # 步骤2：按下划线 "_" 分割文件名主体
        name_parts = name_without_ext.split("_")
        # 步骤3：验证格式（必须至少包含 4 部分：前缀_起始日期_终止日期 → 分割后长度 ≥4）
        if len(name_parts) < 3:
            raise ValueError(f"文件名格式错误：{file_name} → 不符合 'xxx_起始日期_终止日期.parquet' 格式")
        # 步骤4：提取日期部分（倒数第三部分=起始日期，倒数第二部分=终止日期）
        start_date_str = name_parts[-2]
        end_date_str = name_parts[-1]

        # 步骤5：验证日期格式（8 位数字 + 合法日期）
        def validate_date(date_str: str) -> bool:
            if not (len(date_str) == 8 and date_str.isdigit()):
                return False
            try:
                datetime.strptime(date_str, "%Y%m%d")
                return True
            except ValueError:
                return False

        if not validate_date(start_date_str):
            raise ValueError(f"起始日期格式错误：{start_date_str} → 必须是 8 位合法日期（如 20220101）")
        if not validate_date(end_date_str):
            raise ValueError(f"终止日期格式错误：{end_date_str} → 必须是 8 位合法日期（如 20251114）")
        return start_date_str, end_date_str

    # 传入文件名前缀，读取文件，多个文件报错
    def _utils_read_matched_csv_by_prefix(self,save_dir_download, prefix):
        """
        仅传入文件前缀（如 "all_returns_adjusted_"），自动在当前文件夹查找并读取格式为「前缀_起始日期_终止日期.csv」的CSV文件
        若存在多个符合条件的文件，直接报错（避免误读）
        :param prefix: 文件前缀（必须包含末尾的下划线，如 "all_returns_adjusted_"）
        :return: 读取后的 DataFrame
        """
        suffix = ".parquet"
        # 步骤1：获取当前文件夹下所有文件
        all_files = [f for f in os.listdir(save_dir_download) if os.path.isfile(os.path.join(save_dir_download, f))]
        # 步骤2：筛选「前缀匹配+格式合法」的文件
        matched_files = []  # 存储所有符合条件的文件（用于检测重复）
        valid_date_ranges = []  # 存储对应文件的日期范围
        for file_name in all_files:
            # 条件1：以指定前缀开头，且以 .parquet 结尾（不区分大小写）
            if not (file_name.startswith(prefix) and file_name.lower().endswith(suffix)):
                continue
            # 条件2：格式拆分后符合「前缀_起始日期_终止日期.parquet」
            name_core = file_name[len(prefix):-len(suffix)]  # 去掉前缀和 .parquet 后缀
            date_parts = name_core.split("_")
            # 条件3：中间部分必须是两个日期（分割后长度=2）
            if len(date_parts) != 2:
                continue
            start_date_str, end_date_str = date_parts

            # 条件4：日期必须是 8 位数字且合法
            def is_valid_date(d_str: str) -> bool:
                if len(d_str) != 8 or not d_str.isdigit():
                    return False
                try:
                    datetime.strptime(d_str, "%Y%m%d")
                    return True
                except ValueError:
                    return False

            if is_valid_date(start_date_str) and is_valid_date(end_date_str):
                # 符合所有条件，加入列表
                matched_files.append(file_name)
                valid_date_ranges.append((start_date_str, end_date_str))
        # 步骤3：处理匹配结果（核心：检测多个匹配文件）
        if len(matched_files) == 0:
            # raise FileNotFoundError(f"当前文件夹未找到符合格式的文件！\n")
            print("当前文件夹未找到符合格式的文件！")
            return None, pd.DataFrame()
        elif len(matched_files) >= 2:
            # 存在多个符合条件的文件，直接报错
            error_msg = "当前文件夹存在多个符合条件的匹配文件，无法确定读取哪一个！\n"
            for i, (file, date_range) in enumerate(zip(matched_files, valid_date_ranges), 1):
                error_msg += f"{i}. 文件名：{file}\n"
            error_msg += "\n请删除多余文件，或修改前缀以区分不同文件后重试"
            raise ValueError(error_msg)
        else:
            # 仅找到一个符合条件的文件，读取并返回
            target_file = matched_files[0]
            target_date_range = valid_date_ranges[0]
            file_path = os.path.join(save_dir_download, target_file)
            print(f"找到唯一匹配文件：{target_file}")
            try:
                df = pd.read_parquet(file_path, engine='pyarrow')
                return target_file, df
            except Exception as e:
                print(f"文件读取失败：{str(e)}（可能是文件损坏、编码问题或权限不足）")
                return None, pd.DataFrame()

    def _utils_update_file_date_name(self, original_file_name: str, new_start_date: str, new_end_date: str) -> str:
        """
        :param original_file_name: 原始文件名（含后缀，如"all_returns_adjusted_20220101_20251114.csv"）
        :param new_start_date: 新的起始日期（字符串，格式：YYYYMMDD，如"20220101"）
        :param new_end_date: 新的结束日期（字符串，格式：YYYYMMDD，如"20251130"）
        :return: 更新后的新文件名
        """

        # ---------------------- 1. 严格验证输入参数格式 ----------------------
        # 验证新日期格式（8位数字，YYYYMMDD）
        def validate_date(date_str: str, date_desc: str) -> None:
            if not (isinstance(date_str, str) and len(date_str) == 8 and date_str.isdigit()):
                raise ValueError(f"{date_desc}格式错误！请输入YYYYMMDD格式的8位数字字符串，当前输入：{date_str}")

        validate_date(new_start_date, "新起始日期")
        validate_date(new_end_date, "新结束日期")
        # 验证原始文件名（必须含后缀）
        if "." not in original_file_name:
            raise ValueError(
                f"原始文件名缺少后缀！当前输入：{original_file_name}（需符合「任意前缀_原起始日期_原结束日期.后缀」格式）")
        # ---------------------- 2. 分割文件名（保留前缀和后缀） ----------------------
        # 步骤1：分割文件名主体（不含后缀）和后缀（按最后一个"."分割，避免前缀含"."的异常）
        name_without_ext, file_ext = original_file_name.rsplit(".", 1)  # 例：("a_b_c_20230101_20251114", "csv")
        # 步骤2：分割文件名主体（按下划线，兼容前缀含多个下划线）
        name_parts = name_without_ext.split("_")  # 例：["a", "b", "c", "20230101", "20251114"]
        # 验证原始文件名格式（必须至少包含「前缀+2个原日期」，分割后至少3部分）
        if len(name_parts) < 3:
            raise ValueError(f"原始文件名格式错误！需符合「任意前缀_原起始日期_原结束日期.后缀」（日期为8位数字），"
                             f"当前输入：{original_file_name}（分割后仅{len(name_parts)}部分）")
        # ---------------------- 3. 强制替换日期（核心逻辑） ----------------------
        # 提取前缀（除最后两个原日期外的所有部分）+ 拼接新日期
        prefix_parts = name_parts[:-2]  # 前缀部分（如["a", "b", "c"]）
        new_name_parts = prefix_parts + [new_start_date, new_end_date]  # 前缀 + 新起始日期 + 新结束日期
        # ---------------------- 4. 构造新文件名 ----------------------
        new_name_without_ext = "_".join(new_name_parts)  # 拼接主体（例："a_b_c_20220101_20251130"）
        new_file_name = f"{new_name_without_ext}.{file_ext}"  # 加上后缀（例："a_b_c_20220101_20251130.csv"）
        # 日志输出（便于调试）
        print(f"原始文件名：{original_file_name},新文件名 ：{new_file_name}")
        return new_file_name

    def _utils_delete_old_file(self,save_dir_download, old_file_name: str):
        """
        安全删除老文件：自动拼接 self.save_dir_download 和文件名 → 验证路径 → 删除
        :param old_file_name: 仅文件名（不含路径，如 "all_returns_adjusted_20220101_20251114.csv"）
        """
        # 1. 拼接完整路径（目录 + 文件名）
        old_file_path = os.path.join(save_dir_download, old_file_name)
        # print(f"准备删除老文件：{old_file_path}")
        # 2. 安全校验
        # 验证目录是否存在（避免目录被误删导致路径无效）
        if not os.path.exists(save_dir_download):
            print(f"警告：老文件保存目录不存在 → {save_dir_download}，无需删除")
            return
        # 验证文件是否存在
        if not os.path.exists(old_file_path):
            print(f"警告：老文件不存在 → {old_file_path}，无需删除")
            return
        # 验证路径是文件（避免误删目录）
        if os.path.isdir(old_file_path):
            print(f"错误：无法删除，路径是目录 → {old_file_path}")
            return
        # 3. 尝试删除（捕获异常）
        try:
            os.remove(old_file_path)
            # print(f"成功删除老文件 → {old_file_path}")
        except OSError as e:
            # 处理权限不足、文件被占用等常见错误
            print(f"错误：删除老文件失败 → {old_file_path}，原因：{str(e)}")

    def _utils_get_missing_date_ranges(self, old_start_str: str, old_end_str: str, new_start_str: str,new_end_str: str) -> List[List[str]]:
        """
        计算未获得的数据区间（仅处理有重叠场景，无重叠返回空列表）
        重叠定义：最小=首尾值相同，最大=相互包含
        :param old_start_str: 老数据起始（YYYYMMDD）
        :param old_end_str: 老数据结束（YYYYMMDD）
        :param new_start_str: 新目标起始（YYYYMMDD）
        :param new_end_str: 新目标结束（YYYYMMDD）
        :return: 未获得的数据区间列表，格式 [[begin1, end1], [begin2, end2]]（空列表=无缺失/无重叠）
        """

        # ---------------- 1. 日期校验+转换（避免类型错误） ----------------
        def validate_date(date_str: str, date_desc: str) -> Optional[pd.Timestamp]:
            if not (isinstance(date_str, str) and len(date_str) == 8 and date_str.isdigit()):
                print(f"日期格式错误：{date_desc}={date_str}（需YYYYMMDD 8位数字）")
                return None
            try:
                return pd.to_datetime(date_str, format='%Y%m%d')
            except ValueError:
                print(f"日期无效：{date_desc}={date_str}（不存在该日期）")
                return None

        # 转换并校验4个日期
        O_s = validate_date(old_start_str, "老数据起始")
        O_e = validate_date(old_end_str, "老数据结束")
        N_s = validate_date(new_start_str, "新目标起始")
        N_e = validate_date(new_end_str, "新目标结束")

        # 校验失败/范围无效，返回空列表
        if any(date is None for date in [O_s, O_e, N_s, N_e]) or O_s > O_e or N_s > N_e:
            return []
        # 校验：老数据/新数据自身的时间范围合法（起始≤结束）
        if O_s > O_e:
            print(f"老数据时间范围无效：起始日期{old_start_str} > 结束日期{old_end_str}")
            return []
        if N_s > N_e:
            print(f"新数据时间范围无效：起始日期{new_start_str} > 结束日期{new_end_str}")
            return []
        # ---------------- 2. 核心前提：判断是否有重叠（满足最小重叠要求） ----------------
        # 有重叠条件：新起始≤老结束 AND 新结束≥老起始（覆盖首尾相同、部分重叠、完全包含）
        has_intersection = (N_s <= O_e) and (N_e >= O_s)
        if not has_intersection:
            print(
                f"新目标范围[{new_start_str}-{new_end_str}]与老数据[{old_start_str}-{old_end_str}]无重叠（首尾也不同），不计算缺失区间")
            return []
        # ---------------- 3. 仅处理有重叠的场景，计算缺失区间 ----------------
        missing_ranges = []

        # 场景1：新目标完全在老数据内部（老包含新，最大重叠）→ 无缺失
        if N_s >= O_s and N_e <= O_e:
            print(
                f"新目标范围[{new_start_str}-{new_end_str}]完全在老数据[{old_start_str}-{old_end_str}]内（老包含新），无获得数据")
            return missing_ranges

        # 场景2：新目标部分重叠老数据前面（最小重叠=新结束=老起始）→ 缺失：[N_s, O_s-1]
        elif N_s < O_s and N_e <= O_e:
            missing_start = new_start_str
            missing_end = (O_s - pd.DateOffset(days=0)).strftime('%Y%m%d')
            missing_ranges.append([missing_start, missing_end])
            print(
                f"获得数据区间：新目标部分重叠老数据前面（重叠={N_e.strftime('%Y%m%d')}={O_s.strftime('%Y%m%d')}）→ [{missing_start}-{missing_end}]")

        # 场景3：新目标部分重叠老数据后面（最小重叠=新起始=老结束）→ 缺失：[O_e+1, N_e]
        elif N_s >= O_s and N_e > O_e:
            missing_start = (O_e + pd.DateOffset(days=0)).strftime('%Y%m%d')
            missing_end = new_end_str
            missing_ranges.append([missing_start, missing_end])
            print(
                f"获得数据区间：新目标部分重叠老数据后面（重叠={N_s.strftime('%Y%m%d')}={O_e.strftime('%Y%m%d')}）→ [{missing_start}-{missing_end}]")

        # 场景4：新目标完全包含老数据（新包含老，最大重叠）→ 缺失：[N_s, O_s-1] + [O_e+1, N_e]
        elif N_s < O_s and N_e > O_e:
            # 前面缺失区间
            missing_front_start = new_start_str
            missing_front_end = (O_s - pd.DateOffset(days=0)).strftime('%Y%m%d')
            # 后面缺失区间
            missing_back_start = (O_e + pd.DateOffset(days=0)).strftime('%Y%m%d')
            missing_back_end = new_end_str
            missing_ranges.append([missing_front_start, missing_front_end])
            missing_ranges.append([missing_back_start, missing_back_end])
            print(
                f"获得数据区间：新目标完全包含老数据（新包含老）→ [{missing_front_start}-{missing_front_end}] 和 [{missing_back_start}-{missing_back_end}]")

        # ---------------- 4. 结果整理（去重+排序） ----------------
        # 转换为元组进行去重，再转回列表
        unique_ranges = list(set(tuple(item) for item in missing_ranges))
        # 按起始日期排序
        unique_ranges.sort(key=lambda x: x[0])
        # 转回列表格式
        return [list(item) for item in unique_ranges]

    def _utils_convert_to_json(self, df_weight: pd.DataFrame) -> Dict[str, List[str]]:
        """
        将 DataFrame 转换为每日成分股 JSON 格式
        Args:
            df: index_weight 返回的 DataFrame
        Returns:
            {date: [stock1, stock2, ...]}
        """
        if df_weight.empty:
            return {}
        df_weight['con_code'] = df_weight['con_code'].str.split('.', expand=True).pipe(lambda df: df[1].str.upper() + df[0]).where(df_weight['con_code'].str.split('.').str.len() == 2, df_weight['con_code'])
        # 按日期分组
        result = {}
        for trade_date, group in df_weight.groupby('trade_date'):
            stocks = sorted(group['con_code'].tolist())
            result[trade_date] = stocks
        return result

    # -----↑↑↑↑------功能函数------↑↑↑↑------

    def download_baostock_basic_mins(self, start_date_str, end_date_str):
        # 2. 基础参数配置
        obtain_date_str = (pd.to_datetime(start_date_str, format='%Y%m%d') - pd.DateOffset(months=self.more_month)).strftime('%Y%m%d')  # '20220901'

        # 文件名定义
        file_name_adj = os.path.join(self.save_dir_download, f'download_adj_df_{start_date_str}_{end_date_str}.parquet')
        file_name_mins_basic = os.path.join(self.save_dir_download,f'download_baostock_30mins_basic_df_{start_date_str}_{end_date_str}.parquet')
        file_name_mins_adjusted = os.path.join(self.save_dir_download,f'all_returns_adjusted_30mins_{start_date_str}_{end_date_str}.parquet')

        file_name_adj_exists = os.path.exists(file_name_adj)
        file_name_mins_basic_exists = os.path.exists(file_name_mins_basic)
        file_name_mins_adjusted_exists = os.path.exists(file_name_mins_adjusted)
        if file_name_adj_exists and file_name_mins_basic_exists and file_name_mins_adjusted_exists:
            print("[download_baostock_basic_mins]以下三个文件均已存在，无需重复处理，程序退出!")
            return

        file_name_mins_basic_prefix = 'all_returns_adjusted_30mins_'
        file_name_mins_basic_exist, temp1 = self._utils_read_matched_csv_by_prefix(self.save_dir_download,file_name_mins_basic_prefix)
        temp1 = pd.DataFrame()
        if file_name_mins_basic_exist:
            print(f"{file_name_mins_basic_exist}文件已存在，请使用增量下载")
            return

        # 3. 分批获取指数权重
        index_weight_df = self._get_index_weight_df('000852.SH',obtain_date_str, end_date_str)
        if len(index_weight_df) == 0:
            print(f"[download_baostock_basic_mins] index_weight_df 获取失败，退出!")
            return

        # 4 获取所有需查询的股票
        all_stocks = index_weight_df['con_code'].dropna().unique().tolist()
        if len(all_stocks) == 0:
            print("无有效成分股数据，程序终止")
            return

        #  5 获取日线行情
        def get_mins_basic_df():
            if os.path.exists(file_name_mins_basic):
                file_name_mins_basic_df = pd.read_parquet(file_name_mins_basic, engine='pyarrow')
                if len(file_name_mins_basic_df) == 0:
                    print("本地日线数据为空，退出!")
                    return
            else:
                file_name_mins_basic_df = self._get_baostock_daily_basic_mins(all_stocks, obtain_date_str, end_date_str,frequency="60")
                if len(file_name_mins_basic_df) == 0:
                    print("远程日线数据为空，退出!")
                    return
                file_name_mins_basic_df.to_parquet(file_name_mins_basic, index=False, engine='pyarrow')
                return file_name_mins_basic_df

        #  6 获取复权因子
        def get_mins_adj_df():
            if os.path.exists(file_name_adj):
                file_name_adj_df = pd.read_parquet(file_name_adj, engine='pyarrow')
                if len(file_name_adj_df) == 0:
                    print("本地复权因子数据为空，退出!")
                    return
            else:
                file_name_adj_df = self._get_adj_factor(all_stocks, obtain_date_str, end_date_str)
                if len(file_name_adj_df) == 0:
                    print("远程复权因子数据为空，退出!")
                    return
                # file_name_adj_df.to_parquet(file_name_adj, index=False, engine='pyarrow')
            return file_name_adj_df

        with ThreadPoolExecutor(max_workers=2) as executor:
            # 提交任务
            file_name_mins_basic_df = executor.submit(get_mins_basic_df)
            file_name_adj_df = executor.submit(get_mins_adj_df)
            # 获取结果
            file_name_mins_basic_df = file_name_mins_basic_df.result()
            file_name_adj_df = file_name_adj_df.result()

        #  7 复权处理+合并数据
        file_name_mins_adjusted_df = file_name_mins_basic_df
        if self.add_adj_comlums_flag:
            file_name_mins_adjusted_df = self._add_adj_comlums(file_name_adj_df, file_name_mins_basic_df)

        # 9 增加月度调仓flag
        if self.compute_change_position_flag:
            file_name_mins_adjusted_df = self._change_position_time(index_weight_df, file_name_mins_adjusted_df,"month", 1, start_date_str, end_date_str)

        if self.add_comlums_shenewn_flag:
            file_name_mins_adjusted_df = self.add_wanshen_classify(file_name_mins_adjusted_df)

        # 8 保存 adjusted 数据
        file_name_mins_adjusted_df['ts_code'] = file_name_mins_adjusted_df['ts_code'].apply(lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")
        file_name_mins_adjusted_df.to_parquet(file_name_mins_adjusted, index=False, engine='pyarrow')
        print(f"调整后的数据已保存至{file_name_mins_adjusted}，共{len(file_name_mins_adjusted_df)}条记录")

        # 10 按股票代码分组，批量保存
        self._save_substock_data(self.save_dir_basic_60mins,file_name_mins_adjusted_df,0)

    def download_tushare_shenwan_classify(self):
        self.add_wanshen_classify(pd.DataFrame())
        file_name_shenwan_L1 = os.path.join(self.save_dir_shenwan, f'download_shenwan_classify_df_L1.csv')
        file_name_shenwan_L2 = os.path.join(self.save_dir_shenwan, f'download_shenwan_classify_df_L2.csv')
        file_name_shenwan_L3 = os.path.join(self.save_dir_shenwan, f'download_shenwan_classify_df_L3.csv')
        # 1. 基础参数配置
        fine_exists_L1 = os.path.exists(file_name_shenwan_L1)
        fine_exists_L2 = os.path.exists(file_name_shenwan_L2)
        fine_exists_L3 = os.path.exists(file_name_shenwan_L3)
        if fine_exists_L1 and fine_exists_L2 and fine_exists_L3:
            print("[download_tushare_shenwan_classify] 三个文件已存在，无需重复处理，程序退出!")
            return

        #  3 获取行业分类
        file_name_shenwan_L1_df = self.pro.index_classify(level='L1', src='SW2021')
        file_name_shenwan_L2_df = self.pro.index_classify(level='L2', src='SW2021')
        file_name_shenwan_L3_df = self.pro.index_classify(level='L3', src='SW2021')

        # 4 保存
        file_name_shenwan_L1_df.to_csv(file_name_shenwan_L1, index=False, encoding='utf-8-sig')
        print(f"数据保存成功：{file_name_shenwan_L1}")
        file_name_shenwan_L2_df.to_csv(file_name_shenwan_L2, index=False, encoding='utf-8-sig')
        print(f"数据保存成功：{file_name_shenwan_L2_df}")
        file_name_shenwan_L3_df.to_csv(file_name_shenwan_L3, index=False, encoding='utf-8-sig')
        print(f"数据保存成功：{file_name_shenwan_L3_df}")
        

    def add_wanshen_classify(self, file_name_df):
        # 1. 定义文件路径
        shenwan_industry_file = os.path.join(self.save_dir_shenwan, 'download_shenwan_stock_df_L1_L2_L3.csv')
        shenwan_industry_df = pd.DataFrame()  # 存储完整的行业分类数据
        missing_stocks = []  # 需要补充分类的股票列表
        # 2. 提取原始数据中所有有效股票
        if len(file_name_df)>0:
            raw_stocks = file_name_df['ts_code'].dropna().drop_duplicates().tolist()
        else:
            file_name_basic_df = self._get_stock_basic_df()
            raw_stocks = file_name_basic_df['ts_code'].dropna().drop_duplicates().tolist()
        try:
            # 3. 读取已有分类文件（若存在），计算缺失股票
            if os.path.exists(shenwan_industry_file):
                # 读取并清洗已有分类数据（去重、去空，避免无效数据干扰）
                shenwan_industry_df = pd.read_csv(shenwan_industry_file, encoding='utf-8-sig')
                shenwan_industry_df = shenwan_industry_df.dropna(subset=['ts_code']).drop_duplicates('ts_code')
                existing_stocks = shenwan_industry_df['ts_code'].tolist()
                # 计算「原始数据有但分类文件没有」的缺失股票
                missing_stocks = list(set(raw_stocks) - set(existing_stocks))
                if missing_stocks:
                    print(f" {len(missing_stocks)} 只股票未包含在分类文件中，将补充下载")
            else:
                # 分类文件不存在：所有原始股票均为缺失股票，需全量下载
                missing_stocks = raw_stocks

            # 4. 下载缺失股票的分类并补充到文件
            if missing_stocks:
                # 调用接口获取缺失股票的分类（假设返回含 ts_code 和行业列的DataFrame）
                add_industry_df = self._get_index_member_all(missing_stocks)
                add_industry_df = add_industry_df.dropna(subset=['ts_code']).drop_duplicates('ts_code')
                if len(add_industry_df) == 0:
                    print("未获取到有效补充分类数据，跳过更新文件")
                else:
                    # 合并已有数据和补充数据（纵向拼接，确保无重复）
                    shenwan_industry_df = pd.concat([shenwan_industry_df, add_industry_df], axis=0,
                                                    ignore_index=True).drop_duplicates('ts_code')  # 最终去重，保证 ts_code 唯一
                    # 覆盖保存，确保文件始终是完整最新的
                    shenwan_industry_df.to_csv(shenwan_industry_file, index=False, encoding='utf-8-sig')
        except Exception as e:
            print(f"补充申万行业分类时发生错误：{str(e)}")
            # 异常时仍尝试读取已有文件（若存在），避免完全无法返回分类
            if os.path.exists(shenwan_industry_file):
                shenwan_industry_df = pd.read_csv(shenwan_industry_file, encoding='utf-8-sig').dropna(subset=['ts_code']).drop_duplicates('ts_code')
            else:
                print("无任何可用的行业分类数据，返回原数据")
                return file_name_df
        # 5. 合并分类到原始数据框（左连接，保留原始所有记录，缺失分类为NaN）
        if len(file_name_df>0):
            file_name_df = file_name_df.merge(shenwan_industry_df, on='ts_code', how='left')
        else:
            file_name_df = shenwan_industry_df
        return file_name_df

    def _get_index_member_all(self, stock_list):
        all_industry_dfs = []
        industry_df = pd.DataFrame()
        max_retry = self.MAX_RETRY  # 最大重试次数
        for stock in tqdm(stock_list, desc="万申分类"):
            retry_count = 0
            success = False
            batch_df = None
            retry_delay = 1  # 每只股票重置初始重试间隔为1秒
            while retry_count < max_retry and not success:
                try:
                    batch_df = self.pro.index_member_all(ts_code=stock)
                    success = True
                    all_industry_dfs.append(batch_df)
                # 只捕获网络相关异常，非网络异常直接跳过重试
                except (NameResolutionError, MaxRetryError, ConnectionError,TimeoutError,ConnectionResetError,requests.exceptions.RequestException) as e:
                    retry_count += 1
                    if retry_count < max_retry:
                        print(f"_get_index_member_all 获取{stock}数据失败（网络错误）：{str(e)[:50]}... 第{retry_count}次重试，等待{retry_delay}秒")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避，间隔翻倍
                    else:
                        print(f"_get_index_member_all 获取{stock}数据失败：{str(e)[:50]}... 已重试{max_retry}次，跳过该股票")
                except Exception as e:
                    retry_count += 1
                    print(f"_get_index_member_all 获取 {stock} 分类失败,非网络错误，直接跳过")
                    break
            time.sleep(0.13)
        if not all_industry_dfs:
            print("所有批次查询失败，返回原表")
            return pd.DataFrame()
        # 合并+去重（保留最新行业记录）
        industry_df = pd.concat(all_industry_dfs, ignore_index=True)
        industry_df = industry_df.drop_duplicates('ts_code', keep='first')
        industry_df = industry_df[['ts_code', 'l1_name', 'l2_name', 'l3_name']]
        return industry_df


    def _save_substock_data(self, save_dir, file_name_df, pbar_position):
        # 1. 清空并重建目录
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        # 2. 提前分组
        grouped = file_name_df.groupby('ts_code')
        total_stocks = len(grouped)
        pbar = tqdm(total=total_stocks, desc=f'{os.path.basename(save_dir)} 按股票保存数据',  # 只显示目录名，更简洁
                    position=pbar_position,  # 关键：每个任务分配唯一位置（0/1/2）
                    leave=True,  # 任务完成后保留进度条
                    dynamic_ncols=True, mininterval=0.1)
        # 3. 多线程保存（合并单股逻辑）
        with ThreadPoolExecutor(max_workers=64) as inner_executor:
            futures = []
            for stock_code, group_data in grouped:
                def task(sc=stock_code, gd=group_data):  # 绑定变量，避免闭包延迟
                    try:
                        filepath = os.path.join(save_dir, f"{sc}.parquet")
                        # 兼容低版本：移除 write_metadata_file
                        gd.to_parquet(filepath, index=False, engine='pyarrow', compression='snappy')
                    except Exception as e:
                        # 错误信息换行+指定位置，避免覆盖进度条
                        print(f"\n[{sc}] 保存失败：{str(e)}")
                    finally:
                        pbar.update(1)  # 独立更新各自的进度条
                futures.append(inner_executor.submit(task))
            # 等待所有任务完成
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"\n[任务异常]：{str(e)}")
                    pbar.update(1)
        pbar.close()
        print(f"\n{os.path.basename(save_dir)} 所有股票数据保存完成")


    def updates_baostock_basic_mins(self, start_date_str, end_date_str):
        # 2. 基础参数配置
        file_name_30mins_basic_df_prefix = 'download_baostock_30mins_basic_df_'
        file_name_adj_df_prefix = 'download_adj_df_'
        file_name_30mins_adjusted_df_prefix = 'all_returns_adjusted_30mins_'

        file_name_30mins_basic, file_name_30mins_basic_df = self._utils_read_matched_csv_by_prefix(self.save_dir_download,file_name_30mins_basic_df_prefix)
        if len(file_name_30mins_basic_df) == 0:
            print(f"未读取到文件：{file_name_30mins_basic_df_prefix},请先下载原始数据")
            return
        file_name_adj, file_name_adj_df = self._utils_read_matched_csv_by_prefix(self.save_dir_download,file_name_adj_df_prefix)
        if len(file_name_adj_df) == 0:
            print(f"未读取到文件：{file_name_adj_df_prefix},请先下载原始数据")
            return
        file_name_30mins_adjusted, file_name_30mins_adjusted_df = self._utils_read_matched_csv_by_prefix(self.save_dir_download,file_name_30mins_adjusted_df_prefix)
        if len(file_name_30mins_adjusted_df) == 0:
            print(f"未读取到文件：{file_name_30mins_adjusted_df_prefix},请先下载原始数据")
            return
        file_name_30mins_adjusted_df = pd.DataFrame()  # 此文件为basic_df和adj_df 合并而来，设置为空防止占内存，后期赋值保存

        old_start_str, old_end_str = self._utils_extract_date_from_filename(file_name_30mins_adjusted)
        if file_name_30mins_adjusted:
            if (end_date_str == old_start_str and start_date_str < end_date_str) or (start_date_str == old_end_str and end_date_str > start_date_str):
                print("下载日期正确")
            else:
                print("[ updates_baostock_basic_mins ]下载日期设置错误")
                return
        missing_start_end_list = self._utils_get_missing_date_ranges(old_start_str, old_end_str, start_date_str,end_date_str)
        begin_all_str = min(old_start_str, old_end_str, start_date_str, end_date_str)  # 所有日期首位相连或者重叠后的最边际日期
        end_all_str = max(old_start_str, old_end_str, start_date_str, end_date_str)
        obtain_begin_all_str = (pd.to_datetime(begin_all_str, format='%Y%m%d') - pd.DateOffset(months=self.more_month)).strftime('%Y%m%d')  # '20220901'

        # # 0 分批获取指数权重
        index_weight_df = self._get_index_weight_df('000852.SH',begin_all_str, end_all_str)
        if len(index_weight_df) == 0:
            print(f"[updates_tushare_basic] index_weight_df 获取失败，退出!")
            return
        #
        # 2 获取所有需查询的股票
        all_stocks = index_weight_df['con_code'].dropna().unique().tolist()
        if len(all_stocks) == 0:
            print("无有效成分股数据，程序终止")
            return
        old_stock_list = file_name_30mins_basic_df['ts_code'].dropna().unique().tolist()  # 老文件股列表
        all_stocks = index_weight_df['con_code'].dropna().unique().tolist()  # 新日期内股票列表
        new_stock_list = list(set(all_stocks) - set(old_stock_list))

        # 1、从未出现过的股票，日期最小取到最大
        if len(new_stock_list) > 0:
            print("-----获取新股票所有日期数据-----：")
            with ThreadPoolExecutor(max_workers=2) as executor:
                new_30mins_basic_df = executor.submit(self._get_baostock_daily_basic_mins, new_stock_list, obtain_begin_all_str,end_all_str, frequency="60")
                new_adj_df = executor.submit(self._get_adj_factor, new_stock_list, begin_all_str, end_all_str)
                new_30mins_basic_df = new_30mins_basic_df.result()
                new_adj_df = new_adj_df.result()
            # 1.1获取基础日线行情
            new_30mins_basic_df['ts_code'] = new_30mins_basic_df['ts_code'].apply(lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")  # 新数据转换，保持与老文件同格式
            if len(new_30mins_basic_df) == 0:
                print("远程日线数据为空，退出!")
                return
            file_name_30mins_basic_df = pd.concat(
                [file_name_30mins_basic_df, new_30mins_basic_df],
                ignore_index=True,  # 重置索引（重要，避免索引冲突）
                axis=0)  # 纵向追加（行级追加）
            # 1.2 获取复权因子
            # new_adj_df = self._get_adj_factor(new_stock_list, begin_all_str, end_all_str)
            file_name_adj_df = pd.concat(
                [file_name_adj_df, new_adj_df],  # 先加原始数据，后加新数据
                ignore_index=True,  # 重置索引（重要，避免索引冲突）
                axis=0)  # 纵向追加（行级追加）

        # 2、对于老股票，补充缺失的时间段数据
        print("-----获取旧股票新日期数据-----：")
        for missing_start_str, missing_end_str in missing_start_end_list:
            obtain_missing_date_str = (pd.to_datetime(missing_start_str, format='%Y%m%d') - pd.DateOffset(months=self.more_month)).strftime('%Y%m%d')  # '20220901'
            with ThreadPoolExecutor(max_workers=2) as executor:
                new_30mins_basic_df = executor.submit(self._get_baostock_daily_basic_mins, old_stock_list,obtain_missing_date_str, missing_end_str, frequency="60")
                new_adj_df = executor.submit(self._get_adj_factor, old_stock_list, missing_start_str, missing_end_str)
                new_30mins_basic_df = new_30mins_basic_df.result()
                new_adj_df = new_adj_df.result()
            # 2.1获取基础日线行情
            new_30mins_basic_df['ts_code'] = new_30mins_basic_df['ts_code'].apply(lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")  # 新数据转换，保持与老文件同格式
            file_name_30mins_basic_df = pd.concat([file_name_30mins_basic_df, new_30mins_basic_df], ignore_index=True,axis=0)  # 纵向追加（行级追加）
            # 2.2 获取复权因子
            file_name_adj_df = pd.concat([file_name_adj_df, new_adj_df], ignore_index=True, axis=0)  # 纵向追加（行级追加）

        # # 3 排序并去重
        file_name_30mins_basic_df = file_name_30mins_basic_df.sort_values(by=['ts_code', 'trade_date']).drop_duplicates(keep="first", ignore_index=True)  # 保留第一次出现的重复行（可改为"last"保留最后一次）# 去重后重置索引
        file_name_adj_df = file_name_adj_df.sort_values(by=['ts_code', 'trade_date']).drop_duplicates(keep="first",ignore_index=True)  # 保留第一次出现的重复行（可改为"last"保留最后一次）# 去重后重置索引

        # 4 保存并删除
        self._to_new_csv_and_delete_old(file_name_30mins_basic, file_name_30mins_basic_df, begin_all_str, end_all_str,self.save_dir_download )
        self._to_new_csv_and_delete_old(file_name_adj, file_name_adj_df, begin_all_str, end_all_str,self.save_dir_download)

        file_name_30mins_basic_df['ts_code'] = file_name_30mins_basic_df['ts_code'].apply(lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")
        file_name_adj_df['ts_code'] = file_name_adj_df['ts_code'].apply(lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")

        file_name_30mins_adjusted_df = file_name_30mins_basic_df
        #  5 复权处理+合并数据
        if self.add_adj_comlums_flag:
            file_name_30mins_adjusted_df = self._add_adj_comlums(file_name_adj_df, file_name_30mins_basic_df)

        # 6 月度调仓处理,增加 列['adjusted_month', 'is_current_stock'] 到file_name_adjusted_df表
        if self.compute_change_position_flag:
            file_name_30mins_adjusted_df = self._change_position_time(index_weight_df, file_name_30mins_adjusted_df,"month", 1, start_date_str, end_date_str)

        file_name_30mins_adjusted_df['time'] = pd.to_datetime(file_name_30mins_adjusted_df['time'].astype(str).str[:14],format='%Y%m%d%H%M%S', errors='coerce')
        # 7 保存并删除
        self._to_new_csv_and_delete_old(file_name_30mins_adjusted, file_name_30mins_adjusted_df, begin_all_str,end_all_str,self.save_dir_download)
        # 8 单独保存至单股文件夹
        self._save_substock_data(self.save_dir_basic_60mins,file_name_30mins_adjusted_df,0)


    def _get_baostock_daily_basic_mins_old(self, all_stocks, start_date_str, end_date_str, frequency="60"):
        lg = bs.login()
        if lg.error_code != '0':
            print(f"BaoStock登录失败：{lg.error_msg}") 
        # 日期转换(baostock 使用 %Y-%m-%d， Tushare使用 %Y%m%d)
        start_date_str = datetime.strptime(start_date_str, "%Y%m%d").strftime("%Y-%m-%d")
        end_date_str = datetime.strptime(end_date_str, "%Y%m%d").strftime("%Y-%m-%d")
        # 转换股票代码格式：适配baostock（如 600000.SH → sh.600000）
        all_stocks=[f"{code.split('.')[1].lower()}.{code.split('.')[0]}" if len(code.split('.'))==2 else code for code in all_stocks]
        
        basic_mins_all = []
        basic_mins_df = pd.DataFrame()
        max_retry = 5#self.MAX_RETRY  # 从实例属性获取最大重试次数
        retry_delay_base = 1  # 初始重试间隔为1秒
        
        for i, stock_code in enumerate(tqdm(all_stocks, desc=f'获取{frequency}分钟行情'), 1):
            retry_count = 0
            success = False
            retry_delay = retry_delay_base  # 每只股票重置初始重试间隔
            while retry_count < max_retry and not success:
                try:
                    # 调用BaoStock 分钟接口(adjustflag="3"默认不复权,1：后复权；2：前复权)
                    rs = bs.query_history_k_data_plus(
                        stock_code,"date,time,code,open,high,low,close,volume,amount",
                        start_date=start_date_str,end_date=end_date_str,frequency=frequency,adjustflag="3")
                    success = True  # 成功调用接口，退出重试循环
                    
                    # 检查接口返回是否正常
                    if rs.error_code != '0':
                        print(f"BaoStock查询失败（{stock_code}）：{rs.error_msg}")
                        continue
                    
                    # 提取数据
                    data_list = []
                    while rs.next():
                        data_list.append(rs.get_row_data())
                    
                    if len(data_list) == 0:
                        print(f"{frequency}分钟线数据daily_basic为空（{stock_code}）")
                        continue
                    
                    # 格式化数据
                    df_temp = pd.DataFrame(data_list, columns=rs.fields)
                    # 转为数值型（避免字符串），无效值转为NaN
                    df_temp['close'] = pd.to_numeric(df_temp['close'], errors='coerce')
                    df_temp['open'] = pd.to_numeric(df_temp['open'], errors='coerce')
                    df_temp['high'] = pd.to_numeric(df_temp['high'], errors='coerce')
                    df_temp['low'] = pd.to_numeric(df_temp['low'], errors='coerce')
                    df_temp['volume'] = pd.to_numeric(df_temp['volume'], errors='coerce')
                    df_temp['amount'] = pd.to_numeric(df_temp['amount'], errors='coerce')
                                        
                    basic_mins_all.append(df_temp)
                    
                # 只捕获网络相关异常，进行重试
                except (NameResolutionError, MaxRetryError, ConnectionError,TimeoutError,ConnectionResetError,requests.exceptions.RequestException) as e:
                    retry_count += 1
                    if retry_count < max_retry:
                        print(f"获取{stock_code} {frequency}分钟数据失败（网络错误）：{str(e)[:50]}... 第{retry_count}次重试，等待{retry_delay}秒")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避，间隔翻倍
                    else:
                        print(f"获取{stock_code} {frequency}分钟数据失败：{str(e)[:50]}... 已重试{max_retry}次，跳过该股票")
                
                # 非网络异常：直接终止重试，跳过该股票
                except Exception as e:
                    print(f"获取{stock_code} {frequency}分钟行情失败：{e}，非网络错误，直接跳过")
                    break  # 跳出while重试循环

        # 合并基础行情（过滤无效数据）
        if basic_mins_all:
            basic_mins_df = pd.concat(basic_mins_all, ignore_index=True)
            # 转换回Tushare代码格式（sh.600000 → 600000.SH）
            basic_mins_df['code'] = basic_mins_df['code'].str.split('.').str[1] + '.' + basic_mins_df['code'].str.split('.').str[0].str.upper()
            basic_mins_df.rename(columns={'time': 'trade_date','code':'ts_code'}, inplace=True)
            basic_mins_df.drop(columns=['date'], inplace=True)
            basic_mins_df['trade_date']=pd.to_datetime(basic_mins_df['trade_date'],format='%Y%m%d%H%M%S%f',errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
            basic_mins_df = basic_mins_df.dropna(subset=['trade_date'])
            basic_mins_df['factor']=1
            print(f"共获取{len(basic_mins_df)}条{frequency}分钟行情数据")
        else:
            print(f"未获取到任何{frequency}分钟行情数据")
        return basic_mins_df
    
    def _get_baostock_daily_basic_mins(self, all_stocks, start_date_str, end_date_str, frequency):
        # -------------------------- 1. 基础格式转换（增加异常处理+参数校验） --------------------------
        self.VALID_FREQUENCIES = {"5", "15", "30", "60"}  # Baostock合法分钟线频率
        # 日期转换(baostock 使用 %Y-%m-%d， Tushare使用 %Y%m%d)
        start_date_bs = datetime.strptime(start_date_str, "%Y%m%d").strftime("%Y-%m-%d")
        end_date_bs = datetime.strptime(end_date_str, "%Y%m%d").strftime("%Y-%m-%d")
        
        # 校验分钟线频率
        if str(frequency) not in self.VALID_FREQUENCIES:
            print(f"非法的分钟线频率：{frequency}，仅支持{self.VALID_FREQUENCIES}")
            return pd.DataFrame()
        
        # 转换股票代码格式：适配baostock（增加合法性校验）
        all_stocks_bs = [f"{code.split('.')[1].lower()}.{code.split('.')[0]}" if len(code.split('.')) == 2 else code for code in all_stocks ]

        all_stocks_bs = [conv_code for code in all_stocks if (conv_code := (f"{code.split('.')[1].lower()}.{code.split('.')[0]}" if len(code.split('.')) == 2 else code.lower())) and len(parts:=conv_code.split('.'))==2 and parts[0] in ['sz','sh'] and parts[1].isdigit() and len(parts[1])==6]
        
        if not all_stocks_bs:
            print("无合法的股票代码，终止拉取")
            return pd.DataFrame()
        
        # -------------------------- 2. 分批逻辑（保留你的原始逻辑） --------------------------
        max_workers = self.MAX_WORKERS  # 从类属性获取进程数
        batch_size = len(all_stocks_bs) // max_workers
        if batch_size == 0:
            batch_size = 1
        
        stock_batches = [all_stocks_bs[i:i + batch_size] for i in range(0, len(all_stocks_bs), batch_size)]

        # -------------------------- 4. 多进程执行（增加异常处理） --------------------------
        # 构造进程池参数（每个批次对应一个参数元组）
        pool_args = [(batch, start_date_bs, end_date_bs, frequency, self.MAX_RETRY) for batch in stock_batches]
        
        # 启动进程池，处理所有批次
        all_batch_results = []
        with Pool(processes=max_workers) as pool:
            try:
                # 遍历进程结果（带进度条）
                for batch_df in pool.imap_unordered(self._process_single_batch, pool_args):
                    if not batch_df.empty:
                        all_batch_results.append(batch_df)
            except Exception as e:
                print(f"进程池执行异常：{type(e).__name__} - {str(e)}")
                pool.terminate()  # 终止所有子进程，避免挂死
                return pd.DataFrame()

        # -------------------------- 5. 最终数据合并与格式化（优化时间格式容错） --------------------------
        if not all_batch_results:
            print(f"未获取到任何{frequency}分钟数据")
            return pd.DataFrame()
        print(f"所有进程结束，处理数据...")
        # 合并所有批次数据             
        basic_mins_df = pd.concat(all_batch_results, ignore_index=True)
        for col in ['close', 'open', 'high', 'low', 'volume', 'amount']:
            basic_mins_df[col] = pd.to_numeric(basic_mins_df[col], errors='coerce')
        # 还原股票代码为Tushare格式（sh.600000 → 600000.SH）
        #basic_mins_df['code'] = basic_mins_df['code'].apply(lambda x: f"{x.split('.')[1]}.{x.split('.')[0].upper()}" if len(x.split('.'))==2 else x)
        # basic_mins_df['code'] = basic_mins_df['code'].str.split('.', expand=True).pipe(lambda df: df[1] + '.' + df[0].str.upper()).where(basic_mins_df['code'].str.count('.') == 1, basic_mins_df['code'])
        basic_mins_df['code'] = (lambda s: s[1] + '.' + s[0].str.upper()).__call__(basic_mins_df['code'].str.split('.', expand=True)).where(basic_mins_df['code'].str.split('.').str.len() == 2, basic_mins_df['code'])
        basic_mins_df.rename(columns={'time': 'trade_date', 'code': 'ts_code'}, inplace=True)
        basic_mins_df.drop(columns=['date'], inplace=True)
        basic_mins_df['trade_date'] = pd.to_datetime(basic_mins_df['trade_date'],format='%Y%m%d%H%M%S%f',errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # 过滤空值+添加factor列
        basic_mins_df = basic_mins_df.drop_duplicates(subset=['ts_code','trade_date'], keep="first",ignore_index=True)
        basic_mins_df['factor'] = 1

        print(f"拉取完成！共获取{len(basic_mins_df)}条{frequency}分钟数据")
        return basic_mins_df
  
    # -------------------------- 3. 进程处理核心函数（修复重试逻辑+补全导入） --------------------------
    @staticmethod
    def _process_single_batch(batch_args):# 不支持类方法，因为每个进程需要独立登录Baostock
        """
        子进程处理单个批次（独立登录Baostock，隔离Socket）
        :param batch_args: tuple - (batch_stocks, start_bs, end_bs, frequency, max_retry)
        :return: pd.DataFrame - 该批次的分钟线数据
        """
        import baostock as bs   #!!!独立修改过底层库文件，以支持多进程
        import random
        time.sleep(random.uniform(0, 2))
        batch_stocks, start_bs, end_bs, frequency, max_retry = batch_args
        # 子进程独立登录Baostock（关键：每个进程专属Socket）
        pid = os.getpid() #获取当前进程ID
        batch_data = []
        retry_delay_base = 1  # 初始重试间隔
        lg = None  # 登录实例
        def login_with_retry():
            """Baostock登录，失败时重试"""
            login_retry = 30 #登录重试次数
            nonlocal lg
            login_retry_count = 0
            login_delay = 1
            while login_retry_count < login_retry:
                try:
                    # # 先登出旧会话（避免残留）
                    # try:
                    #     bs.logout()
                    # except:
                    #     pass
                    # 重新登录
                    lg = bs.login()
                    if lg.error_code == '0':
                        return True
                    else:
                        print(f"Pid:{pid}登录失败（第{login_retry_count+1}次）：{lg.error_msg}")
                except Exception as e:
                    print(f"Pid:{pid}登录异常（第{login_retry_count+1}次）：{str(e)[:50]}")
                
                login_retry_count += 1
                if login_retry_count < login_retry:
                    time.sleep(login_delay)
                    login_delay *= 1.5  # 指数退避
            
            print(f"Pid:{pid}登录重试{login_retry}次失败，放弃该批次处理")
            return False
        
        # -------------------------- 初始登录（带重试） --------------------------
        if not login_with_retry():
            return pd.DataFrame()

        # 遍历批次内的每只股票
        for i, stock_code in  enumerate(tqdm(batch_stocks, desc=f'Pid:{pid} 获取{frequency}mins 数据')):#:
            retry_count = 0
            success = False
            retry_delay = retry_delay_base
            while retry_count < max_retry and not success:
                try:
                    # 调用Baostock分钟接口
                    rs = bs.query_history_k_data_plus(stock_code,"date,time,code,open,high,low,close,volume,amount",start_date=start_bs,end_date=end_bs,frequency=frequency,adjustflag="3")# 不复权
                    
                    # 检查接口返回（先判断错误，再标记success）
                    if rs.error_code != '0':
                        print(f"Pid:{pid} 【{stock_code}】接口返回错误：{rs.error_msg}")
                        if "未登录" in rs.error_msg:
                            print(f"Pid:{pid} 【{stock_code}】会话失效，重新登录...")
                            login_with_retry()
                        retry_count += 1  # 接口错误也重试（如临时限流）
                        if retry_count < max_retry:
                            time.sleep(retry_delay)
                            retry_delay *= 2
                        continue
                    
                    # 提取数据
                    data_list = []
                    while rs.next():
                        data_list.append(rs.get_row_data())
                    
                    if len(data_list) == 0:
                        print(f"Pid:{pid} 【{stock_code}】无{frequency}分钟数据")
                        success = True
                        continue
                    
                    # 格式化数据
                    df_temp = pd.DataFrame(data_list, columns=rs.fields)
                    # 数值型转换
                    batch_data.append(df_temp)
                    success = True  # 只有处理成功才标记success
                    time.sleep(random.uniform(0.05, 0.15))
                # 网络异常重试（指数退避）
                except (NameResolutionError, MaxRetryError, ConnectionError, TimeoutError,ConnectionResetError, requests.exceptions.RequestException) as e:
                    retry_count += 1
                    if retry_count < max_retry:
                        print(f"Pid:{pid} 【{stock_code}】网络错误({str(e)[:50]})，第{retry_count}次重试（等待{retry_delay}秒）")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        print(f"Pid:{pid} 【{stock_code}】重试{max_retry}次失败，跳过")
                
                # 非网络异常直接跳过
                except Exception as e:
                    print(f"Pid:{pid} 【{stock_code}】异常：{e}，跳过")
                    success = True  # 非网络异常，终止重试
                    break
        
        # # 子进程登出Baostock
        # bs.logout()
        
        # 合并批次内数据
        if batch_data:
            return pd.concat(batch_data, ignore_index=True)
        return pd.DataFrame()



    def updates_tushare_A_basic(self, start_date_str, end_date_str):
        file_name_basic_df_prefix = 'download_A_basic_df_'
        file_name_adj_df_prefix = 'download_A_adj_df_'
        file_name_daily_df_prefix = 'download_A_daily_df_'
        file_name_updown_limit_df_prefix = 'download_A_updown_limit_df_'
        file_name_moneyflow_df_prefix = 'download_A_moneyflow_df_'
        file_name_adjusted_df_prefix = 'all_returns_A_adjusted_'

        file_name_basic, file_name_basic_df = self._utils_read_matched_csv_by_prefix(self.save_dir_download,file_name_basic_df_prefix)
        if len(file_name_basic_df) == 0:
            print(f"未读取到文件：{file_name_basic_df_prefix},请先下载原始数据")
            return
        file_name_adj, file_name_adj_df = self._utils_read_matched_csv_by_prefix(self.save_dir_download,file_name_adj_df_prefix)
        if len(file_name_adj_df) == 0:
            print(f"未读取到文件：{file_name_adj_df_prefix},请先下载原始数据")
            return
        file_name_daily, file_name_daily_df = self._utils_read_matched_csv_by_prefix(self.save_dir_download,file_name_daily_df_prefix)
        if len(file_name_daily_df) == 0:
            print(f"未读取到文件：{file_name_daily_df_prefix},请先下载原始数据")
            return
        file_name_updown,file_name_updown_limit_df = self._utils_read_matched_csv_by_prefix(self.save_dir_download,file_name_updown_limit_df_prefix)
        if len(file_name_updown_limit_df) == 0:
            print(f"未读取到文件：{file_name_updown_limit_df_prefix},请先下载原始数据")
            return
        file_name_moneyflow, file_name_moneyflow_df = self._utils_read_matched_csv_by_prefix(self.save_dir_download,file_name_moneyflow_df_prefix)
        if len(file_name_moneyflow_df) == 0:
            print(f"未读取到文件：{file_name_moneyflow_df_prefix},请先下载原始数据")
            return
        file_name_adjusted, file_name_adjusted_df = self._utils_read_matched_csv_by_prefix(self.save_dir_download,file_name_adjusted_df_prefix)
        if len(file_name_adjusted_df) == 0:
            print(f"未读取到文件：{file_name_adjusted_df_prefix},请先下载原始数据")
            return
        

        file_name_adjusted_df = pd.DataFrame()  # 此文件为basic_df和adj_df 合并而来，设置为空防止占内存，后期赋值保存
        old_start_str, old_end_str = self._utils_extract_date_from_filename(file_name_adjusted)
        if file_name_adjusted:
            if (end_date_str == old_start_str and start_date_str < end_date_str) or (start_date_str == old_end_str and end_date_str > start_date_str):
                print("下载日期正确")
            else:
                print("[ download_tushare_A_basic ]下载日期设置错误")
                return
        missing_start_end_list = self._utils_get_missing_date_ranges(old_start_str, old_end_str, start_date_str,end_date_str)
        begin_all_str = min(old_start_str, old_end_str, start_date_str, end_date_str)  # 所有日期首位相连或者重叠后的最边际日期
        end_all_str = max(old_start_str, old_end_str, start_date_str, end_date_str)
        obtain_begin_all_str = (pd.to_datetime(begin_all_str, format='%Y%m%d') - pd.DateOffset(months=self.more_month)).strftime('%Y%m%d')  # '20220901'

        # 0 获取全部股票列表
        stock_basic_df = self._get_stock_basic_df()
        if len(stock_basic_df) == 0:
            print(f"[updates_tushare_A_basic] stock_basic_df 获取失败，退出!")
            return

        old_stock_list = file_name_basic_df['ts_code'].dropna().unique().tolist()  # 老文件股列表
        all_stocks = stock_basic_df['ts_code'].dropna().unique().tolist()  # 新日期内股票列表
        new_stock_list = list(set(all_stocks) - set(old_stock_list))

        # 1、从未出现过的股票，日期最小取到最大
        if len(new_stock_list) > 0:
            print("-----获取新股票所有日期数据-----：")
            with ThreadPoolExecutor(max_workers=5) as executor:
                new_basic_df = executor.submit(self._get_daily_basic, new_stock_list, obtain_begin_all_str, end_all_str)
                new_adj_df = executor.submit(self._get_adj_factor, new_stock_list, obtain_begin_all_str, end_all_str)
                new_daily_df = executor.submit(self._get_daily, new_stock_list, obtain_begin_all_str, end_all_str)
                new_updown_limit_df = executor.submit(self._get_updown_limit, new_stock_list, obtain_begin_all_str, end_all_str)
                new_moneyflow_df = executor.submit(self._get_moneyflow, new_stock_list, obtain_begin_all_str, end_all_str)
                new_basic_df = new_basic_df.result()
                new_adj_df = new_adj_df.result()
                new_daily_df = new_daily_df.result()
                new_updown_limit_df = new_updown_limit_df.result()
                new_moneyflow_df = new_moneyflow_df.result()

            # 1.1获取基础日线行情
            if len(new_basic_df) == 0:
                print("new_basic_df 获取数据为空")
            file_name_basic_df = pd.concat([file_name_basic_df, new_basic_df], ignore_index=True, axis=0)  # 纵向追加（行级追加）
            # 1.2 获取复权因子
            if len(new_adj_df) == 0:
                print("new_adj_df 获取数据为空")
            file_name_adj_df = pd.concat([file_name_adj_df, new_adj_df], ignore_index=True, axis=0)  # 纵向追加（行级追加）
            # 1.2 获取日线行情
            if len(new_daily_df) == 0:
                print("new_daily_df 获取数据为空")
            file_name_daily_df = pd.concat([file_name_daily_df, new_daily_df], ignore_index=True, axis=0)  # 纵向追加（行级追加）
            # 1.3 获取涨跌停数据
            if len(new_updown_limit_df) == 0:
                print("new_updown_limit_df 获取数据为空")
            file_name_updown_limit_df = pd.concat([file_name_updown_limit_df, new_updown_limit_df], ignore_index=True, axis=0)  # 纵向追加（行级追加）
            # 1.4 获取资金流向数据
            if len(new_moneyflow_df) == 0:
                print("new_moneyflow_df 获取数据为空")
            file_name_moneyflow_df = pd.concat([file_name_moneyflow_df, new_moneyflow_df], ignore_index=True, axis=0)  # 纵向追加（行级追加）

        # 2、对于老股票，补充缺失的时间段数据
        print("-----获取旧股票新日期数据-----：")
        for missing_start_str, missing_end_str in missing_start_end_list:
            obtain_missing_date_str = (pd.to_datetime(missing_start_str, format='%Y%m%d') - pd.DateOffset(months=self.more_month)).strftime('%Y%m%d')  # '20220901'
            with ThreadPoolExecutor(max_workers=5) as executor:
                basic_df = executor.submit(self._get_daily_basic, old_stock_list, obtain_missing_date_str,missing_end_str)
                adj_df = executor.submit(self._get_adj_factor, old_stock_list, obtain_missing_date_str, missing_end_str)
                daily_df = executor.submit(self._get_daily, old_stock_list, obtain_missing_date_str, missing_end_str)
                updown_limit_df = executor.submit(self._get_updown_limit, old_stock_list, obtain_missing_date_str, missing_end_str)
                moneyflow_df = executor.submit(self._get_moneyflow, old_stock_list, obtain_missing_date_str, missing_end_str)
                basic_df = basic_df.result()
                adj_df = adj_df.result()
                daily_df = daily_df.result()
                updown_limit_df = updown_limit_df.result()
                moneyflow_df = moneyflow_df.result()
            # 2.1获取基础日线行情
            file_name_basic_df = pd.concat(
                [file_name_basic_df, basic_df],  # 先加原始数据，后加新数据
                ignore_index=True,  # 重置索引（重要，避免索引冲突）
                axis=0)  # 纵向追加（行级追加）
            # 2.2 获取复权因子
            file_name_adj_df = pd.concat(
                [file_name_adj_df, adj_df],  # 先加原始数据，后加新数据
                ignore_index=True,  # 重置索引（重要，避免索引冲突）
                axis=0)  # 纵向追加（行级追加）
            # 2.3 获取日线行情
            file_name_daily_df = pd.concat(
                [file_name_daily_df, daily_df],  # 先加原始数据，后加新数据
                ignore_index=True,  # 重置索引（重要，避免索引冲突）
                axis=0)  # 纵向追加（行级追加）
            # 2.4 获取涨跌停数据
            file_name_updown_limit_df = pd.concat(
                [file_name_updown_limit_df, updown_limit_df],  # 先加原始数据，后加新数据
                ignore_index=True,  # 重置索引（重要，避免索引冲突）
                axis=0)  # 纵向追加（行级追加）
            # 2.5 获取资金流向数据
            file_name_moneyflow_df = pd.concat(
                [file_name_moneyflow_df, moneyflow_df],  # 先加原始数据，后加新数据
                ignore_index=True,  # 重置索引（重要，避免索引冲突
                axis=0)  # 纵向追加（行级追加）

        # 3 排序并去重
        file_name_basic_df = file_name_basic_df.sort_values(by=['ts_code', 'trade_date']).drop_duplicates(keep="first",ignore_index=True)  # 保留第一次出现的重复行（可改为"last"保留最后一次）# 去重后重置索引
        file_name_adj_df = file_name_adj_df.sort_values(by=['ts_code', 'trade_date']).drop_duplicates(keep="first",ignore_index=True)  # 保留第一次出现的重复行（可改为"last"保留最后一次）# 去重后重置索引
        file_name_daily_df = file_name_daily_df.sort_values(by=['ts_code', 'trade_date']).drop_duplicates(keep="first",ignore_index=True)
        file_name_updown_limit_df = file_name_updown_limit_df.sort_values(by=['ts_code', 'trade_date']).drop_duplicates(keep="first",ignore_index=True)
        file_name_moneyflow_df = file_name_moneyflow_df.sort_values(by=['ts_code', 'trade_date']).drop_duplicates(keep="first",ignore_index=True)
        # 4 保存并删除
        self._to_new_csv_and_delete_old(file_name_basic, file_name_basic_df, begin_all_str, end_all_str,self.save_dir_download)
        self._to_new_csv_and_delete_old(file_name_adj, file_name_adj_df, begin_all_str, end_all_str,self.save_dir_download)
        self._to_new_csv_and_delete_old(file_name_daily, file_name_daily_df, begin_all_str, end_all_str,self.save_dir_download)
        self._to_new_csv_and_delete_old(file_name_updown, file_name_updown_limit_df, begin_all_str, end_all_str,self.save_dir_download)
        self._to_new_csv_and_delete_old(file_name_moneyflow, file_name_moneyflow_df, begin_all_str, end_all_str,self.save_dir_download)

        file_name_basic_df['ts_code'] = file_name_basic_df['ts_code'].apply(lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")
        file_name_adj_df['ts_code'] = file_name_adj_df['ts_code'].apply(lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")
        file_name_daily_df['ts_code'] = file_name_daily_df['ts_code'].apply(lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")
        file_name_updown_limit_df['ts_code'] = file_name_updown_limit_df['ts_code'].apply(lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")
        file_name_moneyflow_df['ts_code'] = file_name_moneyflow_df['ts_code'].apply(lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")


        file_name_adjusted_df = file_name_basic_df
        #  5 复权处理+合并数据
        if self.add_adj_comlums_flag:
            file_name_adjusted_df = self._add_adj_comlums(file_name_adj_df, file_name_basic_df)

        # 6 月度调仓处理,增加 列['adjusted_month', 'is_current_stock'] 到file_name_adjusted_df表
        if self.compute_change_position_flag:
            file_name_adjusted_df = self._change_position_time(stock_basic_df, file_name_adjusted_df, "month", 1,start_date_str, end_date_str)

        if self.add_comlums_shenewn_flag:
            file_name_adjusted_df = self.add_wanshen_classify(file_name_adjusted_df)

        # 7 保存并删除
        self._to_new_csv_and_delete_old(file_name_adjusted, file_name_adjusted_df, begin_all_str, end_all_str,self.save_dir_download)

        file_name_daily_df.rename(columns={'vol': 'volume'}, inplace=True)
        file_name_adj_df.rename(columns={'adj_factor': 'factor'}, inplace=True)

        # # 8 单独保存至单股文件夹
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.submit(self._save_substock_data,self.save_dir_basic,file_name_basic_df,0)
            executor.submit(self._save_substock_data,self.save_dir_daily, file_name_daily_df,1)
            if not self.add_adj_comlums_flag:
                executor.submit(self._save_substock_data,self.save_dir_adj, file_name_adj_df,2)
            executor.submit(self._save_substock_data,self.save_dir_updown_limit, file_name_updown_limit_df,3)
            executor.submit(self._save_substock_data,self.save_dir_moneyflow, file_name_moneyflow_df,4)


    def updates_tushare_A_finance(self, start_date_str, end_date_str):
        file_name_fina_df_prefix = f'download_A_fina_df_'
        file_name_income_df_prefix = f'download_A_income_df_'
        file_name_balance_df_prefix = f'download_A_balance_df_'

        file_name_fina, file_name_fina_df = self._utils_read_matched_csv_by_prefix(self.save_dir_download,file_name_fina_df_prefix)
        if len(file_name_fina_df) == 0:
            print(f"未读取到文件：{file_name_fina_df_prefix}")
            return

        file_name_income, file_name_income_df = self._utils_read_matched_csv_by_prefix(self.save_dir_download,file_name_income_df_prefix)
        if len(file_name_income_df) == 0:
            print(f"未读取到文件：{file_name_income_df_prefix}")
            return

        file_name_balance, file_name_balance_df = self._utils_read_matched_csv_by_prefix(self.save_dir_download,file_name_balance_df_prefix)
        if len(file_name_balance_df) == 0:
            print(f"未读取到文件：{file_name_balance_df_prefix}")
            return

        old_start_str, old_end_str = self._utils_extract_date_from_filename(file_name_fina)  # 原始文件的start_date和end_date
        if file_name_fina:
            if (end_date_str == old_start_str and start_date_str < end_date_str) or (start_date_str == old_end_str and end_date_str > start_date_str):
                print("下载日期正确")
            else:
                print("[ updates_tushare_A_finance ]下载日期设置错误")
                return
        missing_start_end_list = self._utils_get_missing_date_ranges(old_start_str, old_end_str, start_date_str,end_date_str)
        begin_all_str = min(old_start_str, old_end_str, start_date_str, end_date_str)  # 所有日期首位相连或者重叠后的最边际日期
        end_all_str = max(old_start_str, old_end_str, start_date_str, end_date_str)
        #
        # 0 分批获取指数权重 日期从最小取最大，从而获取所有的新stock list
        stock_basic_df = self._get_stock_basic_df()
        if (len(stock_basic_df) > 0):
            stock_basic_df['ts_code'] = stock_basic_df['ts_code'].apply(lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")
        old_stock_list = list(set(file_name_fina_df['ts_code'].dropna().unique()) | set(file_name_income_df['ts_code'].dropna().unique()) | set(file_name_balance_df['ts_code'].dropna().unique()))  # 老文件股列表
        all_stocks = stock_basic_df['ts_code'].dropna().unique().tolist()  # 新日期内股票列表
        new_stock_list = list(set(all_stocks) - set(old_stock_list))
        # 1、从未出现过的股票，日期最小取到最大
        if len(new_stock_list) > 0:
            with ThreadPoolExecutor(max_workers=3) as executor:
                fina_df_temp = executor.submit(self._update_get_finace_df, new_stock_list, "fina_indicator",begin_all_str, end_all_str)
                income_df_temp = executor.submit(self._update_get_finace_df, new_stock_list, "income", begin_all_str,end_all_str)
                balance_df_temp = executor.submit(self._update_get_finace_df, new_stock_list, "balance", begin_all_str,end_all_str)
                fina_df_temp = fina_df_temp.result()
                income_df_temp = income_df_temp.result()
                balance_df_temp = balance_df_temp.result()
            # 1.1  获取财务指标数据
            # fina_df_temp = self._update_get_finace_df(new_stock_list,"fina_indicator",begin_all_str,end_all_str)
            if len(fina_df_temp) == 0:
                print("fina_df_temp 获取数据为空")
            file_name_fina_df = pd.concat([file_name_fina_df, fina_df_temp], ignore_index=True, axis=0)
            # 1.2 获取利润数据
            # income_df_temp = self._update_get_finace_df(new_stock_list,"income", begin_all_str, end_all_str)
            if len(income_df_temp) == 0:
                print("income_df_temp 获取数据为空")
            file_name_income_df = pd.concat([file_name_income_df, income_df_temp], ignore_index=True, axis=0)
            # 1.3 获取资产负债数据
            # balance_df_temp = self._update_get_finace_df(new_stock_list,"balance", begin_all_str, end_all_str)
            if len(balance_df_temp) == 0:
                print("balance_df_temp 获取数据为空")
            file_name_balance_df = pd.concat([file_name_balance_df, balance_df_temp], ignore_index=True, axis=0)
        print("new_stock_list 新股票添加完成")
        # 2、对于老股票，补充缺失的时间段数据
        for missing_start_str, missing_end_str in missing_start_end_list:
            with ThreadPoolExecutor(max_workers=3) as executor:
                fina_df_temp_1 = executor.submit(self._update_get_finace_df, old_stock_list, "fina_indicator",missing_start_str, missing_end_str)
                income_df_temp_1 = executor.submit(self._update_get_finace_df, old_stock_list, "income",missing_start_str, missing_end_str)
                balance_df_temp_1 = executor.submit(self._update_get_finace_df, old_stock_list, "balance",missing_start_str, missing_end_str)
                fina_df_temp_1 = fina_df_temp_1.result()
                income_df_temp_1 = income_df_temp_1.result()
                balance_df_temp_1 = balance_df_temp_1.result()
            # 2.1  获取财务指标数据
            # fina_df_temp_1 = self._update_get_finace_df(old_stock_list, "fina_indicator", missing_start_str, missing_end_str)
            if len(fina_df_temp_1) == 0:
                print("fina_df_temp_1 获取数据为空,可能无增量数据")
            file_name_fina_df = pd.concat([file_name_fina_df, fina_df_temp_1], ignore_index=True, axis=0)
            # 2.2 获取利润数据
            # income_df_temp_1 = self._update_get_finace_df(old_stock_list, "income", missing_start_str, missing_end_str)
            if len(income_df_temp_1) == 0:
                print("income_df_temp_1 获取数据为空,可能无增量数据")
            file_name_income_df = pd.concat([file_name_income_df, income_df_temp_1], ignore_index=True, axis=0)
            # 2.3 获取资产负债数据
            # balance_df_temp_1 = self._update_get_finace_df(old_stock_list, "balance", missing_start_str, missing_end_str)
            if len(balance_df_temp_1) == 0:
                print("balance_df_temp 获取数据为空,可能无增量数据")
            file_name_balance_df = pd.concat([file_name_balance_df, balance_df_temp_1], ignore_index=True, axis=0)
        print("missing_start_end_list 老股票更新完成")
        # 3、排序并去重
        file_name_fina_df = file_name_fina_df.sort_values(by=['ts_code', 'ann_date']).drop_duplicates(keep="first",ignore_index=True)
        file_name_income_df = file_name_income_df.sort_values(by=['ts_code', 'ann_date']).drop_duplicates(keep="first",ignore_index=True)
        file_name_balance_df = file_name_balance_df.sort_values(by=['ts_code', 'ann_date']).drop_duplicates(keep="first", ignore_index=True)

        # 4、保存新文件并删除旧文件
        self._to_new_csv_and_delete_old(file_name_fina, file_name_fina_df, begin_all_str, end_all_str,self.save_dir_download)
        self._to_new_csv_and_delete_old(file_name_income, file_name_income_df, begin_all_str, end_all_str,self.save_dir_download)
        self._to_new_csv_and_delete_old(file_name_balance, file_name_balance_df, begin_all_str, end_all_str,self.save_dir_download)

        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.submit(self._save_substock_data, self.save_dir_fina_indicator, file_name_fina_df, 0)
            executor.submit(self._save_substock_data, self.save_dir_income, file_name_income_df, 1)
            executor.submit(self._save_substock_data, self.save_dir_balancesheet, file_name_balance_df, 2)

    def updates_tushare_A_basic_mins(self, start_date_str, end_date_str):
        # 2. 基础参数配置
        file_name_60mins_basic_df_prefix = 'download_A_60mins_basic_df_'
        file_name_5mins_basic_df_prefix = 'download_A_5mins_basic_df_'

        file_name_60mins_basic, file_name_60mins_basic_df = self._utils_read_matched_csv_by_prefix(self.save_dir_download,file_name_60mins_basic_df_prefix)
        if len(file_name_60mins_basic_df) == 0:
            print(f"未读取到文件：{file_name_60mins_basic_df_prefix},请先下载原始数据")
            return
        file_name_5mins_basic, file_name_5mins_basic_df = self._utils_read_matched_csv_by_prefix(self.save_dir_download,file_name_5mins_basic_df_prefix)
        if len(file_name_5mins_basic_df) == 0:
            print(f"未读取到文件：{file_name_5mins_basic_df_prefix},请先下载原始数据")
            return

        old_start_str, old_end_str = self._utils_extract_date_from_filename(file_name_60mins_basic)
        if (end_date_str == old_start_str and start_date_str < end_date_str) or (start_date_str == old_end_str and end_date_str > start_date_str):
            print("下载日期正确")
        else:
            print("[ updates_tushare_A_basic_mins ]下载日期设置错误")
            return
        old_start_str, old_end_str = self._utils_extract_date_from_filename(file_name_5mins_basic)
        if (end_date_str == old_start_str and start_date_str < end_date_str) or (start_date_str == old_end_str and end_date_str > start_date_str):
            print("下载日期正确")
        else:
            print("[ updates_tushare_A_basic_mins ]下载日期设置错误")
            return
        
        missing_start_end_list = self._utils_get_missing_date_ranges(old_start_str, old_end_str, start_date_str, end_date_str)
        begin_all_str = min(old_start_str, old_end_str, start_date_str, end_date_str)  # 所有日期首位相连或者重叠后的最边际日期
        end_all_str = max(old_start_str, old_end_str, start_date_str, end_date_str)
        obtain_begin_all_str = (pd.to_datetime(begin_all_str, format='%Y%m%d') - pd.DateOffset(months=self.more_month)).strftime('%Y%m%d')  # '20220901'

        # 0 获取全部股票列表
        stock_basic_df = self._get_stock_basic_df()
        if len(stock_basic_df) == 0:
            print(f"[updates_tushare_A_basic] stock_basic_df 获取失败，退出!")
            return
        #
        # 2 获取所有需查询的股票
        all_stocks = stock_basic_df['ts_code'].dropna().unique().tolist()
        all_stocks = [code for code in all_stocks if code.lower().endswith(('.sh','.sz'))]
        if len(all_stocks) == 0:
            print("无有效成分股数据，程序终止")
            return
        old_stock_list = file_name_60mins_basic_df['ts_code'].dropna().unique().tolist() # 老文件股列表
        old_stock_list = [code for code in old_stock_list if code.lower().endswith(('.sh', '.sz'))  ]
        all_stocks = stock_basic_df['ts_code'].dropna().unique().tolist()  # 新日期内股票列表
        new_stock_list = list(set(all_stocks) - set(old_stock_list))
        new_stock_list = [code for code in new_stock_list if code.lower().endswith(('.sh', '.sz'))  ]
        # 1、从未出现过的股票，日期最小取到最大
        if len(new_stock_list) > 0:
            print("-----获取新股票所有日期数据-----：")
            with ThreadPoolExecutor(max_workers=1) as executor:
                new_60mins_basic_df = executor.submit(self._get_baostock_daily_basic_mins, new_stock_list, obtain_begin_all_str, end_all_str, frequency="60")
                new_5mins_basic_df = executor.submit(self._get_baostock_daily_basic_mins, new_stock_list, obtain_begin_all_str, end_all_str, frequency="5")

                new_60mins_basic_df = new_60mins_basic_df.result()
                new_5mins_basic_df = new_5mins_basic_df.result()
            # 1.1获取基础日线行情
            if len(new_60mins_basic_df) == 0:
                print("远程日线mins数据为空")
            file_name_60mins_basic_df = pd.concat(
                [file_name_60mins_basic_df, new_60mins_basic_df],
                ignore_index=True,  # 重置索引（重要，避免索引冲突）
                axis=0)  # 纵向追加（行级追加）
            
            if len(new_5mins_basic_df) == 0:
                print("远程5分钟mins数据为空")
            file_name_5mins_basic_df = pd.concat(
                [file_name_5mins_basic_df, new_5mins_basic_df],
                ignore_index=True,  # 重置索引（重要，避免索引冲突）
                axis=0)  # 纵向追加（行级追加）

        # 2、对于老股票，补充缺失的时间段数据
        print("-----获取旧股票新日期数据-----：")
        for missing_start_str, missing_end_str in missing_start_end_list:
            obtain_missing_date_str = (pd.to_datetime(missing_start_str, format='%Y%m%d') - pd.DateOffset(months=self.more_month)).strftime('%Y%m%d')  # '20220901'
            with ThreadPoolExecutor(max_workers=1) as executor:
                new_60mins_basic_df = executor.submit(self._get_baostock_daily_basic_mins, old_stock_list, obtain_missing_date_str, missing_end_str, frequency="60")
                new_5mins_basic_df = executor.submit(self._get_baostock_daily_basic_mins, old_stock_list, obtain_missing_date_str, missing_end_str, frequency="5")
                new_60mins_basic_df = new_60mins_basic_df.result()
                new_5mins_basic_df = new_5mins_basic_df.result()

            # 2.1获取基础日线行情
            file_name_60mins_basic_df = pd.concat([file_name_60mins_basic_df, new_60mins_basic_df], ignore_index=True, axis=0)  # 纵向追加（行级追加）
            file_name_5mins_basic_df = pd.concat([file_name_5mins_basic_df, new_5mins_basic_df], ignore_index=True, axis=0)  # 纵向追加（行级追加）

        # # 3 排序并去重
        file_name_60mins_basic_df = file_name_60mins_basic_df.drop_duplicates(subset=['ts_code', 'trade_date'], keep="first",ignore_index=True)#.sort_values(by=['ts_code', 'trade_date'])
        file_name_60mins_basic_df['trade_date']=pd.to_datetime(file_name_60mins_basic_df['trade_date'],errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

        file_name_5mins_basic_df = file_name_5mins_basic_df.drop_duplicates(subset=['ts_code', 'trade_date'], keep="first",ignore_index=True)#.sort_values(by=['ts_code', 'trade_date'])
        file_name_5mins_basic_df['trade_date']=pd.to_datetime(file_name_5mins_basic_df['trade_date'],errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

        # 4 保存并删除
        self._to_new_csv_and_delete_old(file_name_60mins_basic, file_name_60mins_basic_df, begin_all_str, end_all_str,self.save_dir_download)
        self._to_new_csv_and_delete_old(file_name_5mins_basic, file_name_5mins_basic_df, begin_all_str, end_all_str,self.save_dir_download)
        
        split_ts_mins = file_name_60mins_basic_df['ts_code'].str.split('.', expand=True)
        file_name_60mins_basic_df['ts_code'] = split_ts_mins[1] + split_ts_mins[0]
        
        split_ts_5mins = file_name_5mins_basic_df['ts_code'].str.split('.', expand=True)
        file_name_5mins_basic_df['ts_code'] = split_ts_5mins[1] + split_ts_5mins[0]

        # 8 单独保存至单股文件夹
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(self._save_substock_data, self.save_dir_basic_60mins, file_name_60mins_basic_df, 0)
            executor.submit(self._save_substock_data, self.save_dir_basic_5mins, file_name_5mins_basic_df, 0)

    def download_baostock_A_basic_mins(self, start_date_str, end_date_str):

        # 2. 基础参数配置
        obtain_date_str = (pd.to_datetime(start_date_str, format='%Y%m%d') - pd.DateOffset(months=self.more_month)).strftime('%Y%m%d')  # '20220901'

        # 文件名定义
        file_name_mins_basic = os.path.join(self.save_dir_download, f'download_A_60mins_basic_df_{start_date_str}_{end_date_str}.parquet')
        file_name_5mins_basic = os.path.join(self.save_dir_download, f'download_A_5mins_basic_df_{start_date_str}_{end_date_str}.parquet')


        file_name_mins_basic_exists = os.path.exists(file_name_mins_basic)
        file_name_5mins_basic_exists = os.path.exists(file_name_5mins_basic)
        if file_name_mins_basic_exists and file_name_5mins_basic_exists:
            print("[download_baostock_A_basic_mins] 以下2个文件均已存在，无需重复处理，程序退出!")
            return

        # 3. 分批获取股票列表
        stock_basic_df = self._get_stock_basic_df()
        if len(stock_basic_df) == 0:
            print(f"[download_baostock_A_basic_mins] stock_basic_df 获取失败，退出!")
            return

        # 4 获取所有需查询的股票
        all_stocks = stock_basic_df['ts_code'].dropna().unique().tolist()
        # df = pd.read_csv(r"/home/yunbo/project/quantitative/data/ts_trade_count.csv", usecols=['ts_code'])#手动查看缺失的数据，行数整数250，调试补充股票数据用
        # all_stocks = df['ts_code'].dropna().drop_duplicates().tolist()
        if len(all_stocks) == 0:
            print("无有效成分股数据，程序终止")
            return

        #  5 获取日线行情
        def get_mins_basic_df():
            if os.path.exists(file_name_mins_basic):
                file_name_mins_basic_df = pd.read_parquet(file_name_mins_basic, engine='pyarrow')
                if len(file_name_mins_basic_df) == 0:
                    print("本地 60分钟线数据为空")
                    return pd.DataFrame()
            else:
                file_name_mins_basic_df = self._get_baostock_daily_basic_mins(all_stocks, obtain_date_str, end_date_str,frequency="60")
                if len(file_name_mins_basic_df) == 0:
                    print("远程 60分钟线数据为空")
                    return pd.DataFrame()
                file_name_mins_basic_df.to_parquet(file_name_mins_basic, index=False, engine='pyarrow')
            return file_name_mins_basic_df
        
        def get_5mins_basic_df():
            if os.path.exists(file_name_5mins_basic):
                file_name_5mins_basic_df = pd.read_parquet(file_name_5mins_basic, engine='pyarrow')
                if len(file_name_5mins_basic_df) == 0:
                    print("本地 5分钟线数据为空")
                    return pd.DataFrame()
            else:
                file_name_5mins_basic_df = self._get_baostock_daily_basic_mins(all_stocks, obtain_date_str, end_date_str,frequency="5")
                if len(file_name_5mins_basic_df) == 0:
                    print("远程 5分钟线数据为空")
                    return pd.DataFrame()
                file_name_5mins_basic_df.to_parquet(file_name_5mins_basic, index=False, engine='pyarrow')
            return file_name_5mins_basic_df


        with ThreadPoolExecutor(max_workers=2) as executor:
            # 提交任务
            file_name_mins_basic_df = executor.submit(get_mins_basic_df)
            file_name_5mins_basic_df = executor.submit(get_5mins_basic_df)
            # 获取结果
            file_name_mins_basic_df = file_name_mins_basic_df.result()
            file_name_5mins_basic_df = file_name_5mins_basic_df.result()

        # 处理60分钟数据的ts_code（向量化替代apply）
        split_ts_mins = file_name_mins_basic_df['ts_code'].str.split('.', expand=True)
        file_name_mins_basic_df['ts_code'] = split_ts_mins[1] + split_ts_mins[0]
        file_name_mins_basic_df = file_name_mins_basic_df.drop_duplicates(subset=['ts_code', 'trade_date'], keep="first",ignore_index=True)#.sort_values(by=['ts_code', 'trade_date'])

        # 处理5分钟数据的ts_code（向量化替代apply）
        split_ts_5mins = file_name_5mins_basic_df['ts_code'].str.split('.', expand=True)#仅拆分一次ts_code列（避免重复计算，核心提速点）
        file_name_5mins_basic_df['ts_code'] = split_ts_5mins[1] + split_ts_5mins[0]
        file_name_5mins_basic_df = file_name_5mins_basic_df.drop_duplicates(subset=['ts_code', 'trade_date'], keep="first",ignore_index=True)#.sort_values(by=['ts_code', 'trade_date'])

        print(f"60mins数据已保存至{file_name_mins_basic}，共{len(file_name_mins_basic_df)}条记录")
        print(f"5mins数据已保存至{file_name_5mins_basic}，共{len(file_name_5mins_basic_df)}条记录")

        # 10 按股票代码分组，批量保存
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(self._save_substock_data,self.save_dir_basic_60mins, file_name_mins_basic_df, 0)
            executor.submit(self._save_substock_data,self.save_dir_basic_5mins, file_name_5mins_basic_df, 0)




    def download_index(self,start_date_str,end_date_str):
        index_mapping = self.index_mapping
        code_to_csi = self.code_to_csi
        if end_date_str is None:
            start_date_str = time.strftime("%Y%m%d")  # 默认当前日期
        os.makedirs(self.save_dir_index_weight, exist_ok=True)
        for index_code in  tqdm(index_mapping.values(), desc='获取各种指数数据'):
            result_df = self._get_index_weight_df(index_code,start_date_str,end_date_str)
            csv_path = os.path.join(self.save_dir_download_index, f"index_basic_{index_code}_{start_date_str}_{end_date_str}.parquet")
            result_df.to_parquet(csv_path, index=False, engine='pyarrow')
            
            # 指数列表成分股单独转到json，方便O(1)读取使用
            result_df['trade_date'] = result_df['trade_date'].dt.strftime('%Y-%m-%d')
            new_data = self._utils_convert_to_json(result_df)
            sorted_data = dict(sorted(new_data.items()))
            csi_name = code_to_csi[index_code]
            # 保存 JSON 文件
            filepath = os.path.join(self.save_dir_index_weight, f"{csi_name}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(sorted_data, f, ensure_ascii=False, indent=2)
            print(f"已保存: {csv_path},{filepath}")  
        
        

    def update_index(self, start_date_str, end_date_str):
        index_mapping = self.index_mapping
        code_to_csi = self.code_to_csi
        for index_name, index_code in tqdm(index_mapping.items(), desc="更新指数数据"):
            file_prefix = f"index_basic_{index_code}_"
            old_file_name, old_df = self._utils_read_matched_csv_by_prefix(self.save_dir_download_index,file_prefix)
            if old_file_name is None:
                print(f"\n指数{index_name}({index_code})无有效旧文件，跳过更新")
                continue
            try:
                old_start_str, old_end_str = self._utils_extract_date_from_filename(old_file_name)
                if (end_date_str == old_start_str and start_date_str < end_date_str) or (start_date_str == old_end_str and end_date_str > start_date_str):
                    print("下载日期正确")
                else:
                    print("[ update_index ]下载日期设置错误")
                    continue
            except ValueError as e:
                print(f"\n指数{index_name}({index_code})提取旧文件日期失败：{e}，跳过")
                continue
            missing_ranges = self._utils_get_missing_date_ranges(old_start_str, old_end_str, start_date_str, end_date_str)
            begin_all_str = min(old_start_str, old_end_str, start_date_str, end_date_str)  # 所有日期首位相连或者重叠后的最边际日期
            end_all_str = max(old_start_str, old_end_str, start_date_str, end_date_str)
            all_missing_df = pd.DataFrame()
            for miss_start, miss_end in missing_ranges:
                all_missing_df = self._get_index_weight_df(index_code,miss_start,miss_end)
            old_df = pd.concat([old_df, all_missing_df], ignore_index=True)
            if {"con_code", "trade_date"}.issubset(old_df.columns):
                old_df = old_df.drop_duplicates(subset=["con_code", "trade_date"],keep="last")
            # 按日期排序
            if "trade_date" in old_df.columns:
                old_df["trade_date"] = pd.to_datetime(old_df["trade_date"])
                old_df = old_df.sort_values("trade_date").reset_index(drop=True)
            self._to_new_csv_and_delete_old(old_file_name,old_df, begin_all_str, end_all_str,self.save_dir_download_index)

                        # 指数列表成分股单独转到json，方便O(1)读取使用
            old_df['trade_date'] = old_df['trade_date'].dt.strftime('%Y-%m-%d')
            new_data = self._utils_convert_to_json(old_df)
            sorted_data = dict(sorted(new_data.items()))
            csi_name = code_to_csi[index_code]
            # 保存 JSON 文件
            filepath = os.path.join(self.save_dir_index_weight, f"{csi_name}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(sorted_data, f, ensure_ascii=False, indent=2)
            print(f"已保存: {old_file_name},{filepath}") 
        print("\n所有指数更新流程执行完毕！")


    def download_index_daily(self, start_date_str, end_date_str):
        index_mapping = self.index_mapping
        if end_date_str is None:
            start_date_str = time.strftime("%Y%m%d")  # 默认当前日期
        for index_code in  tqdm(index_mapping.values(), desc='获取各种指数数据'):
            result_df = self._get_index_daily_df(index_code,start_date_str,end_date_str)
            result_df.rename(columns={'vol': 'volume'}, inplace=True)
            #保存到本地origin下载
            csv_path = os.path.join(self.save_dir_download_index, f"index_daily_{index_code}_{start_date_str}_{end_date_str}.parquet")
            result_df.to_parquet(csv_path, index=False, engine='pyarrow')
            # 保存到qlib转换文件夹
            qlib_index_code = index_code.split('.')[1] + index_code.split('.')[0]
            csv_path = os.path.join(self.save_dir_index_daily, f"{qlib_index_code}.parquet")
            result_df.to_parquet(csv_path, index=False, engine='pyarrow')
        print("获取各种指数日线行情数据成功")

    def update_index_daily(self, start_date_str, end_date_str):
        index_mapping = self.index_mapping
        for index_name, index_code in tqdm(index_mapping.items(), desc="更新指数数据"):
            file_prefix = f"index_daily_{index_code}_"
            old_file_name, old_df = self._utils_read_matched_csv_by_prefix(self.save_dir_download_index,file_prefix)
            if old_file_name is None:
                print(f"\n指数{index_name}({index_code})无有效旧文件，跳过更新")
                continue
            try:
                old_start_str, old_end_str = self._utils_extract_date_from_filename(old_file_name)
                if (end_date_str == old_start_str and start_date_str < end_date_str) or (start_date_str == old_end_str and end_date_str > start_date_str):
                    print("下载日期正确")
                else:
                    print("[ update_index_daily ]下载日期设置错误")
                    continue
            except ValueError as e:
                print(f"\n指数{index_name}({index_code})提取旧文件日期失败：{e}，跳过")
                continue
            missing_ranges = self._utils_get_missing_date_ranges(old_start_str, old_end_str, start_date_str, end_date_str)
            begin_all_str = min(old_start_str, old_end_str, start_date_str, end_date_str)  # 所有日期首位相连或者重叠后的最边际日期
            end_all_str = max(old_start_str, old_end_str, start_date_str, end_date_str)
            all_missing_df = pd.DataFrame()
            for miss_start, miss_end in missing_ranges:
                all_missing_df = self._get_index_daily_df(index_code,miss_start,miss_end)
                all_missing_df.rename(columns={'vol': 'volume'}, inplace=True)
            old_df = pd.concat([old_df, all_missing_df], ignore_index=True)
            if {"ts_code", "trade_date"}.issubset(old_df.columns):
                old_df = old_df.drop_duplicates(subset=["ts_code", "trade_date"],keep="last")
            # 按日期排序
            if "trade_date" in old_df.columns:
                old_df["trade_date"] = pd.to_datetime(old_df["trade_date"])
                old_df = old_df.sort_values("trade_date").reset_index(drop=True)
            self._to_new_csv_and_delete_old(old_file_name,old_df, begin_all_str, end_all_str,self.save_dir_download_index)

            # 保存到qlib转换文件夹
            qlib_index_code = index_code.split('.')[1] + index_code.split('.')[0]
            csv_path = os.path.join(self.save_dir_index_daily, f"{qlib_index_code}.parquet")
            old_df.to_parquet(csv_path, index=False, engine='pyarrow')
        print("\n所有指数更新流程执行完毕！")

    def _get_index_daily_df(self,index_code, start_date_str, end_date_str):
        current_start = pd.to_datetime(start_date_str)
        total_end = pd.to_datetime(end_date_str)
        daily_df = pd.DataFrame()  # 初始化DataFrame（避免后续 concat 报错）
        max_retry = self.MAX_RETRY  # 最大重试次数
        while current_start <= total_end:
            retry_count = 0
            success = False
            temp_df = None
            retry_delay = 1  # 每只股票重置初始重试间隔为1秒
            current_end = current_start + pd.DateOffset(months=3)  # 每3个月一批
            if current_end > total_end:
                current_end = total_end
            curr_start_str = current_start.strftime('%Y%m%d')
            curr_end_str = current_end.strftime('%Y%m%d')
            while retry_count < max_retry and not success:
                try:
                    temp_df = self.pro.index_daily(ts_code=index_code, start_date=curr_start_str,end_date=curr_end_str)
                    success = True  # 成功获取，退出重试循环
                    if not temp_df.empty:
                        daily_df = pd.concat([daily_df, temp_df], ignore_index=True)
                # 只捕获网络相关异常，非网络异常直接跳过重试
                except (NameResolutionError, MaxRetryError, ConnectionError,TimeoutError,ConnectionResetError,requests.exceptions.RequestException) as e:
                    retry_count += 1
                    if retry_count < max_retry:
                        print(f"获取{index_code}数据失败（网络错误）：{str(e)[:50]}... 第{retry_count}次重试，等待{retry_delay}秒")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避，间隔翻倍
                    else:
                        print(f"获取{index_code}数据失败：{str(e)[:50]}... 已重试{max_retry}次，跳过该股票")
                except Exception as e:
                    print(f"获取指数日线行情失败（{curr_start_str}-{curr_end_str}）：{e}")
                    break
            current_start = current_end + pd.DateOffset(days=1)  # 避免重复日期
        # 权重数据清洗
        index_daily_df = daily_df.drop_duplicates(subset=['trade_date', 'ts_code'])  # 去重
        index_daily_df = index_daily_df.sort_values('trade_date')
        index_daily_df['trade_date'] = pd.to_datetime(index_daily_df['trade_date'], format='%Y%m%d')  # 统一日期
        print(f"index_daily_df 获取{len(index_daily_df)}条数据")
        return index_daily_df



    def download_tushare_shenwan_daily(self, start_date_str, end_date_str):
        file_name_shenwan_daily = os.path.join(self.save_dir_shenwan, f'download_shenwan_daily_df_{start_date_str}_{end_date_str}.parquet')
        shenwan_exists = os.path.exists(file_name_shenwan_daily)
        if shenwan_exists:
            print("[download_tushare_shenwan_daily]文件已存在，无需重复处理，程序退出!")
            return
        # 日期子集跳出
        file_name_shenwan_daily_df_prefix = 'download_shenwan_daily_df_'
        file_name_shenwan_df_exist, temp1 = self._utils_read_matched_csv_by_prefix(self.save_dir_shenwan,file_name_shenwan_daily_df_prefix)
        temp1 = pd.DataFrame()
        if file_name_shenwan_df_exist:
            print(f"{file_name_shenwan_df_exist}文件已存在，请使用增量下载")
            return
        # 分批获取sw指数权重
        file_name_shenwan_daily_df = self._get_shenwan_daily_df( start_date_str, end_date_str)
        if len(file_name_shenwan_daily_df) == 0:
            print("获取日线数据为空，退出!")
            return
        file_name_shenwan_daily_df.to_parquet(file_name_shenwan_daily, index=False, engine='pyarrow')
        file_name_shenwan_daily_df['ts_code'] = file_name_shenwan_daily_df['ts_code'].apply(lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")
        print(f"数据已保存至{file_name_shenwan_daily}，共{len(file_name_shenwan_daily_df)}条记录")
        # 单独保存至单股文件夹
        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.submit(self._save_substock_data, self.save_dir_shenwan_daily, file_name_shenwan_daily_df, 0)
        print("download_tushare_shenwan_daily 下载完成")

    def _get_shenwan_daily_df(self, start_date_str, end_date_str):
        shenwan_df = pd.DataFrame()
        date_range = pd.date_range(start=start_date_str, end=end_date_str, freq='D').strftime('%Y%m%d')
        max_retry = self.MAX_RETRY  # 最大重试次数
        # 使用 tqdm 创建进度条
        for curr_date_str in tqdm(date_range, desc="获取申万指数数据"):
            retry_count = 0
            success = False
            temp_df = None
            retry_delay = 1  # 每只股票重置初始重试间隔为1秒
            while retry_count < max_retry and not success:
                try:
                    temp_df = self.pro.sw_daily(trade_date=curr_date_str, fields='ts_code,trade_date,open,close')
                    # temp_df['trade_date'] = curr_date_str
                    success = True  # 成功获取，退出重试循环
                    if not temp_df.empty:
                        shenwan_df = pd.concat([shenwan_df, temp_df], ignore_index=True)
                    time.sleep(0.33)# 一分钟200次
                # 只捕获网络相关异常，非网络异常直接跳过重试
                except (NameResolutionError, MaxRetryError, ConnectionError,TimeoutError,ConnectionResetError,requests.exceptions.RequestException) as e:
                    retry_count += 1
                    if retry_count < max_retry:
                        print(f"获取{curr_date_str}数据失败（网络错误）：{str(e)[:50]}... 第{retry_count}次重试，等待{retry_delay}秒")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避，间隔翻倍
                    else:
                        print(f"获取{curr_date_str}数据失败：{str(e)[:50]}... 已重试{max_retry}次，跳过该股票")
                except Exception as e:
                    print(f"获取指数日线失败（{curr_date_str}：{e}")
        # 数据清洗
        if not shenwan_df.empty:
            index_shenwan_df = shenwan_df.drop_duplicates(subset=['ts_code','trade_date'])
            index_shenwan_df = index_shenwan_df.sort_values(by='trade_date')
            index_shenwan_df['trade_date'] = pd.to_datetime(index_shenwan_df['trade_date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
            print(f"\n申万指数数据获取完成，共获取 {len(index_shenwan_df)} 条数据")
        else:
            index_shenwan_df = pd.DataFrame()
            print("\n未获取到任何数据")
        return index_shenwan_df



    def update_tushare_shenwan_daily(self,start_date_str, end_date_str):
        file_name_shenwan_daily_df_prefix = 'download_shenwan_daily_df_'
        file_name_shenwan_daily, file_name_shenwan_daily_df = self._utils_read_matched_csv_by_prefix(self.save_dir_shenwan, file_name_shenwan_daily_df_prefix)
        if len(file_name_shenwan_daily_df) == 0:
            print(f"未读取到文件：{file_name_shenwan_daily_df_prefix},请先下载原始数据")
            return

        old_start_str, old_end_str = self._utils_extract_date_from_filename(file_name_shenwan_daily)
        if file_name_shenwan_daily:
            if (end_date_str == old_start_str and start_date_str < end_date_str) or (start_date_str == old_end_str and end_date_str > start_date_str):
                print("下载日期正确")
            else:
                print("[ update_tushare_shenwan_daily ]下载日期设置错误")
                return

        missing_start_end_list = self._utils_get_missing_date_ranges(old_start_str, old_end_str, start_date_str, end_date_str)
        begin_all_str = min(old_start_str, old_end_str, start_date_str, end_date_str)  # 所有日期首位相连或者重叠后的最边际日期
        end_all_str = max(old_start_str, old_end_str, start_date_str, end_date_str)
        for missing_start_str, missing_end_str in missing_start_end_list:
            tmp_df = self._get_shenwan_daily_df(start_date_str, end_date_str)
            if len(tmp_df) == 0:
                print("获取日线数据为空{start_date_str} - {end_date_str}")

            file_name_shenwan_daily_df = pd.concat(
                [file_name_shenwan_daily_df, tmp_df],  # 先加原始数据，后加新数据
                ignore_index=True,  # 重置索引（重要，避免索引冲突）
                axis=0)  # 纵向追加（行级追加）
        self._to_new_csv_and_delete_old(file_name_shenwan_daily, file_name_shenwan_daily_df, begin_all_str, end_all_str, self.save_dir_shenwan)
        file_name_shenwan_daily_df['ts_code'] = file_name_shenwan_daily_df['ts_code'].apply(lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")
        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.submit(self._save_substock_data, self.save_dir_shenwan_daily, file_name_shenwan_daily_df, 0)
        print("update_tushare_shenwan_daily 下载完成")

    def _get_updown_limit(self, stock_list, start_date_str, end_date_str):
        updown_limit_all = []
        updown_limit_df = pd.DataFrame()
        max_retry = self.MAX_RETRY  # 最大重试次数
        for i, stock_code in enumerate(tqdm(stock_list, desc='获取涨跌停数据'), 1):
            retry_count = 0
            success = False
            tmp = None
            retry_delay = 1  # 每只股票重置初始重试间隔为1秒
            while retry_count < max_retry and not success:
                try:
                    tmp = self.pro.stk_limit(ts_code=stock_code, start_date=start_date_str, end_date=end_date_str)
                    success = True  # 成功获取，退出重试循环
                    if not tmp.empty:
                        updown_limit_all.append(tmp)
                    else:
                        print(f"_get_updown_limit 未查询到股票{stock_code}数据，{start_date_str}---{end_date_str}")
                except (NameResolutionError, MaxRetryError, ConnectionError,TimeoutError,ConnectionResetError,requests.exceptions.RequestException) as e:
                    retry_count += 1
                    if retry_count < max_retry:
                        print(f"_get_updown_limit 获取{stock_code}数据失败（网络错误）：{str(e)[:50]}... 第{retry_count}次重试，等待{retry_delay}秒")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避，间隔翻倍
                    else:
                        print(f"_get_updown_limit 获取{stock_code}数据失败：{str(e)[:50]}... 已重试{max_retry}次，跳过该股票")
                # 非网络异常：直接终止重试，跳过该股票
                except Exception as e:
                    print(f"_get_updown_limit 获取涨跌停数据失败（{stock_code}）：{e}，非网络错误，直接跳过")
                    break  # 跳出while重试循环

        if updown_limit_all:  # 合并复权因子（过滤无效数据）
            updown_limit_all = [df for df in updown_limit_all if not df.empty]
            updown_limit_df = pd.concat(updown_limit_all, ignore_index=True)
            updown_limit_df =  updown_limit_df[["ts_code", "trade_date", "up_limit", "down_limit"]]
            updown_limit_df = updown_limit_df.dropna(subset=['ts_code', 'trade_date'])  # 过滤NaN
            updown_limit_df['trade_date'] = pd.to_datetime(updown_limit_df['trade_date']).dt.strftime('%Y-%m-%d')
            print(f"共获取{len(updown_limit_df)}条涨跌停数据")
        return updown_limit_df
    

    def _get_moneyflow(self, stock_list, start_date_str, end_date_str):
        moneyflow_all = []
        moneyflow_df = pd.DataFrame()
        max_retry = self.MAX_RETRY  # 最大重试次数
        for i, stock_code in enumerate(tqdm(stock_list, desc='获取个股资金流向数据'), 1):
            retry_count = 0
            success = False
            tmp = None
            retry_delay = 1  # 每只股票重置初始重试间隔为1秒
            while retry_count < max_retry and not success:
                try:
                    tmp = self.pro.moneyflow(ts_code=stock_code,start_date=start_date_str,end_date=end_date_str)
                    success = True  # 成功获取，退出重试循环
                    if not tmp.empty:
                        moneyflow_all.append(tmp)
                    else:
                        print(f"_get_moneyflow 未查询到股票{stock_code}数据，{start_date_str}---{end_date_str}")
                # 只捕获网络相关异常，非网络异常直接跳过重试
                except (NameResolutionError, MaxRetryError, ConnectionError,TimeoutError,ConnectionResetError,requests.exceptions.RequestException) as e:
                    retry_count += 1
                    if retry_count < max_retry:
                        print(f"_get_moneyflow 获取{stock_code}数据失败（网络错误）：{str(e)[:50]}... 第{retry_count}次重试，等待{retry_delay}秒")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避，间隔翻倍
                    else:
                        print(f"_get_moneyflow 获取{stock_code}数据失败：{str(e)[:50]}... 已重试{max_retry}次，跳过该股票")
                # 非网络异常：直接终止重试，跳过该股票
                except Exception as e:
                    print(f"_get_moneyflow 获取个股资金流向失败（{stock_code}）：{e}，非网络错误，直接跳过")
                    break  # 跳出while重试循环

        if moneyflow_all:  # 合并复权因子（过滤无效数据）
            moneyflow_all = [df for df in moneyflow_all if not df.empty]
            moneyflow_df = pd.concat(moneyflow_all, ignore_index=True)
            moneyflow_df = moneyflow_df.dropna(subset=['ts_code', 'trade_date'])  # 过滤NaN
            moneyflow_df['trade_date'] = pd.to_datetime(moneyflow_df['trade_date']).dt.strftime('%Y-%m-%d')
            print(f"共获取{len(moneyflow_df)}条个股资金流向数据")
        return moneyflow_df
    

    # 暂时不使用baostock复权因子，统一使用tushare复权因子
    def _get_baostock_adj_factor(self, stock_list,start_date_str, end_date_str):
        import baostock as bs
        start_date_str = datetime.strptime(start_date_str, "%Y%m%d").strftime("%Y-%m-%d")
        end_date_str = datetime.strptime(end_date_str, "%Y%m%d").strftime("%Y-%m-%d")
        stock_list = [f"{code.split('.')[1].lower()}.{code.split('.')[0]}" if len(code.split('.'))==2 else code for code in stock_list]
        adj_all = []
        adj_df = pd.DataFrame()
        max_retry = self.MAX_RETRY  # 最大重试次数
        for i, stock_code in enumerate(tqdm(stock_list, desc='获取baostock复权因子'), 1):
            retry_count = 0
            success = False
            tmp = None
            retry_delay = 1  # 每只股票重置初始重试间隔为1秒
            while retry_count < max_retry and not success:
                try:
                    # 调用Tushare复权因子接口
                    rs = bs.query_adjust_factor(code=stock_code, start_date=start_date_str, end_date=end_date_str)
                    if rs.error_code != '0':
                        print(f"BaoStock查询失败（{stock_code}）：{rs.error_msg}")
                        continue
                    # 提取数据
                    data_list = []
                    while rs.next():
                        data_list.append(rs.get_row_data())
                    if len(data_list) == 0:
                        print(f"分钟线数据adj_factor为空（{stock_code}）")
                        continue
                    tmp = pd.DataFrame(data_list, columns=rs.fields)
                    adj_all.append(tmp)
                    success = True  # 成功获取，退出重试循环
                # 只捕获网络相关异常，非网络异常直接跳过重试
                except (NameResolutionError, MaxRetryError, ConnectionError,TimeoutError,ConnectionResetError,requests.exceptions.RequestException) as e:
                    retry_count += 1
                    if retry_count < max_retry:
                        print(f"_get_baostock_adj_factor 获取{stock_code}数据失败（网络错误）：{str(e)[:50]}... 第{retry_count}次重试，等待{retry_delay}秒")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避，间隔翻倍
                    else:
                        print(f"_get_baostock_adj_factor 获取{stock_code}数据失败：{str(e)[:50]}... 已重试{max_retry}次，跳过该股票")
                except Exception as e:
                    print(f"_get_baostock_adj_factor 获取baostock复权因子失败（{stock_code}）：{e}")
                    break
        if adj_all:  # 合并复权因子（过滤无效数据）
            adj_all = [df for df in adj_all if not df.empty]
            adj_df = pd.concat(adj_all, ignore_index=True)
            adj_df.rename(columns={'code':'ts_code','dividOperateDate':'trade_date'},inplace=True)
            print(f"共获取{len(adj_df)}条复权因子数据")
        return adj_df
        


