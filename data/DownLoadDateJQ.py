
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import os
import sys
import jqdatasdk as jq
import json
from urllib3.exceptions import NameResolutionError, MaxRetryError, ConnectionError
import requests
import pandas as pd
from tqdm import tqdm
import time
from typing import Dict,List

from DataBase_ClickHouse import ClickHouseDB


class DownloadDataFromJointQuant:
    def __init__(self):
        self.config_path = './config.json'
        self.MAX_RETRY = 10  # 最大重试次数
        self.save_dir_index_weight = './download_Jqdata/index_weight'

        self.database_name = 'quantitative'
        self.table_name_A_basic_df = 'A_basic_df'
        self.table_name_A_basic_1min_df = 'A_basic_1min_df'
        self.table_name_A_basic_5min_df = 'A_basic_5min_df'
        self.table_name_A_basic_15min_df = 'A_basic_15min_df'
        self.table_name_A_basic_30min_df = 'A_basic_30min_df'
        self.table_name_A_basic_60min_df = 'A_basic_60min_df'

        self.table_name_A_daily_basic_df = 'A_daily_basic_df'
        self.table_name_index_daily_basic_df = 'index_daily_basic_df'

        self.table_name_index_daily_df = 'index_daily_df'
        self.table_name_index_daily_1min_df = 'index_daily_1min_df'
        self.table_name_index_daily_5min_df = 'index_daily_5min_df'
        self.table_name_index_daily_15min_df = 'index_daily_15min_df'
        self.table_name_index_daily_30min_df = 'index_daily_30min_df'
        self.table_name_index_daily_60min_df = 'index_daily_60min_df'

        self.table_name_shenwan_index_basic_df = 'shenwan_index_basic_df'
        self.table_name_shewan_index_basic_1min_df = 'shenwan_index_basic_1min_df'
        self.table_name_shewan_index_basic_5min_df = 'shenwan_index_basic_5min_df'
        self.table_name_shewan_index_basic_15min_df = 'shenwan_index_basic_15min_df'
        self.table_name_shewan_index_basic_30min_df = 'shenwan_index_basic_30min_df'
        self.table_name_shewan_index_basic_60min_df = 'shenwan_index_basic_60min_df'

        self.table_name_index_constituent_stock_df = 'index_constituent_stock_df'
        self.table_name_shenwan_index_constituent_stock_df = 'shenwan_index_constituent_stock_df'
        self.table_name_indicator = 'indicator_df'
        self.table_name_balance = 'balance_df'
        self.table_name_income = 'income_df'
        self.table_name_cashflow = 'cashflow_df'

        self.table_name_sw_l1 = 'sw_l1'
        self.table_name_sw_l2 = 'sw_l2'
        self.table_name_sw_l3 = 'sw_l3'

        self.table_name_index_daily = 'download_index_daily_df'
        self.table_name_shenwan_daily = 'download_shenwan_daily_df'
        self.table_name_shenwan_stock_industry_df = 'shenwan_stock_industry_df'



        self.index_mapping = {
            "上证50":    "000016.XSHG",
            "沪深300":   "000300.XSHG",
            "科创50":    "000688.XSHG", 
            "中证1000":  "000852.XSHG",  
            "中证500":   "000905.XSHG",   
            "中证800":   "000906.XSHG",  
            "中证全指":  "000985.XSHG", 
            "创业板指":  "399006.XSHE" 
        }
        self.code_to_csi = {
            "000016.XSHG": "csi50",  # 上证50
            "000300.XSHG": "csi300",  # 沪深300
            "000688.XSHG": "csi50kechuang",  # 科创50
            "000852.XSHG": "csi1000",  # 中证1000
            "000905.XSHG": "csi500",  # 中证500
            "000906.XSHG": "csi800",  # 中证800
            "000985.XSHG": "csiall",  # 中证全指
            "399006.XSHE": "csigg",  # 创业板指
        }
        self.init_jointquant()

        self.db = ClickHouseDB()
        if not self.db.connect_flag:
            print("数据库连接失败")
            return

    
    def init_jointquant(self):
        config = self._load_config()
        jq_username = config.get('jq_username', '') if config else ''
        jq_password = config.get('jq_password', '') if config else ''
        if not jq_username or not jq_password:
            raise ValueError("JointQuant用户名或密码未配置")
        try:
            jq.auth(jq_username, jq_password)
            print("聚宽数据登录成功")
        except Exception as e:
            raise ValueError(f"聚宽数据登录失败：{str(e)}")
            
        

    def _load_config(self):      
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                # 获取clickhouse部分的配置
                if 'jointquant' in config_data:
                    return config_data['jointquant']
                else:
                    return config_data
        except Exception as e:
            print(f"加载配置文件 {self.config_path} 时出错: {e}")
            return {}

    
    def download_A_basic(self,start_date_str,end_date_str):
        # 统一日期格式为datetime
        if isinstance(start_date_str, str):
            start_date_str = datetime.strptime(start_date_str, '%Y-%m-%d')
        if isinstance(end_date_str, str):
            end_date_str = datetime.strptime(end_date_str, '%Y-%m-%d')

        all_stocks = self.get_all_stocks()[:3] # 获取所有股票代码

        index_codes = list(self.index_mapping.values()) # 获取所有指数代码

        shenwan_index_codes = self._get_all_shenwan_index_codes() # 获取申万指数代码

        #获取申万指数代码

        if len(all_stocks) == 0:
            print("无有效成分股数据，程序终止")
            return
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            # 提交任务
            basic_df_result = executor.submit(self._get_daily, all_stocks, start_date_str, end_date_str,'daily',self.table_name_A_basic_df,['trade_date','code'])# 股票基础数据 ok
            basic_df_1min_result = executor.submit(self._get_daily, all_stocks, start_date_str, end_date_str,'1m',self.table_name_A_basic_1min_df,['trade_date','code'])# 股票1分钟级数据 ok
            basic_df_5min_result = executor.submit(self._get_daily, all_stocks, start_date_str, end_date_str,'5m',self.table_name_A_basic_5min_df,['trade_date','code'])# 股票5分钟级数据 ok
            basic_df_15min_result = executor.submit(self._get_daily, all_stocks, start_date_str, end_date_str,'15m',self.table_name_A_basic_15min_df,['trade_date','code'])# 股票15分钟级数据 ok
            basic_df_30min_result = executor.submit(self._get_daily, all_stocks, start_date_str, end_date_str,'30m',self.table_name_A_basic_30min_df,['trade_date','code'])# 股票30分钟级数据 ok
            basic_df_60min_result = executor.submit(self._get_daily, all_stocks, start_date_str, end_date_str,'60m',self.table_name_A_basic_60min_df,['trade_date','code'])# 股票60分钟级数据  ok

            basic_df_daily_result = executor.submit(self._get_daily_basic, "stock", all_stocks, start_date_str, end_date_str,self.table_name_A_daily_basic_df,['trade_date','code'])# 股票日线basic数据 ok
            index_basic_df_result = executor.submit(self._get_daily_basic, "index", index_codes, start_date_str, end_date_str,self.table_name_index_daily_basic_df,['trade_date','code'])# 指数日线basic数据 ok


            index_daily_df_result = executor.submit(self._get_daily,index_codes, start_date_str, end_date_str,'daily',self.table_name_index_daily_df,['trade_date','code']) # 指数日线行情  ok
            index_daily_1min_df_result = executor.submit(self._get_daily,index_codes, start_date_str, end_date_str,'1m',self.table_name_index_daily_1min_df,['trade_date','code']) # 指数1分钟级数据 ok
            index_daily_5min_df_result = executor.submit(self._get_daily,index_codes, start_date_str, end_date_str,'5m',self.table_name_index_daily_5min_df,['trade_date','code']) # 指数5分钟级数据 ok
            index_daily_15min_df_result = executor.submit(self._get_daily,index_codes, start_date_str, end_date_str,'15m',self.table_name_index_daily_15min_df,['trade_date','code']) # 指数15分钟级数据 ok
            index_daily_30min_df_result = executor.submit(self._get_daily,index_codes, start_date_str, end_date_str,'30m',self.table_name_index_daily_30min_df,['trade_date','code']) # 指数30分钟级数据 ok
            index_daily_60min_df_result = executor.submit(self._get_daily,index_codes, start_date_str, end_date_str,'60m',self.table_name_index_daily_60min_df,['trade_date','code']) # 指数60分钟级数据   ok

            index_constituent_stock_df_result = executor.submit(self._get_index_constituent_stock, start_date_str, end_date_str) # 指数成分股  ok
            # shewan_index_constituent_stock_df_result = executor.submit(self._get_shenwan_index_constituent_stock, shenwan_index_codes,start_date_str, end_date_str) # 申万指数成分股
            shenwan_classify_df_result = executor.submit(self._get_shenwan_classify) # 申万分类数据，行业分类，股票-行业分类 ok

            fundamentals_df_result = executor.submit(self._get_history_fundamentals, all_stocks, start_date_str,end_date_str,"indicator",self.table_name_indicator,["code","statDate","pubDate"])# indicator 数据  ok
            balance_df_result = executor.submit(self._get_history_fundamentals, all_stocks, start_date_str,end_date_str,"balance",self.table_name_balance,["code","statDate","pubDate"])# 财务数据  ok
            income_df_result = executor.submit(self._get_history_fundamentals, all_stocks, start_date_str,end_date_str,"income",self.table_name_income,["code","statDate","pubDate"])# 财务数据  ok
            cashflow_df_result = executor.submit(self._get_history_fundamentals, all_stocks, start_date_str,end_date_str,"cashflow",self.table_name_cashflow,["code","statDate","pubDate"])# 财务数据  ok

            # 获取结果

    def _get_daily(self, stock_list, start_date_str, end_date_str, frequency_str, table_name, primary_key_list):
        basic_all = []
        basic_df = pd.DataFrame()
        max_retry = self.MAX_RETRY  # 最大重试次数
        # 根据频率确定批次大小
        if frequency_str == 'daily':
            batch_days = 5000  # 5000条，5000天
        elif frequency_str == '1m':
            batch_days = 18  # 18天 
        elif frequency_str == '5m':
            batch_days = 80  # 80天
        elif frequency_str == '15m':
            batch_days = 200  # 10天
        elif frequency_str == '30m':
            batch_days = 400  # 400天
        elif frequency_str == '60m':
            batch_days = 800  # 800天
        else:
            raise ValueError(f"不支持的频率: {frequency_str}")
        for i, stock_code in enumerate(tqdm(stock_list, desc=f'获取日线数据{frequency_str}'), 1):
            retry_count = 0
            success = False
            tmp = None
            retry_delay = 1  # 每只股票重置初始重试间隔为1秒
            # 将日期范围拆分为小批次
            current_start = pd.to_datetime(start_date_str)
            end_date = pd.to_datetime(end_date_str)
            while current_start <= end_date:
                # 计算当前批次的结束日期
                current_end = min(current_start + pd.DateOffset(days=batch_days), end_date)
                current_start_str = current_start.strftime('%Y-%m-%d')
                current_end_str = current_end.strftime('%Y-%m-%d')
                retry_count = 0  # 每个批次重置重试计数
                success = False
                while retry_count < max_retry and not success:
                    try:
                        # 查询该股票在当前批次日期范围内的数据
                        tmp = jq.get_price(security=stock_code, start_date=current_start_str, end_date=current_end_str, frequency=frequency_str, skip_paused=False, fq='post', count=None, round=True)
                        success = True  # 成功获取，退出重试循环
                        if len(tmp) > 0:
                            tmp['code'] = stock_code
                            tmp = tmp.reset_index(names='trade_date')
                            if frequency_str == 'daily':
                                tmp['trade_date'] = tmp['trade_date'].dt.strftime('%Y-%m-%d')
                            else:
                                tmp['trade_date'] = pd.to_datetime(tmp['trade_date']).dt.tz_localize(None)
                            self.save_data_to_database_with_threadpool(tmp, primary_key_list, table_name)
                            basic_all.append(tmp)
                        else:
                            print(f"daily_basic_ 未查询到股票{stock_code}在{current_start_str}至{current_end_str}的数据")
                        
                    # 只捕获网络相关异常，非网络异常直接跳过重试
                    except Exception as e:
                        retry_count += 1
                        if retry_count < max_retry:
                            print(f"_get_daily 获取{stock_code}在{current_start_str}至{current_end_str}的数据失败（网络错误）：{str(e)[:50]}... 第{retry_count}次重试，等待{retry_delay}秒")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # 指数退避，间隔翻倍
                        else:
                            print(f"_get_daily 获取{stock_code}在{current_start_str}至{current_end_str}的数据失败：{str(e)[:50]}... 已重试{max_retry}次，跳过该批次")
                # 移动到下一个批次
                current_start = current_end + pd.DateOffset(days=1)
                # 重置延迟时间
                retry_delay = 1
        # 合并结果
        if basic_all:
            basic_all = [df for df in basic_all if not df.empty]
            basic_df = pd.concat(basic_all, ignore_index=True)
            print(f"\n共获取{len(basic_df)}条数据")
        return basic_df

    def _get_history_fundamentals(self, stock_list, stat_date_str, end_date_str, table_name_jq, table_name_local,primary_key_list):
        exclude_fields = ['id', 'day']  # 可根据需求扩展
        table_info = jq.get_table_info(table_name_jq)
        field_names = table_info['name_en'].tolist()
        field_names = [f for f in field_names if f not in exclude_fields]
        field_names = [f"{table_name_jq}.{f}" for f in field_names]
        max_retry = self.MAX_RETRY  # 最大重试次数
    
        start_year = stat_date_str.year
        end_year = end_date_str.year
        years = list(range(start_year, end_year + 1))  # 按年遍历的列表
        
        for i, stock_code in enumerate(tqdm(stock_list, desc=f'获取财务数据{table_name_jq}'), 1):
            stock_data_list = []  # 存储单只股票所有年份的数据
            retry_delay = 1       # 每只股票重置初始重试间隔为1秒
            
            # 按年份循环拉取
            for year in years:
                retry_count = 0
                success = False
                tmp = None
                while retry_count < max_retry and not success:
                    try:
                        # 构造当年的查询日期（如2023年查2023-12-31）
                        year_watch_date = f"{year}-12-31"
                        # 查询该股票当年的财务数据（按季度拉取，count=4表示4个季度）
                        tmp = jq.get_history_fundamentals(
                            security=stock_code,
                            watch_date=year_watch_date,  # 按年份查询
                            fields=field_names,
                            count=4,                     # 拉取当年4个季度数据
                            interval='1q',
                            stat_by_year=False
                        )
                        success = True  # 成功获取，退出重试循环
                        if not tmp.empty:
                            stock_data_list.append(tmp)
                            self.save_data_to_database_with_threadpool(tmp, primary_key_list, table_name_local)
                        else:
                            print(f"未查询到股票{stock_code} {year}年的{table_name_jq}数据")
                            
                    except (NameResolutionError, MaxRetryError, ConnectionError, TimeoutError,ConnectionResetError, requests.exceptions.RequestException) as e:
                        retry_count += 1
                        if retry_count < max_retry:
                            print(f"获取{stock_code} {year}年{table_name_jq}数据失败（网络错误）：{str(e)[:50]}... 第{retry_count}次重试，等待{retry_delay}秒")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # 指数退避，间隔翻倍
                        else:
                            print(f"获取{stock_code} {year}年{table_name_jq}数据失败：{str(e)[:50]}... 已重试{max_retry}次，跳过该年份数据")
                            break
                    except Exception as e:
                        print(f"获取{stock_code} {year}年{table_name_jq}数据失败（非网络错误）：{e}，跳过该年份数据")
                        break  # 跳出while重试循环
            if stock_data_list:
                stock_data = pd.concat(stock_data_list, ignore_index=True)
            else:
                print(f"_get_history_fundamentals 未查询到股票{stock_code} {start_year}-{end_year}年的{table_name_jq}数据")

    def _get_daily_basic(self, stock_type:str, stock_list, start_date_str, end_date_str, table_name, primary_key_list):
        basic_all = []
        basic_df = pd.DataFrame()
        max_retry = self.MAX_RETRY  # 最大重试次数
        for i, stock_code in enumerate(tqdm(stock_list, desc='获取日线数据'), 1):
            retry_count = 0
            success = False
            tmp = None
            retry_delay = 1  # 每只股票重置初始重试间隔为1秒
            # 将日期范围拆分为小批次
            current_start = pd.to_datetime(start_date_str)
            end_date = pd.to_datetime(end_date_str)
            while current_start <= end_date:
                # 计算当前批次的结束日期
                current_end = min(current_start + pd.DateOffset(days=90), end_date)
                current_start_str = current_start.strftime('%Y-%m-%d')
                current_end_str = current_end.strftime('%Y-%m-%d')
                retry_count = 0  # 每个批次重置重试计数
                success = False
                while retry_count < max_retry and not success:
                    try:
                        # 查询该股票在当前批次日期范围内的数据
                        if stock_type =="stock":
                            tmp = jq.get_valuation(security_list=stock_code, start_date=current_start_str, end_date=current_end_str)
                        elif stock_type == "index":
                            tmp = jq.get_index_valuation(security_list=stock_code, start_date=current_start_str, end_date=current_end_str)
                        else:
                            raise ValueError(f"_get_daily_basic 未知的股票类型{stock_type}")
                        tmp.rename(columns={"day": "trade_date"}, inplace=True)
                        success = True  # 成功获取，退出重试循环
                        if len(tmp) > 0:
                            self.save_data_to_database_with_threadpool(tmp, primary_key_list, table_name)
                            basic_all.append(tmp)
                        else:
                            print(f"daily_basic_ 未查询到股票{stock_code}在{current_start_str}至{current_end_str}的数据")
                    # 只捕获网络相关异常，非网络异常直接跳过重试
                    except (NameResolutionError, MaxRetryError, ConnectionError, TimeoutError, ConnectionResetError, requests.exceptions.RequestException) as e:
                        retry_count += 1
                        if retry_count < max_retry:
                            print(f"_get_daily_basic 获取{stock_code}在{current_start_str}至{current_end_str}的数据失败（网络错误）：{str(e)[:50]}... 第{retry_count}次重试，等待{retry_delay}秒")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # 指数退避，间隔翻倍
                        else:
                            print(f"_get_daily_basic 获取{stock_code}在{current_start_str}至{current_end_str}的数据失败：{str(e)[:50]}... 已重试{max_retry}次，跳过该批次")
                    except Exception as e:
                        print(f"_get_daily_basic 获取{stock_code}在{current_start_str}至{current_end_str}的数据失败（非网络错误）：{e}，跳过该批次")
                        break  # 跳出while重试循环，进入下一批次
                # 移动到下一个批次
                current_start = current_end + pd.DateOffset(days=1)
                # 重置延迟时间
                retry_delay = 1
        # 合并结果
        if basic_all:
            basic_all = [df for df in basic_all if not df.empty]
            basic_df = pd.concat(basic_all, ignore_index=True)
            print(f"\n共获取{len(basic_df)}条数据")
        return basic_df


    def _get_index_constituent_stock(self,start_date_str,end_date_str):
        index_mapping = self.index_mapping
        code_to_csi = self.code_to_csi
        os.makedirs(self.save_dir_index_weight, exist_ok=True)
        for index_code in  tqdm(index_mapping.values(), desc='获取各种指数成分股'):
            result_df = self._get_index_stocks(index_code,start_date_str,end_date_str)

            self.save_data_to_database_with_threadpool(result_df,['index_code','code','date'],self.table_name_index_constituent_stock_df)
            
            # 指数列表成分股单独转到json，方便O(1)读取使用
            result_df = result_df.reset_index(drop=True)
            new_data = self._utils_convert_to_json(result_df)
            sorted_data = dict(sorted(new_data.items()))
            csi_name = code_to_csi[index_code]
            # 保存 JSON 文件
            filepath = os.path.join(self.save_dir_index_weight, f"{csi_name}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(sorted_data, f, ensure_ascii=False, indent=2)
            print(f"已保存: {filepath}")
        

    def _get_index_stocks(self, index_code, start_date_str, end_date_str):  # 获取指数成分股
        """
        获取指定指数在指定日期范围内的成分股
        :param index_code: 指数代码
        :param start_date_str: 开始日期，格式如'2020-01-01'
        :param end_date_str: 结束日期，格式如'2020-12-31'
        :return: 包含index_code, code, date列的DataFrame
        """
        # 参数验证
        if not all([index_code, start_date_str, end_date_str]):
            print("❌ 参数缺失，无法获取指数成分股数据")
            return pd.DataFrame(columns=['index_code', 'code', 'date'])
        current_start = pd.to_datetime(start_date_str)
        total_end = pd.to_datetime(end_date_str)
        index_list = []
        # 检查日期范围有效性
        if current_start > total_end:
            print("❌ 开始日期晚于结束日期")
            return pd.DataFrame(columns=['index_code', 'code', 'date'])
        # 生成日期范围，使用pandas的date_range更高效
        date_range = pd.date_range(start=current_start, end=total_end, freq='D')
        # 按天逐批遍历（核心：每天查询一个日期的数据）
        for current_date in tqdm(date_range, desc=f'获取指数 {index_code} 成分股'):
            curr_date_str = current_date.strftime('%Y%m%d')
            try:
                temp_list = jq.get_index_stocks(index_symbol=index_code, date=curr_date_str)
                if temp_list and len(temp_list) > 0:
                    # 批量添加数据，减少循环内操作
                    index_list.extend([
                        {'index_code': index_code, 'code': code, 'date': curr_date_str}
                        for code in temp_list  ])
            except Exception as e:
                print(f"❌ 获取{curr_date_str}指数列表失败：{e}")
        if index_list:
            index_df = pd.DataFrame(index_list)
            index_df['date'] = pd.to_datetime(index_df['date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
        else:
            index_df = pd.DataFrame(columns=['index_code', 'code', 'date'])            
        return index_df


    def _get_shenwan_index_constituent_stock(self, shenwan_index_codes, start_date_str, end_date_str):
        """
        批量获取申万指数成分股并保存到数据库
        :param shenwan_index_codes: 申万指数代码列表
        :param start_date_str: 开始日期，格式如'2020-01-01'
        :param end_date_str: 结束日期，格式如'2020-12-31'
        """
        # 参数验证
        if not shenwan_index_codes:
            print("❌ 未提供申万指数代码列表")
            return
        all_results = []
        success_count = 0
        
        for index_code in tqdm(shenwan_index_codes, desc='获取申万指数成分股'):
            result_df = self._get_index_stocks(index_code, start_date_str, end_date_str)
            if result_df is not None and not result_df.empty:
                all_results.append(result_df)
                success_count += 1
            else:
                print(f"⚠️ 指数 {index_code} 在 {start_date_str} 到 {end_date_str} 期间没有成分股数据")
        # 批量处理和保存数据，减少数据库连接次数
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            self.save_data_to_database_with_threadpool(
                combined_df,['index_code', 'code', 'date'],self.table_name_shenwan_index_constituent_stock_df)
        else:
            print(f"\n📊 未获取到任何申万指数成分股数据")

    def get_all_stocks(self):
        """
        获取所有股票列表
        """
        try:
            # 获取所有A股股票
            all_stocks = jq.get_all_securities(types=['stock'], date=None)
            # 重置索引，使股票代码成为普通列
            all_stocks_reset = all_stocks.reset_index()
            all_stocks_reset.columns = ['code', 'display_name', 'name', 'start_date', 'end_date', 'type']
            return all_stocks_reset['code'].tolist()
        except Exception as e:
            print(f"获取股票列表失败: {e}")
            return None
    
    def _get_all_shenwan_index_codes(self):
        """
        获取所有申万指数代码（一级+二级+三级），返回list格式
        :return: 所有申万指数代码列表 ['index_code1', 'index_code2', ...]
        """
        try:
            sw_l1 = jq.get_industries(name='sw_l1').reset_index(names='index_code')
            sw_l2 = jq.get_industries(name='sw_l2').reset_index(names='index_code')
            sw_l3 = jq.get_industries(name='sw_l3').reset_index(names='index_code')
            sw_l1_codes = sw_l1['index_code'].tolist()
            sw_l2_codes = sw_l2['index_code'].tolist()
            sw_l3_codes = sw_l3['index_code'].tolist()
            all_shenwan_index_codes = list(set(sw_l1_codes + sw_l2_codes + sw_l3_codes))
            return all_shenwan_index_codes
        except Exception as e:
            print(f"❌ 获取申万指数代码失败：{str(e)[:100]}")
            return []  # 异常时返回空列表，避免后续代码报错

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
        df_weight = self.convert_jq_code_suffix(df_weight)
        # 按日期分组
        result = {}
        for trade_date, group in df_weight.groupby('date'):
            stocks = sorted(group['code'].tolist())
            result[trade_date] = stocks
        return result
    
    def _get_shenwan_classify(self):
        """
        获取申万分类数据
        """
        sw_l1 = jq.get_industries(name='sw_l1').reset_index(names='index_code')
        sw_l2 = jq.get_industries(name='sw_l2').reset_index(names='index_code')
        sw_l3 = jq.get_industries(name='sw_l3').reset_index(names='index_code')
        # 获取所有股票代码
        all_stock_codes = self.get_all_stocks()
        stock_industry_dict = jq.get_industry(all_stock_codes)
        stock_industry_df = pd.json_normalize(stock_industry_dict.values()).assign(stock_code=stock_industry_dict.keys()).reset_index(drop=True)

        sw_l1['start_date'] = sw_l1['start_date'].dt.strftime('%Y-%m-%d')
        sw_l2['start_date'] = sw_l2['start_date'].dt.strftime('%Y-%m-%d')
        sw_l3['start_date'] = sw_l3['start_date'].dt.strftime('%Y-%m-%d')
        self.save_data_to_database_with_threadpool(sw_l1,["index_code"],self.table_name_sw_l1)
        self.save_data_to_database_with_threadpool(sw_l2,["index_code"],self.table_name_sw_l2)
        self.save_data_to_database_with_threadpool(sw_l3,["index_code"],self.table_name_sw_l3)
        self.save_data_to_database_with_threadpool(stock_industry_df,["stock_code"],self.table_name_shenwan_stock_industry_df)
        print("\n📊 申万分类数据获取完成")

    def _get_shenwan_index(self):
        """
        获取申万指数数据
        """
        all_market_indexes = jq.get_all_securities(types=['index'])  # 无参数=返回全市场所有指数
        shenwan_index = all_market_indexes[all_market_indexes.index.str.startswith('80')]
        return shenwan_index
    
    def convert_jq_code_suffix(self,input_data, col_name: str = 'code', with_dot: bool = True):
        """
        转换聚宽（JoinQuant）股票代码后缀：.XSHG→sh/.sh，.XSHE→sz/.sz
        :param input_data: 输入数据，支持3种类型：
                        - 单个字符串（如'600000.XSHG'）
                        - pandas Series（如df['code']）
                        - pandas DataFrame（需指定col_name）
        :param col_name: 仅DataFrame时生效，指定要处理的列名，默认'code'
        :param with_dot: 后缀是否带点，默认False（如600000sh）；True则为600000.sh
        :return: 处理后的结果（同输入类型：字符串/Series/DataFrame）
        :raises TypeError: 输入类型不支持时抛出异常
        """
        # 定义替换规则（根据with_dot调整）
        replace_rules = {
            '.XSHG': '.SH' if with_dot else 'SH',
            '.XSHE': '.SZ' if with_dot else 'SZ'
        }
        
        # 处理单个字符串
        if isinstance(input_data, str):
            code = input_data
            for old_suffix, new_suffix in replace_rules.items():
                code = code.replace(old_suffix, new_suffix)
            return code
        
        # 处理pandas Series
        elif isinstance(input_data, pd.Series):
            series_data = input_data.copy()  # 避免修改原数据
            for old_suffix, new_suffix in replace_rules.items():
                # regex=False：精准匹配字符串，避免.被正则解析
                series_data = series_data.str.replace(old_suffix, new_suffix, regex=False)
            return series_data
        
        # 处理pandas DataFrame
        elif isinstance(input_data, pd.DataFrame):
            df_data = input_data.copy()  # 避免修改原数据
            if col_name not in df_data.columns:
                raise ValueError(f"DataFrame中不存在列名：{col_name}，请检查col_name参数")
            
            # 批量替换指定列的后缀
            for old_suffix, new_suffix in replace_rules.items():
                df_data[col_name] = df_data[col_name].str.replace(old_suffix, new_suffix, regex=False)
            return df_data
    
        # 不支持的输入类型
        else:
            raise TypeError(
                f"不支持的输入类型：{type(input_data)}！\n"
                "请输入以下类型：单个字符串 / pandas Series / pandas DataFrame"
            )

    def save_data_to_database_with_threadpool_old(self, data_df, primary_keys, table_name, max_workers=40, operation_type='update'):
        """
        使用线程池将数据保存到数据库
        :param data_df: 要保存的数据框
        :param primary_keys_map: 主键列表
        :param table_name: 表名
        :param max_workers: 线程池最大工作线程数
        :param operation_type: 操作类型 ('update', 'replace', 'append')
        """
        def save_single_table(df, primary_keys, table_name):
            """单个表的保存函数"""
            if df is not None and not df.empty:
                try:
                    # 使用save_df_to_clickhouse方法保存数据到ClickHouse
                    success = self.db.save_df_to_clickhouse(
                        df=df,
                        table_name=table_name,
                        primary_keys=primary_keys,
                        operation_type=operation_type
                    )
                    if success:
                        return True
                    else:
                        return False
                except Exception as e:
                    return False
            else:
                return False

        # 使用线程池执行数据库保存操作
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future = executor.submit(save_single_table, data_df, primary_keys, table_name)
            try:
                result = future.result()
                if result is False:
                    print(f"❌ 数据库保存失败！{table_name} 表")
            except Exception as e:
                print(f"❌ 处理 {table_name} 表时发生异常: {e}")

    def save_data_to_database_with_threadpool(self, data_df, primary_keys, table_name, 
                                            max_workers=10, operation_type='update', 
                                            batch_size=10000, show_progress=False):
        """
        使用线程池将数据保存到数据库，支持大数据集分批并行处理
        :param data_df: 要保存的数据框
        :param primary_keys: 主键列表
        :param table_name: 表名
        :param max_workers: 线程池最大工作线程数
        :param operation_type: 操作类型 ('update', 'replace', 'append')
        :param batch_size: 每批处理的数据量
        :param show_progress: 是否显示进度条
        :return: 总体成功状态
        """
        if data_df is None or data_df.empty:
            return True

        def save_single_batch(batch_df, batch_index):
            """
            保存单个数据批次到数据库
            :param batch_df: 批次数据
            :param batch_index: 批次索引
            :return: (批次索引, 成功状态, 数据行数)
            """
            try:
                if batch_df is not None and not batch_df.empty:
                    # 使用save_df_to_clickhouse方法保存数据到ClickHouse
                    success = self.db.save_df_to_clickhouse(
                        df=batch_df,
                        table_name=table_name,
                        primary_keys=primary_keys,
                        operation_type=operation_type
                    )
                    return (batch_index, success, len(batch_df))
                else:
                    return (batch_index, True, 0)  # 空数据批次视为成功
            except Exception as e:
                print(f"❌ 批次 {batch_index} 保存失败: {str(e)[:100]}...")
                return (batch_index, False, len(batch_df))

        # 计算批次数量
        total_rows = len(data_df)
        num_batches = (total_rows + batch_size - 1) // batch_size  # 向上取整

        # 分割数据为多个批次
        batches = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_rows)
            batch_df = data_df.iloc[start_idx:end_idx].copy()
            batches.append((batch_df, i))

        # 使用线程池执行数据库保存操作
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有批次任务
            future_to_batch = {
                executor.submit(save_single_batch, batch_df, batch_idx): batch_idx
                for batch_df, batch_idx in batches
            }

            # 处理完成的任务
            if show_progress:
                progress_bar = tqdm(total=num_batches, desc=f"💾 保存到 {table_name}", unit="批")

            for future in as_completed(future_to_batch):
                batch_result = future.result()
                results.append(batch_result)
                if show_progress:
                    progress_bar.update(1)

            if show_progress:
                progress_bar.close()

        # 统计结果
        total_success = sum(1 for r in results if r[1])
        total_failed = num_batches - total_success
        total_saved_rows = sum(r[2] for r in results if r[1])

        # 如果有失败批次，尝试重新保存失败的批次
        if total_failed > 0:
            print(f"🔄 尝试重新保存 {total_failed} 个失败的批次...")
            failed_batches = [(batches[batch_idx][0], batch_idx) for batch_idx, success, _ in results if not success]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                retry_futures = {
                    executor.submit(save_single_batch, batch_df, batch_idx): batch_idx
                    for batch_df, batch_idx in failed_batches
                }

                retry_results = []
                if show_progress:
                    retry_progress = tqdm(total=len(failed_batches), desc=f"🔄 重试保存到 {table_name}", unit="批")

                for future in as_completed(retry_futures):
                    retry_result = future.result()
                    retry_results.append(retry_result)
                    if show_progress:
                        retry_progress.update(1)

                if show_progress:
                    retry_progress.close()

            # 统计重试结果
            retry_success = sum(1 for r in retry_results if r[1])
            retry_failed = len(failed_batches) - retry_success
            retry_saved_rows = sum(r[2] for r in retry_results if r[1])

            print(f"\n🔄 重试结果统计 ({table_name}):")
            print(f"   重试批次: {len(failed_batches)}")
            print(f"   重试成功: {retry_success}")
            print(f"   重试失败: {retry_failed}")
            print(f"   重试成功保存行数: {retry_saved_rows}")

            # 更新总统计
            total_success += retry_success
            total_saved_rows += retry_saved_rows

        # 最终结果判断
        if total_success == num_batches:
            return True
        else:
            print(f"⚠️ 部分批次数据保存失败，请检查日志。")
            return False


    def update_A_basic(self, start_date_str, end_date_str):
        pass



