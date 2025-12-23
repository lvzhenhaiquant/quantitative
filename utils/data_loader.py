"""
数据加载工具
负责从QLib加载股票数据、指数数据等
"""

import qlib
from qlib.data import D
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Union
import yaml
from .logger import get_logger

# 设置日志
logger = get_logger("data.DataLoader")


class DataLoader:
    """
    QLib数据加载器

    功能：
    1. 加载个股价格数据
    2. 加载基准指数数据
    3. 加载市值数据
    4. 加载行业分类数据
    5. 加载股票池（instruments）
    """

    def __init__(self, config_path: str = None):
        """
        初始化数据加载器

        Parameters
        ----------
        config_path : str, optional
            配置文件路径，默认读取 configs/data_config.yaml
        """
        # 加载配置
        if config_path is None:
            config_path = "/home/zhenhai1/quantitative/configs/data_config.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 初始化QLib
        qlib_path = self.config['data']['qlib_data_path']
        qlib.init(provider_uri=qlib_path, region='cn')

        logger.info(f"QLib initialized with path: {qlib_path}")

        # 加载申万行业数据
        self.industry_data = self._load_industry_data()

    def _load_industry_data(self) -> pd.DataFrame:
        """加载申万行业分类数据"""
        shenwan_path = self.config['data']['shenwan_path']
        industry_df = pd.read_csv(shenwan_path)

        # 转换股票代码格式：000001.SZ → sz000001
        def convert_code(ts_code):
            if '.' in ts_code:
                code, market = ts_code.split('.')
                market = market.lower()
                return f"{market}{code}"
            return ts_code

        industry_df['code'] = industry_df['ts_code'].apply(convert_code)
        industry_df.set_index('code', inplace=True)

        logger.info(f"Loaded industry data for {len(industry_df)} stocks")
        return industry_df

    def load_stock_prices(self,
                          stock_list: List[str],
                          start_date: str,
                          end_date: str,
                          fields: List[str] = None,
                          lookback_days: int = 0,
                          adjust: bool = True) -> pd.DataFrame:
        """
        加载个股价格数据

        Parameters
        ----------
        stock_list : list
            股票代码列表（QLib格式，如 ['sz000001', 'sh600000']）
        start_date : str
            开始日期 'YYYY-MM-DD'
        end_date : str
            结束日期 'YYYY-MM-DD'
        fields : list, optional
            需要加载的字段，默认为 ['$close']
        lookback_days : int, optional
            向前回看的天数，默认为0
        adjust : bool, optional
            是否后复权，默认为 True

        Returns
        -------
        pd.DataFrame
            价格数据，MultiIndex (instrument, datetime)
        """
        if fields is None:
            fields = ['$close']

        # 计算实际的开始日期（考虑lookback）
        if lookback_days > 0:
            actual_start = pd.to_datetime(start_date) - timedelta(days=lookback_days * 2)  # *2 考虑节假日
            actual_start = actual_start.strftime('%Y-%m-%d')
        else:
            actual_start = start_date

        logger.info(f"Loading stock prices: {len(stock_list)} stocks, {actual_start} ~ {end_date}")

        # 如果需要复权，自动加载 $factor
        fields_to_load = list(fields)
        if adjust and '$factor' not in fields_to_load:
            fields_to_load.append('$factor')

        try:
            df = D.features(
                stock_list,
                fields_to_load,
                start_time=actual_start,
                end_time=end_date,
                freq='day'
            )

            # 后复权处理
            if adjust and '$factor' in df.columns:
                price_fields = ['$open', '$high', '$low', '$close', '$vwap']
                for field in price_fields:
                    if field in df.columns:
                        df[field] = df[field] * df['$factor']

                # 如果用户没有请求 $factor，删除它
                if '$factor' not in fields:
                    df.drop(columns=['$factor'], inplace=True)

                logger.info(f"已应用后复权")

            logger.info(f"Loaded {len(df)} records")
            return df

        except Exception as e:
            logger.error(f"Error loading stock prices: {e}")
            raise

    def load_benchmark_prices(self,
                              benchmark: str,
                              start_date: str,
                              end_date: str,
                              fields: List[str] = None,
                              lookback_days: int = 0) -> pd.Series:
        """
        加载基准指数价格数据

        Parameters
        ----------
        benchmark : str
            基准指数代码（如 'sh000300'）
        start_date : str
            开始日期
        end_date : str
            结束日期
        fields : list, optional
            字段列表，默认为 ['$close']
        lookback_days : int, optional
            向前回看天数

        Returns
        -------
        pd.Series
            指数收盘价序列，index为datetime
        """
        if fields is None:
            fields = ['$close']

        # 计算实际的开始日期
        if lookback_days > 0:
            actual_start = pd.to_datetime(start_date) - timedelta(days=lookback_days * 2)
            actual_start = actual_start.strftime('%Y-%m-%d')
        else:
            actual_start = start_date

        logger.info(f"Loading benchmark {benchmark}: {actual_start} ~ {end_date}")

        try:
            df = D.features(
                [benchmark],
                fields,
                start_time=actual_start,
                end_time=end_date,
                freq='day'
            )

            # 提取为Series（去掉instrument层级）
            prices = df.loc[benchmark, fields[0]]

            logger.info(f"Loaded {len(prices)} benchmark records")
            return prices

        except Exception as e:
            logger.error(f"Error loading benchmark prices: {e}")
            raise

    def load_market_cap(self,
                        stock_list: List[str],
                        start_date: str,
                        end_date: str,
                        cap_type: str = 'total_mv') -> pd.DataFrame:
        """
        加载市值数据

        Parameters
        ----------
        stock_list : list
            股票代码列表
        start_date : str
            开始日期
        end_date : str
            结束日期
        cap_type : str, optional
            市值类型：'total_mv'（总市值）或 'circ_mv'（流通市值）

        Returns
        -------
        pd.DataFrame
            市值数据
        """
        field = f'${cap_type}'

        logger.info(f"Loading market cap ({cap_type}): {len(stock_list)} stocks")

        try:
            df = D.features(
                stock_list,
                [field],
                start_time=start_date,
                end_time=end_date,
                freq='day'
            )

            return df

        except Exception as e:
            logger.error(f"Error loading market cap: {e}")
            raise

    def load_instruments(self, instrument_name: str = 'csi1000') -> pd.DataFrame:
        """
        加载股票池（instruments）

        Parameters
        ----------
        instrument_name : str, optional
            股票池名称，如 'csi1000', 'csi300' 等

        Returns
        -------
        pd.DataFrame
            股票池数据，columns=['code', 'start_date', 'end_date']
        """
        inst_path = self.config['data']['instruments'][instrument_name]

        logger.info(f"Loading instrument: {instrument_name} from {inst_path}")

        try:
            # 读取instruments文件（格式：code  start_date  end_date）
            df = pd.read_csv(inst_path, sep='\t', header=None, names=['code', 'start_date', 'end_date'])

            logger.info(f"Loaded {len(df)} stocks from {instrument_name}")
            return df

        except Exception as e:
            logger.error(f"Error loading instruments: {e}")
            raise

    def get_stock_list_by_date(self,
                                instrument_name: str,
                                target_date: str) -> List[str]:
        """
        获取指定日期的股票池成分股

        Parameters
        ----------
        instrument_name : str
            股票池名称
        target_date : str
            目标日期

        Returns
        -------
        list
            股票代码列表
        """
        df = self.load_instruments(instrument_name)

        # 筛选在target_date有效的股票
        target_date = pd.to_datetime(target_date)
        valid_stocks = df[
            (pd.to_datetime(df['start_date']) <= target_date) &
            (pd.to_datetime(df['end_date']) >= target_date)
        ]

        stock_list = valid_stocks['code'].tolist()

        logger.info(f"Found {len(stock_list)} stocks in {instrument_name} at {target_date}")

        return stock_list

    def get_industry(self, stock_code: str, level: int = 1) -> str:
        """
        获取股票的行业分类

        Parameters
        ----------
        stock_code : str
            股票代码（QLib格式）
        level : int, optional
            行业等级：1（一级）、2（二级）、3（三级）

        Returns
        -------
        str
            行业名称
        """
        level_map = {1: 'l1_name', 2: 'l2_name', 3: 'l3_name'}
        col = level_map[level]

        if stock_code in self.industry_data.index:
            return self.industry_data.loc[stock_code, col]
        else:
            logger.warning(f"Industry not found for {stock_code}")
            return "Unknown"

    def __repr__(self):
        return f"DataLoader(qlib_path={self.config['data']['qlib_data_path']})"
