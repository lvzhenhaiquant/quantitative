"""
数据加载工具
负责从 Parquet 文件加载股票数据、指数数据等
"""

import json
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Union, Dict
import yaml
from .logger import get_logger

# 设置日志
logger = get_logger("data.DataLoader")


class DataLoader:
    """
    Parquet 数据加载器

    功能：
    1. 加载个股价格数据
    2. 加载基准指数数据
    3. 加载市值数据
    4. 加载行业分类数据
    5. 加载股票池（instruments）
    """

    # 指数代码映射 (小写 -> 大写)
    INDEX_CODE_MAP = {
        'sh000300': 'SH000300',  # 沪深300
        'sh000905': 'SH000905',  # 中证500
        'sh000852': 'SH000852',  # 中证1000
        'sh000985': 'SH000985',  # 中证全指
    }

    # 股票池名称映射
    POOL_MAP = {
        'csi1000': 'csi1000',
        'csi500': 'csi500',
        'csi300': 'csi300',
        'csiall': 'csiall',
    }

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

        # 数据路径
        self.data_path = Path(self.config['data']['raw_data_path'])
        self.daily_path = self.data_path / 'daily'
        self.basic_path = self.data_path / 'basic'
        self.adj_path = self.data_path / 'adj'
        self.index_daily_path = self.data_path / 'index_daily'
        self.index_weight_path = self.data_path / 'index_weight'

        logger.info(f"DataLoader initialized with path: {self.data_path}")

        # 缓存
        self._pool_cache: Dict[str, Dict] = {}
        self._available_stocks = None

        # 加载申万行业数据
        self.industry_data = self._load_industry_data()

    def _get_available_stocks(self) -> set:
        """获取所有可用的股票代码"""
        if self._available_stocks is None:
            stocks = set()
            for f in self.daily_path.glob('*.parquet'):
                stock = f.stem  # SZ000001
                stocks.add(stock)
            self._available_stocks = stocks
        return self._available_stocks

    def _load_industry_data(self) -> pd.DataFrame:
        """加载申万行业分类数据"""
        shenwan_path = self.config['data']['shenwan_path']

        try:
            industry_df = pd.read_csv(shenwan_path)

            # 转换股票代码格式：000001.SZ → SZ000001
            def convert_code(ts_code):
                if '.' in ts_code:
                    code, market = ts_code.split('.')
                    return f"{market}{code}"
                return ts_code

            industry_df['code'] = industry_df['ts_code'].apply(convert_code)
            industry_df.set_index('code', inplace=True)

            logger.info(f"Loaded industry data for {len(industry_df)} stocks")
            return industry_df

        except Exception as e:
            logger.warning(f"Failed to load industry data: {e}")
            return pd.DataFrame()

    def _load_single_stock(
        self,
        stock: str,
        source_path: Path,
        columns: List[str],
        start_date: str,
        end_date: str
    ) -> Optional[pl.DataFrame]:
        """加载单只股票的数据"""
        filepath = source_path / f"{stock}.parquet"

        if not filepath.exists():
            return None

        try:
            cols_to_read = ['ts_code', 'trade_date'] + columns
            df = pl.read_parquet(filepath, columns=cols_to_read)

            # 处理日期格式
            if df['trade_date'].dtype == pl.Utf8:
                # 检查日期格式
                sample = df['trade_date'][0]
                if '-' in sample:
                    # 格式: YYYY-MM-DD
                    df = df.filter(
                        (pl.col('trade_date') >= start_date) &
                        (pl.col('trade_date') <= end_date)
                    )
                    df = df.with_columns(
                        pl.col('trade_date').str.to_date('%Y-%m-%d').alias('date')
                    )
                else:
                    # 格式: YYYYMMDD
                    start_str = start_date.replace('-', '')
                    end_str = end_date.replace('-', '')
                    df = df.filter(
                        (pl.col('trade_date') >= start_str) &
                        (pl.col('trade_date') <= end_str)
                    )
                    df = df.with_columns(
                        pl.col('trade_date').str.to_date('%Y%m%d').alias('date')
                    )
            else:
                df = df.with_columns(
                    pl.col('trade_date').cast(pl.Date).alias('date')
                )
                start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
                end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
                df = df.filter(
                    (pl.col('date') >= start_dt) &
                    (pl.col('date') <= end_dt)
                )

            df = df.with_columns(pl.col('ts_code').alias('stock'))
            df = df.drop(['ts_code', 'trade_date'])

            return df if len(df) > 0 else None

        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")
            return None

    def load_stock_prices(
        self,
        stock_list: List[str],
        start_date: str,
        end_date: str,
        fields: List[str] = None,
        lookback_days: int = 0,
        adjust: bool = True
    ) -> pd.DataFrame:
        """
        加载个股价格数据

        Parameters
        ----------
        stock_list : list
            股票代码列表（如 ['SZ000001', 'SH600000'] 或 ['sz000001', 'sh600000']）
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
            actual_start = pd.to_datetime(start_date) - timedelta(days=lookback_days * 2)
            actual_start = actual_start.strftime('%Y-%m-%d')
        else:
            actual_start = start_date

        logger.info(f"Loading stock prices: {len(stock_list)} stocks, {actual_start} ~ {end_date}")

        # 统一股票代码为大写
        stock_list = [s.upper() for s in stock_list]

        # 过滤出实际有数据的股票
        available = self._get_available_stocks()
        stock_list = [s for s in stock_list if s in available]

        if not stock_list:
            logger.warning("No valid stocks found")
            return pd.DataFrame()

        # 字段映射
        field_map = {
            '$open': 'open', '$high': 'high', '$low': 'low',
            '$close': 'close', '$volume': 'volume', '$amount': 'amount',
            '$pre_close': 'pre_close', '$change': 'change', '$pct_chg': 'pct_chg',
        }

        # 确定需要加载的列
        daily_cols = [field_map[f] for f in fields if f in field_map]

        need_adjust = adjust and any(f in ['$open', '$high', '$low', '$close'] for f in fields)

        # 加载数据
        all_dfs = []

        for stock in stock_list:
            # 加载日线数据
            df = self._load_single_stock(stock, self.daily_path, daily_cols, actual_start, end_date)

            if df is None:
                continue

            # 加载复权因子
            if need_adjust:
                df_adj = self._load_single_stock(stock, self.adj_path, ['factor'], actual_start, end_date)
                if df_adj is not None:
                    df = df.join(df_adj.select(['stock', 'date', 'factor']), on=['stock', 'date'], how='left')

            all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame()

        result = pl.concat(all_dfs)

        # 后复权处理
        if need_adjust and 'factor' in result.columns:
            result = result.with_columns(
                pl.col('factor').fill_null(strategy='forward').over('stock')
            )

            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in result.columns:
                    result = result.with_columns(
                        (pl.col(col) * pl.col('factor')).alias(col)
                    )

            result = result.drop('factor')
            logger.info("已应用后复权")

        # 转换为 pandas DataFrame with MultiIndex
        df_pd = result.to_pandas()

        # 重命名列，添加 $ 前缀
        rename_map = {v: k for k, v in field_map.items() if v in df_pd.columns}
        df_pd.rename(columns=rename_map, inplace=True)

        # 转换股票代码为小写 (兼容原有格式)
        df_pd['stock'] = df_pd['stock'].str.lower()

        # 设置 MultiIndex
        df_pd.set_index(['stock', 'date'], inplace=True)

        logger.info(f"Loaded {len(df_pd)} records")
        return df_pd

    def load_benchmark_prices(
        self,
        benchmark: str,
        start_date: str,
        end_date: str,
        fields: List[str] = None,
        lookback_days: int = 0
    ) -> pd.Series:
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

        # 转换指数代码
        if benchmark.lower() in self.INDEX_CODE_MAP:
            index_code = self.INDEX_CODE_MAP[benchmark.lower()]
        else:
            index_code = benchmark.upper()

        filepath = self.index_daily_path / f"{index_code}.parquet"

        if not filepath.exists():
            logger.error(f"Benchmark file not found: {filepath}")
            raise FileNotFoundError(f"Benchmark file not found: {filepath}")

        try:
            df = pl.read_parquet(filepath, columns=['trade_date', 'close'])

            # 处理日期格式
            if df['trade_date'].dtype == pl.Utf8:
                sample = df['trade_date'][0]
                if '-' in sample:
                    # 格式: YYYY-MM-DD
                    df = df.filter(
                        (pl.col('trade_date') >= actual_start) &
                        (pl.col('trade_date') <= end_date)
                    )
                    df = df.with_columns(
                        pl.col('trade_date').str.to_date('%Y-%m-%d').alias('date')
                    )
                else:
                    # 格式: YYYYMMDD
                    start_str = actual_start.replace('-', '')
                    end_str = end_date.replace('-', '')
                    df = df.filter(
                        (pl.col('trade_date') >= start_str) &
                        (pl.col('trade_date') <= end_str)
                    )
                    df = df.with_columns(
                        pl.col('trade_date').str.to_date('%Y%m%d').alias('date')
                    )
            else:
                df = df.with_columns(
                    pl.col('trade_date').cast(pl.Date).alias('date')
                )
                start_dt = datetime.strptime(actual_start, '%Y-%m-%d').date()
                end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
                df = df.filter(
                    (pl.col('date') >= start_dt) &
                    (pl.col('date') <= end_dt)
                )

            df = df.sort('date')
            df_pd = df.to_pandas()

            # 返回 Series，index 为 datetime
            prices = pd.Series(df_pd['close'].values, index=pd.to_datetime(df_pd['date']))
            prices.name = fields[0]

            logger.info(f"Loaded {len(prices)} benchmark records")
            return prices

        except Exception as e:
            logger.error(f"Error loading benchmark prices: {e}")
            raise

    def load_market_cap(
        self,
        stock_list: List[str],
        start_date: str,
        end_date: str,
        cap_type: str = 'total_mv'
    ) -> pd.DataFrame:
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
            市值数据，MultiIndex (instrument, datetime)
        """
        logger.info(f"Loading market cap ({cap_type}): {len(stock_list)} stocks")

        # 统一股票代码为大写
        stock_list = [s.upper() for s in stock_list]

        all_dfs = []

        for stock in stock_list:
            df = self._load_single_stock(stock, self.basic_path, [cap_type], start_date, end_date)
            if df is not None:
                all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame()

        result = pl.concat(all_dfs)
        df_pd = result.to_pandas()

        # 重命名列
        df_pd.rename(columns={cap_type: f'${cap_type}'}, inplace=True)

        # 转换股票代码为小写
        df_pd['stock'] = df_pd['stock'].str.lower()

        # 设置 MultiIndex
        df_pd.set_index(['stock', 'date'], inplace=True)

        return df_pd

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
            股票池数据，从 JSON 生成伪 DataFrame
            columns=['code', 'start_date', 'end_date']
        """
        pool_name = self.POOL_MAP.get(instrument_name, instrument_name)
        json_file = self.index_weight_path / f"{pool_name}.json"

        logger.info(f"Loading instrument: {instrument_name} from {json_file}")

        if not json_file.exists():
            logger.error(f"Instrument file not found: {json_file}")
            raise FileNotFoundError(f"Instrument file not found: {json_file}")

        # 加载 JSON
        if pool_name not in self._pool_cache:
            with open(json_file, 'r') as f:
                self._pool_cache[pool_name] = json.load(f)

        pool_data = self._pool_cache[pool_name]

        # 从 JSON 构建股票的起止日期
        # {date: [stocks]} -> {stock: [dates]}
        stock_dates = {}
        for date_str, stocks in pool_data.items():
            for stock in stocks:
                if stock not in stock_dates:
                    stock_dates[stock] = []
                stock_dates[stock].append(date_str)

        # 构建 DataFrame
        records = []
        for stock, dates in stock_dates.items():
            sorted_dates = sorted(dates)
            # 转换股票代码为小写 (兼容原有格式)
            records.append({
                'code': stock.lower(),
                'start_date': sorted_dates[0],
                'end_date': sorted_dates[-1]
            })

        df = pd.DataFrame(records)

        logger.info(f"Loaded {len(df)} stocks from {instrument_name}")
        return df

    def get_stock_list_by_date(
        self,
        instrument_name: str,
        target_date: str
    ) -> List[str]:
        """
        获取指定日期的股票池成分股

        Parameters
        ----------
        instrument_name : str
            股票池名称
        target_date : str
            目标日期 'YYYY-MM-DD'

        Returns
        -------
        list
            股票代码列表 (小写格式)
        """
        pool_name = self.POOL_MAP.get(instrument_name, instrument_name)
        json_file = self.index_weight_path / f"{pool_name}.json"

        if not json_file.exists():
            logger.warning(f"Instrument file not found: {json_file}")
            return []

        # 加载 JSON
        if pool_name not in self._pool_cache:
            with open(json_file, 'r') as f:
                self._pool_cache[pool_name] = json.load(f)

        pool_data = self._pool_cache[pool_name]

        # 直接查找日期
        if target_date in pool_data:
            stocks = pool_data[target_date]
        else:
            # 找最近的之前的日期
            dates = sorted(pool_data.keys())
            stocks = []
            for d in reversed(dates):
                if d <= target_date:
                    stocks = pool_data[d]
                    break

        # 转换为小写 (兼容原有格式)
        stock_list = [s.lower() for s in stocks]

        logger.info(f"Found {len(stock_list)} stocks in {instrument_name} at {target_date}")
        return stock_list

    def get_industry(self, stock_code: str, level: int = 1) -> str:
        """
        获取股票的行业分类

        Parameters
        ----------
        stock_code : str
            股票代码
        level : int, optional
            行业等级：1（一级）、2（二级）、3（三级）

        Returns
        -------
        str
            行业名称
        """
        level_map = {1: 'l1_name', 2: 'l2_name', 3: 'l3_name'}
        col = level_map[level]

        # 尝试大写和小写
        codes_to_try = [stock_code, stock_code.upper(), stock_code.lower()]

        for code in codes_to_try:
            if code in self.industry_data.index:
                return self.industry_data.loc[code, col]

        logger.warning(f"Industry not found for {stock_code}")
        return "Unknown"

    def __repr__(self):
        return f"DataLoader(data_path={self.data_path})"
