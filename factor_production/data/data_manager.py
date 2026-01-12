"""
数据管理器
从 Parquet 文件加载数据，转换为 Polars DataFrame
"""
import json
import polars as pl
import pandas as pd
from typing import Union, List, Optional, Set
from pathlib import Path
from datetime import datetime


class DataManager:
    """
    统一数据加载接口

    - 从 Parquet 文件加载各类数据
    - 支持股票池名称或股票列表
    - 输出 Polars DataFrame
    """

    # 股票池名称映射
    POOL_MAP = {
        'csi1000': 'csi1000',
        'csi500': 'csi500',
        'csi300': 'csi300',
        'csiall': 'csiall',
        'all': 'csiall',
        '中证1000': 'csi1000',
        '中证500': 'csi500',
        '沪深300': 'csi300',
        '全A': 'csiall',
    }

    # 字段来源映射
    # daily: 日线行情 (open, high, low, close, volume, amount)
    # basic: 每日指标 (pe_ttm, pb, turnover_rate_f, total_mv, circ_mv, ...)
    # adj: 复权因子 (factor)
    FIELD_SOURCE = {
        # 价格字段 -> daily
        '$open': ('daily', 'open'),
        '$high': ('daily', 'high'),
        '$low': ('daily', 'low'),
        '$close': ('daily', 'close'),
        '$volume': ('daily', 'volume'),
        '$amount': ('daily', 'amount'),
        '$pre_close': ('daily', 'pre_close'),
        '$change': ('daily', 'change'),
        '$pct_chg': ('daily', 'pct_chg'),
        # 复权因子
        '$factor': ('adj', 'factor'),
        # 估值指标 -> basic
        '$pe': ('basic', 'pe'),
        '$pe_ttm': ('basic', 'pe_ttm'),
        '$pb': ('basic', 'pb'),
        '$ps': ('basic', 'ps'),
        '$ps_ttm': ('basic', 'ps_ttm'),
        '$dv_ratio': ('basic', 'dv_ratio'),
        '$dv_ttm': ('basic', 'dv_ttm'),
        # 换手率
        '$turnover_rate': ('basic', 'turnover_rate'),
        '$turnover_rate_f': ('basic', 'turnover_rate_f'),
        '$volume_ratio': ('basic', 'volume_ratio'),
        # 市值
        '$total_mv': ('basic', 'total_mv'),
        '$circ_mv': ('basic', 'circ_mv'),
        # 股本
        '$total_share': ('basic', 'total_share'),
        '$float_share': ('basic', 'float_share'),
        '$free_share': ('basic', 'free_share'),
    }

    # 指数代码映射 (QLib格式 -> Tushare格式)
    INDEX_CODE_MAP = {
        'sh000300': 'SH000300',  # 沪深300
        'sh000905': 'SH000905',  # 中证500
        'sh000852': 'SH000852',  # 中证1000
        'sh000985': 'SH000985',  # 中证全指
    }

    def __init__(self, data_path: str = None):
        """
        初始化数据管理器

        Args:
            data_path: 数据根目录，默认为 download_data
        """
        if data_path is None:
            data_path = '/home/yunbo/project/quantitative/data/download_data'

        self.data_path = Path(data_path)
        self.daily_path = self.data_path / 'daily'
        self.basic_path = self.data_path / 'basic'
        self.adj_path = self.data_path / 'adj'
        self.index_daily_path = self.data_path / 'index_daily'
        self.index_weight_path = self.data_path / 'index_weight'

        # 缓存股票池数据
        self._pool_cache = {}

        # 缓存可用股票列表
        self._available_stocks = None

        print(f"DataManager 初始化完成，数据路径: {data_path}")

    def _get_available_stocks(self) -> Set[str]:
        """获取所有可用的股票代码"""
        if self._available_stocks is None:
            stocks = set()
            for f in self.daily_path.glob('*.parquet'):
                # SZ000001.parquet -> SZ000001
                stock = f.stem
                stocks.add(stock)
            self._available_stocks = stocks
        return self._available_stocks

    def _convert_stock_code(self, code: str, to_upper: bool = True) -> str:
        """
        转换股票代码格式

        Args:
            code: 股票代码
            to_upper: True -> SZ000001, False -> sz000001
        """
        if to_upper:
            return code.upper()
        else:
            return code.lower()

    def _load_parquet_for_stocks(
        self,
        stocks: List[str],
        source_path: Path,
        columns: List[str],
        start: str,
        end: str
    ) -> pl.DataFrame:
        """
        加载多只股票的 parquet 数据

        Args:
            stocks: 股票代码列表 (大写格式 SZ000001)
            source_path: 数据目录
            columns: 需要的列
            start: 开始日期 'YYYY-MM-DD'
            end: 结束日期 'YYYY-MM-DD'

        Returns:
            合并后的 DataFrame
        """
        all_dfs = []

        for stock in stocks:
            filepath = source_path / f"{stock}.parquet"

            if not filepath.exists():
                continue

            try:
                # 只读取需要的列
                cols_to_read = ['ts_code', 'trade_date'] + columns
                df = pl.read_parquet(filepath, columns=cols_to_read)

                # 筛选日期范围
                # trade_date 可能是 str (YYYY-MM-DD 或 YYYYMMDD) 或 datetime
                if df['trade_date'].dtype == pl.Utf8:
                    # 检查日期格式
                    sample = df['trade_date'][0]
                    if '-' in sample:
                        # 格式: YYYY-MM-DD
                        df = df.filter(
                            (pl.col('trade_date') >= start) &
                            (pl.col('trade_date') <= end)
                        )
                        df = df.with_columns(
                            pl.col('trade_date').str.to_date('%Y-%m-%d').alias('date')
                        )
                    else:
                        # 格式: YYYYMMDD
                        start_compact = start.replace('-', '')
                        end_compact = end.replace('-', '')
                        df = df.filter(
                            (pl.col('trade_date') >= start_compact) &
                            (pl.col('trade_date') <= end_compact)
                        )
                        df = df.with_columns(
                            pl.col('trade_date').str.to_date('%Y%m%d').alias('date')
                        )
                else:
                    df = df.with_columns(
                        pl.col('trade_date').cast(pl.Date).alias('date')
                    )
                    start_date = datetime.strptime(start, '%Y-%m-%d').date()
                    end_date = datetime.strptime(end, '%Y-%m-%d').date()
                    df = df.filter(
                        (pl.col('date') >= start_date) &
                        (pl.col('date') <= end_date)
                    )

                # 添加股票代码列
                df = df.with_columns(
                    pl.col('ts_code').alias('stock')
                )

                # 删除原始列
                df = df.drop(['ts_code', 'trade_date'])

                if len(df) > 0:
                    all_dfs.append(df)

            except Exception as e:
                print(f"警告: 加载 {filepath} 失败: {e}")
                continue

        if not all_dfs:
            return pl.DataFrame()

        return pl.concat(all_dfs)

    def load(
        self,
        stocks: Union[str, List[str]],
        start: str,
        end: str,
        fields: List[str],
        adjust: bool = True
    ) -> pl.DataFrame:
        """
        加载数据

        Args:
            stocks: 股票池名称 ('csi1000', 'csi500', 'csi300' 等)
                    或股票列表 ['SZ000001', 'SH600000', ...]
            start: 开始日期 'YYYY-MM-DD'
            end: 结束日期 'YYYY-MM-DD'
            fields: 字段列表，如 ['$close', '$turnover_rate_f', '$pe_ttm']
            adjust: 是否后复权，默认为 True

        Returns:
            Polars DataFrame，含 ['stock', 'date'] + 字段列
        """
        # 1. 解析股票列表
        if isinstance(stocks, str):
            stock_list = self.get_pool_stocks(stocks, start, end)
            print(f"股票池 {stocks}: {len(stock_list)} 只股票 ({start} ~ {end})")
        else:
            # 统一转换为大写
            stock_list = [self._convert_stock_code(s, to_upper=True) for s in stocks]

        if len(stock_list) == 0:
            print("警告: 股票列表为空")
            return pl.DataFrame()

        # 过滤出实际有数据的股票
        available = self._get_available_stocks()
        stock_list = [s for s in stock_list if s in available]

        if len(stock_list) == 0:
            print("警告: 没有找到任何股票数据")
            return pl.DataFrame()

        # 2. 分析需要加载的数据源
        need_adjust = adjust and any(f in ['$open', '$high', '$low', '$close', '$vwap'] for f in fields)

        # 按数据源分组字段
        source_fields = {'daily': [], 'basic': [], 'adj': []}

        for field in fields:
            if field in self.FIELD_SOURCE:
                source, col = self.FIELD_SOURCE[field]
                source_fields[source].append(col)

        # 如果需要复权，添加 factor
        if need_adjust:
            source_fields['adj'].append('factor')

        # 3. 加载各数据源
        result_df = None

        # 加载 daily 数据
        if source_fields['daily']:
            df_daily = self._load_parquet_for_stocks(
                stock_list, self.daily_path, source_fields['daily'], start, end
            )
            if not df_daily.is_empty():
                result_df = df_daily

        # 加载 basic 数据
        if source_fields['basic']:
            df_basic = self._load_parquet_for_stocks(
                stock_list, self.basic_path, source_fields['basic'], start, end
            )
            if not df_basic.is_empty():
                if result_df is None:
                    result_df = df_basic
                else:
                    result_df = result_df.join(
                        df_basic, on=['stock', 'date'], how='left'
                    )

        # 加载 adj 数据
        if source_fields['adj']:
            df_adj = self._load_parquet_for_stocks(
                stock_list, self.adj_path, source_fields['adj'], start, end
            )
            if not df_adj.is_empty():
                if result_df is None:
                    result_df = df_adj
                else:
                    result_df = result_df.join(
                        df_adj, on=['stock', 'date'], how='left'
                    )

        if result_df is None or result_df.is_empty():
            print("警告: 加载的数据为空")
            return pl.DataFrame()

        # 4. 后复权处理
        if need_adjust and 'factor' in result_df.columns:
            # 前向填充 factor
            result_df = result_df.with_columns(
                pl.col('factor').fill_null(strategy='forward').over('stock')
            )

            # 对价格字段应用复权
            price_cols = ['open', 'high', 'low', 'close', 'vwap']
            for col in price_cols:
                if col in result_df.columns:
                    result_df = result_df.with_columns(
                        (pl.col(col) * pl.col('factor')).alias(col)
                    )

            # 如果用户没有请求 factor，删除它
            if '$factor' not in fields:
                result_df = result_df.drop('factor')

            print("已应用后复权")

        # 5. 排序
        result_df = result_df.sort(['stock', 'date'])

        # 6. 去重（按 stock + date 唯一，保留第一条）
        before_dedup = result_df.shape[0]
        result_df = result_df.unique(subset=['stock', 'date'], keep='first')
        after_dedup = result_df.shape[0]
        if before_dedup != after_dedup:
            print(f"去重: {before_dedup} -> {after_dedup} (删除 {before_dedup - after_dedup} 条重复)")

        print(f"数据加载完成: {result_df.shape[0]} 行, {result_df.shape[1]} 列")

        return result_df

    def get_pool_stocks(self, pool: str, start: str, end: str) -> List[str]:
        """
        获取某段时间内所有曾为成分股的股票

        Args:
            pool: 股票池名称 ('csi1000', 'csi500', 'csi300' 等)
            start: 开始日期 'YYYY-MM-DD'
            end: 结束日期 'YYYY-MM-DD'

        Returns:
            股票代码列表 (大写格式 SZ000001)
        """
        # 映射股票池名称
        pool_name = self.POOL_MAP.get(pool, pool)

        # 检查缓存
        cache_key = pool_name
        if cache_key not in self._pool_cache:
            # 加载 JSON 文件
            json_file = self.index_weight_path / f"{pool_name}.json"

            if not json_file.exists():
                print(f"警告: 股票池文件不存在 {json_file}")
                return []

            with open(json_file, 'r') as f:
                self._pool_cache[cache_key] = json.load(f)

        pool_data = self._pool_cache[cache_key]
        all_dates = sorted(pool_data.keys())

        # 收集时间范围内的所有股票
        stocks = set()

        # 找范围内的日期
        for date_str in all_dates:
            if start <= date_str <= end:
                stocks.update(pool_data[date_str])

        # 如果范围内没有数据，使用最近的之前的日期
        if not stocks:
            for date_str in reversed(all_dates):
                if date_str <= end:
                    stocks.update(pool_data[date_str])
                    break

        return sorted(list(stocks))

    def get_pool_stocks_by_date(self, pool: str, date: str) -> List[str]:
        """
        获取指定日期的成分股

        Args:
            pool: 股票池名称
            date: 日期 'YYYY-MM-DD'

        Returns:
            股票代码列表
        """
        pool_name = self.POOL_MAP.get(pool, pool)

        # 检查缓存
        cache_key = pool_name
        if cache_key not in self._pool_cache:
            json_file = self.index_weight_path / f"{pool_name}.json"

            if not json_file.exists():
                print(f"警告: 股票池文件不存在 {json_file}")
                return []

            with open(json_file, 'r') as f:
                self._pool_cache[cache_key] = json.load(f)

        pool_data = self._pool_cache[cache_key]

        # 查找指定日期或最近的日期
        if date in pool_data:
            return pool_data[date]

        # 如果没有精确匹配，找最近的之前的日期
        dates = sorted(pool_data.keys())
        for d in reversed(dates):
            if d <= date:
                return pool_data[d]

        return []

    def get_trading_days(self, start: str, end: str) -> List[str]:
        """
        获取交易日历

        Args:
            start: 开始日期
            end: 结束日期

        Returns:
            交易日列表
        """
        # 从实际数据中获取交易日
        # 读取一个指数的日期作为交易日历
        index_file = self.index_daily_path / 'SH000300.parquet'

        if not index_file.exists():
            print(f"警告: 指数数据文件不存在 {index_file}")
            return []

        df = pl.read_parquet(index_file, columns=['trade_date'])

        # 处理日期格式
        if df['trade_date'].dtype == pl.Utf8:
            df = df.with_columns(
                pl.col('trade_date').str.to_date('%Y%m%d').alias('date')
            )
        else:
            df = df.with_columns(
                pl.col('trade_date').cast(pl.Date).alias('date')
            )

        # 筛选日期范围
        start_date = datetime.strptime(start, '%Y-%m-%d').date()
        end_date = datetime.strptime(end, '%Y-%m-%d').date()

        df = df.filter(
            (pl.col('date') >= start_date) &
            (pl.col('date') <= end_date)
        )

        # 排序并返回
        dates = df.sort('date')['date'].to_list()

        return [d.strftime('%Y-%m-%d') for d in dates]

    def load_benchmark(
        self,
        code: str,
        start: str,
        end: str
    ) -> pl.DataFrame:
        """
        加载基准指数数据

        Args:
            code: 指数代码，如 'sh000852' (中证1000)
            start: 开始日期
            end: 结束日期

        Returns:
            Polars DataFrame [stock, date, close]
        """
        # 转换指数代码格式
        if code.lower() in self.INDEX_CODE_MAP:
            index_code = self.INDEX_CODE_MAP[code.lower()]
        else:
            # 尝试直接转换 sh000300 -> SH000300
            index_code = code.upper()

        filepath = self.index_daily_path / f"{index_code}.parquet"

        if not filepath.exists():
            print(f"警告: 指数数据文件不存在 {filepath}")
            return pl.DataFrame()

        df = pl.read_parquet(filepath, columns=['ts_code', 'trade_date', 'close'])

        # 处理日期格式
        if df['trade_date'].dtype == pl.Utf8:
            df = df.with_columns(
                pl.col('trade_date').str.to_date('%Y%m%d').alias('date')
            )
        else:
            df = df.with_columns(
                pl.col('trade_date').cast(pl.Date).alias('date')
            )

        # 筛选日期范围
        start_date = datetime.strptime(start, '%Y-%m-%d').date()
        end_date = datetime.strptime(end, '%Y-%m-%d').date()

        df = df.filter(
            (pl.col('date') >= start_date) &
            (pl.col('date') <= end_date)
        )

        # 重命名列
        df = df.select([
            pl.col('ts_code').alias('stock'),
            pl.col('date'),
            pl.col('close')
        ])

        return df.sort('date')

    def load_financial(
        self,
        stocks: Union[str, List[str]],
        start: str,
        end: str,
        fields: List[str]
    ) -> pl.DataFrame:
        """
        加载财报数据（多进程并行读取 + 向量化前向填充）

        数据来源:
        - income/: 利润表字段 (non_oper_income, non_oper_exp, total_profit, ...)
        - fina_indicator/: 财务指标 (netprofit_yoy, roe, ...)

        Args:
            stocks: 股票池名称或股票列表
            start: 开始日期 'YYYY-MM-DD'
            end: 结束日期 'YYYY-MM-DD'
            fields: 财务字段列表，如 ['non_oper_income', 'total_profit']

        Returns:
            Polars DataFrame，含 ['stock', 'date'] + 字段列（日频，已前向填充）
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        import multiprocessing as mp

        # 字段来源映射
        FIELD_SOURCE = {
            # 利润表字段 (income)
            'non_oper_income': 'income',
            'non_oper_exp': 'income',
            'total_profit': 'income',
            'n_income': 'income',
            'revenue': 'income',
            'total_cogs': 'income',
            'operate_profit': 'income',
            'total_revenue': 'income',
            # 财务指标 (fina_indicator)
            'netprofit_yoy': 'fina_indicator',
            'or_yoy': 'fina_indicator',
            'roe': 'fina_indicator',
            'roa': 'fina_indicator',
            'grossprofit_margin': 'fina_indicator',
            'debt_to_assets': 'fina_indicator',
        }

        # 解析股票列表
        if isinstance(stocks, str):
            stock_list = set(self.get_pool_stocks(stocks, start, end))
        else:
            stock_list = set([self._convert_stock_code(s, to_upper=True) for s in stocks])

        # 获取交易日历
        trade_dates_df = self._get_trade_dates_df(start, end)

        if trade_dates_df.is_empty():
            print("警告: 无法获取交易日历")
            return pl.DataFrame()

        # 按来源分组字段
        source_fields = {}
        for field in fields:
            source = FIELD_SOURCE.get(field, 'income')
            if source not in source_fields:
                source_fields[source] = []
            source_fields[source].append(field)

        all_data = []
        n_workers = mp.cpu_count()

        # 从每个数据源并行读取
        for source, src_fields in source_fields.items():
            data_dir = self.data_path / source

            if not data_dir.exists():
                print(f"警告: 数据目录不存在 {data_dir}")
                continue

            # 只读取股票池中的文件
            files_to_read = [data_dir / f"{stock}.parquet" for stock in stock_list]
            files_to_read = [f for f in files_to_read if f.exists()]

            print(f"加载 {source} 数据 ({len(files_to_read)} 个文件, {n_workers} 核心)...")

            # 定义读取单个文件的函数
            def read_single_file(filepath):
                try:
                    df = pl.read_parquet(filepath)
                    stock = filepath.stem  # SZ000001

                    if 'ann_date' not in df.columns:
                        return None

                    # 选择需要的列
                    cols = ['ann_date'] + [f for f in src_fields if f in df.columns]
                    if len(cols) <= 1:
                        return None

                    df = df.select(cols)
                    df = df.with_columns(pl.lit(stock).alias('stock'))
                    return df
                except:
                    return None

            # 多线程并行读取
            dfs = []
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(read_single_file, f): f for f in files_to_read}
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc=f"  读取 {source}", ncols=80):
                    result = future.result()
                    if result is not None and len(result) > 0:
                        dfs.append(result)

            if not dfs:
                continue

            # 合并所有数据
            df = pl.concat(dfs)

            # 转换公告日期格式
            sample = str(df['ann_date'][0])
            if '-' in sample:
                df = df.with_columns(pl.col('ann_date').str.to_date('%Y-%m-%d'))
            else:
                df = df.with_columns(pl.col('ann_date').str.to_date('%Y%m%d'))

            # 向量化前向填充到日频
            available_fields = [f for f in src_fields if f in df.columns]
            df_daily = self._forward_fill_financial_vectorized(
                df, trade_dates_df, available_fields
            )

            if len(df_daily) > 0:
                all_data.append(df_daily)

            print(f"  {source}: {df_daily['stock'].n_unique()} 只股票, {len(df_daily)} 行")

        if not all_data:
            print("警告: 无财报数据")
            return pl.DataFrame()

        # 合并所有数据
        result = pl.concat(all_data)

        # 如果有多个来源的字段，需要按 stock + date 合并
        if len(source_fields) > 1:
            result = result.group_by(['stock', 'date']).agg(
                [pl.col(f).first() for f in fields if f in result.columns]
            )

        result = result.sort(['stock', 'date'])

        print(f"财报数据加载完成: {len(result)} 行, {result['stock'].n_unique()} 只股票")

        return result

    def _get_trade_dates_df(self, start: str, end: str) -> pl.DataFrame:
        """获取交易日 DataFrame"""
        # 从 daily 目录读取一个文件获取交易日
        daily_dir = self.data_path / 'daily'
        sample_file = next(daily_dir.glob('*.parquet'), None)

        if sample_file is None:
            return pl.DataFrame()

        df = pl.read_parquet(sample_file)

        if 'trade_date' in df.columns:
            date_col = 'trade_date'
        else:
            date_col = 'date'

        # 获取唯一日期
        dates_df = df.select(pl.col(date_col).alias('date')).unique()

        # 转换日期格式
        sample = str(dates_df['date'][0])
        if isinstance(dates_df['date'][0], str):
            if '-' in sample:
                dates_df = dates_df.with_columns(pl.col('date').str.to_date('%Y-%m-%d'))
            else:
                dates_df = dates_df.with_columns(pl.col('date').str.to_date('%Y%m%d'))

        # 筛选日期范围
        start_date = pl.lit(start).str.to_date('%Y-%m-%d')
        end_date = pl.lit(end).str.to_date('%Y-%m-%d')

        dates_df = dates_df.filter(
            (pl.col('date') >= start_date) & (pl.col('date') <= end_date)
        ).sort('date')

        return dates_df

    def _forward_fill_financial_vectorized(
        self,
        df_report: pl.DataFrame,
        trade_dates_df: pl.DataFrame,
        fields: List[str]
    ) -> pl.DataFrame:
        """
        向量化前向填充财报数据到日频（利用 Polars join_asof）

        逻辑: 财报在公告日才公开，使用 asof join 找到每个交易日之前最近的公告
        """
        import warnings

        # 去重：同一股票同一公告日保留最新的
        df_report = df_report.sort(['stock', 'ann_date']).unique(
            subset=['stock', 'ann_date'], keep='last'
        )

        # 为每只股票创建交易日框架
        stocks = df_report['stock'].unique()
        daily_frame = stocks.to_frame().join(trade_dates_df, how='cross')
        daily_frame = daily_frame.sort(['stock', 'date'])

        # 右表按 stock + ann_date 排序
        df_report = df_report.sort(['stock', 'ann_date'])

        # 使用 asof join
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = daily_frame.join_asof(
                df_report,
                left_on='date',
                right_on='ann_date',
                by='stock',
                strategy='backward'
            )

        # 删除没有匹配到的行
        result = result.drop_nulls(subset=fields)

        # 只保留需要的列
        result = result.select(['stock', 'date'] + fields)

        return result
