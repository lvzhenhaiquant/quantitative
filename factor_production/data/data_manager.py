"""
数据管理器
从 QLib 加载数据，转换为 Polars DataFrame
"""
import polars as pl
import pandas as pd
import qlib
from qlib.data import D
from typing import Union, List
from pathlib import Path


class DataManager:
    """
    统一数据加载接口

    - 从 QLib 加载各类数据
    - 支持股票池名称或股票列表
    - 输出 Polars DataFrame
    """

    # 股票池名称映射
    POOL_MAP = {
        'csi1000': 'csi1000',
        'csi500': 'csi500',
        'csi300': 'csi300',
        'csi800': 'csi800',
        'csi50': 'csi50',
        '中证1000': 'csi1000',
        '中证500': 'csi500',
        '沪深300': 'csi300',
        '中证800': 'csi800',
        '上证50': 'csi50',
    }

    def __init__(self, qlib_path: str = None):
        """
        初始化数据管理器

        Args:
            qlib_path: QLib 数据路径，默认从配置文件读取
        """
        if qlib_path is None:
            qlib_path = '/home/zhenhai1/quantitative/qlib_data/cn_data'

        # 初始化 QLib
        qlib.init(provider_uri=qlib_path, region='cn')
        self.qlib_path = qlib_path
        self.instruments_path = Path(qlib_path) / 'instruments'

        print(f"DataManager 初始化完成，QLib路径: {qlib_path}")

    def load(self,
             stocks: Union[str, List[str]],
             start: str,
             end: str,
             fields: List[str],
             adjust: bool = True) -> pl.DataFrame:
        """
        加载数据

        Args:
            stocks: 股票池名称 ('csi1000', 'csi500', 'csi300' 等)
                    或股票列表 ['sz000001', 'sh600000', ...]
            start: 开始日期 'YYYY-MM-DD'
            end: 结束日期 'YYYY-MM-DD'
            fields: 字段列表，如 ['$close', '$turnover_rate_f', '$pe_ttm']
            adjust: 是否后复权，默认为 True

        Returns:
            Polars DataFrame，含 ['date', 'stock'] + 字段列
        """
        # 1. 解析股票列表
        if isinstance(stocks, str):
            stock_list = self.get_pool_stocks(stocks, start, end)
            print(f"股票池 {stocks}: {len(stock_list)} 只股票 ({start} ~ {end})")
        else:
            stock_list = stocks

        if len(stock_list) == 0:
            print("警告: 股票列表为空")
            return pl.DataFrame()

        # 2. 如果需要复权，自动加载 $factor
        fields_to_load = list(fields)
        if adjust and '$factor' not in fields_to_load:
            fields_to_load.append('$factor')

        # 3. 从 QLib 加载数据
        try:
            df_pd = D.features(
                instruments=stock_list,
                fields=fields_to_load,
                start_time=start,
                end_time=end,
                freq='day'
            )
        except Exception as e:
            print(f"QLib 数据加载失败: {e}")
            return pl.DataFrame()

        if df_pd is None or len(df_pd) == 0:
            print("警告: 加载的数据为空")
            return pl.DataFrame()

        # 4. 后复权处理
        if adjust and '$factor' in df_pd.columns:
            price_fields = ['$open', '$high', '$low', '$close', '$vwap']
            for field in price_fields:
                if field in df_pd.columns:
                    df_pd[field] = df_pd[field] * df_pd['$factor']

            # 如果用户没有请求 $factor，删除它
            if '$factor' not in fields:
                df_pd.drop(columns=['$factor'], inplace=True)
                fields_to_load.remove('$factor')

            print("已应用后复权")

        # 5. 转换为 Polars DataFrame
        df_pd = df_pd.reset_index()
        # 使用实际加载的字段（可能包含或不包含 factor）
        output_fields = [f for f in fields_to_load if f in df_pd.columns or f.replace('$', '') in df_pd.columns]
        df_pd.columns = ['stock', 'date'] + [f.replace('$', '') for f in output_fields]

        df = pl.from_pandas(df_pd)

        # 6. 确保日期格式正确
        df = df.with_columns(
            pl.col('date').cast(pl.Date)
        )

        print(f"数据加载完成: {df.shape[0]} 行, {df.shape[1]} 列")

        return df

    def get_pool_stocks(self, pool: str, start: str, end: str) -> List[str]:
        """
        获取某段时间内所有曾为成分股的股票

        Args:
            pool: 股票池名称 ('csi1000', 'csi500', 'csi300' 等)
            start: 开始日期
            end: 结束日期

        Returns:
            股票代码列表
        """
        # 映射股票池名称
        pool_name = self.POOL_MAP.get(pool, pool)

        # 读取 instruments 文件
        inst_file = self.instruments_path / f"{pool_name}.txt"

        if not inst_file.exists():
            print(f"警告: 股票池文件不存在 {inst_file}")
            return []

        # 解析 instruments 文件
        # 格式: stock \t start_date \t end_date
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)

        stocks = set()

        with open(inst_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    stock = parts[0]
                    stock_start = pd.Timestamp(parts[1])
                    stock_end = pd.Timestamp(parts[2])

                    # 检查时间范围是否有交集
                    if stock_start <= end_dt and stock_end >= start_dt:
                        stocks.add(stock)

        return sorted(list(stocks))

    def get_trading_days(self, start: str, end: str) -> List[str]:
        """
        获取交易日历

        Args:
            start: 开始日期
            end: 结束日期

        Returns:
            交易日列表
        """
        calendar_file = Path(self.qlib_path) / 'calendars' / 'day.txt'

        if not calendar_file.exists():
            print(f"警告: 交易日历文件不存在 {calendar_file}")
            return []

        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)

        trading_days = []

        with open(calendar_file, 'r') as f:
            for line in f:
                day = pd.Timestamp(line.strip())
                if start_dt <= day <= end_dt:
                    trading_days.append(day.strftime('%Y-%m-%d'))

        return trading_days

    def load_benchmark(self,
                       code: str,
                       start: str,
                       end: str) -> pl.DataFrame:
        """
        加载基准指数数据

        Args:
            code: 指数代码，如 'sh000852' (中证1000)
            start: 开始日期
            end: 结束日期

        Returns:
            Polars DataFrame
        """
        try:
            df_pd = D.features(
                instruments=[code],
                fields=['$close'],
                start_time=start,
                end_time=end,
                freq='day'
            )
        except Exception as e:
            print(f"基准数据加载失败: {e}")
            return pl.DataFrame()

        if df_pd is None or len(df_pd) == 0:
            return pl.DataFrame()

        df_pd = df_pd.reset_index()
        df_pd.columns = ['stock', 'date', 'close']

        df = pl.from_pandas(df_pd)
        df = df.with_columns(pl.col('date').cast(pl.Date))

        return df

    def load_financial(self,
                       stocks: Union[str, List[str]],
                       start: str,
                       end: str,
                       fields: List[str]) -> pl.DataFrame:
        """
        加载财报数据（已前向填充到日频）

        Args:
            stocks: 股票池名称或股票列表
            start: 开始日期 'YYYY-MM-DD'
            end: 结束日期 'YYYY-MM-DD'
            fields: 财务字段列表，如 ['netprofit_yoy']

        Returns:
            Polars DataFrame，含 ['date', 'stock'] + 字段列
        """
        cache_dir = Path('/home/zhenhai1/quantitative/factor_production/cache/financial')

        # 解析股票列表
        if isinstance(stocks, str):
            stock_list = set(self.get_pool_stocks(stocks, start, end))
        else:
            stock_list = set(stocks)

        # 加载每个字段的 parquet 文件
        all_dfs = []

        for field in fields:
            filepath = cache_dir / f"{field}.parquet"

            if not filepath.exists():
                print(f"警告: 财报数据文件不存在 {filepath}")
                print(f"请先运行: python factor_production/data/download_financial.py")
                continue

            df = pl.read_parquet(filepath)

            # 统一股票代码为大写
            df = df.with_columns(pl.col('stock').str.to_uppercase())

            # 筛选股票和日期范围
            df = df.filter(
                (pl.col('stock').is_in(list(stock_list))) &
                (pl.col('date') >= pl.lit(start).str.to_date()) &
                (pl.col('date') <= pl.lit(end).str.to_date())
            )

            all_dfs.append(df)

        if not all_dfs:
            return pl.DataFrame()

        # 合并多个字段
        result = all_dfs[0]
        for df in all_dfs[1:]:
            result = result.join(df, on=['stock', 'date'], how='outer')

        print(f"财报数据加载完成: {len(result)} 行, {len(result.columns)} 列")

        return result
