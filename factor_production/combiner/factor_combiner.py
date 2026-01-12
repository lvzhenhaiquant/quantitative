"""
因子IC加权合成模块

基于历史Spearman IC对多因子进行加权合成

使用方法:
    from factor_production.combiner import FactorCombiner

    combiner = FactorCombiner(
        factors=['turnover_ratio', 'history_sigma', 'return_var'],
        ic_window=63,  # 3个月
    )

    # 合成综合因子
    combined = combiner.combine('2025-12-31')

    # 查看权重
    weights = combiner.get_weights('2025-12-31')
"""
import polars as pl
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from scipy import stats


class FactorCombiner:
    """
    因子IC加权合成器

    特点:
    - 使用排序计算Spearman IC（自动标准化+处理方向）
    - 权重每周更新
    - IC窗口可配置（默认63天/3个月）
    """

    # 因子缓存目录
    FACTOR_CACHE_DIR = '/home/zhenhai1/quantitative/factor_production/cache'

    # 收益率数据目录
    DATA_DIR = '/home/zhenhai1/quantitative/data/download_data'

    def __init__(
        self,
        factors: List[str],
        ic_window: int = 63,
        ret_period: int = 21,
        weight_update_freq: str = 'weekly',
    ):
        """
        初始化

        Args:
            factors: 因子名称列表
            ic_window: 计算IC的回看窗口（交易日），默认63天（3个月）
            ret_period: IC计算使用的收益率周期（交易日），默认21天（月度调仓）
            weight_update_freq: 权重更新频率，'weekly' 或 'monthly'
        """
        self.factors = factors
        self.ic_window = ic_window
        self.ret_period = ret_period
        self.weight_update_freq = weight_update_freq

        # 缓存
        self._factor_data_cache: Dict[str, pl.DataFrame] = {}
        self._return_data_cache: Optional[pl.DataFrame] = None
        self._weights_cache: Dict[str, Dict[str, float]] = {}  # {date: {factor: weight}}
        self._ic_history: Dict[str, pl.DataFrame] = {}  # {factor: ic_df}

        print(f"FactorCombiner 初始化完成")
        print(f"  因子: {factors}")
        print(f"  IC窗口: {ic_window} 天")
        print(f"  收益率周期: {ret_period} 天")
        print(f"  权重更新: {weight_update_freq}")

    def _load_factor(self, factor_name: str) -> pl.DataFrame:
        """加载单个因子数据"""
        if factor_name in self._factor_data_cache:
            return self._factor_data_cache[factor_name]

        cache_dir = Path(self.FACTOR_CACHE_DIR)
        files = list(cache_dir.glob(f"{factor_name}_*.parquet"))

        if not files:
            raise FileNotFoundError(f"未找到因子文件: {factor_name}")

        latest_file = sorted(files)[-1]
        df = pl.read_parquet(latest_file)

        # 确保有必要的列
        if 'stock' not in df.columns or 'date' not in df.columns:
            raise ValueError(f"因子文件缺少 stock 或 date 列: {factor_name}")

        if factor_name not in df.columns:
            raise ValueError(f"因子文件缺少因子列: {factor_name}")

        # 只保留需要的列
        df = df.select(['stock', 'date', factor_name])

        # 去重
        df = df.unique(subset=['stock', 'date'], keep='first')

        # 过滤无效值
        df = df.filter(
            pl.col(factor_name).is_not_null() &
            pl.col(factor_name).is_not_nan()
        )

        self._factor_data_cache[factor_name] = df
        print(f"  加载因子 {factor_name}: {len(df)} 行")
        return df

    def _load_returns(self, start_date: str, end_date: str) -> pl.DataFrame:
        """
        加载收益率数据（未来N日收益）

        Returns:
            DataFrame: [stock, date, next_ret] - next_ret 是未来 ret_period 个交易日的收益率
        """
        if self._return_data_cache is not None:
            return self._return_data_cache

        print("加载收益率数据...")

        # 从 daily 数据计算收益率
        daily_dir = Path(self.DATA_DIR) / 'daily'

        all_dfs = []
        for f in daily_dir.glob('*.parquet'):
            try:
                df = pl.read_parquet(f, columns=['ts_code', 'trade_date', 'close'])

                # 处理日期格式
                sample = str(df['trade_date'][0])
                if '-' in sample:
                    df = df.with_columns(pl.col('trade_date').str.to_date('%Y-%m-%d').alias('date'))
                else:
                    df = df.with_columns(pl.col('trade_date').str.to_date('%Y%m%d').alias('date'))

                df = df.with_columns(pl.col('ts_code').alias('stock'))
                df = df.select(['stock', 'date', 'close'])

                all_dfs.append(df)
            except:
                continue

        if not all_dfs:
            raise ValueError("无法加载收益率数据")

        df = pl.concat(all_dfs)
        df = df.sort(['stock', 'date'])

        # 计算未来N日收益率（与调仓周期匹配）
        df = df.with_columns(
            (pl.col('close').shift(-self.ret_period).over('stock') / pl.col('close') - 1).alias('next_ret')
        )

        df = df.select(['stock', 'date', 'next_ret'])
        df = df.drop_nulls(subset=['next_ret'])

        self._return_data_cache = df
        print(f"收益率数据: {len(df)} 行, {df['stock'].n_unique()} 只股票, 收益周期: {self.ret_period}天")

        return df

    def _calc_daily_ic(
        self,
        factor_name: str,
        date,
        factor_df: pl.DataFrame,
        return_df: pl.DataFrame
    ) -> Optional[float]:
        """
        计算单日 Spearman IC

        IC = corr(当日因子值排名, 未来N日收益排名)
        N = ret_period（与调仓周期匹配）
        """
        # 筛选当日数据（date 可能是 str 或 datetime.date）
        if isinstance(date, str):
            target_date = datetime.strptime(date, '%Y-%m-%d').date()
        else:
            target_date = date

        factor_day = factor_df.filter(pl.col('date') == target_date)
        return_day = return_df.filter(pl.col('date') == target_date)

        if len(factor_day) == 0 or len(return_day) == 0:
            return None

        # 合并（只按 stock 合并，因为日期已筛选）
        factor_day = factor_day.select(['stock', factor_name])
        return_day = return_day.select(['stock', 'next_ret'])

        merged = factor_day.join(return_day, on='stock', how='inner')

        if len(merged) < 30:  # 样本太少
            return None

        # 计算 Spearman IC（排序相关）
        factor_vals = merged[factor_name].to_numpy()
        return_vals = merged['next_ret'].to_numpy()

        # 过滤 NaN
        mask = ~(np.isnan(factor_vals) | np.isnan(return_vals))
        factor_vals = factor_vals[mask]
        return_vals = return_vals[mask]

        if len(factor_vals) < 30:
            return None

        ic, _ = stats.spearmanr(factor_vals, return_vals)

        return ic if not np.isnan(ic) else None

    def calc_ic_series(
        self,
        factor_name: str,
        start_date: str,
        end_date: str
    ) -> pl.DataFrame:
        """
        计算因子IC序列

        Returns:
            DataFrame: [date, ic]
        """
        factor_df = self._load_factor(factor_name)
        return_df = self._load_returns(start_date, end_date)

        # 获取所有交易日
        dates = factor_df.filter(
            (pl.col('date') >= datetime.strptime(start_date, '%Y-%m-%d').date()) &
            (pl.col('date') <= datetime.strptime(end_date, '%Y-%m-%d').date())
        )['date'].unique().sort()

        ic_records = []
        for d in dates.to_list():
            ic = self._calc_daily_ic(factor_name, d, factor_df, return_df)
            if ic is not None:
                ic_records.append({'date': d, 'ic': ic})

        return pl.DataFrame(ic_records)

    def _get_rebalance_date(self, date: str) -> str:
        """
        获取最近的调仓日（每周一）

        Args:
            date: 目标日期 'YYYY-MM-DD'

        Returns:
            最近的周一日期
        """
        dt = datetime.strptime(date, '%Y-%m-%d')

        # 找到本周或上周一
        days_since_monday = dt.weekday()
        monday = dt - timedelta(days=days_since_monday)

        return monday.strftime('%Y-%m-%d')

    def calc_weights(self, date: str) -> Dict[str, float]:
        """
        计算截至 date 的因子权重

        使用最近 ic_window 天的平均IC作为权重
        权重归一化: w_i = IC_i / sum(|IC_i|)

        Returns:
            {factor_name: weight}
        """
        # 检查缓存（按调仓日缓存）
        rebalance_date = self._get_rebalance_date(date)
        if rebalance_date in self._weights_cache:
            return self._weights_cache[rebalance_date]

        print(f"计算权重 (截至 {date}, 调仓日 {rebalance_date})...")

        # 计算回看窗口的起止日期
        end_dt = datetime.strptime(date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=self.ic_window * 2)  # 多取一些，确保有足够交易日
        start_date = start_dt.strftime('%Y-%m-%d')

        # 计算每个因子的平均IC
        ic_means = {}

        for factor_name in self.factors:
            print(f"  计算 {factor_name} IC...")

            ic_df = self.calc_ic_series(factor_name, start_date, date)

            if len(ic_df) == 0:
                print(f"    警告: {factor_name} 无IC数据")
                ic_means[factor_name] = 0.0
                continue

            # 取最近 ic_window 天
            ic_df = ic_df.sort('date', descending=True).head(self.ic_window)

            ic_mean = ic_df['ic'].mean()
            ic_std = ic_df['ic'].std()
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0

            ic_means[factor_name] = ic_mean

            print(f"    IC均值: {ic_mean:.4f}, IC_IR: {ic_ir:.4f}, 样本: {len(ic_df)}")

        # 归一化权重
        total_abs = sum(abs(v) for v in ic_means.values())

        if total_abs == 0:
            # 所有IC都是0，使用等权
            weights = {f: 1.0 / len(self.factors) for f in self.factors}
        else:
            weights = {f: ic / total_abs for f, ic in ic_means.items()}

        print(f"  权重: {weights}")

        # 缓存
        self._weights_cache[rebalance_date] = weights

        return weights

    def combine(self, date: str, stocks: Optional[List[str]] = None) -> pl.DataFrame:
        """
        合成综合因子

        Args:
            date: 目标日期 'YYYY-MM-DD'
            stocks: 股票列表，None 表示全部

        Returns:
            DataFrame: [stock, date, combined_factor]
        """
        print(f"\n合成综合因子 ({date})...")

        # 1. 获取权重
        weights = self.calc_weights(date)

        # 2. 加载并处理各因子数据
        target_date = datetime.strptime(date, '%Y-%m-%d').date()

        factor_dfs = []
        for factor_name in self.factors:
            df = self._load_factor(factor_name)

            # 筛选日期
            df = df.filter(pl.col('date') == target_date)

            # 筛选股票
            if stocks:
                df = df.filter(pl.col('stock').is_in(stocks))

            # 去重
            df = df.unique(subset=['stock'], keep='first')

            # 过滤 null
            df = df.drop_nulls(subset=[factor_name])

            # 计算排名（百分位，0-1）
            df = df.with_columns(
                (pl.col(factor_name).rank() / pl.col(factor_name).count()).alias(f'{factor_name}_rank')
            )

            df = df.select(['stock', f'{factor_name}_rank'])
            factor_dfs.append(df)

        if not factor_dfs:
            return pl.DataFrame()

        # 3. 合并所有因子（使用 coalesce 处理 outer join）
        result = factor_dfs[0]
        for df in factor_dfs[1:]:
            result = result.join(df, on='stock', how='outer_coalesce')

        # 4. 加权合成
        # combined = sum(w_i * rank_i)
        weighted_sum_expr = pl.lit(0.0)
        for factor_name in self.factors:
            rank_col = f'{factor_name}_rank'
            if rank_col in result.columns:
                w = weights.get(factor_name, 0)
                weighted_sum_expr = weighted_sum_expr + pl.col(rank_col).fill_null(0.5) * w

        result = result.with_columns(
            weighted_sum_expr.alias('combined_factor'),
            pl.lit(target_date).alias('date')
        )

        # 只保留需要的列
        result = result.select(['stock', 'date', 'combined_factor'])
        result = result.drop_nulls(subset=['combined_factor'])
        result = result.sort('combined_factor', descending=True)  # 综合因子越大越好

        print(f"合成完成: {len(result)} 只股票")

        return result

    def get_weights(self, date: str) -> Dict[str, float]:
        """获取指定日期的权重"""
        rebalance_date = self._get_rebalance_date(date)

        if rebalance_date in self._weights_cache:
            return self._weights_cache[rebalance_date]

        return self.calc_weights(date)

    def select_stocks(
        self,
        date: str,
        n_stocks: int = 30,
        stocks: Optional[List[str]] = None
    ) -> List[str]:
        """
        基于综合因子选股

        Args:
            date: 目标日期
            n_stocks: 选股数量
            stocks: 股票池，None 表示全部

        Returns:
            选中的股票列表
        """
        combined = self.combine(date, stocks)

        if len(combined) == 0:
            return []

        # 选综合因子最大的 n 只
        top_stocks = combined.head(n_stocks)['stock'].to_list()

        return top_stocks

    def summary(self, date: str) -> None:
        """打印因子权重摘要"""
        weights = self.get_weights(date)

        print(f"\n{'='*50}")
        print(f"因子权重摘要 ({date})")
        print(f"{'='*50}")

        for factor, weight in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True):
            direction = "选大" if weight > 0 else "选小"
            print(f"  {factor:20s}: {weight:+.4f} ({direction})")

        print(f"{'='*50}")


if __name__ == '__main__':
    # 测试
    combiner = FactorCombiner(
        factors=['turnover_ratio', 'history_sigma', 'return_var'],
        ic_window=63
    )

    # 计算权重
    weights = combiner.calc_weights('2025-12-31')
    print(f"\n权重: {weights}")

    # 选股
    stocks = combiner.select_stocks('2025-12-31', n_stocks=30)
    print(f"\n选股: {stocks[:10]}...")
