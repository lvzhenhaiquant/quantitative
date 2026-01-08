"""
因子中性化模块
支持行业中性化、市值中性化
"""
import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
from sklearn.linear_model import LinearRegression
from typing import List, Optional


class FactorNeutralizer:
    """
    因子中性化器

    使用回归法消除因子的行业和市值暴露：
        原始因子 = α + β₁×行业哑变量 + β₂×ln(市值) + ε
        中性化因子 = ε (残差)

    使用示例:
        neutralizer = FactorNeutralizer()
        df = neutralizer.neutralize(df, 'history_sigma', how=['industry', 'market_cap'])
    """

    def __init__(self,
                 shenwan_path: str = None,
                 industry_level: int = 2):
        """
        初始化

        Args:
            shenwan_path: 申万行业分类文件路径
            industry_level: 行业级别 (1=一级, 2=二级, 3=三级)
        """
        if shenwan_path is None:
            shenwan_path = '/home/yunbo/project/quantitative/data/download_data/shenwan/download_shenwan_stock_df_L1_L2_L3.csv'

        self.industry_level = industry_level
        self.industry_col = f'l{industry_level}_name'

        # 加载行业数据
        self._load_industry_data(shenwan_path)

    def _load_industry_data(self, path: str):
        """加载申万行业分类"""
        df = pd.read_csv(path, encoding='utf-8-sig')

        # 转换股票代码格式: 000006.SZ -> SZ000006
        def convert_code(ts_code):
            if '.' in ts_code:
                code, market = ts_code.split('.')
                return f"{market}{code}"
            return ts_code

        df['stock'] = df['ts_code'].apply(convert_code)

        # 构建映射字典
        self.stock_to_industry = dict(zip(df['stock'], df[self.industry_col]))

        # 统计
        n_industries = df[self.industry_col].nunique()
        print(f"加载申万{self.industry_level}级行业: {n_industries} 个行业, {len(df)} 只股票")

    def get_industry(self, stock: str) -> Optional[str]:
        """获取股票的行业分类"""
        # 尝试大写和小写
        stock_upper = stock.upper()
        if stock_upper in self.stock_to_industry:
            return self.stock_to_industry[stock_upper]
        return None

    def neutralize(self,
                   df: pl.DataFrame,
                   factor_col: str,
                   market_cap_col: str = 'circ_mv',
                   how: List[str] = ['industry', 'market_cap'],
                   standardize: bool = True,
                   n_jobs: int = -1) -> pl.DataFrame:
        """
        因子中性化（截面回归法，多核并行）

        Args:
            df: 因子数据，需含 ['stock', 'date', factor_col] 列
            factor_col: 因子列名
            market_cap_col: 市值列名 (默认用流通市值 circ_mv)
            how: 中性化方式
                - ['industry']: 仅行业中性化
                - ['market_cap']: 仅市值中性化
                - ['industry', 'market_cap']: 同时中性化 (推荐)
            standardize: 是否对残差标准化
            n_jobs: 并行核心数 (-1 = 全部核心)

        Returns:
            添加 '{factor_col}_neutral' 列的 DataFrame
        """
        from joblib import Parallel, delayed
        import os

        if n_jobs == -1:
            n_jobs = os.cpu_count()

        print(f"\n开始中性化: {factor_col}")
        print(f"  方式: {how}")
        print(f"  并行核心数: {n_jobs}")

        # 转为 pandas 处理
        df_pd = df.to_pandas()

        # 添加行业列
        if 'industry' in how:
            df_pd['_industry'] = df_pd['stock'].apply(
                lambda x: self.stock_to_industry.get(x.upper(), None)
            )

        dates = sorted(df_pd['date'].unique())

        # 定义单日处理函数
        def process_single_day(date):
            df_day = df_pd[df_pd['date'] == date].copy()

            # 筛选有效数据
            valid_mask = df_day[factor_col].notna()

            if 'market_cap' in how:
                valid_mask &= df_day[market_cap_col].notna() & (df_day[market_cap_col] > 0)

            if 'industry' in how:
                valid_mask &= df_day['_industry'].notna()

            df_valid = df_day[valid_mask].copy()

            if len(df_valid) < 50:
                return None

            # 构建回归特征 X
            X_parts = []

            # 行业哑变量
            if 'industry' in how:
                industry_dummies = pd.get_dummies(
                    df_valid['_industry'],
                    prefix='ind',
                    drop_first=True,
                    dtype=float
                )
                X_parts.append(industry_dummies)

            # 对数市值
            if 'market_cap' in how:
                log_cap = np.log(df_valid[market_cap_col].values).reshape(-1, 1)
                X_parts.append(pd.DataFrame(
                    log_cap,
                    columns=['log_cap'],
                    index=df_valid.index
                ))

            if not X_parts:
                return None

            X = pd.concat(X_parts, axis=1).values
            y = df_valid[factor_col].values

            # 回归
            try:
                model = LinearRegression()
                model.fit(X, y)
                residuals = y - model.predict(X)

                # 标准化
                if standardize:
                    std = residuals.std()
                    if std > 1e-8:
                        residuals = (residuals - residuals.mean()) / std

                df_valid[f'{factor_col}_neutral'] = residuals
                return df_valid[['stock', 'date', f'{factor_col}_neutral']]

            except Exception:
                return None

        # 并行处理所有日期（带进度条）
        from tqdm import tqdm
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_single_day)(date) for date in tqdm(dates, desc="  中性化进度")
        )

        # 过滤 None 结果
        result_list = [r for r in results if r is not None]
        success_count = len(result_list)
        skip_count = len(dates) - success_count

        print(f"  完成: {success_count} 天, 跳过: {skip_count} 天")

        if not result_list:
            print("  警告: 无有效数据")
            return df

        # 合并结果
        neutral_df = pd.concat(result_list, ignore_index=True)
        neutral_pl = pl.from_pandas(neutral_df)

        # Join 回原数据
        result = df.join(neutral_pl, on=['stock', 'date'], how='left')

        # 统计
        neutral_col = f'{factor_col}_neutral'
        valid_count = result[neutral_col].drop_nulls().len()
        print(f"  中性化因子有效值: {valid_count}")

        return result


def neutralize_factor(
    df: pl.DataFrame,
    factor_col: str,
    market_cap_col: str = 'circ_mv',
    how: List[str] = ['industry', 'market_cap'],
    industry_level: int = 2
) -> pl.DataFrame:
    """
    便捷函数：因子中性化

    Args:
        df: 因子数据
        factor_col: 因子列名
        market_cap_col: 市值列名
        how: ['industry', 'market_cap']
        industry_level: 申万行业级别 (1/2/3)

    Returns:
        添加中性化因子列的 DataFrame
    """
    neutralizer = FactorNeutralizer(industry_level=industry_level)
    return neutralizer.neutralize(df, factor_col, market_cap_col, how)


if __name__ == '__main__':
    # 测试
    import sys
    sys.path.insert(0, '/home/yunbo/project/quantitative')

    from factor_production import DataManager

    # 加载因子
    cache_dir = Path('/home/yunbo/project/quantitative/factor_production/cache')
    factor_file = list(cache_dir.glob('history_sigma_*.parquet'))[0]
    df = pl.read_parquet(factor_file)
    print(f"加载因子: {len(df)} 行")

    # 加载市值
    dm = DataManager()
    dates = df['date'].unique().to_list()
    start = str(min(dates))
    end = str(max(dates))

    stocks = df['stock'].unique().to_list()
    df_cap = dm.load(stocks, start, end, ['$circ_mv'])
    df_cap = df_cap.rename({'circ_mv': 'circ_mv'})

    # 合并
    df = df.join(df_cap.select(['stock', 'date', 'circ_mv']), on=['stock', 'date'], how='left')

    # 中性化
    neutralizer = FactorNeutralizer(industry_level=2)
    df = neutralizer.neutralize(df, 'history_sigma', how=['industry', 'market_cap'])

    # 对比
    print("\n原始因子 vs 中性化因子:")
    sample = df.filter(pl.col('history_sigma_neutral').is_not_null()).head(10)
    print(sample.select(['stock', 'date', 'history_sigma', 'history_sigma_neutral']))
