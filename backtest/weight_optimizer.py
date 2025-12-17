"""
权重优化模块
支持最小方差(min_vol)、最大夏普(max_sharpe)等优化方法
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import warnings
warnings.filterwarnings('ignore')


class WeightOptimizer:
    """权重优化器"""

    def __init__(self,
                 method: str = 'min_vol',
                 lookback_days: int = 252,
                 max_weight: float = 0.05,
                 min_weight: float = 0.0,
                 risk_free_rate: float = 0.02,
                 use_dynamic_rf: bool = False):
        """
        初始化权重优化器

        Args:
            method: 优化方法 ('min_vol', 'max_sharpe', 'equal')
            lookback_days: 回看天数，用于计算协方差矩阵
            max_weight: 单只股票最大权重
            min_weight: 单只股票最小权重
            risk_free_rate: 无风险利率（用于max_sharpe，当use_dynamic_rf=False时使用）
            use_dynamic_rf: 是否使用动态SHIBOR作为无风险利率
        """
        self.method = method
        self.lookback_days = lookback_days
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.risk_free_rate = risk_free_rate
        self.use_dynamic_rf = use_dynamic_rf
        self.shibor_df = None  # SHIBOR数据，由外部设置

    def set_shibor_data(self, shibor_df: pd.DataFrame):
        """
        设置SHIBOR数据

        Args:
            shibor_df: SHIBOR数据，需包含 'date' 和 '1y' 列
        """
        self.shibor_df = shibor_df
        if shibor_df is not None:
            print(f"已加载SHIBOR数据: {len(shibor_df)}条, "
                  f"范围: {shibor_df['date'].min()} ~ {shibor_df['date'].max()}")

    def _get_risk_free_rate(self, current_date: pd.Timestamp) -> float:
        """
        获取无风险利率（避免未来函数）

        Args:
            current_date: 当前调仓日期

        Returns:
            无风险利率（年化）
        """
        if not self.use_dynamic_rf or self.shibor_df is None:
            return self.risk_free_rate

        # 只使用 current_date 之前的数据（不含当天，避免未来函数）
        available = self.shibor_df[self.shibor_df['date'] < current_date]

        if len(available) > 0:
            # 取最近一天的 SHIBOR 1年期利率
            rf = available.iloc[-1]['1y'] / 100  # 转为小数
            return rf
        else:
            # 没有可用数据，使用默认值
            return self.risk_free_rate

    def optimize(self,
                 price_df: pd.DataFrame,
                 selected_stocks: List[str],
                 current_date: pd.Timestamp) -> Dict[str, float]:
        """
        计算最优权重

        Args:
            price_df: 价格数据，index为日期，columns为股票代码
            selected_stocks: 选中的股票列表
            current_date: 当前日期（调仓日）

        Returns:
            权重字典 {股票代码: 权重}
        """
        if self.method == 'equal':
            return self._equal_weight(selected_stocks)
        elif self.method == 'min_vol':
            return self._min_volatility(price_df, selected_stocks, current_date)
        elif self.method == 'max_sharpe':
            return self._max_sharpe(price_df, selected_stocks, current_date)
        else:
            raise ValueError(f"不支持的优化方法: {self.method}")

    def _equal_weight(self, selected_stocks: List[str]) -> Dict[str, float]:
        """等权重"""
        n = len(selected_stocks)
        if n == 0:
            return {}
        weight = 1.0 / n
        return {stock: weight for stock in selected_stocks}

    def _get_historical_prices(self,
                               price_df: pd.DataFrame,
                               selected_stocks: List[str],
                               current_date: pd.Timestamp) -> pd.DataFrame:
        """
        获取历史价格数据（用于计算协方差）

        注意：使用 current_date - 1 之前的数据，避免未来函数
        """
        # 筛选选中的股票
        available_stocks = [s for s in selected_stocks if s in price_df.columns]
        if len(available_stocks) == 0:
            return pd.DataFrame()

        hist_prices = price_df[available_stocks].copy()

        # 只使用 current_date 之前的数据（不包含当天）
        hist_prices = hist_prices[hist_prices.index < current_date]

        # 取最近 lookback_days 天
        hist_prices = hist_prices.tail(self.lookback_days)

        # 删除全为NaN的列
        hist_prices = hist_prices.dropna(axis=1, how='all')

        # 前向填充缺失值
        hist_prices = hist_prices.ffill()

        return hist_prices

    def _min_volatility(self,
                        price_df: pd.DataFrame,
                        selected_stocks: List[str],
                        current_date: pd.Timestamp) -> Dict[str, float]:
        """最小方差优化"""
        hist_prices = self._get_historical_prices(price_df, selected_stocks, current_date)

        if len(hist_prices) < 60 or len(hist_prices.columns) < 2:
            # 数据不足，使用等权
            return self._equal_weight(selected_stocks)

        try:
            # 计算协方差矩阵
            S = risk_models.sample_cov(hist_prices)

            # 最小方差优化
            ef = EfficientFrontier(None, S, weight_bounds=(self.min_weight, self.max_weight))
            ef.min_volatility()
            weights = ef.clean_weights()

            # 转换为字典，只保留权重>0的股票
            result = {stock: w for stock, w in weights.items() if w > 1e-6}

            # 如果优化后没有股票被选中，回退到等权
            if len(result) == 0:
                return self._equal_weight(selected_stocks)

            return result

        except Exception as e:
            print(f"  最小方差优化失败: {e}, 使用等权")
            return self._equal_weight(selected_stocks)

    def _max_sharpe(self,
                    price_df: pd.DataFrame,
                    selected_stocks: List[str],
                    current_date: pd.Timestamp) -> Dict[str, float]:
        """最大夏普比率优化"""
        hist_prices = self._get_historical_prices(price_df, selected_stocks, current_date)

        if len(hist_prices) < 60 or len(hist_prices.columns) < 2:
            # 数据不足，使用等权
            return self._equal_weight(selected_stocks)

        try:
            # 计算预期收益和协方差
            mu = expected_returns.mean_historical_return(hist_prices)
            S = risk_models.sample_cov(hist_prices)

            # 获取无风险利率（动态SHIBOR或固定值）
            rf = self._get_risk_free_rate(current_date)

            # 最大夏普比率优化
            ef = EfficientFrontier(mu, S, weight_bounds=(self.min_weight, self.max_weight))
            ef.max_sharpe(risk_free_rate=rf)
            weights = ef.clean_weights()

            # 转换为字典，只保留权重>0的股票
            result = {stock: w for stock, w in weights.items() if w > 1e-6}

            # 如果优化后没有股票被选中，回退到等权
            if len(result) == 0:
                return self._equal_weight(selected_stocks)

            return result

        except Exception as e:
            print(f"  最大夏普优化失败: {e}, 使用等权")
            return self._equal_weight(selected_stocks)


class RiskParityOptimizer:
    """风险平价优化器"""

    def __init__(self,
                 lookback_days: int = 252,
                 max_weight: float = 0.10,
                 min_weight: float = 0.0):
        """
        Args:
            lookback_days: 回看天数
            max_weight: 单只股票最大权重
            min_weight: 单只股票最小权重
        """
        self.lookback_days = lookback_days
        self.max_weight = max_weight
        self.min_weight = min_weight

    def optimize(self,
                 price_df: pd.DataFrame,
                 selected_stocks: List[str],
                 current_date: pd.Timestamp) -> Dict[str, float]:
        """
        风险平价优化：让每只股票对组合风险的贡献相等

        简化实现：使用波动率倒数加权
        """
        # 获取历史数据
        available_stocks = [s for s in selected_stocks if s in price_df.columns]
        if len(available_stocks) == 0:
            return {}

        hist_prices = price_df[available_stocks].copy()
        hist_prices = hist_prices[hist_prices.index < current_date]
        hist_prices = hist_prices.tail(self.lookback_days)
        hist_prices = hist_prices.dropna(axis=1, how='all').ffill()

        if len(hist_prices) < 60:
            # 数据不足，等权
            n = len(selected_stocks)
            return {stock: 1.0/n for stock in selected_stocks}

        try:
            # 计算收益率
            returns = hist_prices.pct_change().dropna()

            # 计算波动率
            volatilities = returns.std() * np.sqrt(252)

            # 波动率倒数作为权重
            inv_vol = 1.0 / volatilities
            inv_vol = inv_vol.replace([np.inf, -np.inf], 0)

            # 归一化
            weights = inv_vol / inv_vol.sum()

            # 应用权重限制
            weights = weights.clip(self.min_weight, self.max_weight)
            weights = weights / weights.sum()  # 重新归一化

            return weights.to_dict()

        except Exception as e:
            print(f"  风险平价优化失败: {e}, 使用等权")
            n = len(selected_stocks)
            return {stock: 1.0/n for stock in selected_stocks}
