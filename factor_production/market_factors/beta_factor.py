"""
Beta因子计算
衡量个股相对市场基准的敏感度（系统性风险）
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Dict, Union
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from factor_production.base_factor import BaseFactor


class BetaFactor(BaseFactor):
    """
    Beta因子

    原理：
    通过个股收益率对市场收益率的加权线性回归，得到Beta系数
    Beta > 1: 个股波动大于市场（高风险）
    Beta < 1: 个股波动小于市场（低风险）
    Beta = 1: 个股波动等于市场

    回归模型：
    R_stock = alpha + beta * R_market + epsilon

    加权方式：
    使用半衰权重，近期数据权重更大
    """

    def __init__(self, params: Dict):
        """
        初始化Beta因子

        Parameters
        ----------
        params : dict
            参数字典，包括：
            - lookback_days: 回看天数（默认251天=1年）
            - half_life: 半衰期（默认63天=1季度）
            - benchmark: 基准指数（默认'sh000300'）
            - min_valid_days: 最少有效数据点（默认50）
        """
        super().__init__(name='beta', params=params)

        # 提取参数
        self.min_valid_days = params.get('min_valid_days', 50)

    def _calculate_core(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Beta因子核心计算逻辑

        Parameters
        ----------
        data : dict
            包含：
            - 'stock_prices': DataFrame, 个股价格
            - 'benchmark_prices': Series, 基准价格

        Returns
        -------
        pd.DataFrame
            Beta因子值，columns=['factor_value']
        """
        stock_prices = data['stock_prices']
        benchmark_prices = data['benchmark_prices']

        self.logger.info("开始计算Beta因子...")

        # Step 1: 计算收益率
        stock_returns = self._calculate_returns(stock_prices)
        benchmark_returns = self._calculate_returns(benchmark_prices)

        # 获取实际数据长度
        actual_length = len(benchmark_returns)

        # Step 2: 生成半衰权重（使用实际数据长度）
        weights = self._generate_half_decay_weights(
            half_life=self.half_life,
            length=actual_length
        )

        self.logger.info(f"实际数据长度: {actual_length}, 权重范围: [{weights.min():.6f}, {weights.max():.6f}]")

        # Step 3: 逐只股票计算Beta
        beta_dict = {}
        valid_count = 0
        skip_count = 0

        # 获取所有股票列表
        if isinstance(stock_returns.index, pd.MultiIndex):
            stock_list = stock_returns.index.get_level_values(0).unique()
        else:
            stock_list = [stock_returns.name] if hasattr(stock_returns, 'name') else []

        for stock in stock_list:
            try:
                # 提取单只股票的收益率
                if isinstance(stock_returns.index, pd.MultiIndex):
                    y = stock_returns.loc[stock].values
                else:
                    y = stock_returns.values

                # 对齐日期
                x = benchmark_returns.values

                # 处理长度不一致的情况
                min_len = min(len(x), len(y))
                x = x[:min_len]
                y = y[:min_len]
                w = weights[:min_len]

                # 去除NaN
                valid_mask = ~(np.isnan(x) | np.isnan(y))

                if valid_mask.sum() < self.min_valid_days:
                    skip_count += 1
                    continue

                x_valid = x[valid_mask].reshape(-1, 1)
                y_valid = y[valid_mask]
                w_valid = w[valid_mask]

                # 加权线性回归
                model = LinearRegression()
                model.fit(x_valid, y_valid, sample_weight=w_valid)

                # 提取Beta系数
                beta = model.coef_[0]

                beta_dict[stock] = beta
                valid_count += 1

            except Exception as e:
                self.logger.warning(f"计算Beta失败 {stock}: {e}")
                skip_count += 1
                continue

        self.logger.info(f"Beta计算完成: 成功{valid_count}只, 跳过{skip_count}只")

        # 转换为DataFrame
        beta_df = pd.DataFrame.from_dict(
            beta_dict,
            orient='index',
            columns=['factor_value']
        )

        # 打印统计信息
        if len(beta_df) > 0:
            self.logger.info(f"Beta统计 - 均值: {beta_df['factor_value'].mean():.3f}, "
                           f"中位数: {beta_df['factor_value'].median():.3f}, "
                           f"范围: [{beta_df['factor_value'].min():.3f}, {beta_df['factor_value'].max():.3f}]")

        return beta_df

    def _calculate_returns(self, prices: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """
        计算对数收益率

        Parameters
        ----------
        prices : DataFrame or Series
            价格数据

        Returns
        -------
        DataFrame or Series
            对数收益率
        """
        # 对于DataFrame（多只股票）
        if isinstance(prices, pd.DataFrame):
            if isinstance(prices.index, pd.MultiIndex):
                # MultiIndex情况：按stock分组计算
                returns = prices.groupby(level='instrument').apply(
                    lambda x: np.log(x / x.shift(1))
                ).iloc[:, 0]  # 只取第一列（$close）
            else:
                returns = np.log(prices / prices.shift(1))

        # 对于Series（单只股票或指数）
        else:
            returns = np.log(prices / prices.shift(1))

        # 去掉第一行NaN
        returns = returns.dropna()

        return returns

    def _generate_half_decay_weights(self, half_life: int, length: int) -> np.ndarray:
        """
        生成半衰权重序列

        Parameters
        ----------
        half_life : int
            半衰期（天数）
        length : int
            总长度（天数）

        Returns
        -------
        np.ndarray
            权重序列，近期权重更大

        原理：
        w(t) = (0.5)^((T-t)/half_life)
        其中 T 为最后一天，t 为当前天
        """
        decay_factor = 0.5 ** (1.0 / half_life)
        weights = np.array([decay_factor ** (length - i - 1) for i in range(length)])

        # 归一化（使权重和为1）
        weights = weights / weights.sum()

        return weights

    def __repr__(self):
        return f"BetaFactor(lookback={self.lookback_days}, half_life={self.half_life}, benchmark={self.benchmark})"


# 为了兼容性，也创建HistorySigmaFactor
class HistorySigmaFactor(BaseFactor):
    """
    历史波动率因子（History_Sigma）

    原理：
    通过个股收益率对市场收益率回归后，计算残差的标准差
    衡量个股的特质风险（非系统性风险）
    """

    def __init__(self, params: Dict):
        super().__init__(name='history_sigma', params=params)
        self.min_valid_days = params.get('min_valid_days', 50)

    def _calculate_core(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """History_Sigma因子核心计算逻辑"""
        stock_prices = data['stock_prices']
        benchmark_prices = data['benchmark_prices']

        self.logger.info("开始计算History_Sigma因子...")

        # 计算收益率
        stock_returns = self._calculate_returns(stock_prices)
        benchmark_returns = self._calculate_returns(benchmark_prices)

        # 获取实际数据长度
        actual_length = len(benchmark_returns)

        # 生成半衰权重（使用实际数据长度）
        weights = self._generate_half_decay_weights(self.half_life, actual_length)

        # 逐只股票计算History_Sigma
        sigma_dict = {}
        valid_count = 0
        skip_count = 0

        # 获取股票列表
        if isinstance(stock_returns.index, pd.MultiIndex):
            stock_list = stock_returns.index.get_level_values(0).unique()
        else:
            stock_list = [stock_returns.name] if hasattr(stock_returns, 'name') else []

        for stock in stock_list:
            try:
                # 提取收益率
                if isinstance(stock_returns.index, pd.MultiIndex):
                    y = stock_returns.loc[stock].values
                else:
                    y = stock_returns.values

                x = benchmark_returns.values

                # 对齐长度
                min_len = min(len(x), len(y))
                x = x[:min_len]
                y = y[:min_len]
                w = weights[:min_len]

                # 去除NaN
                valid_mask = ~(np.isnan(x) | np.isnan(y))

                if valid_mask.sum() < self.min_valid_days:
                    skip_count += 1
                    continue

                x_valid = x[valid_mask].reshape(-1, 1)
                y_valid = y[valid_mask]
                w_valid = w[valid_mask]

                # 加权线性回归
                model = LinearRegression()
                model.fit(x_valid, y_valid, sample_weight=w_valid)

                # 计算残差
                y_pred = model.predict(x_valid)
                residual = y_valid - y_pred

                # 残差的标准差 = History_Sigma
                history_sigma = np.std(residual)

                sigma_dict[stock] = history_sigma
                valid_count += 1

            except Exception as e:
                self.logger.warning(f"计算History_Sigma失败 {stock}: {e}")
                skip_count += 1
                continue

        self.logger.info(f"History_Sigma计算完成: 成功{valid_count}只, 跳过{skip_count}只")

        # 转换为DataFrame
        sigma_df = pd.DataFrame.from_dict(
            sigma_dict,
            orient='index',
            columns=['factor_value']
        )

        if len(sigma_df) > 0:
            self.logger.info(f"History_Sigma统计 - 均值: {sigma_df['factor_value'].mean():.6f}, "
                           f"中位数: {sigma_df['factor_value'].median():.6f}")

        return sigma_df

    def _calculate_returns(self, prices):
        """计算对数收益率（同BetaFactor）"""
        if isinstance(prices, pd.DataFrame):
            if isinstance(prices.index, pd.MultiIndex):
                returns = prices.groupby(level='instrument').apply(
                    lambda x: np.log(x / x.shift(1))
                ).iloc[:, 0]
            else:
                returns = np.log(prices / prices.shift(1))
        else:
            returns = np.log(prices / prices.shift(1))

        returns = returns.dropna()
        return returns

    def _generate_half_decay_weights(self, half_life: int, length: int) -> np.ndarray:
        """生成半衰权重（同BetaFactor）"""
        decay_factor = 0.5 ** (1.0 / half_life)
        weights = np.array([decay_factor ** (length - i - 1) for i in range(length)])
        weights = weights / weights.sum()
        return weights


if __name__ == "__main__":
    # 简单测试
    import yaml

    # 加载配置
    with open("/home/zhenhai1/quantitative/configs/factor_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    params = config['beta_factor']['params']

    # 创建因子实例
    beta_factor = BetaFactor(params)
    print(beta_factor)
