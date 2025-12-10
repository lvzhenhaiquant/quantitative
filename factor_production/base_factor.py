"""
因子基类
定义因子计算的标准流程和接口
"""

import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any
import sys
import os

# 添加utils到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger


class BaseFactor(ABC):
    """
    因子计算基类

    所有因子都应该继承这个基类，并实现 _calculate_core 方法

    标准流程：
    1. 数据加载（由子类通过data_loader获取）
    2. 数据预处理（去极值、缺失值处理）
    3. 因子计算（核心逻辑，子类实现）
    4. 数据后处理（标准化、中性化）
    5. 结果缓存（可选）
    """

    def __init__(self, name: str, params: Dict[str, Any]):
        """
        初始化因子

        Parameters
        ----------
        name : str
            因子名称
        params : dict
            因子参数配置
        """
        self.name = name
        self.params = params
        self.logger = get_logger(f"factor.{self.__class__.__name__}")

        # 从params中提取常用参数
        self.lookback_days = params.get('lookback_days', 251)
        self.half_life = params.get('half_life', 63)
        self.benchmark = params.get('benchmark', 'sh000300')

        self.logger.info(f"初始化因子: {name}")
        self.logger.info(f"参数: {params}")

    @abstractmethod
    def _calculate_core(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        因子计算核心逻辑（子类必须实现）

        Parameters
        ----------
        data : dict
            包含计算所需的所有数据，格式为：
            {
                'stock_prices': DataFrame,  # 个股价格数据
                'benchmark_prices': DataFrame,  # 基准价格数据
                ...
            }

        Returns
        -------
        pd.DataFrame
            因子值，index为股票代码，columns为['factor_value']
        """
        raise NotImplementedError("子类必须实现 _calculate_core 方法")

    def calculate(self,
                  stock_list: list,
                  start_date: str,
                  end_date: str,
                  data_loader) -> pd.DataFrame:
        """
        计算因子（完整流程）

        Parameters
        ----------
        stock_list : list
            股票列表
        start_date : str
            开始日期
        end_date : str
            结束日期
        data_loader : DataLoader
            数据加载器实例

        Returns
        -------
        pd.DataFrame
            因子值，index为股票代码，columns为[self.name]
        """
        self.logger.info(f"开始计算因子 {self.name}")
        self.logger.info(f"股票数量: {len(stock_list)}, 日期范围: {start_date} ~ {end_date}")

        # Step 1: 加载数据
        data = self._load_data(stock_list, start_date, end_date, data_loader)

        # Step 2: 数据预处理
        data = self._preprocess(data)

        # Step 3: 核心计算
        factor_values = self._calculate_core(data)

        # Step 4: 数据后处理
        factor_values = self._postprocess(factor_values)

        # 重命名列为因子名称
        if 'factor_value' in factor_values.columns:
            factor_values.rename(columns={'factor_value': self.name}, inplace=True)

        self.logger.info(f"因子计算完成，有效股票数: {len(factor_values)}")

        return factor_values

    def _load_data(self,
                   stock_list: list,
                   start_date: str,
                   end_date: str,
                   data_loader) -> Dict[str, pd.DataFrame]:
        """
        加载计算所需的数据

        默认实现：加载个股价格和基准价格
        子类可以重写此方法以加载更多数据
        """
        self.logger.info("加载数据...")

        # 加载个股价格
        stock_prices = data_loader.load_stock_prices(
            stock_list,
            start_date,
            end_date,
            lookback_days=self.lookback_days
        )

        # 加载基准价格
        benchmark_prices = data_loader.load_benchmark_prices(
            self.benchmark,
            start_date,
            end_date,
            lookback_days=self.lookback_days
        )

        return {
            'stock_prices': stock_prices,
            'benchmark_prices': benchmark_prices
        }

    def _preprocess(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        数据预处理

        包括：去极值、缺失值处理等
        子类可以重写此方法
        """
        # 默认不做预处理，直接返回
        return data

    def _postprocess(self, factor_values: pd.DataFrame) -> pd.DataFrame:
        """
        因子后处理

        包括：标准化、中性化等
        子类可以重写此方法
        """
        # 默认不做后处理，直接返回
        return factor_values

    def save(self, factor_values: pd.DataFrame, output_path: str):
        """保存因子值"""
        factor_values.to_csv(output_path, index=True)
        self.logger.info(f"因子已保存到: {output_path}")

    def __str__(self):
        return f"Factor({self.name}, params={self.params})"

    def __repr__(self):
        return self.__str__()
