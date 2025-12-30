"""
股票过滤器

提供ST、停牌、涨跌停股票的过滤功能
数据存储格式: {date_str: [stock_list]}
"""

import os
import json
from typing import List, Set, Optional
from datetime import datetime
import pandas as pd

# 数据目录
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


class StockFilter:
    """
    股票过滤器

    用于过滤ST、停牌、涨跌停股票

    使用示例:
        filter = StockFilter()

        # 检查单只股票
        if filter.is_tradable('000001.SZ', '2024-01-15'):
            ...

        # 批量过滤
        valid_stocks = filter.filter_stocks(stock_list, '2024-01-15')
    """

    def __init__(self, data_dir: str = None):
        """
        初始化过滤器

        Args:
            data_dir: 数据目录路径，默认为 backtest/filters/data
        """
        self.data_dir = data_dir or DATA_DIR

        # 缓存数据
        self._st_data: dict = None
        self._suspend_data: dict = None
        self._limit_data: dict = None

        # 加载数据
        self._load_data()

    def _load_data(self):
        """加载所有过滤数据"""
        self._st_data = self._load_json('st_stocks.json')
        self._suspend_data = self._load_json('suspend_stocks.json')
        self._limit_data = self._load_json('limit_stocks.json')

        # 统计
        st_days = len(self._st_data) if self._st_data else 0
        suspend_days = len(self._suspend_data) if self._suspend_data else 0
        limit_days = len(self._limit_data) if self._limit_data else 0

        print(f"StockFilter 加载完成: ST({st_days}天), 停牌({suspend_days}天), 涨跌停({limit_days}天)")

    def _load_json(self, filename: str) -> dict:
        """加载JSON文件"""
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _normalize_date(self, date) -> str:
        """
        标准化日期格式为 YYYYMMDD

        支持输入: '2024-01-15', '20240115', datetime, pd.Timestamp
        """
        if isinstance(date, str):
            return date.replace('-', '')
        elif isinstance(date, (datetime, pd.Timestamp)):
            return date.strftime('%Y%m%d')
        return str(date)

    def _normalize_stock(self, stock: str) -> str:
        """
        标准化股票代码为 Tushare 格式 (000001.SZ)

        支持输入: 'sz000001', 'SZ000001', '000001.SZ'
        """
        stock = stock.upper()

        # 已经是标准格式
        if '.' in stock:
            return stock

        # sz000001 -> 000001.SZ
        if stock.startswith('SZ'):
            return stock[2:] + '.SZ'
        elif stock.startswith('SH'):
            return stock[2:] + '.SH'

        # 纯数字，根据首位判断
        if stock.isdigit():
            if stock.startswith(('0', '3')):
                return stock + '.SZ'
            elif stock.startswith(('6', '9')):
                return stock + '.SH'

        return stock

    def get_st_stocks(self, date) -> Set[str]:
        """获取指定日期的ST股票集合"""
        date_str = self._normalize_date(date)
        stocks = self._st_data.get(date_str, [])
        return set(stocks)

    def get_suspend_stocks(self, date) -> Set[str]:
        """获取指定日期的停牌股票集合"""
        date_str = self._normalize_date(date)
        stocks = self._suspend_data.get(date_str, [])
        return set(stocks)

    def get_limit_up_stocks(self, date) -> Set[str]:
        """获取指定日期的涨停股票集合"""
        date_str = self._normalize_date(date)
        data = self._limit_data.get(date_str, {})
        return set(data.get('limit_up', []))

    def get_limit_down_stocks(self, date) -> Set[str]:
        """获取指定日期的跌停股票集合"""
        date_str = self._normalize_date(date)
        data = self._limit_data.get(date_str, {})
        return set(data.get('limit_down', []))

    def get_all_limit_stocks(self, date) -> Set[str]:
        """获取指定日期的所有涨跌停股票集合"""
        return self.get_limit_up_stocks(date) | self.get_limit_down_stocks(date)

    def is_st(self, stock: str, date) -> bool:
        """检查股票是否为ST"""
        stock = self._normalize_stock(stock)
        return stock in self.get_st_stocks(date)

    def is_suspended(self, stock: str, date) -> bool:
        """检查股票是否停牌"""
        stock = self._normalize_stock(stock)
        return stock in self.get_suspend_stocks(date)

    def is_limit_up(self, stock: str, date) -> bool:
        """检查股票是否涨停"""
        stock = self._normalize_stock(stock)
        return stock in self.get_limit_up_stocks(date)

    def is_limit_down(self, stock: str, date) -> bool:
        """检查股票是否跌停"""
        stock = self._normalize_stock(stock)
        return stock in self.get_limit_down_stocks(date)

    def is_kcbj(self, stock: str) -> bool:
        """
        检查是否为科创板或北交所股票

        科创板: 688xxx (上海)
        北交所: 43xxxx, 83xxxx, 87xxxx
        """
        stock = self._normalize_stock(stock)
        code = stock.split('.')[0]

        # 科创板: 688xxx
        if code.startswith('688'):
            return True
        # 北交所: 43xxxx, 83xxxx, 87xxxx
        if code.startswith('43') or code.startswith('83') or code.startswith('87'):
            return True
        return False

    def is_tradable(self, stock: str, date,
                    exclude_st: bool = True,
                    exclude_suspend: bool = True,
                    exclude_limit: bool = True,
                    exclude_kcbj: bool = False) -> bool:
        """
        检查股票是否可交易

        Args:
            stock: 股票代码
            date: 日期
            exclude_st: 是否排除ST
            exclude_suspend: 是否排除停牌
            exclude_limit: 是否排除涨跌停
            exclude_kcbj: 是否排除科创板和北交所

        Returns:
            True 如果可交易
        """
        stock = self._normalize_stock(stock)

        if exclude_kcbj and self.is_kcbj(stock):
            return False

        if exclude_st and self.is_st(stock, date):
            return False

        if exclude_suspend and self.is_suspended(stock, date):
            return False

        if exclude_limit:
            if self.is_limit_up(stock, date) or self.is_limit_down(stock, date):
                return False

        return True

    def filter_stocks(self, stocks: List[str], date,
                      exclude_st: bool = True,
                      exclude_suspend: bool = True,
                      exclude_limit: bool = True,
                      exclude_kcbj: bool = False) -> List[str]:
        """
        批量过滤股票

        Args:
            stocks: 股票列表
            date: 日期
            exclude_st: 是否排除ST
            exclude_suspend: 是否排除停牌
            exclude_limit: 是否排除涨跌停
            exclude_kcbj: 是否排除科创板和北交所

        Returns:
            过滤后的股票列表
        """
        date_str = self._normalize_date(date)

        # 获取需要排除的股票集合
        exclude_set = set()

        if exclude_st:
            exclude_set |= self.get_st_stocks(date_str)

        if exclude_suspend:
            exclude_set |= self.get_suspend_stocks(date_str)

        if exclude_limit:
            exclude_set |= self.get_all_limit_stocks(date_str)

        # 过滤
        result = []
        for stock in stocks:
            normalized = self._normalize_stock(stock)
            if normalized not in exclude_set:
                # 科创板/北交所过滤（不需要日期，直接判断代码）
                if exclude_kcbj and self.is_kcbj(normalized):
                    continue
                result.append(stock)  # 保持原格式

        return result

    def get_filter_stats(self, stocks: List[str], date) -> dict:
        """
        获取过滤统计信息

        Args:
            stocks: 股票列表
            date: 日期

        Returns:
            统计字典
        """
        date_str = self._normalize_date(date)
        normalized_stocks = {self._normalize_stock(s) for s in stocks}

        st_stocks = self.get_st_stocks(date_str)
        suspend_stocks = self.get_suspend_stocks(date_str)
        limit_stocks = self.get_all_limit_stocks(date_str)

        st_filtered = normalized_stocks & st_stocks
        suspend_filtered = normalized_stocks & suspend_stocks
        limit_filtered = normalized_stocks & limit_stocks

        return {
            'total': len(stocks),
            'st_count': len(st_filtered),
            'suspend_count': len(suspend_filtered),
            'limit_count': len(limit_filtered),
            'st_stocks': list(st_filtered),
            'suspend_stocks': list(suspend_filtered),
            'limit_stocks': list(limit_filtered),
        }


# 便捷函数
_default_filter = None

def get_filter() -> StockFilter:
    """获取默认过滤器实例（单例）"""
    global _default_filter
    if _default_filter is None:
        _default_filter = StockFilter()
    return _default_filter


def filter_stocks(stocks: List[str], date, **kwargs) -> List[str]:
    """便捷函数：过滤股票"""
    return get_filter().filter_stocks(stocks, date, **kwargs)


def is_tradable(stock: str, date, **kwargs) -> bool:
    """便捷函数：检查是否可交易"""
    return get_filter().is_tradable(stock, date, **kwargs)
