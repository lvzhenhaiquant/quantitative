"""
股票过滤模块

提供ST、停牌、涨跌停股票的过滤功能

使用示例:
    from backtest.filters import StockFilter, filter_stocks

    # 方式1: 使用类
    sf = StockFilter()
    valid_stocks = sf.filter_stocks(stock_list, '2024-01-15')

    # 方式2: 使用便捷函数
    valid_stocks = filter_stocks(stock_list, '2024-01-15')
"""

from .stock_filter import (
    StockFilter,
    get_filter,
    filter_stocks,
    is_tradable,
)

__all__ = [
    'StockFilter',
    'get_filter',
    'filter_stocks',
    'is_tradable',
]
