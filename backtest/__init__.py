"""
回测模块

提供简洁的回测接口:
    from backtest import BacktestEngine, backtest

    # 方式1: 使用引擎
    engine = BacktestEngine()
    result = engine.run('volatility', direction='min', weight='equal')

    # 方式2: 快捷函数
    result = backtest('volatility', direction='min')
"""

from .engine import BacktestEngine, backtest
from .backtester import Backtester, Portfolio, PerformanceAnalyzer
from .weight_optimizer import WeightOptimizer, RiskParityOptimizer

__all__ = [
    'BacktestEngine',
    'backtest',
    'Backtester',
    'Portfolio',
    'PerformanceAnalyzer',
    'WeightOptimizer',
    'RiskParityOptimizer',
]
