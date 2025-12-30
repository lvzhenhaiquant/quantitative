"""
因子计算框架 v2

架构:
    data/           数据加载层 (QLib → Polars)
    factors/        因子计算层 (纯函数)
    engine/         调度引擎层 (整合数据+计算+保存)
    cache/          因子缓存

使用示例:
    from factor_production_v2 import DataManager, FactorEngine
    from factor_production_v2.factors import calc_turnover_mean

    dm = DataManager()
    engine = FactorEngine(dm)

    result = engine.run(
        factor_func=calc_turnover_mean,
        stocks='csi1000',
        start='2020-01-01',
        end='2025-12-17',
        fields=['$turnover_rate_f'],
        window=20
    )
"""

from .data import DataManager
from .engine import FactorEngine
from .neutralize import FactorNeutralizer, neutralize_factor

__all__ = ['DataManager', 'FactorEngine', 'FactorNeutralizer', 'neutralize_factor']
