"""
计算过去5年的残差波动率因子
时间范围：2020-12-11 ~ 2025-12-11（约5年）
"""

import sys
import os
import yaml
from datetime import datetime

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import DataLoader
from utils.logger import get_logger
from factor_production.factor_scheduler import FactorScheduler
from factor_production.market_factors.beta_factor import HistorySigmaFactor

def main():
    # 初始化日志
    logger = get_logger('main')

    logger.info("=" * 80)
    logger.info("开始计算5年残差波动率因子")
    logger.info("=" * 80)

    # 时间范围（5年）
    start_date = '2020-12-11'  # 5年前
    end_date = '2025-12-17'    # 今天

    logger.info(f"计算时间范围: {start_date} ~ {end_date}")
    logger.info(f"预计交易日数: ~1255天 (5年 × 251天/年)")

    # 1. 加载配置
    logger.info("\n[1/4] 加载配置...")
    with open('configs/scheduler_config.yaml', 'r') as f:
        scheduler_config = yaml.safe_load(f)

    with open('configs/factor_config.yaml', 'r') as f:
        factor_config = yaml.safe_load(f)

    # 2. 初始化数据加载器
    logger.info("[2/4] 初始化数据加载器...")
    data_loader = DataLoader()

    # 3. 初始化因子调度器
    logger.info("[3/4] 初始化因子调度器...")
    scheduler = FactorScheduler(data_loader, scheduler_config['scheduler'])

    # 4. 创建History_Sigma因子
    logger.info("[4/4] 创建History_Sigma因子...")
    params = factor_config['history_sigma_factor']['params']
    logger.info(f"因子参数:")
    logger.info(f"  - lookback_days: {params['lookback_days']}")
    logger.info(f"  - half_life: {params['half_life']}")
    logger.info(f"  - benchmark: {params['benchmark']}")

    history_sigma_factor = HistorySigmaFactor(params)

    # 5. 开始计算因子（使用多核并行加速）
    logger.info("\n" + "=" * 80)
    logger.info("开始并行计算因子（多核加速）")
    logger.info("=" * 80)

    start_time = datetime.now()

    try:
        save_path = scheduler.daily_calculate_parallel(
            factor=history_sigma_factor,
            instrument_name='csi1000',
            start_date=start_date,
            end_date=end_date,
            n_jobs=-1  # 使用所有CPU核心
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info("\n" + "=" * 80)
        logger.info("✓ 因子计算完成！")
        logger.info("=" * 80)
        logger.info(f"保存路径: {save_path}")
        logger.info(f"总耗时: {duration:.0f}秒 ({duration/60:.1f}分钟)")
        logger.info(f"平均速度: {duration/1255*1000:.0f}毫秒/天")

        return save_path

    except Exception as e:
        logger.error(f"计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()

    if result:
        print("\n" + "=" * 80)
        print("🎉 5年因子数据计算完成！")
        print("=" * 80)
        print(f"因子文件: {result}")
        print("\n下一步可以运行回测：")
        print("  python test_backtest.py")
    else:
        print("\n❌ 计算失败，请检查日志")