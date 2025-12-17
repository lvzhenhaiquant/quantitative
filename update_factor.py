"""
增量更新因子数据
从已有数据的最后日期开始，计算到最新日期
"""

import sys
import os
import yaml
import pandas as pd
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import DataLoader
from utils.logger import get_logger
from factor_production.factor_scheduler import FactorScheduler
from factor_production.market_factors.beta_factor import HistorySigmaFactor


def get_latest_factor_file():
    """获取最新的因子文件"""
    cache_dir = 'factor_production/cache'
    files = [f for f in os.listdir(cache_dir)
             if f.startswith('history_sigma_daily_') and f.endswith('.csv')
             and 'checkpoint' not in f]

    if not files:
        return None, None

    # 按修改时间排序，取最新的
    files.sort(key=lambda x: os.path.getmtime(os.path.join(cache_dir, x)), reverse=True)
    latest_file = os.path.join(cache_dir, files[0])

    # 读取最后日期
    df = pd.read_csv(latest_file)
    last_date = pd.to_datetime(df['date']).max().strftime('%Y-%m-%d')

    return latest_file, last_date


def main():
    logger = get_logger('update_factor')

    logger.info("=" * 80)
    logger.info("增量更新因子数据")
    logger.info("=" * 80)

    # 1. 获取已有因子文件的最后日期
    existing_file, last_date = get_latest_factor_file()

    if existing_file is None:
        logger.error("未找到已有因子文件，请先运行 calculate_5year_factor.py")
        return None

    logger.info(f"已有因子文件: {existing_file}")
    logger.info(f"数据截止日期: {last_date}")

    # 2. 设置更新范围
    start_date = (pd.to_datetime(last_date) + timedelta(days=1)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    logger.info(f"更新范围: {start_date} ~ {end_date}")

    # 检查是否需要更新
    if start_date >= end_date:
        logger.info("因子数据已是最新，无需更新")
        return existing_file

    # 3. 加载配置
    with open('configs/scheduler_config.yaml', 'r') as f:
        scheduler_config = yaml.safe_load(f)

    with open('configs/factor_config.yaml', 'r') as f:
        factor_config = yaml.safe_load(f)

    # 4. 初始化
    logger.info("初始化数据加载器和调度器...")
    data_loader = DataLoader()
    scheduler = FactorScheduler(data_loader, scheduler_config['scheduler'])

    params = factor_config['history_sigma_factor']['params']
    history_sigma_factor = HistorySigmaFactor(params)

    logger.info(f"因子参数: lookback={params['lookback_days']}, half_life={params['half_life']}")

    # 5. 【重要】先读取旧数据到内存（因为调度器会删除旧文件）
    logger.info("读取旧因子数据到内存...")
    old_df = pd.read_csv(existing_file)
    logger.info(f"旧数据: {len(old_df)} 条")

    # 6. 计算新因子数据
    logger.info("\n开始计算新因子数据...")
    start_time = datetime.now()

    try:
        new_factor_path = scheduler.daily_calculate_parallel(
            factor=history_sigma_factor,
            instrument_name='csi1000',
            start_date=start_date,
            end_date=end_date,
            n_jobs=-1
        )

        if new_factor_path is None:
            logger.warning("未生成新数据（可能没有新的交易日）")
            # 旧文件已被删除，需要重新保存
            first_date = pd.to_datetime(old_df['date']).min().strftime('%Y%m%d')
            last_date_old = pd.to_datetime(old_df['date']).max().strftime('%Y%m%d')
            restore_file = f'factor_production/cache/history_sigma_daily_{first_date}_{last_date_old}.csv'
            old_df.to_csv(restore_file, index=False)
            logger.info(f"已恢复旧数据: {restore_file}")
            return restore_file

        # 7. 合并新旧数据
        logger.info("\n合并新旧数据...")
        new_df = pd.read_csv(new_factor_path)

        logger.info(f"旧数据: {len(old_df)} 条")
        logger.info(f"新数据: {len(new_df)} 条")

        # 确定索引列名
        index_col = old_df.columns[0]  # 第一列是股票代码

        # 合并并去重
        combined_df = pd.concat([old_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=[index_col, 'date'], keep='last')
        combined_df = combined_df.sort_values('date')

        # 7. 保存合并后的文件
        first_date = pd.to_datetime(combined_df['date']).min().strftime('%Y%m%d')
        last_date_new = pd.to_datetime(combined_df['date']).max().strftime('%Y%m%d')

        output_file = f'factor_production/cache/history_sigma_daily_{first_date}_{last_date_new}.csv'
        combined_df.to_csv(output_file, index=False)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info("\n" + "=" * 80)
        logger.info("更新完成!")
        logger.info("=" * 80)
        logger.info(f"输出文件: {output_file}")
        logger.info(f"总记录数: {len(combined_df)}")
        logger.info(f"日期范围: {first_date} ~ {last_date_new}")
        logger.info(f"耗时: {duration:.1f}秒")

        # 清理临时文件
        if new_factor_path != output_file and os.path.exists(new_factor_path):
            os.remove(new_factor_path)
            logger.info(f"已清理临时文件: {new_factor_path}")

        return output_file

    except Exception as e:
        logger.error(f"更新失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = main()

    if result:
        print(f"\n因子数据已更新: {result}")
    else:
        print("\n更新失败，请检查日志")
