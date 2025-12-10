"""
因子调度器 - 负责因子的滚动计算和保存

功能:
1. 每日滚动计算因子
2. 保存到因子库 (CSV/HDF5/Parquet)
3. 支持增量更新
4. 断点续算
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import os
import sys
from tqdm import tqdm
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.data_loader import DataLoader
from factor_production.base_factor import BaseFactor


class FactorScheduler:
    """
    因子调度器 - 负责每日因子计算和保存

    主要功能:
    1. daily_calculate: 每日滚动计算因子
    2. incremental_update: 增量更新因子
    3. load_factor_data: 加载已保存的因子数据
    """

    def __init__(self, data_loader: DataLoader, config: Dict[str, Any]):
        """
        初始化因子调度器

        Parameters
        ----------
        data_loader : DataLoader
            数据加载器实例
        config : dict
            调度器配置
        """
        self.data_loader = data_loader
        self.config = config

        # 提取配置参数
        self.factor_save_dir = config.get('factor_save_dir', 'factor_production/cache')
        self.storage_format = config.get('storage', {}).get('format', 'csv')
        self.batch_size = config.get('performance', {}).get('batch_size', 10)
        self.show_progress = config.get('performance', {}).get('show_progress', True)
        self.save_interval = config.get('performance', {}).get('save_interval', 50)
        self.continue_on_error = config.get('error_handling', {}).get('continue_on_error', True)
        self.max_retries = config.get('error_handling', {}).get('max_retries', 3)
        self.retry_delay = config.get('error_handling', {}).get('retry_delay', 5)

        # 创建保存目录
        os.makedirs(self.factor_save_dir, exist_ok=True)

        # 日志
        self.logger = get_logger('scheduler.FactorScheduler')
        self.logger.info(f"FactorScheduler初始化完成")
        self.logger.info(f"因子保存目录: {self.factor_save_dir}")
        self.logger.info(f"存储格式: {self.storage_format}")

    def daily_calculate(self,
                       factor: BaseFactor,
                       instrument_name: str,
                       start_date: str,
                       end_date: str) -> str:
        """
        每日滚动计算因子

        对每个交易日:
        1. 获取这天的成分股
        2. 往前推lookback_days天加载数据
        3. 计算因子值
        4. 保存结果（带日期标记）

        Parameters
        ----------
        factor : BaseFactor
            因子实例
        instrument_name : str
            股票池名称 (例如: 'csi1000')
        start_date : str
            开始日期 'YYYY-MM-DD'
        end_date : str
            结束日期 'YYYY-MM-DD'

        Returns
        -------
        str
            保存的文件路径
        """
        self.logger.info("=" * 80)
        self.logger.info(f"开始每日滚动计算因子: {factor.name}")
        self.logger.info(f"股票池: {instrument_name}")
        self.logger.info(f"日期范围: {start_date} ~ {end_date}")
        self.logger.info("=" * 80)

        # Step 1: 获取交易日历
        trading_days = self._get_trading_days(start_date, end_date)
        total_days = len(trading_days)

        self.logger.info(f"共 {total_days} 个交易日需要计算")

        # Step 2: 滚动计算每个交易日的因子
        all_results = []
        success_count = 0
        fail_count = 0

        # 获取lookback参数
        lookback_days = factor.lookback_days
        buffer_days = 50  # 额外buffer防止数据不足

        # 使用进度条
        iterator = tqdm(enumerate(trading_days, 1), total=total_days,
                       desc=f"计算{factor.name}",
                       disable=not self.show_progress)

        for i, rebalance_date in iterator:
            try:
                rebalance_date_str = rebalance_date.strftime('%Y-%m-%d')

                # 1. 获取这天的成分股
                stock_list = self.data_loader.get_stock_list_by_date(
                    instrument_name,
                    rebalance_date_str
                )

                if len(stock_list) == 0:
                    self.logger.warning(f"[{i}/{total_days}] {rebalance_date_str}: 成分股为空，跳过")
                    fail_count += 1
                    continue

                # 2. 计算数据起始日期（往前推lookback_days + buffer）
                date_index = trading_days.get_loc(rebalance_date)
                start_index = max(0, date_index - lookback_days - buffer_days)
                data_start_date = trading_days[start_index].strftime('%Y-%m-%d')

                # 2.1 【重要】计算因子截止日期 = 调仓日的前一个交易日（避免未来函数）
                # 在调仓日当天，我们只能使用到前一天收盘的数据来计算因子
                if date_index > 0:
                    factor_end_date = trading_days[date_index - 1].strftime('%Y-%m-%d')
                    if i == 1:  # 只在第一次计算时打印说明
                        self.logger.info(f"✓ 避免未来函数: 调仓日{rebalance_date_str}的因子使用截至{factor_end_date}的数据计算")
                else:
                    # 如果是第一个交易日，无法获取前一天数据，跳过
                    self.logger.warning(f"[{i}/{total_days}] {rebalance_date_str}: 是第一个交易日，无法获取前一天数据，跳过")
                    fail_count += 1
                    continue

                # 3. 计算因子（带重试机制）
                # 注意：使用 factor_end_date（前一天）计算因子，但结果标记为 rebalance_date
                factor_values = self._calculate_with_retry(
                    factor=factor,
                    stock_list=stock_list,
                    data_start_date=data_start_date,
                    factor_end_date=factor_end_date,
                    rebalance_date=rebalance_date_str
                )

                if factor_values is None or len(factor_values) == 0:
                    self.logger.warning(f"[{i}/{total_days}] {rebalance_date_str}: 因子计算失败或结果为空")
                    fail_count += 1
                    continue

                # 4. 添加日期列
                factor_values['date'] = rebalance_date_str

                # 5. 添加到结果列表
                all_results.append(factor_values)
                success_count += 1

                # 6. 定期保存（防止中断丢失数据）
                if i % self.save_interval == 0:
                    self._save_checkpoint(all_results, factor.name,
                                        start_date, rebalance_date_str)

                # 更新进度条信息
                iterator.set_postfix({
                    'success': success_count,
                    'fail': fail_count,
                    'stocks': len(factor_values)
                })

            except Exception as e:
                self.logger.error(f"[{i}/{total_days}] {rebalance_date_str}: 发生错误 - {e}")
                fail_count += 1

                if not self.continue_on_error:
                    raise
                continue

        # Step 3: 合并所有结果并保存
        if len(all_results) == 0:
            self.logger.error("没有成功计算任何因子值!")
            return None

        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"滚动计算完成!")
        self.logger.info(f"成功: {success_count}期, 失败: {fail_count}期")
        self.logger.info("=" * 80)

        # 合并结果
        final_result = pd.concat(all_results, axis=0)

        # 保存最终结果
        save_path = self._save_factor_data(
            factor_data=final_result,
            factor_name=factor.name,
            start_date=start_date,
            end_date=end_date,
            freq='daily'
        )

        # 打印统计信息
        self._print_statistics(final_result, factor.name)

        return save_path

    def _get_trading_days(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
        """
        获取交易日历

        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期

        Returns
        -------
        pd.DatetimeIndex
            交易日列表
        """
        self.logger.info("获取交易日历...")

        # 从基准指数获取交易日历
        benchmark_prices = self.data_loader.load_benchmark_prices(
            benchmark='sh000300',
            start_date=start_date,
            end_date=end_date
        )

        trading_days = pd.to_datetime(benchmark_prices.index)

        self.logger.info(f"共 {len(trading_days)} 个交易日")
        self.logger.info(f"首个交易日: {trading_days[0]}")
        self.logger.info(f"最后交易日: {trading_days[-1]}")

        return trading_days

    def _calculate_with_retry(self,
                             factor: BaseFactor,
                             stock_list: List[str],
                             data_start_date: str,
                             factor_end_date: str,
                             rebalance_date: str) -> pd.DataFrame:
        """
        带重试机制的因子计算

        Parameters
        ----------
        factor : BaseFactor
            因子实例
        stock_list : list
            股票列表
        data_start_date : str
            数据起始日期
        factor_end_date : str
            因子计算截止日期（前一个交易日，避免未来函数）
        rebalance_date : str
            调仓日期（仅用于日志记录）

        Returns
        -------
        pd.DataFrame
            因子值
        """
        for attempt in range(self.max_retries):
            try:
                factor_values = factor.calculate(
                    stock_list=stock_list,
                    start_date=data_start_date,
                    end_date=factor_end_date,  # 使用前一天的日期，避免未来函数
                    data_loader=self.data_loader
                )
                return factor_values

            except Exception as e:
                if attempt < self.max_retries - 1:
                    self.logger.warning(f"计算失败，{self.retry_delay}秒后重试 "
                                      f"(第{attempt + 1}/{self.max_retries}次): {e}")
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"计算失败，已达最大重试次数: {e}")
                    raise

        return None

    def _save_checkpoint(self,
                        results: List[pd.DataFrame],
                        factor_name: str,
                        start_date: str,
                        current_date: str):
        """
        保存检查点（防止中断丢失数据）

        Parameters
        ----------
        results : list
            结果列表
        factor_name : str
            因子名称
        start_date : str
            开始日期
        current_date : str
            当前日期
        """
        if len(results) == 0:
            return

        checkpoint_data = pd.concat(results, axis=0)
        checkpoint_path = os.path.join(
            self.factor_save_dir,
            f"{factor_name}_checkpoint_{start_date}_{current_date}.csv"
        )

        checkpoint_data.to_csv(checkpoint_path, index=True)
        self.logger.info(f"检查点已保存: {checkpoint_path}")

    def _save_factor_data(self,
                         factor_data: pd.DataFrame,
                         factor_name: str,
                         start_date: str,
                         end_date: str,
                         freq: str = 'daily') -> str:
        """
        保存因子数据

        Parameters
        ----------
        factor_data : pd.DataFrame
            因子数据
        factor_name : str
            因子名称
        start_date : str
            开始日期
        end_date : str
            结束日期
        freq : str
            频率 ('daily', 'weekly', 'monthly')

        Returns
        -------
        str
            保存路径
        """
        # 生成文件名
        start_date_fmt = start_date.replace('-', '')
        end_date_fmt = end_date.replace('-', '')
        filename = f"{factor_name}_{freq}_{start_date_fmt}_{end_date_fmt}.{self.storage_format}"
        save_path = os.path.join(self.factor_save_dir, filename)

        # 根据格式保存
        if self.storage_format == 'csv':
            factor_data.to_csv(save_path, index=True)
        elif self.storage_format == 'hdf5':
            factor_data.to_hdf(save_path, key='factor', mode='w')
        elif self.storage_format == 'parquet':
            factor_data.to_parquet(save_path, index=True)
        else:
            raise ValueError(f"不支持的存储格式: {self.storage_format}")

        self.logger.info(f"因子数据已保存: {save_path}")

        return save_path

    def _print_statistics(self, factor_data: pd.DataFrame, factor_name: str):
        """打印统计信息"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("因子数据统计")
        self.logger.info("=" * 80)
        self.logger.info(f"总记录数: {len(factor_data)}")
        self.logger.info(f"数据形状: {factor_data.shape}")
        self.logger.info(f"时间范围: {factor_data['date'].min()} ~ {factor_data['date'].max()}")
        self.logger.info(f"总期数: {factor_data['date'].nunique()}")
        self.logger.info(f"每期平均股票数: {len(factor_data) / factor_data['date'].nunique():.0f}")

        # 因子值统计
        if factor_name in factor_data.columns:
            self.logger.info(f"\n因子值统计:")
            self.logger.info(f"  均值: {factor_data[factor_name].mean():.6f}")
            self.logger.info(f"  中位数: {factor_data[factor_name].median():.6f}")
            self.logger.info(f"  标准差: {factor_data[factor_name].std():.6f}")
            self.logger.info(f"  最小值: {factor_data[factor_name].min():.6f}")
            self.logger.info(f"  最大值: {factor_data[factor_name].max():.6f}")
            self.logger.info(f"  缺失值: {factor_data[factor_name].isna().sum()}")

    def load_factor_data(self,
                        factor_name: str,
                        start_date: str = None,
                        end_date: str = None,
                        freq: str = 'daily') -> pd.DataFrame:
        """
        加载已保存的因子数据

        Parameters
        ----------
        factor_name : str
            因子名称
        start_date : str, optional
            开始日期
        end_date : str, optional
            结束日期
        freq : str
            频率

        Returns
        -------
        pd.DataFrame
            因子数据
        """
        # 查找匹配的文件
        import glob
        pattern = os.path.join(self.factor_save_dir, f"{factor_name}_{freq}_*.{self.storage_format}")
        files = glob.glob(pattern)

        if len(files) == 0:
            self.logger.error(f"未找到因子文件: {pattern}")
            return None

        # 加载最新的文件
        latest_file = sorted(files)[-1]
        self.logger.info(f"加载因子数据: {latest_file}")

        if self.storage_format == 'csv':
            factor_data = pd.read_csv(latest_file, index_col=0)
        elif self.storage_format == 'hdf5':
            factor_data = pd.read_hdf(latest_file, key='factor')
        elif self.storage_format == 'parquet':
            factor_data = pd.read_parquet(latest_file)
        else:
            raise ValueError(f"不支持的存储格式: {self.storage_format}")

        # 筛选日期范围
        if start_date or end_date:
            if 'date' in factor_data.columns:
                if start_date:
                    factor_data = factor_data[factor_data['date'] >= start_date]
                if end_date:
                    factor_data = factor_data[factor_data['date'] <= end_date]

        self.logger.info(f"加载完成，共 {len(factor_data)} 条记录")

        return factor_data

    def __repr__(self):
        return (f"FactorScheduler(save_dir='{self.factor_save_dir}', "
                f"format='{self.storage_format}')")


if __name__ == "__main__":
    # 简单测试
    import yaml

    with open('/home/zhenhai1/quantitative/configs/scheduler_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    data_loader = DataLoader()
    scheduler = FactorScheduler(data_loader, config['scheduler'])

    print(scheduler)