"""
因子计算引擎
整合数据加载 + 因子计算 + 结果保存
"""
import polars as pl
from typing import Callable, List, Union, Any
from pathlib import Path
from datetime import datetime, timedelta
import os


class FactorEngine:
    """
    因子计算引擎

    职责:
    1. 调用 DataManager 加载数据
    2. 调用因子函数计算
    3. 保存结果到 cache
    """

    def __init__(self, data_manager, cache_dir: str = None):
        """
        初始化引擎

        Args:
            data_manager: DataManager 实例
            cache_dir: 缓存目录
        """
        self.dm = data_manager

        if cache_dir is None:
            cache_dir = '/home/yunbo/project/quantitative/factor_production_v2/cache'

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"FactorEngine 初始化完成，缓存目录: {self.cache_dir}")

    def run(self,
            factor_func: Callable,
            stocks: Union[str, List[str]],
            start: str,
            end: str,
            fields: List[str],
            factor_col: str = None,
            save: bool = True,
            **kwargs) -> pl.DataFrame:
        """
        运行因子计算

        Args:
            factor_func: 因子计算函数
            stocks: 股票池名称或股票列表
            start: 开始日期
            end: 结束日期
            fields: 需要加载的字段
            factor_col: 因子列名（用于保存），默认用函数名
            save: 是否保存结果
            **kwargs: 传递给因子函数的额外参数

        Returns:
            计算结果 DataFrame
        """
        factor_name = factor_col or factor_func.__name__.replace('calc_', '')

        print(f"\n{'='*60}")
        print(f"开始计算因子: {factor_name}")
        print(f"股票池: {stocks}")
        print(f"日期范围: {start} ~ {end}")
        print(f"字段: {fields}")
        print(f"{'='*60}")

        # 1. 加载数据（后复权）
        df = self.dm.load(stocks, start, end, fields, adjust=True)

        if df.is_empty():
            print("错误: 数据加载失败")
            return pl.DataFrame()

        # 2. 计算因子
        print(f"\n计算因子中...")
        result = factor_func(df, **kwargs)

        # 3. 统计
        if factor_name in result.columns:
            factor_values = result[factor_name].drop_nulls()
            print(f"\n因子统计:")
            print(f"  有效值: {len(factor_values)}")
            print(f"  均值: {factor_values.mean():.4f}")
            print(f"  标准差: {factor_values.std():.4f}")
            print(f"  最小值: {factor_values.min():.4f}")
            print(f"  最大值: {factor_values.max():.4f}")

        # 4. 保存
        if save:
            save_path = self._save(result, factor_name, start, end)
            print(f"\n已保存到: {save_path}")

        return result

    def run_with_benchmark(self,
                           factor_func: Callable,
                           stocks: Union[str, List[str]],
                           start: str,
                           end: str,
                           fields: List[str],
                           benchmark: str = 'sh000852',
                           factor_col: str = None,
                           save: bool = True,
                           **kwargs) -> pl.DataFrame:
        """
        运行需要基准数据的因子计算（如特质波动率）

        Args:
            factor_func: 因子计算函数
            stocks: 股票池名称或股票列表
            start: 开始日期
            end: 结束日期
            fields: 需要加载的字段
            benchmark: 基准指数代码
            factor_col: 因子列名
            save: 是否保存结果
            **kwargs: 传递给因子函数的额外参数

        Returns:
            计算结果 DataFrame
        """
        factor_name = factor_col or factor_func.__name__.replace('calc_', '')

        print(f"\n{'='*60}")
        print(f"开始计算因子: {factor_name}")
        print(f"股票池: {stocks}")
        print(f"基准: {benchmark}")
        print(f"日期范围: {start} ~ {end}")
        print(f"{'='*60}")

        # 1. 加载个股数据
        df = self.dm.load(stocks, start, end, fields)

        if df.is_empty():
            print("错误: 数据加载失败")
            return pl.DataFrame()

        # 2. 加载基准数据
        benchmark_df = self.dm.load_benchmark(benchmark, start, end)

        if benchmark_df.is_empty():
            print("错误: 基准数据加载失败")
            return pl.DataFrame()

        # 3. 计算因子
        print(f"\n计算因子中...")
        result = factor_func(df, benchmark_df, **kwargs)

        # 4. 统计
        if factor_name in result.columns:
            factor_values = result[factor_name].drop_nulls()
            print(f"\n因子统计:")
            print(f"  有效值: {len(factor_values)}")
            print(f"  均值: {factor_values.mean():.4f}")
            print(f"  标准差: {factor_values.std():.4f}")

        # 5. 保存
        if save:
            save_path = self._save(result, factor_name, start, end)
            print(f"\n已保存到: {save_path}")

        return result

    def _save(self,
              df: pl.DataFrame,
              factor_name: str,
              start: str,
              end: str) -> str:
        """
        保存因子数据

        Args:
            df: 因子数据
            factor_name: 因子名称
            start: 开始日期
            end: 结束日期

        Returns:
            保存路径
        """
        # 格式化日期
        start_str = start.replace('-', '')
        end_str = end.replace('-', '')

        # 文件名
        filename = f"{factor_name}_{start_str}_{end_str}.parquet"
        save_path = self.cache_dir / filename

        # 删除旧文件
        for old_file in self.cache_dir.glob(f"{factor_name}_*.parquet"):
            old_file.unlink()
            print(f"  删除旧文件: {old_file.name}")

        # 保存为 parquet（比 CSV 更快更小）
        df.write_parquet(save_path)

        return str(save_path)

    def load(self, factor_name: str) -> pl.DataFrame:
        """
        加载已保存的因子数据

        Args:
            factor_name: 因子名称

        Returns:
            因子数据 DataFrame
        """
        files = list(self.cache_dir.glob(f"{factor_name}_*.parquet"))

        if len(files) == 0:
            print(f"未找到因子: {factor_name}")
            return pl.DataFrame()

        # 取最新的文件
        latest = sorted(files)[-1]
        print(f"加载因子: {latest}")

        return pl.read_parquet(latest)

    def list_factors(self) -> List[str]:
        """列出所有已保存的因子"""
        files = list(self.cache_dir.glob("*.parquet"))
        factors = set()

        for f in files:
            # 提取因子名（去掉日期后缀）
            name = f.stem.rsplit('_', 2)[0]
            factors.add(name)

        return sorted(list(factors))

    def update(self,
               factor_func: Callable,
               stocks: Union[str, List[str]],
               fields: List[str],
               end: str = None,
               lookback: int = 60,
               factor_col: str = None,
               **kwargs) -> pl.DataFrame:
        """
        增量更新因子

        逻辑:
        1. 读取已有因子，获取最后日期
        2. 加载 last_date - lookback 到 end 的数据（滚动计算需要历史数据）
        3. 计算因子
        4. 只保留新日期的结果
        5. 与旧数据合并并保存

        Args:
            factor_func: 因子计算函数
            stocks: 股票池名称或股票列表
            fields: 需要加载的字段
            end: 结束日期，默认今天
            lookback: 回看天数（用于滚动计算）
            factor_col: 因子列名
            **kwargs: 传递给因子函数的额外参数

        Returns:
            更新后的完整因子数据
        """
        factor_name = factor_col or factor_func.__name__.replace('calc_', '')

        if end is None:
            end = datetime.now().strftime('%Y-%m-%d')

        print(f"\n{'='*60}")
        print(f"增量更新因子: {factor_name}")
        print(f"{'='*60}")

        # 1. 读取已有因子数据
        old_df = self.load(factor_name)

        if old_df.is_empty():
            print("未找到已有因子数据，执行全量计算")
            # 没有旧数据，需要指定 start
            print("错误: 请先使用 run() 进行全量计算")
            return pl.DataFrame()

        # 获取已有数据的日期范围
        old_start = old_df['date'].min()
        old_end = old_df['date'].max()

        print(f"已有数据: {old_start} ~ {old_end}")

        # 2. 检查是否需要更新
        end_date = datetime.strptime(end, '%Y-%m-%d').date()

        if old_end >= end_date:
            print(f"数据已是最新，无需更新")
            return old_df

        # 3. 计算需要加载的数据范围
        # 从 old_end - lookback 开始加载，确保滚动计算有足够数据
        load_start = (datetime.combine(old_end, datetime.min.time())
                      - timedelta(days=lookback)).strftime('%Y-%m-%d')

        print(f"加载数据: {load_start} ~ {end}")

        # 4. 加载新数据
        new_df = self.dm.load(stocks, load_start, end, fields)

        if new_df.is_empty():
            print("错误: 新数据加载失败")
            return old_df

        # 5. 计算因子
        print(f"计算因子中...")
        calc_result = factor_func(new_df, **kwargs)

        # 6. 只保留 old_end 之后的新数据
        new_result = calc_result.filter(pl.col('date') > old_end)

        if new_result.is_empty():
            print("没有新数据需要添加")
            return old_df

        print(f"新增数据: {new_result['date'].min()} ~ {new_result['date'].max()}")
        print(f"新增记录: {len(new_result)} 条")

        # 7. 合并
        # 确保列顺序一致
        common_cols = [c for c in old_df.columns if c in new_result.columns]
        old_df = old_df.select(common_cols)
        new_result = new_result.select(common_cols)

        result = pl.concat([old_df, new_result])

        # 8. 统计
        if factor_name in result.columns:
            factor_values = result[factor_name].drop_nulls()
            print(f"\n更新后因子统计:")
            print(f"  总记录: {len(result)}")
            print(f"  有效值: {len(factor_values)}")
            print(f"  日期范围: {result['date'].min()} ~ {result['date'].max()}")

        # 9. 保存
        new_start = str(old_start)
        save_path = self._save(result, factor_name, new_start, end)
        print(f"\n已保存到: {save_path}")

        return result

    def get_factor_info(self, factor_name: str) -> dict:
        """
        获取因子信息

        Args:
            factor_name: 因子名称

        Returns:
            因子信息字典
        """
        files = list(self.cache_dir.glob(f"{factor_name}_*.parquet"))

        if len(files) == 0:
            return {'exists': False}

        latest = sorted(files)[-1]

        # 解析文件名获取日期范围
        parts = latest.stem.split('_')
        start_date = parts[-2]
        end_date = parts[-1]

        # 读取数据获取更多信息
        df = pl.read_parquet(latest)

        return {
            'exists': True,
            'file': str(latest),
            'start': f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}",
            'end': f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}",
            'rows': len(df),
            'stocks': df['stock'].n_unique(),
            'dates': df['date'].n_unique(),
        }
