"""
回测引擎 - 简化版
提供简洁的回测接口，只需指定因子名、方向、权重方法即可运行

使用示例:
    from backtest import BacktestEngine

    engine = BacktestEngine()

    # 最简调用
    result = engine.run('volatility', direction='min')

    # 完整调用
    result = engine.run(
        factor='volatility',
        direction='min',       # 'min' = 选最小的, 'max' = 选最大的
        weight='max_sharpe',   # 'equal' 或 'max_sharpe'
        n_stocks=30,           # 选股数量
        benchmark='sh000852'   # 基准指数
    )
"""
import os
import sys
import pandas as pd
import numpy as np
import polars as pl
from typing import Dict, Optional, Literal
from datetime import datetime
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, '/home/zhenhai1/quantitative')

from utils.data_loader import DataLoader
from backtest.backtester import Backtester, PerformanceAnalyzer
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class BacktestEngine:
    """
    简化版回测引擎

    设计理念:
    - 最少参数原则: 只需因子名+方向即可运行
    - 合理默认值: n_stocks=30, weight='equal', benchmark='sh000852'
    - 自动化: 自动加载因子、回测、计算指标、画图
    """

    # 因子缓存目录
    FACTOR_CACHE_DIR = '/home/zhenhai1/quantitative/factor_production/cache'

    # 默认配置
    DEFAULT_CONFIG = {
        'initial_cash': 10_000_000,
        'backtest_start_date': '2021-01-01',
        'transaction': {
            'commission_rate': 0.0003,
            'min_commission': 5,
            'stamp_tax': 0.001,
            'slippage': 0.001,
        },
        'risk': {
            'max_position': 0.10,
        },
        'output': {
            'save_dir': '/home/zhenhai1/quantitative/backtest/results'
        },
        'execution': {
            'trade_at_open': True,
        }
    }

    def __init__(self):
        """初始化回测引擎"""
        self.data_loader = DataLoader()
        print("BacktestEngine 初始化完成")

    def run(self,
            factor: str,
            direction: Literal['min', 'max'] = 'min',
            weight: Literal['equal', 'max_sharpe', 'min_vol'] = 'equal',
            n_stocks: int = 30,
            benchmark: str = 'sh000852',
            start_date: str = '2021-01-01',
            universe: str = 'csi1000',
            rebalance: Literal['daily', 'weekly', 'monthly'] = 'monthly',
            neutralize: bool = False,
            show_plot: bool = True) -> Dict:
        """
        运行回测

        Args:
            factor: 因子名称 (如 'volatility', 'turnover_mean')
            direction: 选股方向
                - 'min': 选因子值最小的股票 (低波动、低换手等)
                - 'max': 选因子值最大的股票
            weight: 权重方法
                - 'equal': 等权重
                - 'max_sharpe': 最大夏普比率
                - 'min_vol': 最小方差
            n_stocks: 选股数量 (默认30只)
            benchmark: 基准指数代码 (默认中证1000)
            start_date: 回测起始日期
            universe: 股票池名称 (默认 csi1000, 用于避免未来函数)
            rebalance: 调仓频率 (daily/weekly/monthly)
            show_plot: 是否显示图表

        Returns:
            回测结果字典，包含 metrics, portfolio_values, trades 等
        """
        print("\n" + "=" * 70)
        print(f"  回测配置")
        print("=" * 70)
        print(f"  因子: {factor}")
        print(f"  方向: {'选最小' if direction == 'min' else '选最大'}")
        print(f"  权重: {weight}")
        print(f"  选股: {n_stocks} 只")
        print(f"  调仓: {rebalance}")
        print(f"  基准: {benchmark}")
        print(f"  股票池: {universe}")
        print(f"  起始: {start_date}")
        print(f"  中性化: {'开启 (申万2级+流通市值)' if neutralize else '关闭'}")
        print("=" * 70)

        # 1. 加载因子数据
        factor_data = self._load_factor(factor)
        if factor_data is None:
            print(f"错误: 因子 {factor} 加载失败")
            return None

        # 2. 构建回测配置
        config = self._build_config(
            direction=direction,
            weight=weight,
            n_stocks=n_stocks,
            benchmark=benchmark,
            start_date=start_date,
            universe=universe,
            rebalance=rebalance,
            neutralize=neutralize
        )

        # 3. 运行回测
        backtester = Backtester(config, self.data_loader)
        results = backtester.run(factor, factor_data)

        # 4. 打印结果摘要
        if results:
            self._print_summary(results['metrics'], factor, direction, weight, n_stocks)

            # 5. 画图
            if show_plot:
                self._plot_results(results, factor, direction, weight, n_stocks, benchmark)

        return results

    def _load_factor(self, factor_name: str) -> Optional[pd.DataFrame]:
        """
        从缓存加载因子数据

        Args:
            factor_name: 因子名称

        Returns:
            因子数据 DataFrame (pandas格式，适配Backtester)
        """
        cache_dir = Path(self.FACTOR_CACHE_DIR)

        # 查找因子文件
        files = list(cache_dir.glob(f"{factor_name}_*.parquet"))

        if len(files) == 0:
            print(f"未找到因子文件: {factor_name}")
            print(f"请先使用 FactorEngine 计算因子")
            return None

        # 取最新的文件
        latest_file = sorted(files)[-1]
        print(f"加载因子: {latest_file.name}")

        # 读取 parquet (Polars) 并转换为 pandas
        df_pl = pl.read_parquet(latest_file)
        df_pd = df_pl.to_pandas()

        # 转换格式以适配 Backtester
        # Backtester 期望: index=stock, columns=['date', factor_name]
        df_pd['date'] = pd.to_datetime(df_pd['date']).dt.strftime('%Y-%m-%d')
        df_pd = df_pd.set_index('stock')

        print(f"因子数据: {len(df_pd)} 条, {df_pd['date'].nunique()} 个交易日")

        return df_pd

    def _build_config(self,
                      direction: str,
                      weight: str,
                      n_stocks: int,
                      benchmark: str,
                      start_date: str,
                      universe: str,
                      rebalance: str = 'monthly',
                      neutralize: bool = False) -> Dict:
        """构建回测配置"""
        config = self.DEFAULT_CONFIG.copy()

        # 选股配置
        config['selection'] = {
            'method': 'top_n',
            'n_stocks': n_stocks,
            'ascending': direction == 'min',  # min → ascending=True
        }

        # 权重配置
        config['weighting'] = {
            'method': weight,
            'lookback_days': 252,
            'max_weight': 0.10,
            'min_weight': 0.0,
            'risk_free_rate': 0.02,
            'use_dynamic_rf': weight == 'max_sharpe',  # 最大夏普时用动态SHIBOR
        }

        # 调仓频率
        config['rebalance'] = {'freq': rebalance}

        # 基准
        config['benchmark'] = {
            'code': benchmark,
            'name': self._get_benchmark_name(benchmark)
        }

        # 起始日期
        config['backtest_start_date'] = start_date

        # 股票池配置（避免未来函数）
        config['universe'] = {
            'name': universe,
            'filter': True  # 是否按日期筛选成分股
        }

        # 股票过滤配置（ST、停牌、涨跌停）
        config['stock_filter'] = {
            'exclude_st': True,
            'exclude_suspend': True,
            'exclude_limit': True,
        }

        # 中性化配置
        config['neutralize'] = {
            'enabled': neutralize,
            'how': ['industry', 'market_cap'],
            'industry_level': 2,  # 申万2级行业
            'market_cap_col': 'circ_mv',  # 流通市值
        }

        return config

    def _get_benchmark_name(self, code: str) -> str:
        """获取基准名称"""
        names = {
            'sh000852': '中证1000',
            'sh000300': '沪深300',
            'sh000905': '中证500',
            'sh000016': '上证50',
        }
        return names.get(code, code)

    def _print_summary(self, metrics: Dict, factor: str,
                       direction: str, weight: str, n_stocks: int):
        """打印结果摘要"""
        print("\n")
        print("+" + "-" * 50 + "+")
        print("|" + " " * 15 + "回测结果摘要" + " " * 17 + "|")
        print("+" + "-" * 50 + "+")

        # 策略描述
        desc = f"{factor} | {'最小' if direction == 'min' else '最大'}{n_stocks}只 | {weight}"
        print(f"| 策略: {desc:<42} |")
        print("+" + "-" * 50 + "+")

        # 核心指标
        print(f"| 累计收益    | {metrics['total_return']:>12.2%}  |" + " " * 19 + "|")
        print(f"| 年化收益    | {metrics['annual_return']:>12.2%}  |" + " " * 19 + "|")
        print(f"| 基准收益    | {metrics['benchmark_return']:>12.2%}  |" + " " * 19 + "|")
        print(f"| 超额收益    | {metrics['excess_return']:>12.2%}  |" + " " * 19 + "|")
        print("+" + "-" * 50 + "+")
        print(f"| 夏普比率    | {metrics['sharpe_ratio']:>12.4f}  |" + " " * 19 + "|")
        print(f"| 最大回撤    | {metrics['max_drawdown']:>12.2%}  |" + " " * 19 + "|")
        print(f"| 卡玛比率    | {metrics['calmar_ratio']:>12.4f}  |" + " " * 19 + "|")
        print(f"| 信息比率    | {metrics['information_ratio']:>12.4f}  |" + " " * 19 + "|")
        print("+" + "-" * 50 + "+")

    def _plot_results(self, results: Dict, factor: str,
                      direction: str, weight: str, n_stocks: int,
                      benchmark: str):
        """绘制回测结果图表"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        portfolio_values = results['portfolio_values']
        metrics = results['metrics']

        # 准备数据
        dates = pd.to_datetime(portfolio_values['date'])
        values = portfolio_values['total_value'].values
        nav = values / values[0]

        # 计算回撤
        running_max = pd.Series(nav).expanding().max()
        drawdown = (nav - running_max) / running_max

        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                                  gridspec_kw={'height_ratios': [3, 1]})

        # 上图: 净值曲线
        ax1 = axes[0]
        ax1.plot(dates, nav, label='Strategy NAV', linewidth=2, color='#1f77b4')
        ax1.axhline(y=1, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

        # 标题
        title = f"{factor} Strategy | {'Min' if direction == 'min' else 'Max'} {n_stocks} | {weight}"
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_ylabel('NAV', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 添加指标文本
        text = (f"Annual Return: {metrics['annual_return']:.2%}\n"
                f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
                f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
                f"Excess Return: {metrics['excess_return']:.2%}")
        ax1.text(0.02, 0.98, text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 下图: 回撤
        ax2 = axes[1]
        ax2.fill_between(dates, drawdown, 0, alpha=0.3, color='red')
        ax2.plot(dates, drawdown, color='red', linewidth=1)
        ax2.set_ylabel('Drawdown', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # 格式化日期
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        plt.xticks(rotation=45)
        plt.tight_layout()

        # 保存到因子子目录
        save_dir = Path(self.DEFAULT_CONFIG['output']['save_dir']) / factor
        save_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{direction}_{weight}_{n_stocks}.png"
        save_path = save_dir / filename

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n图表已保存: {save_path}")

        plt.close()

    def list_factors(self) -> list:
        """列出所有可用的因子"""
        cache_dir = Path(self.FACTOR_CACHE_DIR)
        files = list(cache_dir.glob("*.parquet"))

        factors = set()
        for f in files:
            # 提取因子名 (去掉日期后缀)
            name = f.stem.rsplit('_', 2)[0]
            factors.add(name)

        factors = sorted(list(factors))

        print("\n可用因子:")
        for i, f in enumerate(factors, 1):
            print(f"  {i}. {f}")

        return factors

    def compare(self,
                factor: str,
                directions: list = ['min', 'max'],
                weights: list = ['equal', 'max_sharpe'],
                n_stocks: int = 30) -> pd.DataFrame:
        """
        比较不同配置的回测结果

        Args:
            factor: 因子名称
            directions: 方向列表
            weights: 权重方法列表
            n_stocks: 选股数量

        Returns:
            比较结果表格
        """
        results = []

        for direction in directions:
            for weight in weights:
                print(f"\n>>> 测试: {direction} + {weight}")
                res = self.run(factor, direction, weight, n_stocks, show_plot=False)

                if res:
                    metrics = res['metrics']
                    results.append({
                        'direction': direction,
                        'weight': weight,
                        'annual_return': metrics['annual_return'],
                        'sharpe': metrics['sharpe_ratio'],
                        'max_drawdown': metrics['max_drawdown'],
                        'excess_return': metrics['excess_return'],
                    })

        df = pd.DataFrame(results)

        print("\n" + "=" * 70)
        print("  对比结果")
        print("=" * 70)
        print(df.to_string(index=False))

        return df


# 便捷函数
def backtest(factor: str,
             direction: Literal['min', 'max'] = 'min',
             weight: Literal['equal', 'max_sharpe'] = 'equal',
             n_stocks: int = 30) -> Dict:
    """
    快速回测函数

    Example:
        from backtest import backtest
        result = backtest('volatility', direction='min', weight='equal')
    """
    engine = BacktestEngine()
    return engine.run(factor, direction, weight, n_stocks)
