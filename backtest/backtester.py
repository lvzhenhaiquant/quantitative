"""
回测系统 - 整合版

包含:
1. Portfolio - 投资组合管理
2. PerformanceAnalyzer - 业绩分析
3. Backtester - 回测引擎
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import os
import sys
import struct
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.data_loader import DataLoader
from backtest.weight_optimizer import WeightOptimizer, RiskParityOptimizer

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


class Portfolio:
    """投资组合管理"""

    def __init__(self, initial_cash: float, config: Dict):
        """
        初始化投资组合

        Parameters
        ----------
        initial_cash : float
            初始资金
        config : dict
            配置参数
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}  # {stock_code: shares}
        self.config = config

        # 交易成本参数
        self.commission_rate = config.get('transaction', {}).get('commission_rate', 0.0003)
        self.min_commission = config.get('transaction', {}).get('min_commission', 5)
        self.stamp_tax = config.get('transaction', {}).get('stamp_tax', 0.001)
        self.slippage = config.get('transaction', {}).get('slippage', 0.001)

        # 风险控制参数
        self.max_position = config.get('risk', {}).get('max_position', 0.05)

        # 记录
        self.trades = []
        self.position_history = []
        self.value_history = []

        self.logger = get_logger('backtest.Portfolio')

    def rebalance(self, target_weights: Dict[str, float], prices: Dict[str, float], date: str):
        """
        调仓到目标权重

        Parameters
        ----------
        target_weights : dict
            目标权重 {stock_code: weight}
        prices : dict
            当前价格 {stock_code: price}
        date : str
            日期
        """
        # 计算当前市值
        current_value = self.calculate_portfolio_value(prices)

        # 计算目标持仓（股数）
        target_positions = {}
        for stock, weight in target_weights.items():
            if stock not in prices or pd.isna(prices[stock]) or prices[stock] <= 0:
                continue

            # 应用最大仓位限制
            weight = min(weight, self.max_position)

            # 计算目标市值
            target_amount = current_value * weight

            # 计算应持有股数（100股为一手）
            shares = int(target_amount / prices[stock] / 100) * 100

            if shares > 0:
                target_positions[stock] = shares

        # 执行交易
        self._execute_trades(target_positions, prices, date)

    def _execute_trades(self, target_positions: Dict[str, int],
                       prices: Dict[str, float], date: str):
        """执行交易"""
        # 1. 卖出不在目标组合中的股票
        for stock in list(self.positions.keys()):
            if stock not in target_positions:
                # 检查价格是否有效
                if stock in prices and not pd.isna(prices[stock]) and prices[stock] > 0:
                    self._sell(stock, self.positions[stock], prices[stock], date)
                else:
                    # 价格无效（如停牌），强制清仓（按0处理或保留持仓）
                    # 这里选择强制清仓，以免影响组合
                    if stock in self.positions:
                        del self.positions[stock]

        # 2. 调整现有持仓
        for stock in list(self.positions.keys()):
            if stock in target_positions:
                # 检查价格是否有效
                if stock not in prices or pd.isna(prices[stock]) or prices[stock] <= 0:
                    continue  # 价格无效，跳过调整

                current_shares = self.positions[stock]
                target_shares = target_positions[stock]

                if target_shares < current_shares:
                    # 减仓
                    self._sell(stock, current_shares - target_shares, prices[stock], date)
                elif target_shares > current_shares:
                    # 加仓
                    self._buy(stock, target_shares - current_shares, prices[stock], date)

        # 3. 买入新股票
        for stock, shares in target_positions.items():
            if stock not in self.positions and shares > 0:
                # 检查价格是否有效
                if stock in prices and not pd.isna(prices[stock]) and prices[stock] > 0:
                    self._buy(stock, shares, prices[stock], date)

    def _buy(self, stock: str, shares: int, price: float, date: str):
        """买入"""
        if shares <= 0 or price <= 0:
            return

        # 考虑滑点
        actual_price = price * (1 + self.slippage)

        # 计算成本
        amount = shares * actual_price
        commission = max(amount * self.commission_rate, self.min_commission)
        total_cost = amount + commission

        # 检查资金是否足够
        if total_cost > self.cash:
            # 资金不足,调整股数
            available_amount = self.cash - self.min_commission
            if available_amount <= 0:
                return
            shares = int((available_amount / (1 + self.commission_rate)) / actual_price / 100) * 100
            if shares <= 0:
                return

            amount = shares * actual_price
            commission = max(amount * self.commission_rate, self.min_commission)
            total_cost = amount + commission

        # 执行买入
        self.cash -= total_cost
        self.positions[stock] = self.positions.get(stock, 0) + shares

        # 记录交易
        self.trades.append({
            'date': date,
            'stock': stock,
            'action': 'BUY',
            'shares': shares,
            'price': actual_price,
            'amount': amount,
            'commission': commission,
            'total_cost': total_cost
        })

    def _sell(self, stock: str, shares: int, price: float, date: str):
        """卖出"""
        if shares <= 0 or stock not in self.positions or price <= 0:
            return

        # 不能卖出超过持有量
        shares = min(shares, self.positions[stock])

        # 考虑滑点
        actual_price = price * (1 - self.slippage)

        # 计算收入
        amount = shares * actual_price
        commission = max(amount * self.commission_rate, self.min_commission)
        stamp_tax = amount * self.stamp_tax
        total_revenue = amount - commission - stamp_tax

        # 执行卖出
        self.cash += total_revenue
        self.positions[stock] -= shares

        if self.positions[stock] <= 0:
            del self.positions[stock]

        # 记录交易
        self.trades.append({
            'date': date,
            'stock': stock,
            'action': 'SELL',
            'shares': shares,
            'price': actual_price,
            'amount': amount,
            'commission': commission,
            'stamp_tax': stamp_tax,
            'total_revenue': total_revenue
        })

    def calculate_portfolio_value(self, prices: Dict[str, float]) -> float:
        """计算组合总市值"""
        position_value = 0
        for stock, shares in self.positions.items():
            if stock in prices and not pd.isna(prices[stock]) and prices[stock] > 0:
                position_value += shares * prices[stock]

        return self.cash + position_value

    def record_state(self, date: str, prices: Dict[str, float]):
        """记录当前状态"""
        portfolio_value = self.calculate_portfolio_value(prices)

        # 记录净值
        self.value_history.append({
            'date': date,
            'cash': self.cash,
            'position_value': portfolio_value - self.cash,
            'total_value': portfolio_value
        })

        # 记录持仓
        for stock, shares in self.positions.items():
            if stock in prices:
                self.position_history.append({
                    'date': date,
                    'stock': stock,
                    'shares': shares,
                    'price': prices[stock],
                    'market_value': shares * prices[stock]
                })


class PerformanceAnalyzer:
    """业绩分析器"""

    def __init__(self):
        self.logger = get_logger('backtest.PerformanceAnalyzer')

    def analyze(self, portfolio_values: pd.DataFrame, benchmark_values: pd.DataFrame) -> Dict:
        """
        分析业绩指标

        Parameters
        ----------
        portfolio_values : DataFrame
            组合净值序列, columns=['date', 'total_value']
        benchmark_values : Series
            基准净值序列, index=日期, values=价格

        Returns
        -------
        dict
            业绩指标
        """
        # 对齐日期：只保留portfolio有数据的日期
        portfolio_dates = pd.to_datetime(portfolio_values['date'])
        benchmark_aligned = benchmark_values.loc[portfolio_dates]

        # 归一化：都除以第一个值，保证同一起点
        portfolio_nav = portfolio_values['total_value'].values / portfolio_values['total_value'].values[0]
        benchmark_nav = benchmark_aligned.values / benchmark_aligned.values[0]

        # 计算收益率（用于波动率、夏普比率等指标）
        portfolio_returns = pd.Series(portfolio_nav).pct_change().fillna(0)
        benchmark_returns = pd.Series(benchmark_nav).pct_change().fillna(0)

        portfolio_returns.index = portfolio_dates
        benchmark_returns.index = portfolio_dates

        # 计算指标
        metrics = {}

        # 1. 累计收益率（减1转为收益率）
        metrics['total_return'] = portfolio_nav[-1] - 1  # 策略累计收益率
        metrics['benchmark_return'] = benchmark_nav[-1] - 1  # 基准累计收益率
        metrics['excess_return'] = (portfolio_nav[-1] - 1) - (benchmark_nav[-1] - 1)  # 超额收益率

        # 2. 年化收益
        metrics['annual_return'] = self._annual_return(portfolio_returns)
        metrics['benchmark_annual_return'] = self._annual_return(benchmark_returns)

        # 3. 波动率
        metrics['volatility'] = self._volatility(portfolio_returns)
        metrics['benchmark_volatility'] = self._volatility(benchmark_returns)

        # 4. 夏普比率
        metrics['sharpe_ratio'] = self._sharpe_ratio(portfolio_returns)

        # 5. 最大回撤
        cumulative_returns = (1 + portfolio_returns).cumprod()
        metrics['max_drawdown'] = self._max_drawdown(cumulative_returns)

        # 6. 信息比率
        metrics['information_ratio'] = self._information_ratio(portfolio_returns, benchmark_returns)

        # 7. 胜率
        metrics['win_rate'] = (portfolio_returns > 0).sum() / len(portfolio_returns)

        # 8. 卡玛比率 (收益/最大回撤)
        metrics['calmar_ratio'] = metrics['annual_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0

        return metrics

    def _calculate_returns(self, values: pd.Series) -> pd.Series:
        """计算收益率"""
        return values.pct_change().fillna(0)

    def _total_return(self, returns: pd.Series) -> float:
        """累计收益"""
        return (1 + returns).prod() - 1

    def _annual_return(self, returns: pd.Series) -> float:
        """年化收益"""
        total_return = self._total_return(returns)
        n_days = len(returns)
        n_years = n_days / 252  # 假设252个交易日/年
        return (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    def _volatility(self, returns: pd.Series) -> float:
        """年化波动率"""
        return returns.std() * np.sqrt(252)

    def _sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.03) -> float:
        """夏普比率"""
        annual_return = self._annual_return(returns)
        volatility = self._volatility(returns)
        return (annual_return - risk_free_rate) / volatility if volatility != 0 else 0

    def _max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """最大回撤"""
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()

    def _information_ratio(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """信息比率"""
        excess_returns = portfolio_returns - benchmark_returns
        return (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252)) if excess_returns.std() != 0 else 0


class Backtester:
    """
    回测引擎

    功能:
    1. 加载因子数据
    2. 根据频率筛选调仓日
    3. 选股 + 构建投资组合
    4. 计算收益和风险指标
    """

    def __init__(self, config: Dict, data_loader: DataLoader):
        """
        初始化回测引擎

        Parameters
        ----------
        config : dict
            回测配置
        data_loader : DataLoader
            数据加载器
        """
        self.config = config
        self.data_loader = data_loader
        self.logger = get_logger('backtest.Backtester')

        # 提取配置
        self.initial_cash = config.get('initial_cash', 10000000)
        self.backtest_start_date = config.get('backtest_start_date', None)  # 回测起始日期
        self.selection_method = config.get('selection', {}).get('method', 'top_n')
        self.n_stocks = config.get('selection', {}).get('n_stocks', 50)
        self.ascending = config.get('selection', {}).get('ascending', True)
        self.weighting_method = config.get('weighting', {}).get('method', 'equal')
        self.rebalance_freq = config.get('rebalance', {}).get('freq', 'weekly')
        self.benchmark_code = config.get('benchmark', {}).get('code', 'sh000852')  # 默认中证1000
        self.trade_at_open = config.get('execution', {}).get('trade_at_open', True)  # 默认开盘价交易

        # 股票池配置（用于避免未来函数）
        self.universe = config.get('universe', {}).get('name', 'csi1000')
        self.filter_by_universe = config.get('universe', {}).get('filter', True)  # 是否按成分股筛选

        # 成分股缓存（避免重复加载）
        self._universe_cache = None

        # 股票过滤配置（ST、停牌、涨跌停）
        filter_config = config.get('stock_filter', {})
        self.filter_st = filter_config.get('exclude_st', True)
        self.filter_suspend = filter_config.get('exclude_suspend', True)
        self.filter_limit = filter_config.get('exclude_limit', True)

        # 初始化过滤器
        self._stock_filter = None
        if self.filter_st or self.filter_suspend or self.filter_limit:
            try:
                from backtest.filters import StockFilter
                self._stock_filter = StockFilter()
            except Exception as e:
                self.logger.warning(f"股票过滤器初始化失败: {e}，将跳过过滤")

        # 中性化配置
        neutralize_config = config.get('neutralize', {})
        self.neutralize_enabled = neutralize_config.get('enabled', False)
        self.neutralize_how = neutralize_config.get('how', ['industry', 'market_cap'])
        self.neutralize_industry_level = neutralize_config.get('industry_level', 2)
        self.neutralize_cap_col = neutralize_config.get('market_cap_col', 'circ_mv')

        # 初始化中性化器
        self._neutralizer = None
        if self.neutralize_enabled:
            try:
                from factor_production.neutralize import FactorNeutralizer
                self._neutralizer = FactorNeutralizer(industry_level=self.neutralize_industry_level)
                self.logger.info(f"中性化: {self.neutralize_how}, 行业级别={self.neutralize_industry_level}")
            except Exception as e:
                self.logger.warning(f"中性化器初始化失败: {e}，将跳过中性化")

        # 创建结果保存目录
        self.save_dir = config.get('output', {}).get('save_dir', 'backtest/results')
        os.makedirs(self.save_dir, exist_ok=True)

        # 权重优化器配置
        weighting_config = config.get('weighting', {})
        self.use_dynamic_rf = weighting_config.get('use_dynamic_rf', False)
        self.weight_optimizer = WeightOptimizer(
            method=weighting_config.get('method', 'equal'),
            lookback_days=weighting_config.get('lookback_days', 252),
            max_weight=weighting_config.get('max_weight', 0.05),
            min_weight=weighting_config.get('min_weight', 0.0),
            risk_free_rate=weighting_config.get('risk_free_rate', 0.02),
            use_dynamic_rf=self.use_dynamic_rf
        )

        # 价格数据缓存（用于权重优化）
        self.price_df_cache = None

        # 交易价格缓存（预加载到内存，加速回测）
        # 格式: {date_str: {stock: {'open': price, 'close': price}}}
        self.trade_price_cache = {}

        # QLib 数据配置
        self.qlib_path = config.get('qlib_path', '/home/yunbo/project/quantitative/qlib_data/cn_data')
        self.use_qlib = config.get('use_qlib', True)  # 默认使用 QLib 数据
        self._qlib_calendar = None
        self._qlib_calendar_index = None

        self.logger.info("Backtester初始化完成")
        self.logger.info(f"交易时机: {'开盘价' if self.trade_at_open else '收盘价'}")
        self.logger.info(f"权重方法: {self.weighting_method}")
        self.logger.info(f"动态无风险利率(SHIBOR): {self.use_dynamic_rf}")
        self.logger.info(f"股票池: {self.universe} (按日筛选: {self.filter_by_universe})")
        self.logger.info(f"过滤: ST={self.filter_st}, 停牌={self.filter_suspend}, 涨跌停={self.filter_limit}")

    def run(self, factor_name: str, factor_data: pd.DataFrame = None) -> Dict:
        """
        运行回测

        Parameters
        ----------
        factor_name : str
            因子名称
        factor_data : DataFrame, optional
            因子数据，如果为None则从文件加载

        Returns
        -------
        dict
            回测结果
        """
        self.logger.info("=" * 80)
        self.logger.info(f"开始回测: {factor_name}")
        self.logger.info(f"调仓频率: {self.rebalance_freq}")
        self.logger.info(f"选股方法: {self.selection_method}  (N={self.n_stocks})")
        self.logger.info("=" * 80)

        # 1. 加载因子数据
        if factor_data is None:
            from factor_production.factor_scheduler import FactorScheduler
            scheduler = FactorScheduler(self.data_loader, {})
            factor_data = scheduler.load_factor_data(factor_name, freq='daily')

        if factor_data is None or len(factor_data) == 0:
            self.logger.error("因子数据加载失败!")
            return None

        # 1.1 根据配置过滤回测起始日期
        if self.backtest_start_date:
            factor_data = factor_data[factor_data['date'] >= self.backtest_start_date]
            self.logger.info(f"回测起始日期: {self.backtest_start_date}")

        self.logger.info(f"因子数据: {len(factor_data)}条记录, "
                       f"{factor_data['date'].nunique()}个交易日")

        # 1.2 因子中性化（行业+市值）
        if self.neutralize_enabled and self._neutralizer is not None:
            self.logger.info("开始因子中性化...")
            factor_data = self._neutralize_factor(factor_data, factor_name)

        # 2. 筛选调仓日
        rebalance_dates = self._get_rebalance_dates(factor_data)
        self.logger.info(f"调仓日数量: {len(rebalance_dates)}")

        # 3. 创建投资组合
        portfolio = Portfolio(self.initial_cash, self.config)

        # 4. 加载基准数据
        start_date = factor_data['date'].min()
        end_date = factor_data['date'].max()

        benchmark_data = self.data_loader.load_benchmark_prices(
            self.benchmark_code,
            start_date,
            end_date
        )

        # 4.1 预加载所有交易价格到内存（加速回测）
        self.logger.info("预加载交易价格数据到内存...")
        all_stocks = factor_data.index.unique().tolist()

        if self.use_qlib:
            # 使用 QLib bin 文件加载（更快）
            self.trade_price_cache = self._load_qlib_prices(all_stocks, start_date, end_date)
        else:
            # 使用 Parquet 加载
            price_data = self.data_loader.load_stock_prices(
                stock_list=all_stocks,
                start_date=start_date,
                end_date=end_date,
                fields=['$open', '$close'],
                adjust=True  # 后复权
            )

            if price_data is not None and len(price_data) > 0:
                # 构建缓存: {date_str: {stock: {'open': price, 'close': price}}}
                self.trade_price_cache = {}
                for (stock, date), row in price_data.iterrows():
                    date_str = date.strftime('%Y-%m-%d')
                    if date_str not in self.trade_price_cache:
                        self.trade_price_cache[date_str] = {}
                    self.trade_price_cache[date_str][stock] = {
                        'open': row['$open'],
                        'close': row['$close']
                    }
                self.logger.info(f"价格缓存: {len(self.trade_price_cache)} 个交易日, {len(all_stocks)} 只股票")
            else:
                self.logger.warning("价格数据加载失败，将使用逐日加载模式")
                self.trade_price_cache = {}

        # 4.2 加载历史价格数据（用于min_vol/max_sharpe权重优化）
        if self.weighting_method in ['min_vol', 'max_sharpe']:
            self.logger.info("加载历史价格数据用于权重优化...")
            # 获取所有可能涉及的股票
            all_stocks = factor_data.index.unique().tolist()
            # 计算需要的历史数据起始日期（lookback_days + buffer）
            lookback_start = pd.Timestamp(start_date) - pd.Timedelta(days=400)
            price_data = self.data_loader.load_stock_prices(
                stock_list=all_stocks,
                start_date=lookback_start.strftime('%Y-%m-%d'),
                end_date=end_date,
                fields=['$close'],
                adjust=True  # 后复权
            )
            # 转换为宽表格式：index=日期, columns=股票代码
            if price_data is not None and len(price_data) > 0:
                price_pivot = price_data['$close'].unstack(level=0)
                self.price_df_cache = price_pivot
                self.logger.info(f"价格数据: {len(price_pivot)}天, {len(price_pivot.columns)}只股票")
            else:
                self.logger.warning("无法加载价格数据，将使用等权")
                self.price_df_cache = None

        # 4.2 加载SHIBOR数据（用于max_sharpe动态无风险利率）
        if self.weighting_method == 'max_sharpe' and self.use_dynamic_rf:
            self.logger.info("加载SHIBOR数据用于动态无风险利率...")
            shibor_df = self._load_shibor_data(start_date, end_date)
            if shibor_df is not None:
                self.weight_optimizer.set_shibor_data(shibor_df)
            else:
                self.logger.warning("SHIBOR数据加载失败，将使用固定无风险利率")

        # 5. 执行回测
        self.logger.info("\n开始模拟交易...")

        # 获取所有交易日（用于每日记录净值）
        all_trading_dates = sorted(factor_data['date'].unique())
        rebalance_set = set(rebalance_dates)  # 转为集合便于查找

        self.logger.info(f"交易日: {len(all_trading_dates)}, 调仓日: {len(rebalance_dates)}")

        for i, current_date in enumerate(all_trading_dates, 1):
            is_rebalance_day = current_date in rebalance_set

            # 加载当日价格（用于记录净值）
            stocks_to_load = list(portfolio.positions.keys())

            # 如果是调仓日，执行调仓
            if is_rebalance_day:
                self.logger.info(f"[调仓] {current_date}")

                # 获取前一交易日的因子值（避免未来函数）
                prev_dates = [d for d in all_trading_dates if d < current_date]
                if len(prev_dates) == 0:
                    self.logger.warning(f"  跳过: 无前一交易日因子数据")
                    prices = self._load_prices(stocks_to_load, current_date) if stocks_to_load else {}
                    portfolio.record_state(current_date, prices)
                    continue

                prev_factor_date = prev_dates[-1]
                factor_values = factor_data[factor_data['date'] == prev_factor_date]

                # 设置股票代码为索引
                if 'stock' in factor_values.columns:
                    factor_values = factor_values.set_index('stock')

                if len(factor_values) == 0:
                    self.logger.warning(f"  跳过: 无因子数据")
                    prices = self._load_prices(stocks_to_load, current_date) if stocks_to_load else {}
                    portfolio.record_state(current_date, prices)
                    continue

                # 选股
                selected_stocks = self._select_stocks(factor_values, factor_name, date=current_date)

                if len(selected_stocks) == 0:
                    self.logger.warning(f"  跳过: 选股为空")
                    prices = self._load_prices(stocks_to_load, current_date) if stocks_to_load else {}
                    portfolio.record_state(current_date, prices)
                    continue

                # 加载价格（包括选中股票和当前持仓）
                stocks_to_load = list(set(selected_stocks + list(portfolio.positions.keys())))
                prices = self._load_prices(stocks_to_load, current_date)

                # 计算目标权重
                target_weights = self._calculate_weights(
                    selected_stocks, factor_values, factor_name,
                    current_date=pd.Timestamp(current_date)
                )

                # 调仓
                portfolio.rebalance(target_weights, prices, current_date)

                self.logger.info(f"  选股: {len(selected_stocks)}只, "
                               f"持仓: {len(portfolio.positions)}只, "
                               f"净值: {portfolio.calculate_portfolio_value(prices)/self.initial_cash:.4f}")
            else:
                # 非调仓日，只加载持仓股票价格
                prices = self._load_prices(stocks_to_load, current_date) if stocks_to_load else {}

            # 每个交易日都记录净值
            portfolio.record_state(current_date, prices)

        # 6. 业绩分析
        self.logger.info("\n开始业绩分析...")

        portfolio_values = pd.DataFrame(portfolio.value_history)
        analyzer = PerformanceAnalyzer()

        metrics = analyzer.analyze(portfolio_values, benchmark_data)

        # 7. 打印结果
        self._print_results(metrics)

        # 8. 保存结果
        results = {
            'metrics': metrics,
            'portfolio_values': portfolio_values,
            'trades': pd.DataFrame(portfolio.trades),
            'positions': pd.DataFrame(portfolio.position_history)
        }

        self._save_results(results, factor_name)

        # 9. 绘制策略曲线
        self._plot_performance(portfolio_values, benchmark_data, factor_name)

        return results

    def _get_rebalance_dates(self, factor_data: pd.DataFrame) -> List[str]:
        """获取调仓日期"""
        all_dates = pd.to_datetime(factor_data['date'].unique()).sort_values()

        if self.rebalance_freq == 'daily':
            return [d.strftime('%Y-%m-%d') for d in all_dates]

        elif self.rebalance_freq == 'weekly':
            # 使用"前后两天连不上"的逻辑
            rebalance_dates = []
            for i in range(len(all_dates)):
                if i == 0:
                    rebalance_dates.append(all_dates[i])
                else:
                    gap = (all_dates[i] - all_dates[i-1]).days
                    if gap > 1:
                        rebalance_dates.append(all_dates[i])

            return [d.strftime('%Y-%m-%d') for d in rebalance_dates]

        elif self.rebalance_freq == 'monthly':
            # 每月第一个交易日
            return factor_data.groupby(pd.to_datetime(factor_data['date']).dt.to_period('M'))['date'].first().tolist()

        else:
            raise ValueError(f"不支持的调仓频率: {self.rebalance_freq}")

    def _get_universe_stocks(self, date: str) -> List[str]:
        """
        获取指定日期的股票池成分股（避免未来函数）

        Parameters
        ----------
        date : str
            日期

        Returns
        -------
        List[str]
            当日有效的成分股列表
        """
        if not self.filter_by_universe:
            return None  # 不筛选，返回 None

        # 使用缓存避免重复加载 instruments 文件
        if self._universe_cache is None:
            try:
                self._universe_cache = self.data_loader.load_instruments(self.universe)
            except Exception as e:
                self.logger.warning(f"加载股票池 {self.universe} 失败: {e}，将不按成分股筛选")
                self.filter_by_universe = False
                return None

        # 筛选在 date 有效的股票
        target_date = pd.to_datetime(date)
        valid_stocks = self._universe_cache[
            (pd.to_datetime(self._universe_cache['start_date']) <= target_date) &
            (pd.to_datetime(self._universe_cache['end_date']) >= target_date)
        ]

        return valid_stocks['code'].tolist()

    def _select_stocks(self, factor_values: pd.DataFrame, factor_name: str,
                       date: str = None) -> List[str]:
        """
        选股（先筛选当日成分股，过滤ST/停牌/涨跌停，再按因子排序）

        Parameters
        ----------
        factor_values : pd.DataFrame
            因子数据
        factor_name : str
            因子名称
        date : str, optional
            调仓日期（用于筛选当日成分股和过滤）

        Returns
        -------
        List[str]
            选中的股票列表
        """
        # DEBUG: 打印初始状态
        # self.logger.debug(f"  [选股] 初始因子数据: {len(factor_values)} 行, 列: {factor_values.columns.tolist()}")

        # 1. 先筛选当日成分股（避免未来函数）
        if date and self.filter_by_universe:
            universe_stocks = self._get_universe_stocks(date)
            if universe_stocks:
                # 转换为大写以匹配因子数据的股票代码格式
                universe_stocks = [s.upper() for s in universe_stocks]
                # 只保留当日是成分股的股票
                before = len(factor_values)
                factor_values = factor_values[factor_values.index.isin(universe_stocks)]
                # self.logger.debug(f"  [选股] 筛选成分股: {before} -> {len(factor_values)}")

        if len(factor_values) == 0:
            self.logger.warning(f"  [选股] 筛选成分股后为空")
            return []

        # 2. 过滤ST、停牌、涨跌停股票
        if date and self._stock_filter is not None:
            all_stocks = factor_values.index.tolist()
            filtered_stocks = self._stock_filter.filter_stocks(
                all_stocks,
                date,
                exclude_st=self.filter_st,
                exclude_suspend=self.filter_suspend,
                exclude_limit=self.filter_limit
            )
            before = len(factor_values)
            factor_values = factor_values[factor_values.index.isin(filtered_stocks)]
            # self.logger.debug(f"  [选股] 过滤ST/停牌/涨跌停: {before} -> {len(factor_values)}")

        if len(factor_values) == 0:
            self.logger.warning(f"  [选股] 过滤ST/停牌/涨跌停后为空")
            return []

        # 3. 过滤因子值为 NaN 的股票
        before = len(factor_values)
        factor_values = factor_values[factor_values[factor_name].notna()]
        if len(factor_values) == 0:
            self.logger.warning(f"  [选股] 因子值全为NaN (原有 {before} 只)")
            return []

        # 4. 按因子排序选股
        if self.selection_method == 'top_n':
            sorted_stocks = factor_values.sort_values(factor_name, ascending=self.ascending)
            return sorted_stocks.head(self.n_stocks).index.tolist()

        else:
            raise ValueError(f"不支持的选股方法: {self.selection_method}")

    def _calculate_weights(self, selected_stocks: List[str],
                          factor_values: pd.DataFrame, factor_name: str,
                          current_date: pd.Timestamp = None) -> Dict[str, float]:
        """
        计算权重

        Parameters
        ----------
        selected_stocks : List[str]
            选中的股票列表
        factor_values : pd.DataFrame
            因子值
        factor_name : str
            因子名称
        current_date : pd.Timestamp, optional
            当前调仓日期（用于min_vol/max_sharpe等优化方法）

        Returns
        -------
        Dict[str, float]
            权重字典
        """
        if len(selected_stocks) == 0:
            return {}

        if self.weighting_method == 'equal':
            # 等权
            weight = 1.0 / len(selected_stocks)
            return {stock: weight for stock in selected_stocks}

        elif self.weighting_method in ['min_vol', 'max_sharpe']:
            # 使用权重优化器
            if self.price_df_cache is None or current_date is None:
                # 没有价格数据，回退到等权
                self.logger.warning("价格数据不可用，使用等权")
                weight = 1.0 / len(selected_stocks)
                return {stock: weight for stock in selected_stocks}

            return self.weight_optimizer.optimize(
                self.price_df_cache,
                selected_stocks,
                current_date
            )

        elif self.weighting_method == 'factor_weighted':
            # 因子加权（因子值归一化作为权重）
            stock_factors = factor_values[factor_values.index.isin(selected_stocks)][factor_name]
            if self.ascending:
                # 低波动策略：因子值越小权重越大
                inv_factors = 1.0 / (stock_factors + 1e-6)
                weights = inv_factors / inv_factors.sum()
            else:
                # 高值策略：因子值越大权重越大
                weights = stock_factors / stock_factors.sum()
            return weights.to_dict()

        else:
            raise ValueError(f"不支持的权重方法: {self.weighting_method}")

    def _load_prices(self, stocks: List[str], date: str) -> Dict[str, float]:
        """加载指定日期的股票价格（优先从缓存读取，大幅提升速度）"""
        # 根据配置选择价格字段
        price_key = 'open' if self.trade_at_open else 'close'

        prices = {}

        # 优先从缓存读取（注意：缓存中股票代码为小写）
        if date in self.trade_price_cache:
            cache_day = self.trade_price_cache[date]
            for stock in stocks:
                # 尝试小写匹配（缓存中是小写）
                stock_lower = stock.lower()
                if stock_lower in cache_day:
                    price = cache_day[stock_lower][price_key]
                    if not pd.isna(price):
                        prices[stock] = price  # 返回原始大小写格式
            return prices

        # 缓存未命中，回退到逐日加载（理论上不应该发生）
        self.logger.warning(f"价格缓存未命中: {date}，使用逐日加载")
        price_field = '$open' if self.trade_at_open else '$close'

        prices_df = self.data_loader.load_stock_prices(
            stock_list=stocks,
            start_date=date,
            end_date=date,
            fields=[price_field]
        )

        for stock in stocks:
            try:
                if stock in prices_df.index.get_level_values(0):
                    price = prices_df.loc[stock][price_field]
                    if isinstance(price, pd.Series):
                        price = price.iloc[0] if len(price) > 0 else np.nan
                    prices[stock] = price
            except:
                pass

        return prices

    def _load_qlib_calendar(self):
        """加载 QLib 交易日历"""
        if self._qlib_calendar is not None:
            return self._qlib_calendar

        cal_file = Path(self.qlib_path) / 'calendars' / 'day.txt'
        with open(cal_file, 'r') as f:
            self._qlib_calendar = [line.strip() for line in f if line.strip()]
        self._qlib_calendar_index = {d: i for i, d in enumerate(self._qlib_calendar)}
        return self._qlib_calendar

    def _read_qlib_bin(self, filepath: Path) -> tuple:
        """读取 QLib bin 文件，返回 (start_idx, data)"""
        if not filepath.exists():
            return None, None

        with open(filepath, 'rb') as f:
            start_idx = struct.unpack('<f', f.read(4))[0]
            data = np.frombuffer(f.read(), dtype=np.float32)

        return int(start_idx), data

    def _load_qlib_prices(self, stocks: List[str], start_date: str, end_date: str) -> dict:
        """
        从 QLib bin 文件加载价格数据到内存（后复权）

        Returns:
            {date_str: {stock: {'open': price, 'close': price}}}
        """
        self.logger.info("从 QLib 加载价格数据（后复权）...")

        calendar = self._load_qlib_calendar()
        start_idx = self._qlib_calendar_index.get(start_date, 0)
        end_idx = self._qlib_calendar_index.get(end_date, len(calendar) - 1)
        dates = calendar[start_idx:end_idx + 1]

        features_path = Path(self.qlib_path) / 'features'

        def load_stock(stock):
            stock_lower = stock.lower()
            stock_dir = features_path / stock_lower

            if not stock_dir.exists():
                return stock, None, None

            # 读取开盘价、收盘价和复权因子
            open_file = stock_dir / 'open.day.bin'
            close_file = stock_dir / 'close.day.bin'
            factor_file = stock_dir / 'factor.day.bin'

            open_start, open_data = self._read_qlib_bin(open_file)
            close_start, close_data = self._read_qlib_bin(close_file)
            factor_start, factor_data = self._read_qlib_bin(factor_file)

            if open_data is None or close_data is None:
                return stock, None, None

            # 提取需要的日期范围（后复权）
            open_result = {}
            close_result = {}
            last_factor = 1.0  # 用于前向填充

            for date in dates:
                cal_idx = self._qlib_calendar_index[date]

                # 获取复权因子（前向填充）
                factor_val = 1.0
                if factor_data is not None:
                    factor_idx = cal_idx - factor_start
                    if 0 <= factor_idx < len(factor_data):
                        val = factor_data[factor_idx]
                        if not np.isnan(val):
                            factor_val = float(val)
                            last_factor = factor_val
                        else:
                            factor_val = last_factor
                    else:
                        factor_val = last_factor

                # 开盘价（后复权）
                if open_data is not None:
                    data_idx = cal_idx - open_start
                    if 0 <= data_idx < len(open_data):
                        val = open_data[data_idx]
                        if not np.isnan(val):
                            open_result[date] = float(val) * factor_val

                # 收盘价（后复权）
                if close_data is not None:
                    data_idx = cal_idx - close_start
                    if 0 <= data_idx < len(close_data):
                        val = close_data[data_idx]
                        if not np.isnan(val):
                            close_result[date] = float(val) * factor_val

            return stock, open_result, close_result

        # 并行加载
        from tqdm import tqdm
        with ThreadPoolExecutor(max_workers=64) as executor:
            results = list(tqdm(executor.map(load_stock, stocks), total=len(stocks), desc="  加载价格"))

        # 构建缓存
        price_cache = {}
        for stock, open_prices, close_prices in results:
            if open_prices is None:
                continue

            for date in dates:
                if date not in price_cache:
                    price_cache[date] = {}

                open_val = open_prices.get(date)
                close_val = close_prices.get(date) if close_prices else None

                if open_val is not None or close_val is not None:
                    price_cache[date][stock.lower()] = {
                        'open': open_val if open_val else np.nan,
                        'close': close_val if close_val else np.nan
                    }

        self.logger.info(f"QLib 价格缓存: {len(price_cache)} 个交易日")
        return price_cache

    def _neutralize_factor(self, factor_data: pd.DataFrame, factor_name: str) -> pd.DataFrame:
        """
        因子中性化（行业+市值）

        Parameters
        ----------
        factor_data : pd.DataFrame
            因子数据
        factor_name : str
            因子名称

        Returns
        -------
        pd.DataFrame
            中性化后的因子数据（用中性化因子值替换原因子列）
        """
        import polars as pl
        from factor_production import DataManager

        # 1. 转换为 Polars
        # 因子数据格式: index=stock, columns=['date', factor_name, ...]
        df_reset = factor_data.reset_index()
        df_pl = pl.from_pandas(df_reset)

        # 2. 加载流通市值
        self.logger.info("  加载流通市值数据...")
        dm = DataManager()
        stocks = df_pl['stock'].unique().to_list()
        start = df_pl['date'].min()
        end = df_pl['date'].max()

        df_cap = dm.load(stocks, start, end, ['$circ_mv'])

        if df_cap.is_empty():
            self.logger.warning("  流通市值数据为空，跳过中性化")
            return factor_data

        # 重命名列以匹配
        df_cap = df_cap.select(['stock', 'date', 'circ_mv'])

        # 统一日期格式为字符串
        df_pl = df_pl.with_columns(pl.col('date').cast(pl.Utf8).alias('date'))
        df_cap = df_cap.with_columns(pl.col('date').cast(pl.Utf8).alias('date'))

        # 3. 合并市值
        df_merged = df_pl.join(df_cap, on=['stock', 'date'], how='left')

        self.logger.info(f"  合并后: {len(df_merged)} 行, 市值有效: {df_merged['circ_mv'].drop_nulls().len()}")

        # 4. 执行中性化
        df_neutral = self._neutralizer.neutralize(
            df_merged,
            factor_name,
            market_cap_col='circ_mv',
            how=self.neutralize_how,
            standardize=True
        )

        # 5. 检查中性化结果
        neutral_col = f'{factor_name}_neutral'
        if neutral_col not in df_neutral.columns:
            self.logger.warning("  中性化失败，使用原始因子")
            return factor_data

        # 6. 用中性化因子替换原因子列
        df_neutral = df_neutral.with_columns(
            pl.col(neutral_col).alias(factor_name)
        )

        # 7. 转回 Pandas 并恢复索引
        df_pd = df_neutral.to_pandas()
        df_pd = df_pd.set_index('stock')

        # 确保只保留需要的列
        cols_to_keep = ['date', factor_name]
        df_pd = df_pd[cols_to_keep]

        self.logger.info(f"  中性化完成: {len(df_pd)} 行")

        return df_pd

    def _load_shibor_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        加载SHIBOR数据

        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期

        Returns
        -------
        pd.DataFrame
            SHIBOR数据，包含 'date' 和 '1y' 列
        """
        try:
            import tushare as ts
            ts.set_token('a79f284e5d10967dacb6531a3c755a701ca79341ff0c60d59f1fcbf1')
            pro = ts.pro_api()

            # 转换日期格式
            start_fmt = start_date.replace('-', '')
            end_fmt = end_date.replace('-', '')

            # 获取SHIBOR数据
            df = pro.shibor(start_date=start_fmt, end_date=end_fmt)

            if df is None or len(df) == 0:
                self.logger.warning("SHIBOR数据为空")
                return None

            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df = df.sort_values('date').reset_index(drop=True)

            self.logger.info(f"加载SHIBOR数据: {len(df)}条, "
                           f"范围: {df['date'].min().strftime('%Y-%m-%d')} ~ "
                           f"{df['date'].max().strftime('%Y-%m-%d')}")

            return df

        except Exception as e:
            self.logger.error(f"加载SHIBOR数据失败: {e}")
            return None

    def _print_results(self, metrics: Dict):
        """打印业绩指标"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("回测结果")
        self.logger.info("=" * 80)
        self.logger.info(f"累计收益率: {metrics['total_return']:.2%}")
        self.logger.info(f"年化收益率: {metrics['annual_return']:.2%}")
        self.logger.info(f"基准收益率: {metrics['benchmark_return']:.2%}")
        self.logger.info(f"超额收益: {metrics['excess_return']:.2%}")
        self.logger.info(f"年化波动率: {metrics['volatility']:.2%}")
        self.logger.info(f"夏普比率: {metrics['sharpe_ratio']:.4f}")
        self.logger.info(f"最大回撤: {metrics['max_drawdown']:.2%}")
        self.logger.info(f"信息比率: {metrics['information_ratio']:.4f}")
        self.logger.info(f"胜率: {metrics['win_rate']:.2%}")
        self.logger.info(f"卡玛比率: {metrics['calmar_ratio']:.4f}")
        self.logger.info("=" * 80)

    def _plot_performance(self, portfolio_values: pd.DataFrame,
                         benchmark_data: pd.Series,
                         factor_name: str):
        """
        Plot strategy performance curves

        Parameters
        ----------
        portfolio_values : DataFrame
            Portfolio value data
        benchmark_data : Series
            Benchmark data
        factor_name : str
            Factor name
        """
        self.logger.info("\nPlotting performance curves...")

        # Prepare data
        dates = pd.to_datetime(portfolio_values['date'])
        strategy_values = portfolio_values['total_value'].values

        # Align benchmark data
        benchmark_aligned = benchmark_data.loc[dates].values

        # Normalize to same starting point (all start from 1)
        strategy_nav = strategy_values / strategy_values[0]
        benchmark_nav = benchmark_aligned / benchmark_aligned[0]
        excess_nav = strategy_nav / benchmark_nav  # Excess return NAV

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot three curves
        ax.plot(dates, strategy_nav, label='Strategy NAV', linewidth=2, color='#1f77b4')
        ax.plot(dates, benchmark_nav, label='Benchmark NAV (CSI 1000)', linewidth=2, color='#ff7f0e', linestyle='--')
        ax.plot(dates, excess_nav, label='Excess Return NAV', linewidth=2, color='#2ca02c', alpha=0.7)

        # Add baseline
        ax.axhline(y=1, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

        # Set title and labels
        ax.set_title(f'{factor_name} Strategy Performance (Benchmark: CSI 1000)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('NAV (Normalized)', fontsize=12)

        # Set legend
        ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

        # Set grid
        ax.grid(True, alpha=0.3, linestyle='--')

        # Format date axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)

        # Tight layout
        plt.tight_layout()

        # Save plot (保存到因子子目录)
        factor_dir = os.path.join(self.save_dir, factor_name)
        os.makedirs(factor_dir, exist_ok=True)
        plot_path = os.path.join(factor_dir, "performance.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Performance plot saved: {plot_path}")

        # Display plot (if in interactive environment)
        # plt.show()
        plt.close()

        return plot_path

    def _save_results(self, results: Dict, factor_name: str):
        """保存回测结果（每个因子一个子目录）"""
        # 创建因子专属目录
        factor_dir = os.path.join(self.save_dir, factor_name)
        os.makedirs(factor_dir, exist_ok=True)

        # 保存净值曲线
        results['portfolio_values'].to_csv(
            os.path.join(factor_dir, "values.csv"),
            index=False
        )

        # 保存交易记录
        if len(results['trades']) > 0:
            results['trades'].to_csv(
                os.path.join(factor_dir, "trades.csv"),
                index=False
            )

        # 保存业绩指标
        metrics_df = pd.DataFrame([results['metrics']])
        metrics_df.to_csv(
            os.path.join(factor_dir, "metrics.csv"),
            index=False
        )

        self.logger.info(f"\n结果已保存到: {factor_dir}")


if __name__ == "__main__":
    print("Backtester module loaded successfully")