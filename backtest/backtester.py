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
from datetime import datetime
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

        self.logger.info("Backtester初始化完成")
        self.logger.info(f"交易时机: {'开盘价' if self.trade_at_open else '收盘价'}")
        self.logger.info(f"权重方法: {self.weighting_method}")
        self.logger.info(f"动态无风险利率(SHIBOR): {self.use_dynamic_rf}")
        self.logger.info(f"股票池: {self.universe} (按日筛选: {self.filter_by_universe})")
    
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
            # from factor_production.factor_scheduler import FactorScheduler
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

        # 4.1 加载历史价格数据（用于min_vol/max_sharpe权重优化）
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
                fields=['$close']
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
        all_stocks = factor_data.index.unique().tolist()
            # 计算需要的历史数据起始日期（lookback_days + buffer）
        price_data_list = self.data_loader.load_stock_prices(
            stock_list=all_stocks,
            start_date=start_date,
            end_date=end_date,
            fields=['$close', '$open']
        )

        for i, rebalance_date in enumerate(rebalance_dates, 1):
            self.logger.info(f"[{i}/{len(rebalance_dates)}] {rebalance_date}")

            # 获取这天的因子值
            factor_values = factor_data[factor_data['date'] == rebalance_date]

            if len(factor_values) == 0:
                self.logger.warning(f"  跳过: 无因子数据")
                continue

            # 选股（传入日期用于筛选当日成分股，避免未来函数）
            selected_stocks = self._select_stocks(factor_values, factor_name, date=rebalance_date)

            if len(selected_stocks) == 0:
                self.logger.warning(f"  跳过: 选股为空（可能当日成分股不足）")
                continue

            # 加载当日价格
            # prices = self._load_prices(selected_stocks + list(portfolio.positions.keys()),rebalance_date)
            price_field = '$open' if self.trade_at_open else '$close'
            prices = price_data_list.loc[pd.IndexSlice[selected_stocks + list(portfolio.positions.keys()), rebalance_date],price_field].reset_index(level=1, drop=True).dropna().to_dict()

            # 计算目标权重
            target_weights = self._calculate_weights(
                selected_stocks, factor_values, factor_name,
                current_date=pd.Timestamp(rebalance_date)
            )

            # 调仓
            portfolio.rebalance(target_weights, prices, rebalance_date)

            # 记录状态
            portfolio.record_state(rebalance_date, prices)

            self.logger.info(f"  选股: {len(selected_stocks)}只, "
                           f"持仓: {len(portfolio.positions)}只, "
                           f"净值: {portfolio.calculate_portfolio_value(prices)/self.initial_cash:.4f}")

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
        选股（先筛选当日成分股，再按因子排序）

        Parameters
        ----------
        factor_values : pd.DataFrame
            因子数据
        factor_name : str
            因子名称
        date : str, optional
            调仓日期（用于筛选当日成分股）

        Returns
        -------
        List[str]
            选中的股票列表
        """
        # 1. 先筛选当日成分股（避免未来函数）
        if date and self.filter_by_universe:
            universe_stocks = self._get_universe_stocks(date)
            if universe_stocks:
                # 只保留当日是成分股的股票
                factor_values = factor_values[factor_values.index.isin(universe_stocks)]

        if len(factor_values) == 0:
            return []

        # 2. 按因子排序选股
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
        """加载指定日期的股票价格（根据配置使用开盘价或收盘价）"""
        # 根据配置选择价格字段
        price_field = '$open' if self.trade_at_open else '$close'

        prices_df = self.data_loader.load_stock_prices(
            stock_list=stocks,
            start_date=date,
            end_date=date,
            fields=[price_field]
        )

        prices = {}
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

        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = os.path.join(self.save_dir, f"{factor_name}_performance_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Performance plot saved: {plot_path}")

        # Display plot (if in interactive environment)
        # plt.show()
        plt.close()

        return plot_path

    def _save_results(self, results: Dict, factor_name: str):
        """保存回测结果（保存前自动清理相同因子的旧文件）"""
        # 清理相同因子名的旧文件（包括CSV和PNG）
        import glob

        patterns = [
            f'{factor_name}_values_*.csv',
            f'{factor_name}_trades_*.csv',
            f'{factor_name}_metrics_*.csv',
            f'{factor_name}_performance_*.png'
        ]

        old_files = []
        for pattern in patterns:
            old_files.extend(glob.glob(os.path.join(self.save_dir, pattern)))

        if old_files:
            self.logger.info(f"\n发现 {len(old_files)} 个旧的 {factor_name} 回测文件，正在清理...")
            for old_file in old_files:
                try:
                    file_size = os.path.getsize(old_file) / (1024 * 1024)  # MB
                    os.remove(old_file)
                    self.logger.info(f"  ✓ 已删除: {os.path.basename(old_file)} ({file_size:.2f} MB)")
                except Exception as e:
                    self.logger.warning(f"  ✗ 删除失败: {os.path.basename(old_file)} - {e}")

        # 生成新的时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存净值曲线
        results['portfolio_values'].to_csv(
            os.path.join(self.save_dir, f"{factor_name}_values_{timestamp}.csv"),
            index=False
        )

        # 保存交易记录
        if len(results['trades']) > 0:
            results['trades'].to_csv(
                os.path.join(self.save_dir, f"{factor_name}_trades_{timestamp}.csv"),
                index=False
            )

        # 保存业绩指标
        metrics_df = pd.DataFrame([results['metrics']])
        metrics_df.to_csv(
            os.path.join(self.save_dir, f"{factor_name}_metrics_{timestamp}.csv"),
            index=False
        )

        self.logger.info(f"\n结果已保存到: {self.save_dir}")


if __name__ == "__main__":
    print("Backtester module loaded successfully")