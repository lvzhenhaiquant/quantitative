"""
回测系统 - 整合版

包含:
1. Portfolio - 投资组合管理
2. PerformanceAnalyzer - 业绩分析
3. Backtester - 回测引擎
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.data_loader import DataLoader


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
                self._sell(stock, self.positions[stock], prices[stock], date)

        # 2. 调整现有持仓
        for stock in list(self.positions.keys()):
            if stock in target_positions:
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
        # 计算收益率
        portfolio_returns = self._calculate_returns(portfolio_values['total_value'])
        benchmark_returns = self._calculate_returns(benchmark_values)

        # 对齐日期
        portfolio_returns.index = pd.to_datetime(portfolio_values['date'])
        benchmark_returns.index = pd.to_datetime(benchmark_values.index)

        # 取交集
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        portfolio_returns = portfolio_returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]

        # 计算指标
        metrics = {}

        # 1. 累计收益
        metrics['total_return'] = self._total_return(portfolio_returns)
        metrics['benchmark_return'] = self._total_return(benchmark_returns)
        metrics['excess_return'] = metrics['total_return'] - metrics['benchmark_return']

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
        self.selection_method = config.get('selection', {}).get('method', 'top_n')
        self.n_stocks = config.get('selection', {}).get('n_stocks', 50)
        self.ascending = config.get('selection', {}).get('ascending', True)
        self.weighting_method = config.get('weighting', {}).get('method', 'equal')
        self.rebalance_freq = config.get('rebalance', {}).get('freq', 'weekly')
        self.benchmark_code = config.get('benchmark', {}).get('code', 'sh000300')

        # 创建结果保存目录
        self.save_dir = config.get('output', {}).get('save_dir', 'backtest/results')
        os.makedirs(self.save_dir, exist_ok=True)

        self.logger.info("Backtester初始化完成")

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

        # 5. 执行回测
        self.logger.info("\n开始模拟交易...")

        for i, rebalance_date in enumerate(rebalance_dates, 1):
            self.logger.info(f"[{i}/{len(rebalance_dates)}] {rebalance_date}")

            # 获取这天的因子值
            factor_values = factor_data[factor_data['date'] == rebalance_date]

            if len(factor_values) == 0:
                self.logger.warning(f"  跳过: 无因子数据")
                continue

            # 选股
            selected_stocks = self._select_stocks(factor_values, factor_name)

            if len(selected_stocks) == 0:
                self.logger.warning(f"  跳过: 选股为空")
                continue

            # 加载当日价格
            prices = self._load_prices(selected_stocks + list(portfolio.positions.keys()),
                                      rebalance_date)

            # 计算目标权重
            target_weights = self._calculate_weights(selected_stocks, factor_values, factor_name)

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

    def _select_stocks(self, factor_values: pd.DataFrame, factor_name: str) -> List[str]:
        """选股"""
        if self.selection_method == 'top_n':
            # TopN选股
            sorted_stocks = factor_values.sort_values(factor_name, ascending=self.ascending)
            return sorted_stocks.head(self.n_stocks).index.tolist()

        else:
            raise ValueError(f"不支持的选股方法: {self.selection_method}")

    def _calculate_weights(self, selected_stocks: List[str],
                          factor_values: pd.DataFrame, factor_name: str) -> Dict[str, float]:
        """计算权重"""
        if self.weighting_method == 'equal':
            # 等权
            weight = 1.0 / len(selected_stocks)
            return {stock: weight for stock in selected_stocks}

        else:
            raise ValueError(f"不支持的权重方法: {self.weighting_method}")

    def _load_prices(self, stocks: List[str], date: str) -> Dict[str, float]:
        """加载指定日期的股票价格"""
        prices_df = self.data_loader.load_stock_prices(
            stock_list=stocks,
            start_date=date,
            end_date=date,
            fields=['$close']
        )

        prices = {}
        for stock in stocks:
            try:
                if stock in prices_df.index.get_level_values(0):
                    price = prices_df.loc[stock]['$close']
                    if isinstance(price, pd.Series):
                        price = price.iloc[0] if len(price) > 0 else np.nan
                    prices[stock] = price
            except:
                pass

        return prices

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

    def _save_results(self, results: Dict, factor_name: str):
        """保存回测结果"""
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