import numpy as np
import pandas as pd

class FactorCalculator:
    """
    因子计算器类，包含所有因子算子的实现
    """
    # 在 calculator.py 中修改 SUPPORTED_OPERATORS
    # 支持的操作符定义，按类型分类
    SUPPORTED_OPERATORS = {
        "unary": ["Abs", "Log"],  # 一元操作符：只需要一个操作数
                                # 实例：
                                # - Abs(close): 计算收盘价的绝对值
                                # - Log(open): 计算开盘价的自然对数
        
        "binary": ["Add", "Sub", "Mul", "Div", "Greater", "Less", "Max", "Min"],  # 二元操作符：需要两个操作数
                                                                                # 实例：
                                                                                # - Add(close, open): 收盘价 + 开盘价
                                                                                # - Sub(close, open): 收盘价 - 开盘价
                                                                                # - Mul(close, volume): 收盘价 * 成交量
                                                                                # - Div(high, low): 最高价 / 最低价
                                                                                # - Greater(close, open): 收盘价 > 开盘价（返回1或0）
                                                                                # - Less(close, open): 收盘价 < 开盘价（返回1或0）
                                                                                # - Max(close, high): 收盘价和最高价中的较大值
                                                                                # - Min(close, low): 收盘价和最低价中的较小值
        
        "rolling": ["Ref", "Mean", "Sum", "Std", "Var", "Med", "Mad", "Delta", "WMA", "EMA"],  # 滚动操作符：对时间序列数据进行滚动计算，需要两个操作数（数据字段和窗口大小）
                                                                                                # 实例：
                                                                                                # - Ref(close, 5): 获取5天前的收盘价
                                                                                                # - Mean(close, 10): 计算10天收盘价的平均值
                                                                                                # - Sum(volume, 20): 计算20天成交量的总和
                                                                                                # - Std(close, 5): 计算5天收盘价的标准差
        
        "pair_rolling": ["Cov", "Corr"]  # 配对滚动操作符：对两个时间序列数据进行滚动计算，需要三个操作数（两个数据字段和窗口大小）
                                        # 实例：
                                        # - Cov(close, volume, 10): 计算10天收盘价和成交量的协方差
                                        # - Corr(close, volume, 20): 计算20天收盘价和成交量的相关系数
    }
    
    def __init__(self):
        pass

    def _compute_operand_values(self, operand, stock_data):
        """
        递归计算操作数在历史数据中的值
        :param operand: 操作数节点
        :param stock_data: 股票历史数据
        :return: 计算结果数组
        """
        operand_type = operand.get('type')
        if operand_type == 'field':
            # 字段节点，返回对应字段的值
            field_name = operand.get('name', '').lower()
            if field_name in stock_data.columns:
                return stock_data[field_name].values.astype(np.float32)
            else:
                return np.zeros(len(stock_data))
        elif operand_type == 'constant':
            # 常量节点，返回常量值
            try:
                value = float(operand.get('value'))
                return np.full(len(stock_data), value, dtype=np.float32)
            except (ValueError, TypeError):
                return np.zeros(len(stock_data))
        elif operand_type == 'operation':
            # 操作符节点，递归计算
            operator = operand.get('operator')
            operands = operand.get('operands', [])
            
            # 递归计算子操作数的值
            sub_values = [self._compute_operand_values(op, stock_data) for op in operands]
            
            if operator in self.SUPPORTED_OPERATORS["unary"]:
                return self.execute_unary_op(operator, sub_values[0])
            elif operator in self.SUPPORTED_OPERATORS["binary"]:
                return self.execute_binary_op(operator, sub_values[0], sub_values[1])
            else:
                return np.zeros(len(stock_data))
        else:
            return np.zeros(len(stock_data))
    
    def execute_unary_op(self, operator, operand):
        """
        执行一元操作符
        :param operator: 操作符名称
        :param operand: 操作数
        :return: 计算结果
        """
        if operator == 'Abs':
            return np.abs(operand)
        elif operator == 'Log':
            # 避免对负数或零取对数
            operand_safe = np.where(operand <= 0, 1e-8, operand)
            return np.log(operand_safe)
        else:
            raise ValueError(f"Unknown unary operator: {operator}")
    
    def execute_binary_op(self, operator, left, right):
        """
        执行二元操作符（match-case 实现 "switch" 逻辑，Python 3.10+）
        :param operator: 操作符名称
        :param left: 左操作数
        :param right: 右操作数
        :return: 计算结果
        """
        match operator:
            case 'Add':
                return left + right
            case 'Sub':
                return left - right
            case 'Mul':
                return left * right
            case 'Div':
                # 保留除零错误保护
                right_safe = np.where(right == 0, 1e-8, right)
                return left / right_safe
            case 'Greater':
                return np.where(left > right, 1.0, 0.0)
            case 'Less':
                return np.where(left < right, 1.0, 0.0)
            case 'Max':
                return np.maximum(left, right)
            case 'Min':
                return np.minimum(left, right)
            case _:  # 通配符，等效原 else
                raise ValueError(f"Unknown binary operator: {operator}")

    
    def execute_rolling_op(self, operator, operands, date_data, date_list, data):
        """
        执行滚动操作符
        :param operator: 操作符名称
        :param operands: 操作数节点列表
        :param date_data: 当前日期的数据
        :param date_list: 日期列表
        :param data: 完整数据集
        :return: 计算结果
        """
        if not date_data.empty:
            current_date = date_data['date'].iloc[0]
            stocks = date_data['stock'].tolist()
            
            # 获取当前日期在日期列表中的索引
            current_date_idx = date_list.index(current_date)
            
            # 必须从操作数中获取窗口大小，不使用默认值
            if len(operands) < 2:
                print(f"滚动操作符{operator}必须提供窗口大小参数")
                return None
            
            # 获取窗口大小
            window = int(operands[1].get('value', 1))  # 第二个操作数是窗口大小
            
            # 确保窗口大小至少为1,并限制上限为已知上限
            window = min(current_date_idx,max(1, window))
            
            # 获取需要的历史数据范围
            start_date_idx = max(0, current_date_idx - window + 1)
            history_dates = date_list[start_date_idx:current_date_idx + 1]
            
            # 获取历史数据
            history_data = data[data['date'].isin(history_dates) & data['stock'].isin(stocks)]
            
            if history_data.empty:
                return np.zeros(len(date_data))
            
            # 按股票分组计算
            result_dict = {}
            
            for stock in stocks:
                stock_data = history_data[history_data['stock'] == stock].sort_values('date')
                
                if stock_data.empty:
                    result_dict[stock] = 0.0
                    continue
                
                # 获取操作数值（递归计算历史值）
                values = self._compute_operand_values(operands[0], stock_data)
                
                if len(values) < window:
                    result_dict[stock] = 0.0
                    continue
                
                # 替换原有 if-elif 块，直接嵌入 execute_rolling_op 函数中
                match operator:
                    case 'Ref':
                        # 引用前N期值（保留原边界判断逻辑）
                        if len(values) > 1:
                            result_dict[stock] = values[-2]  # 前一期
                        else:
                            result_dict[stock] = values[-1]
                    case 'Mean':
                        result_dict[stock] = np.mean(values)
                    case 'Sum':
                        result_dict[stock] = np.sum(values)
                    case 'Std':
                        result_dict[stock] = np.std(values)
                    case 'Var':
                        result_dict[stock] = np.var(values)
                    case 'Med':
                        result_dict[stock] = np.median(values)
                    case 'Mad':
                        result_dict[stock] = np.median(np.abs(values - np.median(values)))
                    case 'Delta':
                        # 计算与前N期的差值（保留原边界判断逻辑）
                        if len(values) > 1:
                            result_dict[stock] = values[-1] - values[-2]
                        else:
                            result_dict[stock] = 0.0
                    case 'WMA':
                        # 加权移动平均（保留原权重计算逻辑）
                        weights = np.arange(1, len(values) + 1)
                        result_dict[stock] = np.sum(values * weights) / np.sum(weights)
                    case 'EMA':
                        # 指数移动平均（保留原迭代计算逻辑）
                        alpha = 2 / (len(values) + 1)
                        ema = values[0]
                        for val in values[1:]:
                            ema = alpha * val + (1 - alpha) * ema
                        result_dict[stock] = ema
                    case _:  # 通配符，等效原 else 分支
                        raise ValueError(f"Unknown execute_rolling_op operator: {operator}")
            
            # 将结果按照date_data中的股票顺序排列
            result = np.array([result_dict.get(stock, 0.0) for stock in stocks], dtype=np.float32)
            return result
        else:
            raise ValueError(f"date_data为空, 无法执行滚动操作符{operator}")
    
    
    def execute_pair_rolling_op(self, operator, operands, date_data, date_list, data):
        """
        执行配对滚动操作符
        :param operator: 操作符名称
        :param operands: 操作数节点列表
        :param date_data: 当前日期的数据
        :param date_list: 日期列表
        :param data: 完整数据集
        :return: 计算结果
        """
        if not date_data.empty:
            current_date = date_data['date'].iloc[0]
            stocks = date_data['stock'].tolist()
            
            # 获取当前日期在日期列表中的索引
            current_date_idx = date_list.index(current_date)
            
            # 必须从操作数中获取窗口大小，不使用默认值
            if len(operands) < 3:
                raise ValueError(f"配对滚动操作符{operator}必须提供窗口大小参数")
            
            # 获取窗口大小
            window = int(operands[2].get('value', 2))  # 第三个操作数是窗口大小
            
            # 确保窗口大小至少为2（计算协方差/相关系数需要至少2个样本）
            window = max(2, window)
            
            # 获取需要的历史数据范围
            start_date_idx = max(0, current_date_idx - window + 1)
            history_dates = date_list[start_date_idx:current_date_idx + 1]
            
            # 获取历史数据
            history_data = data[data['date'].isin(history_dates) & data['stock'].isin(stocks)]
            
            if history_data.empty:
                return np.zeros(len(date_data))
            
            # 按股票分组计算
            result_dict = {}
            
            for stock in stocks:
                stock_data = history_data[history_data['stock'] == stock].sort_values('date')
                
                if stock_data.empty:
                    result_dict[stock] = 0.0
                    continue
                
                # 计算两个操作数的历史值
                values1 = self._compute_operand_values(operands[0], stock_data)
                values2 = self._compute_operand_values(operands[1], stock_data)
                
                if len(values1) < window or len(values2) < window:
                    result_dict[stock] = 0.0
                    continue
                
                # 执行具体的配对滚动操作
                match operator:
                    case 'Cov':
                        # 计算协方差（保留原逻辑）
                        result_dict[stock] = np.cov(values1, values2)[0, 1]
                    case 'Corr':
                        # 计算相关系数（保留原 NaN 处理逻辑）
                        corr = np.corrcoef(values1, values2)[0, 1]
                        result_dict[stock] = corr if not np.isnan(corr) else 0.0
                    case _:
                        raise ValueError(f"Unknown execute_pair_rolling_op operator: {operator}")
            
            # 将结果按照date_data中的股票顺序排列
            result = np.array([result_dict.get(stock, 0.0) for stock in stocks], dtype=np.float32)
            return result
        
        return np.zeros(len(date_data))