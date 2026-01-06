import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from DataLoad import DataProcessor
from FactorToken import RPNEncoder

class FactorRLEnv(py_environment.PyEnvironment):
    """因子生成强化学习环境"""
    
    def __init__(self, token_lib):
        self.token_lib = token_lib
        self.action_space = token_lib.all_tokens
        self.data_processor = DataProcessor()
        self.RPNEncoder = RPNEncoder(token_lib)
        
        # 定义序列最大长度（使用TokenLibrary中的MAX_EXPR_LENGTH=15）
        self.token_len = self.RPNEncoder.max_length
        # 观测维度 = 数据特征维度
        self.observation_dim = len(self.data_processor.fields_origin)
        
        # 定义观测空间和动作空间
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, name='action')
        # 定义观测空间：数据特征 (不包含Token序列)
        self._observation_spec = array_spec.BoundedArraySpec(shape=(self.observation_dim,), dtype=np.float32, name='observation')
        # 初始化状态
        self.token_action = []
        self._episode_ended = False
        self.reward = {
            "success": 10.0,
            "error": -100.0,
            "no_action": 0.0,
            }

        # 加载环境数据
        self.max_stock_num=1000
        self.csi_code = "csi1000"
        self.data_loader = DataProcessor()
        self.data = self.data_processor._preprocess_data(self.data_loader.load_data(self.csi_code,'2020-01-01', '2025-12-31'))
        self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce').dt.date
        self.date_list = self.data['date'].dropna().sort_values().unique().tolist()


        # 当前数据点
        self._current_date_idx = 0
        self._current_data_features = self._get_current_data_features()

        # 日志文件路径
        self.log_file_path = "./RL_Data/factor_expr_reward.csv"
        
        # AlphaPool因子池管理
        self.alpha_pool = []  # 因子池，容量上限10个
        self.alpha_pool_capacity = 10
        self.failed_factor_cache = set()  # 历史失败缓存因子

        # 大模型介入计数器
        self.step_count = 0
        self.large_model_interval = 3000

        
    def action_spec(self):
        return self._action_spec
    def observation_spec(self):
        return self._observation_spec
    def action_size(self):
        return len(self.action_space)
    
    def is_last(self):
        """对外暴露的done判断接口，核心返回self.episode_done"""
        return self._episode_ended
    
    def _reset(self):
        """重置环境"""
        self.token_action = [self.action_space.index(self.token_lib.BEG)]
        # 随机选择新的当前数据点
        self._current_date_idx = (self._current_date_idx + 1) % len(self.date_list)
        self._current_data_features = self._get_current_data_features()
        self._episode_ended = False
        reset_result = ts.restart(self._get_observation())
        return reset_result
    
    def _filter_date_from_pool(self,current_date):
        date_data = self.data.loc[self.data['date'] == current_date]
        constituent_stocks = self.data_loader.get_pool_stocks_by_date(pool=self.csi_code,date=current_date.strftime("%Y-%m-%d"))
        if not constituent_stocks:
            return np.zeros(self.observation_dim)
        # 筛选当前日期的股票数据
        date_data = date_data[date_data['stock'].isin(constituent_stocks)]
        return date_data
    
    def _get_current_data_features(self):
        """获取当前数据点的特征"""
        current_date = self.date_list[self._current_date_idx]
        # 获取当前日期所有股票的特征值
        date_data = self._filter_date_from_pool(current_date)
        # 筛选有效字段
        available_fields = [field for field in self.data_processor.fields_origin if field in date_data.columns]
        if not available_fields:
            print(f"警告: 没有找到任何有效字段 ")
            return np.zeros(self.observation_dim)
         # 填充缺失值（量化中常用0/行业均值，避免NaN）
        raw_features = date_data[available_fields].fillna(0).values
        all_data_fields = self.data[available_fields]
        min_vals = all_data_fields.min().values
        max_vals = all_data_fields.max().values
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1e-8
        # 对每一列（每个特征）标准化，结果还是 (n_stocks, n_fields)
        normalized_features = (raw_features - min_vals) / range_vals

        n_stocks, n_features = normalized_features.shape
        if n_stocks < self.max_stock_num:
            # 填充零数组到最大股票数量
            padding = np.zeros((self.max_stock_num - n_stocks, n_features), dtype=np.float32)
            padded_features = np.vstack([normalized_features, padding])
        else:
            # 如果超过最大数量，截取到最大数量
            padded_features = normalized_features[:self.max_stock_num]
        return padded_features

    def _get_observation(self):
        """获取观测值 - 结合Token序列和当前数据特征"""
        # 填充Token序列到最大长度,能看到一个因子序列历史动作值
        token_seq = np.full(self.token_len, 0, dtype=np.float32)
        token_seq[:len(self.token_action)] = self.token_action
        # 组合Token序列和数据特征
        token_seq_float = token_seq.astype(np.float32)
        current_data_feat = self._get_current_data_features()

        token_seq_float = np.expand_dims(token_seq_float, axis=0)
        current_data_feat = np.expand_dims(current_data_feat, axis=0)
        obs = [current_data_feat,token_seq_float]
        return obs
    

    def _step(self, action):
        """执行动作"""
        if self._episode_ended:
            return self.reset()
        # 检查是否达到最大长度或结束符
        if len(self.token_action) >= self.token_len or self.action_space[action] == self.token_lib.SEP:
            reward = self._calculate_reward()
            self._current_date_idx +=1
            self.token_action = [self.action_space.index(self.token_lib.BEG)] # 重置动作序列
            print(f"_current_date_idx:  {self.date_list[self._current_date_idx]}")
            if self._current_date_idx >= len(self.date_list):
                self._episode_ended = True
            return ts.termination(self._get_observation(), reward=reward)
        else:
            self.token_action.append(action)     
        return ts.transition(self._get_observation(), reward=0.0, discount=1.0)
    def _calculate_reward(self):
        """使用实际金融数据计算因子的奖励""" 
        MAX_RETRY = 3
        retry_count = 0
        error_prompt = ''
        while retry_count < MAX_RETRY:
            # 1. 将动作索引转换为实际token序列
            current_tokens = [self.action_space[i] for i in self.token_action]
            corrected_tokens = current_tokens
            corrected_tokens = self._change_token_from_deepseek(current_tokens,error_prompt)
            # corrected_tokens =['BEG', 'open', '1.0', 'Add', 'Log', 'close', 'Div','SEP'] #['BEG', 'Corr', 'SEP']
            if self.RPNEncoder._is_valid_expression(corrected_tokens):
                try:
                    factor_expr = self.RPNEncoder.decode(corrected_tokens,return_type="dict")
                    factor_values = self._execute_factor(factor_expr)
                    reward = self._evaluate_factor(factor_values)
                    # 记录因子表达式和奖励
                    self._record_factor_expr(corrected_tokens, reward)
                    print(f"right_reward:{reward}")
                    return reward
                except Exception as e:
                    retry_count += 1
                    error_prompt = f"经过一轮大模型修改后：{corrected_tokens},经过计算校验后，不满足因子的基本计算条件，无法生成可执行的有效因子,需要重新修改。"
                    print(f"{factor_expr} :因子执行错误{e},DeepSeek第{retry_count}次重试")
            else:
                retry_count += 1
        reward = self.reward["error"]
        print(f"error_reward:{reward}")
        return reward

    def _execute_factor(self, factor_expr):
        """
        执行因子表达式并返回因子值
        :param data: 当前数据，包含股票数据
        :param factor_expr: 因子表达式字典
        :return: 因子值（numpy数组）
        """
        
        # 获取当前日期的数据
        current_date = self.date_list[self._current_date_idx]
        date_data = self._filter_date_from_pool(current_date)
        return self._execute_node(factor_expr, date_data)


    def _execute_node(self, node, date_data):
        """ 递归执行表达式节点 :param node: 表达式节点 :param data: 当前数据 :return: 计算结果 """
        node_type = node.get('type')
        if node_type == 'field':
            # 字段节点，返回对应字段的值
            field_name = node.get('name', '').lower()
            if field_name in date_data.columns:
                return date_data[field_name].values.astype(np.float32)
            else:
                # 如果字段不存在，返回零数组
                return np.zeros(len(date_data))
        
        elif node_type == 'constant':
            # 常量节点，返回常量值
            try:
                value = float(node.get('value'))
                return np.full(len(date_data), value, dtype=np.float32)
            except (ValueError, TypeError):
                return np.zeros(len(date_data))
        
        elif node_type == 'operation':
            # 操作符节点
            operator = node.get('operator')
            operands = node.get('operands', [])
            
            # 执行操作数
            operand_values = [self._execute_node(op, date_data) for op in operands]
            
            # 根据操作符类型执行运算
            if operator in self.token_lib.OPERATORS["unary"]:
                return self._execute_unary_op(operator, operand_values[0])
            elif operator in self.token_lib.OPERATORS["binary"]:
                return self._execute_binary_op(operator, operand_values[0], operand_values[1])
            elif operator in self.token_lib.OPERATORS["rolling"]:
                # 对于滚动操作符，我们简化处理，只返回操作数的值
                return operand_values[0]
            elif operator in self.token_lib.OPERATORS["pair_rolling"]:
                # 对于配对滚动操作符，我们简化处理，返回第一个操作数的值
                return operand_values[0]
            else:
                raise ValueError(f"Unknown operator: {operator}")
        
        else:
            raise ValueError(f"Unknown node type: {node_type}")
    

    def _execute_unary_op(self, operator, operand):
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

    def _execute_binary_op(self, operator, left, right):
        """
        执行二元操作符
        :param operator: 操作符名称
        :param left: 左操作数
        :param right: 右操作数
        :return: 计算结果
        """
        if operator == 'Add':
            return left + right
        elif operator == 'Sub':
            return left - right
        elif operator == 'Mul':
            return left * right
        elif operator == 'Div':
            # 避免除零错误
            right_safe = np.where(right == 0, 1e-8, right)
            return left / right_safe
        elif operator == 'Greater':
            return np.where(left > right, 1.0, 0.0)
        elif operator == 'Less':
            return np.where(left < right, 1.0, 0.0)
        else:
            raise ValueError(f"Unknown binary operator: {operator}")

    def _evaluate_factor(self, factor_values):
        """
        评估因子质量，返回奖励值
        :param factor_values: 因子值
        :param data: 当前数据
        :return: 奖励值
        """
        if len(factor_values) == 0 or np.all(np.isnan(factor_values)) or np.all(factor_values == 0):
            return self.reward["error"]
        
        # 移除NaN值
        valid_mask = ~(np.isnan(factor_values) | np.isinf(factor_values))
        if not np.any(valid_mask):
            return self.reward["error"]
        
        valid_factor_values = factor_values[valid_mask]
        
        if len(valid_factor_values) == 0:
            return self.reward["error"]
        
        # 计算因子的标准差，如果标准差为0，说明因子没有区分度
        factor_std = np.std(valid_factor_values)
        if factor_std == 0:
            return self.reward["error"]
        
        # 获取当前日期的数据
        current_date = self.date_list[self._current_date_idx]
        date_data = self.data.loc[self.data['date'] == current_date]
        constituent_stocks = self.data_loader.get_pool_stocks_by_date(pool=self.csi_code, date=current_date.strftime("%Y-%m-%d"))
        
        if not constituent_stocks or date_data.empty:
            return self.reward["error"]
        
        # 筛选当前日期的股票数据
        date_data = date_data[date_data['stock'].isin(constituent_stocks)]
        if date_data.empty:
            return self.reward["error"]
        
        # 获取下一个交易日的收益率（作为目标）
        # 简化处理：使用CLOSE的下一日收益率作为目标
        next_date_idx = (self._current_date_idx + 1) % len(self.date_list)
        if next_date_idx == 0:  # 如果是最后一个日期，无法获取下一日数据
            # 使用当前数据的简单指标
            factor_abs_mean = np.mean(np.abs(valid_factor_values))
            reward = min(5.0, factor_abs_mean * 0.1)  # 给一个基础奖励
            return reward
        
        next_date = self.date_list[next_date_idx]
        next_date_data = self.data.loc[self.data['date'] == next_date]
        next_date_data = next_date_data[next_date_data['stock'].isin(constituent_stocks)]
        
        if next_date_data.empty:
            # 如果没有下一日数据，使用当前数据的简单指标
            factor_abs_mean = np.mean(np.abs(valid_factor_values))
            reward = min(5.0, factor_abs_mean * 0.1)  # 给一个基础奖励
            return reward
        
        # 计算IC（信息系数）- 因子值与下期收益率的相关性
        # 首先需要对齐股票代码和因子值
        common_stocks = set(date_data['stock']).intersection(set(next_date_data['stock']))
        if len(common_stocks) < 2:  # 需要至少2个股票来计算相关性
            factor_abs_mean = np.mean(np.abs(valid_factor_values))
            reward = min(5.0, factor_abs_mean * 0.1)
            return reward
        
        # 对齐数据
        current_data_filtered = date_data[date_data['stock'].isin(common_stocks)].copy()
        next_data_filtered = next_date_data[next_date_data['stock'].isin(common_stocks)].copy()
        
        # 按股票代码排序确保对齐
        current_data_filtered = current_data_filtered.sort_values('stock')
        next_data_filtered = next_data_filtered.sort_values('stock')
        
        # 确保因子值与股票数据对齐
        # 创建股票到因子值的映射
        stock_to_factor = {}
        stock_list = date_data['stock'].tolist()
        
        # 根据原始数据的顺序，将因子值与股票对齐
        for i, stock in enumerate(stock_list):
            if i < len(factor_values) and valid_mask[i]:
                stock_to_factor[stock] = factor_values[i]
        
        # 获取对齐的因子值和收益率
        aligned_factor_values = []
        stock_returns = []
        
        for _, row in current_data_filtered.iterrows():
            stock = row['stock']
            if stock in stock_to_factor and stock in next_data_filtered['stock'].values:
                factor_val = stock_to_factor[stock]
                aligned_factor_values.append(factor_val)
                
                # 计算收益率
                next_row = next_data_filtered[next_data_filtered['stock'] == stock].iloc[0]
                current_close = row['close']
                next_close = next_row['close']
                if current_close != 0:
                    ret = (next_close - current_close) / current_close
                    stock_returns.append(ret)
        
        if len(aligned_factor_values) < 2 or len(stock_returns) < 2 or len(aligned_factor_values) != len(stock_returns):
            factor_abs_mean = np.mean(np.abs(valid_factor_values))
            reward = min(5.0, factor_abs_mean * 0.1)
            return reward
        
        aligned_factor_values = np.array(aligned_factor_values)
        stock_returns = np.array(stock_returns)
        
        # 计算IC
        ic = np.corrcoef(aligned_factor_values, stock_returns)[0, 1]
        
        if np.isnan(ic):
            ic = 0
        
        # 根据IC计算奖励 - IC绝对值越大越好
        ic_abs = abs(ic)
        reward = ic_abs * 2000.0  # 放大IC值作为奖励
        
        # 也可以考虑IC的正负性，但通常我们更关心绝对值
        if ic < 0:
            reward = -reward  # 如果是负相关，给予负奖励
            
        # 限制奖励范围
        reward = np.clip(reward, -1000.0, 1000.0)
        
        return reward
    

    def _change_token_from_deepseek(self, tokens_list,error_prompt):
        """调用DeepSeek修正无效的逆波兰表达式（后缀表达式），返回可执行的逆波兰token序列"""
        import os
        from openai import OpenAI
        from openai import APIError, APIConnectionError, RateLimitError

        try:
            client = OpenAI(
                api_key='sk-38219ab20ad04423bebdb9ed1e8e3f50',
                base_url="https://api.deepseek.com"
            )

            # 核心修改：强化提示词，明确要求逆波兰表达式（后缀表达式）
            system_prompt = f"""
            你是金融因子逆波兰表达式（后缀表达式）修正专家，需将无效的逆波兰token序列修正为可执行的版本，严格遵守以下规则：
            【逆波兰表达式核心规则】
            1. 结构：操作数（字段/常量）在前，运算符紧跟在对应操作数之后，无括号，仅通过顺序体现运算优先级；
            2. 运算符匹配：
            - 一元运算符（{self.token_lib.OPERATORS["unary"]}）：必须紧跟1个操作数后（如 Log(close) → close, Log）；
            - 二元运算符（{self.token_lib.OPERATORS["binary"]}）：必须紧跟2个操作数后（如 close+1 → close, 1, Add）；
            - rolling类运算符（{self.token_lib.OPERATORS["rolling"]}）：紧跟1个操作数后；pair_rolling紧跟2个操作数后；
            3. 有效性：最终栈校验需通过（所有运算符有足够前置操作数，最终栈仅1个操作数）；
            【可用token范围】
            - 字段：{self.token_lib.FIELDS}
            - 常量：{[str(c) for c in self.token_lib.CONSTANTS]}
            - 运算符：{self.token_lib.OPERATORS}
            - 特殊标记：{self.token_lib.BEG}（开头）、{self.token_lib.SEP}（结束）
            【输出要求】
            仅返回修正后的逆波兰token序列（用逗号分隔），无需额外解释。
            """
            user_prompt = f"{error_prompt},修正无效的金融因子逆波兰token序列，确保符合上述逆波兰表达式规则：原始无效序列：{tokens_list}"

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],stream=False,temperature=0.1)  # 低随机性，保证逆波兰格式稳定
            
            # 解析修正后的逆波兰token序列
            corrected_str = response.choices[0].message.content.strip()
            corrected_tokens = [token.strip() for token in corrected_str.split(",")]
            # print(f"DeepSeek修正后的逆波兰序列：{corrected_tokens}")
            return corrected_tokens
        except (APIError, APIConnectionError, RateLimitError) as e:
            print(f"DeepSeek API调用失败: {e}")
            return tokens_list
        except Exception as e:
            print(f"DeepSeek逆波兰表达式解析失败: {e}")
            return tokens_list
    
    #['BEG', 'OPEN', '1.0', 'Add', 'Log', 'CLOSE', 'Div']

    def _record_factor_expr(self, corrected_tokens, reward):
        """
        将因子表达式和奖励记录到CSV文件
        :param factor_expr_dict: 因子表达式字典
        :param reward: 奖励值
        """
        try:
            # 将字典形式的factor_expr转换为字符串形式
            factor_expr_str = self.RPNEncoder.decode(corrected_tokens, return_type="string")
            # 创建记录数据
            record = pd.DataFrame({'time':self._current_date_idx,'factor_expr': [factor_expr_str], 'reward': [reward]})
            # 追加到CSV文件
            record.to_csv(self.log_file_path, mode='a', header=False, index=False, encoding='utf-8')
        except Exception as e:
            print(f"记录因子表达式到CSV时出错: {e}")

    