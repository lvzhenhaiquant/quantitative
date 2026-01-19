import hashlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from utils.DataLoad import DataProcessor
from FactorToken import FactorTokenLibrary, RPNEncoder
from utils.calculator import FactorCalculator
import utils.utils_tools as utils_tools

#大模型调用包
import os
from openai import OpenAI
from openai import APIError, APIConnectionError, RateLimitError

class FactorRLEnv():
    """因子生成强化学习环境"""
    
    def __init__(self):
        # 初始化因子库和数据处理器
        self.token_lib = FactorTokenLibrary()
        self.data_processor = DataProcessor()
        self.RPNEncoder = RPNEncoder(self.token_lib)

        # 初始化因子计算器
        self.calculator = FactorCalculator()

        # 定义序列最大长度（使用TokenLibrary中的MAX_EXPR_LENGTH=15）
        self.token_len = self.token_lib.token_max_length
        # 观测维度 = 数据特征维度
        self.observation_dim = len(self.data_processor.fields_origin)
        self.ast_feature_dim = 9  # AST结构特征维度：depth, node_count, operator_count等

        # 使用 n 天的历史截面数据
        self.date_history_window_size = 3
        
        # 定义观测空间和动作空间
        self.action_space = self.token_lib.all_tokens

        # 初始化状态
        self.token_action = []
        self._episode_ended = False
        self.reward = {
            "fail_token_invalid": -8,
            "fail_ic": -3.0,
            "fail_token_incomplete": -9.0,
            "fail_factor": -6.0,
            "fail_history_factor": -10,
            "fail_calculate_reward": -1,
            "fail_token_invalid_partial": -7,
            "structure_reward_weight": 2,  # 结构奖励权重
            "subtree_closed_reward": 5,     # 子树闭合奖励
            "balanced_structure_reward": 3,  # 结构平衡奖励
            "depth_penalty_weight": 3,       # 深度惩罚权重
            "max_depth_limit": 5               # 最大允许深度
        }

        # 加载环境数据
        self.max_stock_num=1000
        self.csi_code = "csi1000"
        self.data_loader = DataProcessor()
        self.data = self.data_processor._preprocess_data(self.data_loader.load_data(self.csi_code,'2020-01-01', '2025-12-31'))
        self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce').dt.date
        self.date_list = self.data['date'].dropna().sort_values().unique().tolist()
        # 按日期缓存数据以加速访问
        self.date_data_cache = {}
        for date in self.date_list:
            self.date_data_cache[date] = self._filter_date_from_pool(date)

        # 当前数据点
        self._current_date_idx = self.date_history_window_size-1 #0 ,  由于历史窗口为 n 天，所以从第 n-1 天开始
        self._current_data_features = self._get_current_data_features()

        # 日志文件路径
        self.reward_file_csv = "./RL_Data/factor_expr_reward.csv"
        self.alpha_pool_csv = "./RL_Data/alpha_pool.csv"
        
        # AlphaPool因子池管理
        self.alpha_pool = []  # 因子池，容量上限10个
        self.alpha_pool_capacity = 10
        self.alpha_remove_num = 3 # 每次大模型介入剔除因子数量
        self.failed_factor_cache = set()  # 历史失败缓存因子

        self.factor_evaluation_cache = {}  # 存储因子在不同时间点的评估结果

        # 大模型介入计数器
        self.step_count = 0
        self.large_model_interval = 10

        self.ic_history = [] #记录ic用于计算icir

    def action_size(self):
        return len(self.action_space)
    
    def feature_size(self):
        return self.observation_dim
    
    def is_last(self):
        """对外暴露的done判断接口，核心返回self.episode_done"""
        return self._episode_ended
    
    def reset(self,date_idx_init = True):
        """重置环境"""
        self.token_action = [self.action_space.index(self.token_lib.BEG)]
        # 重置当前日期索引为0
        if date_idx_init or self._current_date_idx >= len(self.date_list):
            self._current_date_idx = 0
        else:
            self._current_date_idx +=1 #如果一个因子表达式算一个done的话，日期继续推进
        self._current_data_features = self._get_current_data_features()
        self._episode_ended = False
        reset_result = self._get_observation()
        return reset_result , 0, self._episode_ended
    
    def _filter_date_from_pool(self,current_date):
        # 先检查缓存
        if current_date in self.date_data_cache:
            return self.date_data_cache[current_date]
        date_data = self.data.loc[self.data['date'] == current_date]
        constituent_stocks = self.data_loader.get_pool_stocks_by_date(pool=self.csi_code,date=current_date.strftime("%Y-%m-%d"))
        if not constituent_stocks:
            return np.zeros(self.observation_dim)
        # 筛选当前日期的股票数据
        date_data = date_data[date_data['stock'].isin(constituent_stocks)]
        # 缓存结果
        self.date_data_cache[current_date] = date_data
        return date_data

 # 避免除零错误
    def _get_current_data_features(self):
        """获取当前数据点及其历史的特征，返回shape为(history_window_size, max_stock_num, n_features)"""
        # 获取历史窗口内的所有日期
        history_dates = []
        for i in range(self.date_history_window_size):
            date_idx = self._current_date_idx - i
            if date_idx >= 0 and date_idx < len(self.date_list):  # 添加对索引上限的检查
                history_dates.append(self.date_list[date_idx])
            elif date_idx < 0:
                # 如果历史窗口超过数据起始点，填充最早的日期
                history_dates.append(self.date_list[0])
            else:
                # 如果历史窗口超过数据结束点，填充最晚的日期,在step最后一天 self._current_date_idx 仍然会+1，这里修缮一下
                history_dates.append(self.date_list[-1])
        # 反转日期列表，使最新的日期在最后
        history_dates.reverse()
        
        # 存储历史窗口内的所有数据
        history_features = []
        # 对每一列（每个特征）计算全局的min/max用于标准化
        all_data_fields = self.data[self.data_processor.fields_origin]
        min_vals = all_data_fields.min().values
        max_vals = all_data_fields.max().values
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1e-8
        
        for current_date in history_dates:
            # 获取当前日期所有股票的特征值
            date_data = self._filter_date_from_pool(current_date)
            # 筛选有效字段
            available_fields = [field for field in self.data_processor.fields_origin if field in date_data.columns]
            if not available_fields:
                print(f"警告: 没有找到任何有效字段 ")
                day_features = np.zeros((self.max_stock_num, self.observation_dim), dtype=np.float32)
            else:
                # 填充缺失值（量化中常用0/行业均值，避免NaN）
                raw_features = date_data[available_fields].fillna(0).values
                # 标准化特征
                normalized_features = (raw_features - min_vals[:len(available_fields)]) / range_vals[:len(available_fields)]
                n_stocks, n_features = normalized_features.shape
                if n_stocks < self.max_stock_num:
                    # 填充零数组到最大股票数量
                    padding = np.zeros((self.max_stock_num - n_stocks, n_features), dtype=np.float32)
                    padded_features = np.vstack([normalized_features, padding])
                else:
                    # 如果超过最大数量，截取到最大数量
                    padded_features = normalized_features[:self.max_stock_num]            
            history_features.append(padded_features)
        # 将历史特征转换为numpy数组，形状为(history_window_size, max_stock_num, n_features)
        return np.array(history_features, dtype=np.float32)

    def _get_observation(self):
        """获取观测值 - 结合Token序列和当前数据特征"""
        # 填充Token序列到最大长度,能看到一个因子序列历史动作值
        token_seq = np.full(self.token_len, 0, dtype=np.float32)
        token_seq[:len(self.token_action)] = self.token_action
        # 组合Token序列和数据特征
        token_seq_float = token_seq.astype(np.float32)
        current_data_feat = self._get_current_data_features()

        # 获取AST结构特征
        current_tokens = [self.action_space[i] for i in self.token_action]
        ast_features = self.RPNEncoder.step_ast(current_tokens)
        ast_feature_vector = np.array([
            ast_features['depth'],
            ast_features['node_count'],
            ast_features['operator_count'],
            ast_features['unary_operator_count'],
            ast_features['binary_operator_count'],
            ast_features['rolling_operator_count'],
            ast_features['pair_rolling_operator_count'],
            1.0 if ast_features['subtree_closed'] else 0.0,
            1.0 if ast_features['valid_structure'] else 0.0
        ], dtype=np.float32)
        # 标准化AST特征
        ast_feature_vector[0] = ast_feature_vector[0] / 10.0  # 深度标准化到0-1
        ast_feature_vector[1] = ast_feature_vector[1] / self.token_len  # 节点数标准化
        ast_feature_vector[2] = ast_feature_vector[2] / (self.token_len // 2)  # 操作符数标准化

        token_seq_float = np.expand_dims(token_seq_float, axis=0) #历史窗口为n天，不需要扩展维度
        current_data_feat = np.expand_dims(current_data_feat, axis=0)
        ast_feature_vector = np.expand_dims(ast_feature_vector, axis=0) #历史窗口为n天，不需要扩展维度
        obs = [current_data_feat,token_seq_float,ast_feature_vector] #返回当前数据特征，token序列，ast特征
        return obs
    
    def _calculate_structure_reward(self, ast_features):
        """计算AST结构奖励"""
        reward = 0.0
        
        # 子树闭合奖励
        if ast_features['subtree_closed']:
            reward += self.reward['subtree_closed_reward']
        
        # 结构平衡奖励（节点数与操作符数的比例）
        if ast_features['node_count'] > 0:
            operator_ratio = ast_features['operator_count'] / ast_features['node_count']
            # 理想比例约为 0.5（操作符数约为总节点数的一半）
            balance_score = 1.0 - abs(operator_ratio - 0.5)
            reward += balance_score * self.reward['balanced_structure_reward']
        
        # 深度惩罚（避免过深的树结构）
        if ast_features['depth'] > self.reward['max_depth_limit']:
            depth_excess = ast_features['depth'] - self.reward['max_depth_limit']
            reward -= depth_excess * self.reward['depth_penalty_weight']
        
        # 结构有效性惩罚
        if not ast_features['valid_structure']:
            reward -= 5.0
        
        return reward * self.reward['structure_reward_weight']
    def step(self, action):
        """执行动作"""
        if self._episode_ended:
            return self.reset(date_idx_init = False)
        
        # 大模型阶段性介入
        self.step_count += 1
        if self.step_count % self.large_model_interval == 0:
            self._large_model_intervention()
        
        # 添加索引范围检查
        if self._current_date_idx >= len(self.date_list):
            return self.reset(date_idx_init=True)
        current_date = self.date_list[self._current_date_idx]

        # 检查是否达到最大长度或结束符
        if len(self.token_action) >= self.token_len or self.action_space[action] == self.token_lib.SEP:
            reward = self._calculate_reward()
            self._current_date_idx +=1
            #日期结束，重置
            # if self._current_date_idx >= len(self.date_list):
            self._episode_ended = True #表达式结束
            # 重置动作序列
            self.token_action = [self.action_space.index(self.token_lib.BEG)] 
            print(f"_current_date_idx:  {current_date}, action：{action} reward: {reward}")
            return self._get_observation(),reward,self._episode_ended 
        else:
            # 添加动作并计算结构奖励
            self.token_action.append(action)
            current_tokens = [self.action_space[i] for i in self.token_action]
            ast_features = self.RPNEncoder.step_ast(current_tokens)
            structure_reward = self._calculate_structure_reward(ast_features)
            print(f"_current_date_idx:  {current_date}, action：{action} structure_reward: {structure_reward}")
            return self._get_observation(),structure_reward,self._episode_ended 

    def _calculate_reward(self):
        """使用实际金融数据计算因子的奖励""" 
        # 1. 序列验证
        current_tokens = [self.action_space[i] for i in self.token_action]
        # current_tokens = ['BEG', 'close', 'high', 'Max', 'low', 'open', 'Min', 'Sub', '10', 'Mean', 'SEP']  # mean(max(close, high) - min(low), 10)
        
        # 检查是否符合逆波兰表达式规则
        if not self.RPNEncoder._is_valid_expression(current_tokens):
            return self.reward["fail_token_invalid"]  # 无效因子奖励
        
        # 2. AlphaPool环境评估
        try:
            factor_expr = self.RPNEncoder.decode(current_tokens,return_type="dict")
            factor_values = self._execute_factor(factor_expr)
            if len(factor_values) == 0 or np.all(np.isnan(factor_values)) or np.all(factor_values == 0):
                return self.reward["fail_factor"]
            
            # 计算IC和ICIR
            ic, icir = self._calculate_ic_icir(factor_expr,factor_values)
            if np.isnan(ic) or np.isnan(icir):
                return self.reward["fail_ic"]
              
            # 3. 因子池筛选
            factor_hash = hashlib.md5(str(current_tokens).encode()).hexdigest()

            # 检查是否为历史失败因子
            if factor_hash in self.failed_factor_cache:
                return self.reward["fail_history_factor"]
            
            # 新的因子加入因子池
            factor_info = {
                'tokens': current_tokens,
                'hash': factor_hash,
                'ic': ic,
                'icir': icir,
                'timestamp': self.step_count,
            }
            # 2. 计算并添加加权分数
            factor_info['weighted_score'] = self._weighted_score(factor_info)
            

            # 存储当前评估结果到缓存
            self.factor_evaluation_cache[factor_hash] = {
                'current_date': self.date_list[self._current_date_idx],
                'ic': ic,
                'icir': icir,
                'weighted_score': factor_info['weighted_score']
            }

            reward = 0
            if len(self.alpha_pool) < self.alpha_pool_capacity:
                # 成功入池，奖励为当前因子的加权评分
                self.alpha_pool.append(factor_info)
                reward  = self._weighted_score(factor_info)
                # 写入文档
                self._record_factor_expr(factor_info,reward)
                self._save_alpha_pool_to_local()
                return reward
            else:
                # 替换前重新计算所有因子的最新表现
                self._reevaluate_all_factors()
                # 替换池中的劣质因子，奖励为当前因子的加权评分
                worst_factor = min(self.alpha_pool, key=self._weighted_score)
                reward = self._weighted_score(factor_info)
                if reward > self._weighted_score(worst_factor):
                    # 替换
                    self.alpha_pool.remove(worst_factor)
                    self.alpha_pool.append(factor_info)
                    # 成功入池，奖励为当前因子的加权评分
                    self._record_factor_expr(factor_info,reward)
                    self._save_alpha_pool_to_local()
                    return reward
                else:
                    # 无法入池，奖励为当前池最差因子的加权评分
                    reward = self._weighted_score(worst_factor)
                    return reward
        except Exception as e:
            print(f"计算奖励失败: {e}")
            return self.reward["fail_calculate_reward"]
            
    def _weighted_score(self,factor):
            return (0.3 * factor['ic'] + 0.7 * factor['icir']) * 100
    def _reevaluate_all_factors(self):
        """重新评估因子池中的所有因子，使用最新数据"""
        if not self.alpha_pool:
            return
        # print(f"重新评估因子池中的 {len(self.alpha_pool)} 个因子...")
        # 对每个因子重新计算IC和ICIR
        for factor in self.alpha_pool:
            try:
                # 重新执行因子表达式
                factor_expr = self.RPNEncoder.decode(factor['tokens'], return_type="dict")
                factor_values = self._execute_factor(factor_expr)
                
                if len(factor_values) == 0 or np.all(np.isnan(factor_values)) or np.all(factor_values == 0):
                    print(f"_reevaluate_all_factors：_execute_factor 失败: {factor_expr}")
                    # 因子计算失败，设为极低分数
                    factor['ic'] -=1
                    factor['icir'] -=1
                    factor['weighted_score'] -=100
                    continue
                
                # 重新计算IC和ICIR
                ic, icir = self._calculate_ic_icir(factor_expr, factor_values)
                if np.isnan(ic) or np.isnan(icir):
                    print(f"_reevaluate_all_factors：_calculate_ic_icir 失败: {factor_expr}")
                    # IC/ICIR计算失败，设为极低分数
                    factor['ic'] -=1
                    factor['icir'] -=1
                    factor['weighted_score'] -=100
                    continue
                
                # 更新因子的最新表现
                factor['ic'] = ic
                factor['icir'] = icir
                factor['weighted_score'] = self._weighted_score(factor)
                factor['last_evaluated_date'] = self.date_list[self._current_date_idx]
                
                # 更新缓存
                self.factor_evaluation_cache[factor['hash']] = {
                    'current_date': factor['last_evaluated_date'],
                    'ic': ic,
                    'icir': icir,
                    'weighted_score': factor['weighted_score']
                }
                
            except Exception as e:
                print(f"重新评估因子失败: {factor['hash']}, 错误: {e}")
                # 评估失败，设为极低分数
                factor['ic'] -=1
                factor['icir'] -=1
                factor['weighted_score'] -=100
        
        # 按新的评分重新排序因子池
        self.alpha_pool.sort(key=lambda x: x['weighted_score'], reverse=True)

    def _execute_factor(self, factor_expr):
        """
        执行因子表达式并返回因子值
        :param data: 当前数据，包含股票数据
        :param factor_expr: 因子表达式字典
        :return: 因子值（numpy数组）
        """
        # 获取当前日期的数据
        current_date = self.date_list[self._current_date_idx]
        # date_data = self._filter_date_from_pool(current_date)
        date_data = self.date_data_cache[current_date]
        if date_data.empty:
            return np.array([])
        result = self._execute_node(factor_expr, date_data)
        return result

    def _execute_node(self, node, date_data):
        """ 递归执行表达式节点 :param node: 表达式节点 :param data: 当前数据 :return: 计算结果 """
        node_type = node.get('type')
        if node_type == 'field':
            # 字段节点，返回对应字段的值
            field_name = node.get('name', '').lower()
            if field_name in date_data.columns:
                return date_data[field_name].values.astype(np.float32)
            else:
                # 如果字段不存在
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
                return self.calculator.execute_unary_op(operator, operand_values[0])
            elif operator in self.token_lib.OPERATORS["binary"]:
                return self.calculator.execute_binary_op(operator, operand_values[0], operand_values[1])
            elif operator in self.token_lib.OPERATORS["rolling"]:
                # 滚动操作符，传递完整的操作数节点
                return self.calculator.execute_rolling_op(operator, operands, date_data, self.date_list, self.data)
            elif operator in self.token_lib.OPERATORS["pair_rolling"]:
                # 配对滚动操作符，传递完整的操作数节点
                return self.calculator.execute_pair_rolling_op(operator, operands, date_data, self.date_list, self.data)
            else:
                raise ValueError(f"Unknown operator: {operator}")
        else:
            raise ValueError(f"Unknown node type: {node_type}")
 
    def _calculate_ic_icir(self, factor_expr, factor_values): 
        """ 
        计算IC和ICIR 
        IC：信息系数，衡量因子值与下期收益率的相关性 
        ICIR：信息比率，IC的平均值除以IC的标准差 
        """ 
        current_date = self.date_list[self._current_date_idx] 
        date_data = self.date_data_cache[current_date] 
        
        if date_data.empty: 
            return np.nan, np.nan 
        
        # 获取下一个交易日的收益率（作为目标） 
        next_date_idx = self._current_date_idx + 1
        if next_date_idx >= len(self.date_list):  # 如果是最后一个日期，无法获取下一日数据 
            return np.nan, np.nan 
        
        next_date = self.date_list[next_date_idx] 
        next_date_data = self.date_data_cache[next_date] 
        
        if next_date_data.empty: 
            return np.nan, np.nan 
        
        # 计算IC（信息系数）：筛选共同股票并按股票代码排序 
        common_stocks = set(date_data['stock']).intersection(set(next_date_data['stock'])) 
        if len(common_stocks) < 2:  # 至少需要2只股票才能计算相关性 
            return np.nan, np.nan 
        
        current_data_filtered = date_data[date_data['stock'].isin(common_stocks)].sort_values('stock').reset_index(drop=True) 
        next_data_filtered = next_date_data[next_date_data['stock'].isin(common_stocks)].sort_values('stock').reset_index(drop=True) 

        # 确保股票顺序完全一致 
        current_stocks = current_data_filtered['stock'].values 
        next_stocks = next_data_filtered['stock'].values 
        if not np.array_equal(current_stocks, next_stocks): 
            return np.nan, np.nan 

        # 步骤1：先提取收盘价，计算除零掩码（修正顺序：先算mask，再算收益率） 
        current_close = current_data_filtered['close'].values.astype(np.float32) 
        next_close = next_data_filtered['close'].values.astype(np.float32) 
        
        # 定义除零掩码（避免除以零，同时过滤收盘价为NaN的情况） 
        mask = (current_close != 0) & (~np.isnan(current_close)) & (~np.isnan(next_close)) 
        if np.sum(mask) < 2:  # 有效数据不足2个，无法计算相关性 
            return np.nan, np.nan 

        # 步骤2：计算有效股票的下期收益率（修正顺序：使用提前定义的mask） 
        stock_returns = np.zeros_like(current_close, dtype=np.float32) 
        valid_current_close = current_close[mask] 
        valid_next_close = next_close[mask] 
        stock_returns[mask] = (valid_next_close - valid_current_close) / valid_current_close 

        # 步骤3：因子值对齐（严谨版：按长度对齐+过滤NaN） 
        if len(factor_values) >= len(current_data_filtered): 
            aligned_factor_values = factor_values[:len(current_data_filtered)] 
        else: 
            # 补NaN：仅补充缺失长度，不填充有效数值 
            aligned_factor_values = np.pad( 
                factor_values, 
                (0, len(current_data_filtered) - len(factor_values)), 
                'constant', 
                constant_values=np.nan 
            ) 
        aligned_factor_values = aligned_factor_values.astype(np.float32) 

        # 步骤4：过滤双重有效数据（因子值非NaN + 收益率有效） 
        valid_mask = ~np.isnan(aligned_factor_values) & mask 
        valid_factor_values = aligned_factor_values[valid_mask] 
        valid_returns = stock_returns[valid_mask] 
        
        if len(valid_factor_values) < 2:  # 有效数据不足2个，无法计算相关性 
            return np.nan, np.nan 

        # 步骤5：计算当前IC（异常捕获+严谨处理NaN） 
        ic = np.nan 
        try: 
            # 计算皮尔逊相关系数（[0,1] 对应因子与收益率的相关系数） 
            corr_matrix = np.corrcoef(valid_factor_values, valid_returns) 
            ic = corr_matrix[0, 1] 
        except Exception as e: 
            # 捕获相关性计算异常（如所有值相同） 
            print(f"计算IC失败：{e}") 
            ic = np.nan 

        # 步骤6：仅将有效IC值存入历史（过滤NaN，避免稀释结果） 
        if not np.isnan(ic): 
            self.ic_history.append(ic) 
        else: 
            return np.nan, np.nan 

        # 步骤7：维护IC历史窗口大小（先进先出，保持窗口长度不超过ic_history_window） 
        factor_windows = self._extract_factor_period(factor_expr) 
        ic_history_window = min(factor_windows, len(self.ic_history))  # 窗口大小不超过当前历史长度 
        
        # 裁剪历史列表：保持最新的ic_history_window个数据（先进先出） 
        while len(self.ic_history) > ic_history_window: 
            self.ic_history.pop(0) 

        # 步骤8：计算ICIR（鲁棒性处理+避免除以零） 
        icir = 0.0 
        if len(self.ic_history) >= ic_history_window and ic_history_window >= 2: 
            # 截取最新的ic_history_window个IC数据 
            recent_ic_history = self.ic_history[-ic_history_window:] 
            ic_mean = np.mean(recent_ic_history) 
            ic_std = np.std(recent_ic_history) 
            
            # 定义极小值阈值，避免浮点精度问题导致的除以零 
            min_std_threshold = 1e-8 
            if ic_std > min_std_threshold: 
                icir = ic_mean / ic_std 
            else: 
                icir = 0.0 

        return ic*500, icir*500
    def _extract_factor_period(self, factor_expr):
        """
        从因子表达式中提取周期信息（返回所有滚动算子的最短周期）
        :param factor_expr: 因子表达式字典
        :return: 因子周期（天数）
        """
        # 存储所有提取到的有效滚动周期（仅保留窗口大小≥1的）
        valid_periods = []
        # 递归函数，用于遍历表达式树并收集所有有效周期
        def find_period(node):
            node_type = node.get('type')
            if node_type == 'field' or node_type == 'constant':
                return  # 基础字段/常量无滚动周期，跳过
            elif node_type == 'operation':
                operator = node.get('operator')
                operands = node.get('operands', [])
                
                # 对于滚动操作符，提取窗口大小并加入有效周期列表
                if operator in self.token_lib.OPERATORS["rolling"] or operator in self.token_lib.OPERATORS["pair_rolling"]:
                    # 滚动操作符的最后一个操作数通常是窗口大小
                    if operands and len(operands) > 1:
                        window_operand = operands[-1]
                        if window_operand.get('type') == 'constant':
                            try:
                                window = int(float(window_operand.get('value', 1)))
                                # 仅保留有效窗口（≥1）
                                if window >= 1:
                                    valid_periods.append(window)
                            except (ValueError, TypeError):
                                pass  # 窗口解析失败，忽略
                # 递归遍历所有子操作数
                for op in operands:
                    find_period(op)
        
        # 执行递归遍历，收集所有有效周期
        find_period(factor_expr)
        
        # 核心逻辑：返回最短周期，无有效周期则返回默认1
        if valid_periods:
            return min(valid_periods)  # 取所有有效周期的最小值
        else:
            return 1  # 无滚动算子时返回默认周期1

    def _change_token_from_deepseek(self, tokens_list):
        """调用DeepSeek修正无效的逆波兰表达式（后缀表达式），返回可执行的逆波兰token序列"""
        try:
            deepseek_token = utils_tools.load_token_from_txt()
            client = OpenAI(api_key=deepseek_token,
                            base_url="https://api.deepseek.com")

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
            user_prompt = f"修正无效的金融因子逆波兰token序列，确保符合上述逆波兰表达式规则：原始无效序列：{tokens_list}"

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
    

    def _generate_factors_from_deepseek(self, num_factors):
        """调用DeepSeek API生成优质因子
        :param num_factors: 需要生成的因子数量
        :return: 生成的因子列表，每个因子包含tokens、hash、ic、icir、timestamp等字段
        """
        try:
            deepseek_token = utils_tools.load_token_from_txt()
            client = OpenAI(api_key=deepseek_token,
                            base_url="https://api.deepseek.com")

            # 设计生成优质因子的提示词
            system_prompt = f"""
            你是金融因子逆波兰表达式（后缀表达式）生成专家，需生成具有良好预测能力的金融因子，严格遵守以下规则：
            【核心要求】
            1. 因子类型：生成用于股票价格预测的量化因子，重点关注量价指标的创新性组合
            2. 预测能力：因子应具有较高的信息系数（IC）和信息系数秩相关系数（ICIR）
            3. 表达式格式：必须使用逆波兰表达式（后缀表达式）格式
            
            【逆波兰表达式规则】
            1. 结构：操作数（字段/常量）在前，运算符紧跟在对应操作数之后，无括号
            2. 运算符匹配：
            - 一元运算符（{self.token_lib.OPERATORS["unary"]}）：必须紧跟1个操作数后（如 Log(close) → close, Log）；
            - 二元运算符（{self.token_lib.OPERATORS["binary"]}）：必须紧跟2个操作数后（如 close+1 → close, 1, Add）；
            - rolling类运算符（{self.token_lib.OPERATORS["rolling"]}）：紧跟1个操作数后；pair_rolling紧跟2个操作数后；
            3. 有效性：最终栈校验需通过（所有运算符有足够前置操作数，最终栈仅1个操作数）；
            4. 长度限制：token序列长度不超过{self.token_lib.token_max_length}个token
            5. 开头和结尾：必须以{self.token_lib.BEG}开头，以{self.token_lib.SEP}结尾
            
            【可用token范围】
            - 字段：{self.token_lib.FIELDS}
            - 常量：{[str(c) for c in self.token_lib.CONSTANTS]}
            - 运算符：{self.token_lib.OPERATORS}
            - 特殊标记：{self.token_lib.BEG}（开头）、{self.token_lib.SEP}（结束）
            
            【输出要求】
            1. 生成{num_factors}个不同的因子，每个因子占一行
            2. 每行仅返回逆波兰token序列（用逗号分隔），无需额外解释
            3. 确保所有生成的因子都符合上述规则
            4. 因子应具有多样性，避免重复或相似的模式
            """
            
            user_prompt = f"请生成{num_factors}个优质的金融因子逆波兰表达式，每个表达式占一行，仅返回token序列。"

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
                temperature=0.7  # 适当的随机性，保证因子多样性
            )

            # 解析生成的因子
            generated_str = response.choices[0].message.content.strip()
            factor_lines = generated_str.split('\n')
            
            new_factors = []
            for i, line in enumerate(factor_lines):
                if not line.strip():  # 跳过空行
                    continue
                try:
                    tokens = [token.strip() for token in line.split(',')]
                    # 验证生成的token序列
                    if len(tokens) > 15 or tokens[0] != self.token_lib.BEG or tokens[-1] != self.token_lib.SEP:
                        continue
                    # 计算因子哈希
                    factor_hash = hashlib.sha256(str(tokens).encode()).hexdigest()
                    # 初始化IC和ICIR（将在后续评估中更新）
                    new_factor = {
                        'tokens': tokens,
                        'hash': factor_hash,
                        'ic': 0.0,  # 初始值，后续会更新
                        'icir': 0.0,  # 初始值，后续会更新
                        'timestamp': self.step_count
                    }
                    new_factors.append(new_factor)
                    # 达到请求数量时停止
                    if len(new_factors) >= num_factors:
                        break

                except Exception as e:
                    print(f"解析生成的因子时出错: {e}")
                    continue
            return new_factors
        
        except (APIError, APIConnectionError, RateLimitError) as e:
            return None
        except Exception as e:
            return None


    def _large_model_intervention(self):
        """大模型阶段性介入，剔除劣质因子，注入新因子"""        
        if len(self.alpha_pool) < self.alpha_remove_num:
            return
        # 剔除 n 个劣质因子
        self.alpha_pool.sort(key=lambda x: x['icir'])
        removed_factors = self.alpha_pool[:self.alpha_remove_num]
        self.alpha_pool = self.alpha_pool[self.alpha_remove_num:]
        # 记录为历史失败因子
        for factor in removed_factors:
            self.failed_factor_cache.add(factor['hash'])
        # 大模型注入新的优质因子
        new_factors = new_factors = self._generate_factors_from_deepseek(self.alpha_remove_num)
        # 加入因子池
        for factor in new_factors:
            if len(self.alpha_pool) < self.alpha_pool_capacity:
                self.alpha_pool.append(factor)

    def generate_expert_factors(self, num_factors=10):
        """调用DeepSeek API生成专家因子，直接返回与custom_expert_factors格式完全一致的二维列表
        :param num_factors: 需要生成的因子数量，默认为10
        :return: 生成的因子列表，格式与custom_expert_factors完全一致
        """
        try:
            deepseek_token = utils_tools.load_token_from_txt()
            client = OpenAI(api_key=deepseek_token,
                            base_url="https://api.deepseek.com")

            # 设计生成优质因子的提示词
            system_prompt = f"""
            你是金融因子逆波兰表达式（后缀表达式）生成专家，需生成具有良好预测能力的金融因子，严格遵守以下规则：
            【核心要求】
            1. 因子类型：生成用于股票价格预测的量化因子，重点关注量价指标的创新性组合
            2. 预测能力：因子应具有较高的信息系数（IC）和信息系数秩相关系数（ICIR）
            3. 表达式格式：必须使用逆波兰表达式（后缀表达式）格式
            
            【逆波兰表达式规则】
            1. 结构：操作数（字段/常量）在前，运算符紧跟在对应操作数之后，无括号
            2. 运算符匹配：
            - 一元运算符（{self.token_lib.OPERATORS["unary"]}）：必须紧跟1个操作数后（如 Log(close) → close, Log）；
            - 二元运算符（{self.token_lib.OPERATORS["binary"]}）：必须紧跟2个操作数后（如 close+1 → close, 1, Add）；
            - rolling类运算符（{self.token_lib.OPERATORS["rolling"]}）：紧跟1个操作数后；pair_rolling紧跟2个操作数后；
            3. 有效性：最终栈校验需通过（所有运算符有足够前置操作数，最终栈仅1个操作数）；
            4. 长度限制：token序列长度不超过{self.token_lib.token_max_length}个token
            5. 开头和结尾：必须以{self.token_lib.BEG}开头，以{self.token_lib.SEP}结尾
            
            【可用token范围】
            - 字段：{self.token_lib.FIELDS}
            - 常量：{[str(c) for c in self.token_lib.CONSTANTS]}
            - 运算符：{self.token_lib.OPERATORS}
            - 特殊标记：{self.token_lib.BEG}（开头）、{self.token_lib.SEP}（结束）
            
            【输出要求】
            1. 生成{num_factors}个不同的因子，每个因子占一行
            2. 每行仅返回逆波兰token序列（用逗号分隔），无需额外解释
            3. 确保所有生成的因子都符合上述规则
            4. 因子应具有多样性，避免重复或相似的模式
            """
            
            user_prompt = f"请生成{num_factors}个优质的金融因子逆波兰表达式，每个表达式占一行，仅返回token序列。"

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
                temperature=0.7  # 适当的随机性，保证因子多样性
            )
            # 解析生成的因子
            generated_str = response.choices[0].message.content.strip()
            factor_lines = generated_str.split('\n')
            expert_factors = []
            for i, line in enumerate(factor_lines):
                if not line.strip():  # 跳过空行
                    continue
                try:
                    tokens = [token.strip() for token in line.split(',')]
                    # 验证生成的token序列
                    if len(tokens) > 15 or tokens[0] != self.token_lib.BEG or tokens[-1] != self.token_lib.SEP:
                        continue
                    expert_factors.append(tokens)
                    # 达到请求数量时停止
                    if len(expert_factors) >= num_factors:
                        break
                except Exception as e:
                    continue
            return expert_factors
        except (APIError, APIConnectionError, RateLimitError) as e:
            print(f"DeepSeek API调用失败: {e}")
            return []
        except Exception as e:
            print(f"生成因子时出错: {e}")
            return []

    def _save_alpha_pool_to_local(self):
        """ 将因子池保存到本地CSV文件，每个值作为独立列 """
        try:
            # 准备保存数据
            save_data = []
            for factor in self.alpha_pool:
                # 解码因子表达式为可读字符串
                factor_expr_str = self.RPNEncoder.decode(factor['tokens'], return_type="string")
                # 将tokens列表转换为逗号分隔的字符串
                tokens_str = ",".join(factor['tokens'])
                save_entry = {
                    'hash': factor['hash'],
                    'tokens': tokens_str,  # 逗号分隔的字符串格式
                    'factor_expr': factor_expr_str,
                    'ic': factor['ic'],
                    'icir': factor['icir'],
                    'timestamp': factor['timestamp']
                }
                save_data.append(save_entry)
            df = pd.DataFrame(save_data)
            df.to_csv(self.alpha_pool_csv, index=False, encoding='utf-8')
        except Exception as e:
            print(f"保存因子池到本地时出错: {e}")

    def _record_factor_expr(self, factor_info, reward):
        """
        将因子完整信息记录到CSV文件（包含tokens、hash、ic、icir、timestamp、奖励等）
        :param factor_info: 因子信息字典（包含tokens/hash/ic/icir/timestamp等）
        :param reward: 该因子的最终奖励值
        """
        try:
            # 1. 解析因子表达式字符串（复用原有解码逻辑）
            factor_expr_str = self.RPNEncoder.decode(factor_info['tokens'], return_type="string")
            
            # 2. 构建完整的记录数据（包含所有关键字段）
            record_data = {
                'time': [self._current_date_idx],  # 当前日期索引
                'factor_expr': [factor_expr_str],  # 可读的因子表达式
                'ic': [factor_info['ic']],  # 信息系数
                'icir': [factor_info['icir']],  # 信息系数信息比率
                'timestamp': [factor_info['timestamp']],  # 步数/时间戳
                'reward': [reward]  # 最终奖励值
            }
            record = pd.DataFrame(record_data)
            record.to_csv(
                self.reward_file_csv,
                mode='w',
                header=True,
                index=False,
                encoding='utf-8'
            )
        except Exception as e:
            print(f"记录因子信息到CSV时出错: {e}")
    











