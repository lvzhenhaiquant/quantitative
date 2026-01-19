import os
import numpy as np
import tensorflow as tf
from DeepNetwork.tcn_transmfor.DeepNeuralNetworks import tcn_transformer_build_actor, tcn_transformer_build_critic
from utils.utils_tools import show_expert_loss

class PPOAgent:
    """
    实现了完整的PPO算法，包括策略网络、价值网络和训练逻辑
    """
    def __init__(self, env, learning_rate=3e-4, gamma=0.99, lambda_gae=0.95, 
                 clip_epsilon=0.2, ent_coef=0.1, num_epochs=4, batch_size=32):
        self.env = env
        self.obs_feat_dim = self.env.feature_size()  # 特征长度
        self.token_len = self.env.token_len  # Token序列长度
        self.action_dim = self.env.action_size()  # 动作空间大小
        self.date_history_window_size = self.env.date_history_window_size
       
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_epsilon = clip_epsilon
        self.ent_coef = ent_coef #策略动作的随机性
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.path_prefix="./DeepNetwork/tcn_transmfor/model/ppo_model"
        self.pretrain_path_prefix="./DeepNetwork/tcn_transmfor/model/pretrain_model"
        
        # 创建经验回放缓冲区
        self.buffer = ReplayBuffer(batch_size)
        # 初始化多GPU策略
        self._init_gpu()

        # 初始化
        self._initialize_networks()

        self._define_distributed_train_step()
        self._define_distributed_pretrain_step()
    
    def _init_gpu(self):
        # 配置多GPU策略 - 仅使用一半的显卡数量（最多使用一半，至少使用2个）
        physical_gpus = tf.config.list_physical_devices('GPU')
        num_physical_gpus = len(physical_gpus)
        # 计算要使用的GPU数量
        if num_physical_gpus == 0:
            # 没有GPU可用，使用CPU
            self.strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        else:
            # 计算要使用的GPU数量：最多使用一半
            num_gpus_to_use = num_physical_gpus // 2
            # 设置可见的GPU设备
            tf.config.set_visible_devices(physical_gpus[:num_gpus_to_use], 'GPU')
            # 创建MirroredStrategy
            self.strategy = tf.distribute.MirroredStrategy()

        self.num_replicas = self.strategy.num_replicas_in_sync
        # 计算每个副本的批次大小，但当前未使用
        # self.per_replica_batch_size = self.batch_size // self.num_replicas

        # 在策略作用域内构建网络和优化器
        with self.strategy.scope():
            # 构建策略网络和价值网络
            self.actor = tcn_transformer_build_actor(self.obs_feat_dim, self.token_len,self.env.ast_feature_dim, self.action_dim)
            self.critic = tcn_transformer_build_critic(self.obs_feat_dim, self.token_len,self.env.ast_feature_dim, self.action_dim)
             # 分别创建优化器
            self.actor_optimizer = tf.keras.optimizers.Adam(self.learning_rate)    
            self.critic_optimizer = tf.keras.optimizers.Adam(self.learning_rate * 0.5)  # Critic可以使用较小的学习率
            self.pretrain_optimizer = tf.keras.optimizers.Adam(self.learning_rate * 0.1)  # 预训练使用较小的学习率 
        

    def _initialize_networks(self):
        """初始化网络权重"""
        dummy_stock_obs = np.zeros((1,self.date_history_window_size,1, self.obs_feat_dim), dtype=np.float32)
        dummy_token_obs = np.zeros((1,self.token_len), dtype=np.int32)
        dummy_ast_obs = np.zeros((1,self.env.ast_feature_dim), dtype=np.float32)
         # 通过调用一次前向传播来初始化权重
        self.actor([dummy_stock_obs, dummy_token_obs,dummy_ast_obs])
        self.critic([dummy_stock_obs, dummy_token_obs,dummy_ast_obs])
    
    def get_action(self, observation):
        """获取动作和对数概率（适配多输入Actor模型）"""
        logits = self.actor(observation)
        # 计算动作概率、采样动作、计算对数概率
        action_probs = tf.nn.softmax(logits)
        action_dist = tf.random.categorical(logits, num_samples=1)  # 采样1个动作
        action = action_dist.numpy()[0, 0]  # 提取采样的动作值
        log_prob = tf.math.log(action_probs[0, action])  # 计算该动作的对数概率
        return action, log_prob.numpy()
    
    def get_value(self, observation):
        """获取状态值"""
        value = self.critic(observation)
        return value.numpy()[0, 0]
    
    def compute_advantages(self, rewards, values, dones):
        """计算优势函数和TD目标"""
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                delta = rewards[t] - values[t]
            else:
                delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t+1]) - values[t]
            gae = delta + self.gamma * self.lambda_gae * gae * (1 - dones[t])
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns
    
    @tf.function(reduce_retracing=True)
    def _train_step(self, batch_states, batch_actions, batch_old_probs, batch_returns, batch_advantages):
        """优化的训练步骤，减少tf.function重追踪"""
        with tf.GradientTape() as actor_tape:
            # 策略网络前向传播
            logits = self.actor(batch_states)
            action_probs = tf.nn.softmax(logits)
            # 计算动作对数概率
            actions_one_hot = tf.one_hot(batch_actions, depth=self.action_dim)
            log_probs = tf.reduce_sum(actions_one_hot * tf.nn.log_softmax(logits), axis=1)
            # 计算概率比
            ratio = tf.exp(log_probs - batch_old_probs)
            # PPO损失
            surr1 = ratio * batch_advantages
            surr2 = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
         
            # 熵正则化
            entropy = -tf.reduce_mean(tf.reduce_sum(action_probs * tf.nn.log_softmax(logits), axis=1))
            actor_loss -= self.ent_coef * entropy
        with tf.GradientTape() as critic_tape:
            # 价值网络损失
            values = tf.squeeze(self.critic(batch_states), axis=1)
            critic_loss = tf.reduce_mean(tf.square(batch_returns - values))

        # 更新网络
        actor_gradients = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        critic_gradients = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
        return actor_loss, critic_loss, entropy
    
    @tf.function(reduce_retracing=True)
    def _pretrain_step(self, batch_states, batch_target_actions):
        """模仿学习预训练步骤"""
        with tf.GradientTape() as tape:
            # 策略网络前向传播
            logits = self.actor(batch_states)
            # 计算交叉熵损失
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=batch_target_actions, logits=logits))
        # 只更新Actor网络
        gradients = tape.gradient(loss, self.actor.trainable_variables)
        self.pretrain_optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))
        return loss
    
        
    def _define_distributed_train_step(self):
        """定义分布式训练步骤"""
        @tf.function(reduce_retracing=True)
        def distributed_train_step(states, actions, old_probs, returns, advantages):
            per_replica_losses = self.strategy.run(self._train_step,args=(states, actions, old_probs, returns, advantages))
            # 聚合所有replica的损失
            mean_actor_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[0], axis=None) / self.num_replicas
            mean_critic_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[1], axis=None) / self.num_replicas
            mean_entropy = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[2], axis=None) / self.num_replicas
            return mean_actor_loss, mean_critic_loss, mean_entropy
        self.distributed_train_step = distributed_train_step

    def _define_distributed_pretrain_step(self):
        """定义分布式预训练步骤"""
        @tf.function(reduce_retracing=True)
        def distributed_pretrain_step(states, target_actions):
            per_replica_losses = self.strategy.run(self._pretrain_step,args=(states, target_actions))
            # 聚合所有replica的损失
            mean_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None) / self.num_replicas
            return mean_loss
        self.distributed_pretrain_step = distributed_pretrain_step

    def learn(self):
        """训练PPO代理"""
        # 多轮训练
        for _ in range(self.num_epochs):
            batches = self.buffer.generate_batches()  
            for batch in batches:
                states, actions, old_probs, vals, rewards, dones = batch
                stock_states = np.array([single_state[0] for single_state in states], dtype=np.float32)
                token_states = np.array([single_state[1] for single_state in states], dtype=np.float32)
                ast_states = np.array([single_state[2] for single_state in states], dtype=np.float32)
                
                # 计算优势函数和回报
                advantages, returns = self.compute_advantages(rewards, vals, dones)
                # 将数据转为Tensor
                stock_states = tf.convert_to_tensor(np.array(stock_states), dtype=tf.float32)
                token_states = tf.convert_to_tensor(np.array(token_states), dtype=tf.float32)
                ast_states = tf.convert_to_tensor(np.array(ast_states), dtype=tf.float32)
                actions = tf.convert_to_tensor(np.array(actions), dtype=tf.int32)
                old_probs = tf.convert_to_tensor(np.array(old_probs), dtype=tf.float32)
                returns = tf.convert_to_tensor(np.array(returns), dtype=tf.float32)
                advantages = tf.convert_to_tensor(np.array(advantages), dtype=tf.float32)
                # 标准化优势函数
                advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
                # 确保维度正确（移除可能的额外维度）
                stock_states = np.squeeze(stock_states, axis=1)
                token_states = np.squeeze(token_states, axis=1)
                ast_states = np.squeeze(ast_states, axis=1)
                batch_states = [stock_states, token_states, ast_states]
                
                # 使用优化的训练步骤
                actor_loss, critic_loss, entropy = self.distributed_train_step(
                    batch_states, actions, old_probs, returns, advantages)

        return {
            'actor_loss': actor_loss.numpy(),
            'critic_loss': critic_loss.numpy(),
            'entropy': entropy.numpy()
        }

    def collect_experience_for_day(self,observation):
        """在单个交易日内收集经验数据"""
        reward_list = []
        for _ in range(self.token_len):
            # 获取单步动作/价值
            action, log_prob = self.get_action(observation)
            value = self.get_value(observation)
            next_obs, reward, done = self.env.step(action)
            reward_list.append(reward)
            self.buffer.store_memory(state=observation,action=action,probs=log_prob,vals=value,reward=reward,done=done)
            observation = next_obs
            if done:
                break
        return np.mean(reward_list)
        

    def save_model(self):
        actor_path = f"{self.path_prefix}_actor.h5"
        critic_path = f"{self.path_prefix}_critic.h5"
        os.makedirs(os.path.dirname(self.path_prefix), exist_ok=True)
        self.actor.save_weights(actor_path, save_format="h5")
        self.critic.save_weights(critic_path, save_format="h5")
        print(f"普通模型保存完成：\n  Actor: {actor_path}\n  Critic: {critic_path}")

    def save_pretrain_model(self):
        """保存预训练模型（仅Actor）权重（H5格式）"""
        actor_path = f"{self.pretrain_path_prefix}_actor.h5"
        os.makedirs(os.path.dirname(self.pretrain_path_prefix), exist_ok=True)
        self.actor.save_weights(actor_path, save_format="h5")
        print(f"预训练模型保存完成：\n  Actor: {actor_path}")
        
    
    def load_model(self):
        """加载模型：优先普通模型（Actor+Critic），降级加载预训练模型（仅Actor）"""
        # 步骤1：优先加载普通模型
        actor_main_path = f"{self.path_prefix}_actor.h5"
        critic_main_path = f"{self.path_prefix}_critic.h5"
        if os.path.exists(actor_main_path) and os.path.exists(critic_main_path):
            self.actor.load_weights(actor_main_path)
            self.critic.load_weights(critic_main_path)
            print(f"普通模型加载完成（Actor+Critic）")
            return  # 加载成功直接返回，无需后续逻辑
        
        # 步骤2：普通模型不存在，降级加载预训练模型
        actor_pretrain_path = f"{self.pretrain_path_prefix}_actor.h5"
        if os.path.exists(actor_pretrain_path):
            self.actor.load_weights(actor_pretrain_path)
            print(f"预训练模型加载完成（仅Actor）")
            return
        print("警告：未找到普通模型和预训练模型，将使用随机初始化权重")
    
    def load_pretrain_model(self):
        """加载预训练模型权重"""
        path_prefix = self.pretrain_path_prefix
        actor_path = f"{path_prefix}_actor"
        self.actor.load_weights(actor_path)
        print(f"预训练模型已从: {actor_path} 加载")
    
    def imitation_pretrain(self, use_large_model=True, num_factors=50):
        """完整优化的模仿学习预训练函数"""
        actor_pretrain_path = f"{self.pretrain_path_prefix}_actor.h5"
        if os.path.exists(actor_pretrain_path):
            self.actor.load_weights(actor_pretrain_path)
            print(f"预训练模型加载完成（仅Actor）")
        
        print("开始模仿学习预训练...")
        
        # 预计算token索引映射，提高查找效率
        self.token_index_map = {token: idx for idx, token in enumerate(self.env.token_lib.all_tokens)}
        self.token_reverse_map = {idx: token for idx, token in enumerate(self.env.token_lib.all_tokens)}
        
        # 获取专家因子序列
        generated_factors = []
        if use_large_model:
            print("调用大模型生成专家因子序列...")
            generated_factors = self.env.generate_expert_factors(num_factors=num_factors)
            if len(generated_factors) == 0:
                print("警告：大模型生成失败，使用自定义专家因子")
                use_large_model = False
        
        if not use_large_model:
            # 遵循逆波兰表达式规则
            generated_factors = [
                ["BEG", "close", "open", "Sub", "SEP"],  # (close - open)
                ["BEG", "close", "Log", "SEP"],  # Log(close)
                ["BEG", "close", "5", "Mean", "SEP"],  # Mean(close, 5)
                ["BEG", "high", "low", "Sub", "open", "Div", "SEP"],  # (high - low) / open
                ["BEG", "close", "1", "Delta", "Abs", "SEP"],  # Abs(Delta(close, 1))
                ["BEG", "close", "open", "Add", "2", "Div", "SEP"],  # (close + open) / 2
                ["BEG", "close", "10", "Mean", "open", "Sub", "SEP"],  # Mean(close, 10) - open
                ["BEG", "close", "high", "Less", "open", "Mul", "SEP"],  # (close < high) * open
                ["BEG", "close", "low", "Max", "open", "Min", "SEP"],  # Min(Max(close, low), open)
                ["BEG", "close", "5", "Std", "Log", "SEP"],  # Log(Std(close, 5))
                ["BEG", "close", "open", "Div", "10", "Ref", "SEP"],  # Ref(close/open, 10)
                ["BEG", "close", "high", "Mul", "low", "Sub", "SEP"],  # (close * high) - low
                ["BEG", "close", "20", "Mean", "close", "Sub", "Abs", "SEP"],  # Abs(close - Mean(close, 20))
                ["BEG", "close", "5", "EMA", "close", "10", "EMA", "Sub", "SEP"],  # EMA(close, 5) - EMA(close, 10)
            ]
        
        # 将专家因子转换为token索引序列
        expert_token_sequences = []
        for factor in generated_factors:
            # 确保因子序列长度不超过token_len
            if len(factor) > self.token_len:
                continue
            
            # 检查是否符合逆波兰表达式规则
            if not self.env.RPNEncoder._is_valid_expression(factor):
                continue
            
            # 转换为token索引
            token_sequence = []
            for token in factor:
                token_idx = self.token_index_map.get(token, -1)
                if token_idx != -1:
                    token_sequence.append(token_idx)
                else:
                    # 如果token不在库中，跳过该因子
                    break
            
            # 如果所有token都有效，添加到专家序列列表
            if len(token_sequence) == len(factor):
                expert_token_sequences.append(token_sequence)
        
        if len(expert_token_sequences) == 0:
            print("警告：没有有效的专家因子序列，无法进行模仿学习预训练")
            return
        print(f"共转换 {len(expert_token_sequences)} 个有效专家因子序列")
        
        # 收集所有日期的损失数据
        all_losses = []
        
        # 减少训练日期数量（最多使用20个日期）
        total_dates = len(self.env.date_list) // 3
        print(f"使用 {total_dates} 个日期进行训练")
        
        # 批量处理参数
        batch_size = 64
        num_iterations = 10  # 减少迭代次数
        
        for date_idx in range(total_dates):
            # 设置当前日期索引
            self.env._current_date_idx = date_idx
            print(f"\n在日期 {date_idx + 1}/{total_dates} ({self.env.date_list[date_idx]}) 上训练...")
            
            # 手动更新环境的当前数据特征
            self.env._current_data_features = self.env._get_current_data_features()
            
            # 获取当前日期的观测数据
            observation = self.env._get_observation()
            stock_obs = observation[0]
            stock_obs = np.squeeze(stock_obs, axis=0)
            
            epoch_loss = 0.0
            num_samples = 0
            
            # 随机打乱专家序列
            np.random.shuffle(expert_token_sequences)
            
            # AST特征缓存
            ast_feature_cache = {}
            
            for iter_idx in range(num_iterations):
                # 收集批量数据的列表
                batch_stock_obs_list = []
                batch_token_input_list = []
                batch_ast_input_list = []
                batch_target_actions_list = []
                
                for expert_seq in expert_token_sequences:
                    # 逐token生成训练数据
                    for step in range(1, len(expert_seq)):
                        # 输入序列：前step个token
                        input_seq = expert_seq[:step]
                        # 目标token：下一个token
                        target_token = expert_seq[step]
                        
                        # 准备输入数据
                        padded_input = input_seq.copy()
                        pad_index = self.token_index_map.get(self.env.token_lib.PAD, 0)
                        while len(padded_input) < self.token_len:
                            padded_input.append(pad_index)
                        
                        # 计算AST结构特征（使用缓存）
                        cache_key = tuple(input_seq)
                        if cache_key not in ast_feature_cache:
                            tokens_list = [self.token_reverse_map[token_idx] for token_idx in input_seq]
                            ast_features = self.env.RPNEncoder.step_ast(tokens_list)
                            
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
                            ast_feature_vector[0] = ast_feature_vector[0] / 10.0
                            ast_feature_vector[1] = ast_feature_vector[1] / self.token_len
                            ast_feature_vector[2] = ast_feature_vector[2] / (self.token_len // 2)
                            
                            ast_feature_cache[cache_key] = ast_feature_vector
                        else:
                            ast_feature_vector = ast_feature_cache[cache_key]
                        
                        # 创建单样本数据
                        single_stock_obs = np.expand_dims(stock_obs, axis=0)
                        single_token_input = np.expand_dims(padded_input, axis=0).astype(np.int32)
                        single_ast_input = np.expand_dims(ast_feature_vector, axis=0)
                        single_target_actions = np.array([target_token], dtype=np.int32)
                        
                        # 收集到批量列表
                        batch_stock_obs_list.append(single_stock_obs)
                        batch_token_input_list.append(single_token_input)
                        batch_ast_input_list.append(single_ast_input)
                        batch_target_actions_list.append(single_target_actions)
                        
                        # 当达到批量大小时，执行一次训练
                        if len(batch_stock_obs_list) >= batch_size:
                            # 合并批次数据
                            batch_stock_obs_batch = np.concatenate(batch_stock_obs_list, axis=0)
                            batch_token_input_batch = np.concatenate(batch_token_input_list, axis=0)
                            batch_ast_input_batch = np.concatenate(batch_ast_input_list, axis=0)
                            batch_target_actions_batch = np.concatenate(batch_target_actions_list, axis=0)
                            
                            # 执行批量训练
                            states = [batch_stock_obs_batch, batch_token_input_batch, batch_ast_input_batch]
                            loss = self.distributed_pretrain_step(states, batch_target_actions_batch)
                            
                            epoch_loss += loss.numpy() * batch_size
                            num_samples += batch_size
                            
                            # 清空批次列表
                            batch_stock_obs_list.clear()
                            batch_token_input_list.clear()
                            batch_ast_input_list.clear()
                            batch_target_actions_list.clear()
                
                # 处理剩余的小批次
                if len(batch_stock_obs_list) > 0:
                    batch_stock_obs_batch = np.concatenate(batch_stock_obs_list, axis=0)
                    batch_token_input_batch = np.concatenate(batch_token_input_list, axis=0)
                    batch_ast_input_batch = np.concatenate(batch_ast_input_list, axis=0)
                    batch_target_actions_batch = np.concatenate(batch_target_actions_list, axis=0)
                    
                    states = [batch_stock_obs_batch, batch_token_input_batch, batch_ast_input_batch]
                    loss = self.distributed_pretrain_step(states, batch_target_actions_batch)
                    
                    epoch_loss += loss.numpy() * len(batch_stock_obs_list)
                    num_samples += len(batch_stock_obs_list)
                
                # 定期保存和输出损失
                if num_samples > 0:
                    epoch_loss_avg = epoch_loss / num_samples
                    print(f"迭代 {iter_idx + 1}/{num_iterations}, 损失: {epoch_loss_avg:.6f}")
            
            # 保存预训练模型
            self.save_pretrain_model()
            
            if num_samples > 0:
                epoch_loss_avg = epoch_loss / num_samples
                print(f"日期 {date_idx + 1} 训练完成, 平均损失: {epoch_loss_avg:.6f}")
                all_losses.append(epoch_loss_avg)
                
        print("\n模仿学习预训练完成！")
        return all_losses


class ReplayBuffer:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def store_memory(self, state, action, probs, vals, reward, done):  # 将数据加入buffer
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
 # 从buffer中采样数据,数量为batch_size
 
    def generate_batches(self):
        """
        从 buffer 中随机抽取若干个「连续 batch」
        - batch 内顺序不打乱
        - batch 之间起点随机
        """
        n = len(self.actions)
        batches = []
        if n < self.batch_size:
            batches.append([
                self.states,
                self.actions,
                self.probs,
                self.vals,
                self.rewards,
                self.dones,
            ])
            return batches
        # 允许的起点范围
        max_start = n - self.batch_size
        start = np.random.randint(0, max_start + 1)
        end = start + self.batch_size
        batches.append([
            self.states[start:end],
            self.actions[start:end],
            self.probs[start:end],
            self.vals[start:end],
            self.rewards[start:end],
            self.dones[start:end],
        ])

        return batches



    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []