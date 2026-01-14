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
                 clip_epsilon=0.2, ent_coef=0.1, num_epochs=1, batch_size=32):
        self.env = env
        self.obs_dim = env.observation_spec().shape[1]  # 特征长度
        self.token_len = env.token_len  # Token序列长度
        self.date_history_window_size = self.env.date_history_window_size
        self.action_dim = env.action_size()  # 动作空间大小
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

         # 配置多GPU策略
        self.strategy = tf.distribute.MirroredStrategy()
        self.num_replicas = self.strategy.num_replicas_in_sync
        self.per_replica_batch_size = batch_size // self.num_replicas

        # 在策略作用域内构建网络和优化器
        with self.strategy.scope():
            # 构建策略网络和价值网络
            self.actor = tcn_transformer_build_actor(self.obs_dim, self.token_len,self.env.ast_feature_dim, self.action_dim)
            self.critic = tcn_transformer_build_critic(self.obs_dim, self.token_len,self.env.ast_feature_dim, self.action_dim)
            self.optimizer = tf.keras.optimizers.Adam(learning_rate)    
            self.pretrain_optimizer = tf.keras.optimizers.Adam(learning_rate * 0.1)  # 预训练使用较小的学习率    
        # 初始化
        self._initialize_networks()

        self._define_distributed_train_step()
        self._define_distributed_pretrain_step()

    def _initialize_networks(self):
        """初始化网络权重"""
        dummy_stock_obs = np.zeros((1,self.date_history_window_size,1, self.obs_dim), dtype=np.float32)
        dummy_token_obs = np.zeros((1,self.token_len), dtype=np.float32)
        dummy_ast_obs = np.zeros((1,self.env.ast_feature_dim), dtype=np.float32)
         # 通过调用一次前向传播来初始化权重
        self.actor([dummy_stock_obs, dummy_token_obs,dummy_ast_obs])
        self.critic([dummy_stock_obs, dummy_token_obs,dummy_ast_obs])
    
    def get_action(self, observation):
        """获取动作和对数概率（适配多输入Actor模型）"""
        # observation是二元组：(stock_obs, token_obs)
        # stock_obs_batch, token_obs_batch,ast_obs_batch = observation
        # logits = self.actor([stock_obs_batch, token_obs_batch,ast_obs_batch])
        logits = self.actor(observation)
        # 计算动作概率、采样动作、计算对数概率
        action_probs = tf.nn.softmax(logits)
        action_dist = tf.random.categorical(logits, num_samples=1)  # 采样1个动作
        action = action_dist.numpy()[0, 0]  # 提取采样的动作值
        log_prob = tf.math.log(action_probs[0, action])  # 计算该动作的对数概率
        return action, log_prob.numpy()
    
    def get_value(self, observation):
        """获取状态值"""
        # stock_obs_batch, token_obs_batch, ast_obs_batch,= observation
        # value = self.critic([stock_obs_batch, token_obs_batch,ast_obs_batch])
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
    def _train_step(self, batch_stock_states, batch_token_states,batch_ast_states, batch_actions, batch_old_probs, batch_returns, batch_advantages):
        """优化的训练步骤，减少tf.function重追踪"""
        with tf.GradientTape() as tape:
            # 策略网络前向传播
            logits = self.actor([batch_stock_states, batch_token_states,batch_ast_states])
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
            # 价值网络损失
            values = tf.squeeze(self.critic([batch_stock_states, batch_token_states,batch_ast_states]), axis=1)
            critic_loss = tf.reduce_mean(tf.square(batch_returns - values))
            # 总损失
            total_loss = actor_loss + 0.5 * critic_loss
        # 反向传播和优化
        gradients = tape.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables + self.critic.trainable_variables))
        return actor_loss, critic_loss, entropy
    
    @tf.function(reduce_retracing=True)
    def _pretrain_step(self, batch_stock_states, batch_token_states,batch_ast_states, batch_target_actions):
        """模仿学习预训练步骤"""
        with tf.GradientTape() as tape:
            # 策略网络前向传播
            logits = self.actor([batch_stock_states, batch_token_states, batch_ast_states])
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
        def distributed_train_step(stock_states, token_states,ast_states, actions, old_probs, returns, advantages):
            per_replica_losses = self.strategy.run(
                self._train_step,
                args=(stock_states, token_states,ast_states, actions, old_probs, returns, advantages)
            )
            # 聚合所有replica的损失
            mean_actor_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[0], axis=None) / self.num_replicas
            mean_critic_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[1], axis=None) / self.num_replicas
            mean_entropy = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[2], axis=None) / self.num_replicas
            return mean_actor_loss, mean_critic_loss, mean_entropy
        self.distributed_train_step = distributed_train_step

    def _define_distributed_pretrain_step(self):
        """定义分布式预训练步骤"""
        @tf.function(reduce_retracing=True)
        def distributed_pretrain_step(stock_states, token_states, ast_states, target_actions):
            per_replica_losses = self.strategy.run(
                self._pretrain_step,
                args=(stock_states, token_states,ast_states, target_actions)
            )
            # 聚合所有replica的损失
            mean_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None) / self.num_replicas
            return mean_loss
        self.distributed_pretrain_step = distributed_pretrain_step

    def learn(self):
        """训练PPO代理"""
        # 从buffer中获取批次数据
        states, actions, old_probs, vals, rewards, dones, batches = self.buffer.generate_batches()  

        stock_states = np.array([single_state[0] for single_state in states], dtype=np.float32)
        token_states = np.array([single_state[1] for single_state in states], dtype=np.float32)
        ast_states = np.array([single_state[2] for single_state in states], dtype=np.float32)
        # 计算优势函数和回报
        advantages, returns = self.compute_advantages(rewards, vals, dones)
        
        # 将数据转为Tensor
        stock_states = tf.convert_to_tensor(stock_states, dtype=tf.float32)
        token_states = tf.convert_to_tensor(token_states, dtype=tf.float32)
        ast_states = tf.convert_to_tensor(ast_states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        
        # 标准化优势函数
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
        
        # 多轮训练
        for epoch in range(self.num_epochs):
            # 批次训练
            for batch in batches:
                batch_stock_states = tf.gather(stock_states, batch)
                batch_token_states = tf.gather(token_states, batch)
                batch_ast_states = tf.gather(ast_states, batch)
                batch_actions = tf.gather(tf.cast(actions, dtype=tf.int32), batch)
                batch_old_probs = tf.gather(old_probs, batch)
                batch_returns = tf.gather(returns, batch)
                batch_advantages = tf.gather(advantages, batch)
                # 确保维度正确（移除可能的额外维度）
                batch_stock_states = np.squeeze(batch_stock_states, axis=1)
                batch_token_states = np.squeeze(batch_token_states, axis=1)
                batch_ast_states = np.squeeze(batch_ast_states, axis=1)

                # 使用优化的训练步骤
                actor_loss, critic_loss, entropy = self.distributed_train_step(
                    batch_stock_states, batch_token_states, batch_ast_states, batch_actions, 
                    batch_old_probs, batch_returns, batch_advantages)

        return {
            'actor_loss': actor_loss.numpy(),
            'critic_loss': critic_loss.numpy(),
            'entropy': entropy.numpy()
        }

    def collect_experience_for_day(self,observation):
        for index in range(self.token_len):
            # 获取单步动作/价值
            action, log_prob = self.get_action(observation)
            value = self.get_value(observation)
            DRL_env = self.env.step(action)
            next_obs = DRL_env.observation
            reward = DRL_env.reward
            print(f"{index},    action:{action},    reward:{reward}")
            self.buffer.store_memory(state=observation,action=action,probs=log_prob,vals=value,reward=reward,done=False)
            observation = next_obs
        return reward
        

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

    def imitation_pretrain_old(self,batch_size=64,use_large_model=True, num_factors=50):
        """模仿学习预训练函数，使用内部自定义的专家因子进行预训练"""
        print("开始模仿学习预训练...")
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
                ["BEG", "close", "5", "EMA", "10", "EMA", "Sub", "SEP"],  # EMA(close, 5) - EMA(close, 10)
                ["BEG", "close", "open", "Greater", "1", "0", "If", "SEP"]  # If(close > open, 1, 0)
            ]
        # 将专家因子转换为token索引序列
        expert_token_sequences = []
        for factor in generated_factors:
            # 确保因子序列长度不超过token_len
            if len(factor) > self.token_len:
                continue
            # 检查是否以SEP结尾
            if not factor[-1] == self.env.token_lib.SEP:
                continue
            # 检查是否符合逆波兰表达式规则
            if not self.env.RPNEncoder._is_valid_expression(factor):
                continue

            # 转换为token索引
            token_sequence = []
            for token in factor:
                if token in self.env.token_lib.all_tokens:
                    token_index = self.env.token_lib.all_tokens.index(token)
                    token_sequence.append(token_index)
                else:
                    # 如果token不在库中，跳过该因子
                    break
            # 填充到固定长度
            if len(token_sequence) == len(factor):  # 确保所有token都有效
                while len(token_sequence) < self.token_len:
                     pad_index = self.env.token_lib.all_tokens.index(self.env.token_lib.PAD)  # 使用PAD填充
                     token_sequence.append(pad_index)
                expert_token_sequences.append(token_sequence)
        if len(expert_token_sequences) == 0:
            print("警告：没有有效的专家因子序列，无法进行模仿学习预训练")
            return
        print(f"共转换 {len(expert_token_sequences)} 个有效专家因子序列")

        # 准备训练数据
        expert_token_sequences = np.array(expert_token_sequences, dtype=np.int32)
        expert_token_sequences = tf.convert_to_tensor(expert_token_sequences, dtype=tf.int32)
        num_examples = len(expert_token_sequences)

        # 收集所有日期的损失数据
        all_losses = []
        # 遍历所有日期进行训练
        total_dates = len(self.env.date_list)//3
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
            # 为当前日期创建批次观测数据
            stock_obs_batch = np.repeat(stock_obs[np.newaxis, ...], num_examples, axis=0)
            stock_obs_batch = tf.convert_to_tensor(stock_obs_batch, dtype=tf.float32)
            # 计算批次数量
            num_batches = num_examples // batch_size
            if num_examples % batch_size != 0:
                num_batches += 1
            
            # 开始在当前日期上预训练
            for epoch in range(self.num_epochs):
                epoch_loss = 0.0
                # 随机打乱数据
                indices = np.arange(num_examples)
                np.random.shuffle(indices)
                shuffled_stock_obs = tf.gather(stock_obs_batch, indices)
                shuffled_tokens = tf.gather(expert_token_sequences, indices)
                
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, num_examples)
                    batch_stock_obs = shuffled_stock_obs[start_idx:end_idx]
                    batch_tokens = shuffled_tokens[start_idx:end_idx]
                    batch_target_actions = tf.convert_to_tensor(batch_tokens, dtype=tf.int32)
                    
                    # 训练Actor网络预测下一个token
                    expanded_stock_obs = tf.repeat(batch_stock_obs, self.token_len, axis=0)
                    # 为每个位置创建输入token序列
                    expanded_token_inputs = []
                    for token_seq in batch_tokens.numpy():
                        for t in range(self.token_len):
                            # 输入：前t个Token（到t-1为止），后面用PAD填充到固定长度
                            input_seq = np.copy(token_seq)
                            input_seq[t:] = self.env.token_lib.all_tokens.index(self.env.token_lib.PAD)  # 掩码当前位置之后的token
                            expanded_token_inputs.append(input_seq)
                    
                    expanded_token_inputs = tf.convert_to_tensor(expanded_token_inputs, dtype=tf.float32)
                    expanded_target_actions = tf.reshape(batch_target_actions, [-1])
                    # 执行预训练步骤
                    loss = self.distributed_pretrain_step(expanded_stock_obs, expanded_token_inputs, expanded_target_actions)
                    epoch_loss += loss.numpy()
                epoch_loss /= num_batches
                print(f"  Epoch {epoch+1}/{self.num_epochs}, 损失: {epoch_loss:.6f}")
                # 记录损失
                all_losses.append(epoch_loss)
        
        # 保存预训练模型
        self.save_pretrain_model()
        show_expert_loss(all_losses, window_size=5)
        print("\n模仿学习预训练完成！")
        
    def imitation_pretrain(self, use_large_model=False, num_factors=50):
        actor_pretrain_path = f"{self.pretrain_path_prefix}_actor.h5"
        if os.path.exists(actor_pretrain_path):
            self.actor.load_weights(actor_pretrain_path)
            print(f"预训练模型加载完成（仅Actor）")
        """模仿学习预训练函数，使用内部自定义的专家因子进行预训练"""
        print("开始模仿学习预训练...")
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
                ["BEG", "close", "5", "EMA","close", "10","EMA", "Sub", "SEP"],  # EMA(close, 5) - EMA(close, 10)
            ]
        
        # 将专家因子转换为token索引序列
        expert_token_sequences = []
        for factor in generated_factors:
            # 确保因子序列长度不超过token_len
            if len(factor) > self.token_len:
                continue
            # 检查是否以SEP结尾
            if not factor[-1] == self.env.token_lib.SEP:
                continue
            # 检查是否符合逆波兰表达式规则
            if not self.env.RPNEncoder._is_valid_expression(factor):
                continue

            # 转换为token索引
            token_sequence = []
            for token in factor:
                if token in self.env.token_lib.all_tokens:
                    token_index = self.env.token_lib.all_tokens.index(token)
                    token_sequence.append(token_index)
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
        # 遍历所有日期进行训练
        total_dates = len(self.env.date_list)
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
            for _ in range(100):
                for expert_seq in expert_token_sequences:
                    # 逐token生成训练数据
                    for step in range(1, len(expert_seq)):
                        # 输入序列：前step个token
                        input_seq = expert_seq[:step]
                        # 目标token：下一个token
                        target_token = expert_seq[step]
                        
                        # 准备输入数据
                        padded_input = input_seq.copy()
                        pad_index = self.env.token_lib.all_tokens.index(self.env.token_lib.PAD)
                        while len(padded_input) < self.token_len:
                            padded_input.append(pad_index)
                        
                        # 创建批次数据
                        batch_stock_obs = np.expand_dims(stock_obs, axis=0)
                        batch_stock_obs = np.repeat(batch_stock_obs, 1, axis=0)
                        
                        batch_token_input = np.expand_dims(padded_input, axis=0)
                        batch_token_input = batch_token_input.astype(np.int32)
                        
                        batch_target_actions = np.array([target_token], dtype=np.int32)
                        
                        # 执行预训练步骤
                        loss = self.distributed_pretrain_step(batch_stock_obs, batch_token_input, batch_target_actions)
                        epoch_loss += loss.numpy()
                        num_samples += 1

                if num_samples %100 == 0:
                    # 保存预训练模型
                    self.save_pretrain_model()

                    epoch_loss /= num_samples
                    print(f"损失: {epoch_loss:.6f}")
                    all_losses.append(epoch_loss)                    
                    show_expert_loss(all_losses, window_size=5)
        
        print("\n模仿学习预训练完成！")




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

    def generate_batches(self):  # 从buffer中采样数据,数量为batch_size
        n_states = len(self.actions)
        batch_start = np.arange(0, n_states, self.batch_size)  # 安装batch_size大小分块，0-，1*batch_size-，2*batch_size-
        indices = np.arange(n_states, dtype=np.int64)
        #np.random.shuffle(indices)  # 0 - n_states打乱顺序
        #batches = [indices[i:i + self.batch_size] for i in batch_start if i + self.batch_size <= n_states]
        batches = [indices[i:i + self.batch_size] for i in batch_start]
        return self.states, \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.vals), \
               np.array(self.rewards), \
               np.array(self.dones), \
               batches

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []