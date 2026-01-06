import numpy as np
import tensorflow as tf

class PPOAgent:
    """
    独立的PPO代理类，不依赖tf-agents库
    实现了完整的PPO算法，包括策略网络、价值网络和训练逻辑
    """
    def __init__(self, env, learning_rate=3e-4, gamma=0.99, lambda_gae=0.95, 
                 clip_epsilon=0.2, ent_coef=0.1, num_epochs=4, batch_size=32):
        self.env = env
        self.obs_dim = env.observation_spec().shape[0]  # 特征长度
        self.token_len = env.token_len  # Token序列长度
        self.action_dim = env.action_size()  # 动作空间大小
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_epsilon = clip_epsilon
        self.ent_coef = ent_coef
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        # 创建经验回放缓冲区
        self.buffer = ReplayBuffer(batch_size)
        
        # 构建策略网络和价值网络
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        # 优化器
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)        
        
        # 初始化
        self._initialize_networks()
    
    def _build_actor(self):
        """构建策略网络"""
         # 输入1：个股数值特征 → shape=(None, 952, 4)  None=批次维度（交易日数），952=个股数，4=数值特征数
        stock_input = tf.keras.Input(shape=(None, self.obs_dim), dtype=tf.float32, name='stock_input')

        # 输入2：全局Token序列 → shape=(None, 15)   # None=批次维度，15=Token序列长度
        token_input = tf.keras.Input(shape=(self.token_len,), dtype=tf.float32, name='token_input')

        # ==================== 分支1：处理个股数值特征 ====================3
        numeric_pooled = tf.keras.layers.GlobalAveragePooling1D()(stock_input)  # (None, 4)
        numeric_dense = tf.keras.layers.Dense(32, activation='relu', name='numeric_dense')(numeric_pooled)  # (None, 32)

        # ==================== 分支2：处理全局Token序列 ====================
        token_int = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.int32))(token_input)  # (None, 15)

        # 步骤2：Embedding层编码Token
        token_embedding = tf.keras.layers.Embedding(input_dim=self.action_dim,output_dim=128,name='token_embedding')(token_int)  # (None, 15, 128)

        # 步骤3：平坦化Token特征（聚合为批次级）
        token_flattened = tf.keras.layers.Flatten()(token_embedding)  # (None, 15*128=1920)

        # ==================== 融合两个分支的特征 ====================
        concat_features = tf.keras.layers.Concatenate(name='concat_features')([token_flattened, numeric_dense])  # (None, 1920+32=1952)
          # ==================== 后续全连接层 + 输出 ====================
        dense1 = tf.keras.layers.Dense(128, activation='relu', name='actor_dense1')(concat_features)
        dropout1 = tf.keras.layers.Dropout(0.1, name='actor_dropout1')(dense1)
         # 输出批次级因子/动作概率 → shape=(None, self.action_dim)
        action_logits = tf.keras.layers.Dense(self.action_dim, name='actor_logits')(dropout1)
        model = tf.keras.Model(inputs=[stock_input, token_input], outputs=action_logits, name='multi_input_actor')
        return model
    
    def _build_critic(self):
        """构建价值网络（适配多输入，修正Embedding误用）"""
        stock_input = tf.keras.Input(shape=(None, self.obs_dim), dtype=tf.float32, name='critic_stock_input')
        token_input = tf.keras.Input(shape=(self.token_len,), dtype=tf.float32, name='critic_token_input')

        numeric_pooled = tf.keras.layers.GlobalAveragePooling1D(name='critic_stock_pooling')(stock_input)
        numeric_dense = tf.keras.layers.Dense(32, activation='relu', name='critic_numeric_dense')(numeric_pooled)


        token_float = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(token_input)
        # Embedding编码 → 3D (batch_size, token_len, 128)
        token_embedding = tf.keras.layers.Embedding(
            input_dim=self.action_dim,
            output_dim=128,
            name='critic_embedding'
        )(token_float)
        # 平坦化 → 2D (batch_size, token_len*128)
        token_flattened = tf.keras.layers.Flatten(name='critic_token_flatten')(token_embedding)

        # ========== 融合特征 ==========
        concat_features = tf.keras.layers.Concatenate(name='critic_concat')([token_flattened, numeric_dense])

        # ========== 全连接层输出状态值 ==========
        dense1 = tf.keras.layers.Dense(128, activation='relu', name='critic_dense1')(concat_features)
        dropout1 = tf.keras.layers.Dropout(0.1, name='critic_dropout1')(dense1)
        # 输出状态值（维度1）→ 2D (batch_size, 1)
        value = tf.keras.layers.Dense(1, name='critic_value')(dropout1)

        # 构建多输入价值网络
        model = tf.keras.Model(inputs=[stock_input, token_input], outputs=value, name='multi_input_critic')
        return model
    
    def _initialize_networks(self):
        """初始化网络权重"""
        dummy_stock_obs = np.zeros((1,1, self.obs_dim), dtype=np.float32)
        dummy_token_obs = np.zeros((1,self.token_len), dtype=np.float32)
        self.actor([dummy_stock_obs, dummy_token_obs])
        self.critic([dummy_stock_obs, dummy_token_obs])
    
    def get_action(self, observation):
        """获取动作和对数概率（适配多输入Actor模型）"""
        # observation是二元组：(stock_obs, token_obs)
        stock_obs_batch, token_obs_batch = observation

        logits = self.actor([stock_obs_batch, token_obs_batch])
        # 计算动作概率、采样动作、计算对数概率
        action_probs = tf.nn.softmax(logits)
        action_dist = tf.random.categorical(logits, num_samples=1)  # 采样1个动作
        action = action_dist.numpy()[0, 0]  # 提取采样的动作值
        log_prob = tf.math.log(action_probs[0, action])  # 计算该动作的对数概率
        return action, log_prob.numpy()
    
    def get_value(self, observation):
        """获取状态值"""
        stock_obs_batch, token_obs_batch = observation
        value = self.critic([stock_obs_batch, token_obs_batch])
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
    
    def learn(self):
        """训练PPO代理"""
        # 从buffer中获取批次数据
        states, actions, old_probs, vals, rewards, dones, batches = self.buffer.generate_batches()  
        # stock_states = []
        # token_states = [] 
        # for idx, single_state in enumerate(states):
        #     stock_state = single_state[0]
        #     token_state = single_state[1]
        #     stock_states.append(np.array(stock_state, dtype=np.float32))
        #     token_states.append(np.array(token_state, dtype=np.float32))

        stock_states = np.array([single_state[0] for single_state in states], dtype=np.float32)
        token_states = np.array([single_state[1] for single_state in states], dtype=np.float32)
        # 计算优势函数和回报
        advantages, returns = self.compute_advantages(rewards, vals, dones)
        
        # 将数据转为Tensor
        stock_states = tf.convert_to_tensor(stock_states, dtype=tf.float32)
        token_states = tf.convert_to_tensor(token_states, dtype=tf.float32)

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

                batch_actions = tf.gather(tf.cast(actions, dtype=tf.int32), batch)
                batch_old_probs = tf.gather(old_probs, batch)
                batch_returns = tf.gather(returns, batch)
                batch_advantages = tf.gather(advantages, batch)

                batch_stock_states = np.squeeze(batch_stock_states, axis=1)
                batch_token_states = np.squeeze(batch_token_states, axis=1)

                
                with tf.GradientTape() as tape:
                    # 策略网络前向传播
                    logits = self.actor([batch_stock_states, batch_token_states])
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
                    values = tf.squeeze(self.critic([batch_stock_states, batch_token_states]))
                    critic_loss = tf.reduce_mean(tf.square(batch_returns - values))
                    
                    # 总损失
                    total_loss = actor_loss + 0.5 * critic_loss
                
                # 反向传播和优化
                gradients = tape.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables + self.critic.trainable_variables))
        
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
            reward = 0
            if index == self.token_len-1:
                reward = DRL_env.reward
            
            self.buffer.store_memory(
                state=observation,
                action=action,
                probs=log_prob,
                vals=value,
                reward=reward,
                done=False
            )                
            observation = next_obs
        return reward
        



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