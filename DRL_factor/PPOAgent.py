import os
import numpy as np
import tensorflow as tf
from keras  import layers, Model, backend as K

from DeepNeuralNetworks import CrossAttentionFusion, TokenCastLayer, TokenTransformerEncoder, build_tcn_subnetwork

class PPOAgent:
    """
    实现了完整的PPO算法，包括策略网络、价值网络和训练逻辑
    """
    def __init__(self, env, learning_rate=3e-4, gamma=0.99, lambda_gae=0.95, 
                 clip_epsilon=0.2, ent_coef=0.1, num_epochs=4, batch_size=32):
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
        self.path_prefix="./RL_Data/ppo_model"
        
        # 创建经验回放缓冲区
        self.buffer = ReplayBuffer(batch_size)

         # 配置多GPU策略
        self.strategy = tf.distribute.MirroredStrategy()
        self.num_replicas = self.strategy.num_replicas_in_sync
        self.per_replica_batch_size = batch_size // self.num_replicas

        # 在策略作用域内构建网络和优化器
        with self.strategy.scope():
            # 构建策略网络和价值网络
            self.actor = self._build_actor(self.obs_dim, self.token_len, self.action_dim)
            self.critic = self._build_critic(self.obs_dim, self.token_len, self.action_dim)
            self.optimizer = tf.keras.optimizers.Adam(learning_rate)        
        # 初始化
        self._initialize_networks()

        self._define_distributed_train_step()


    def _build_actor(self,obs_dim, token_len, action_dim):        
        """构建高级策略网络：TCN（时序）+ Transformer（Token）+ 交叉注意力融合"""
        # 输入1：个股数值特征 → shape=(None, 952, 4) (batch, 个股数, 特征数)
        stock_input = layers.Input(shape=(None,None, obs_dim), dtype=tf.float32, name='stock_input')
        # 输入2：全局Token序列 → shape=(None, 15) (batch, Token长度)
        token_input = layers.Input(shape=(token_len,), dtype=tf.float32, name='token_input')
        # ==================== 分支1：个股数值特征（TCN+池化） ====================
        # 转换为：(batch, num_stocks, days, features)
        stock_reshaped = tf.keras.layers.Permute((2, 1, 3))(stock_input)  # (batch, num_stocks, days, features)
        # 核心修改：用函数式API构建的子网络替换Sequential（解决并行池化问题）
        tcn_submodel = build_tcn_subnetwork(obs_dim=obs_dim)
        stock_features = tf.keras.layers.TimeDistributed(tcn_submodel)(stock_reshaped)
        # 在TCN处理后添加注意力层
        attention = layers.MultiHeadAttention(num_heads=2, key_dim=32)(stock_features, stock_features)
        attended_features = layers.Add()([stock_features, attention])
        attended_features = layers.LayerNormalization(epsilon=1e-6)(attended_features)
        # 全局池化获取批次级特征
        numeric_pooled  = tf.keras.layers.GlobalAveragePooling1D()(attended_features)
        numeric_dense = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(numeric_pooled)
        numeric_dense = layers.Dropout(0.1)(numeric_dense)  # (batch, 64)

        # ==================== 分支2：Token序列（Embedding+Transformer） ====================
        token_int = TokenCastLayer(target_dtype=tf.int32, name='token_cast')(token_input)
        # Embedding层（加入正则化防止过拟合）
        token_embedding = layers.Embedding(input_dim=action_dim,output_dim=128,
            embeddings_regularizer=tf.keras.regularizers.l2(1e-4),name='token_embedding')(token_int)  # (batch, 15, 128)
        # Transformer编码器捕捉Token上下文
        transformer_encoder = TokenTransformerEncoder(embed_dim=128, num_heads=4, ff_dim=256, dropout=0.1)(token_embedding)  # (batch, 15, 128)

        # ==================== 跨分支注意力融合 ====================
        fused_features = CrossAttentionFusion(embed_dim=128, num_heads=2, dropout=0.1)(numeric_dense, transformer_encoder)  # (batch, 15*128=1920)
        # ==================== 输出层（加入正则化） ====================
        dense1 = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(fused_features)
        dropout1 = layers.Dropout(0.1)(dense1)
        dense2 = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(dropout1)
        dropout2 = layers.Dropout(0.1)(dense2)
        # 输出动作概率（logits）
        action_logits = layers.Dense(action_dim, name='actor_logits')(dropout2)
        model = Model(inputs=[stock_input, token_input], outputs=action_logits, name='advanced_actor')
        return model

    # ========== 升级后的Critic网络构建函数 ==========
    def _build_critic(self, obs_dim, token_len, action_dim):
        """构建高级价值网络：TCN（时序）+ Transformer（Token）+ 注意力融合"""
        # 输入定义（和Actor保持一致）
        stock_input = layers.Input(shape=(None, None, obs_dim), dtype=tf.float32, name='critic_stock_input')
        token_input = layers.Input(shape=(token_len,), dtype=tf.float32, name='critic_token_input')

        # ==================== 分支1：个股数值特征（TCN+池化） ====================
        # 与Actor网络保持一致的TCN处理
        stock_reshaped = tf.keras.layers.Permute((2, 1, 3))(stock_input)  # (batch, num_stocks, days, features)
        # 使用从DeepNeuralNetworks导入的TCN子网络
        tcn_submodel = build_tcn_subnetwork(obs_dim=obs_dim)
        stock_features = tf.keras.layers.TimeDistributed(tcn_submodel)(stock_reshaped)  # (batch, num_stocks, 64)
        # 在TCN处理后添加注意力层
        attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)(stock_features, stock_features)
        attended_features = layers.Add()([stock_features, attention])
        attended_features = layers.LayerNormalization(epsilon=1e-6)(attended_features)
        # 全局池化获取批次级特征
        numeric_pooled = tf.keras.layers.GlobalAveragePooling1D()(attended_features)  # (batch, 64)
        numeric_dense = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(numeric_pooled)
        numeric_dense = layers.Dropout(0.1)(numeric_dense)  # (batch, 64)

        # ==================== 分支2：Token序列（Embedding+Transformer） ====================
        token_int = TokenCastLayer(target_dtype=tf.int32, name='critic_token_cast')(token_input)
        token_embedding = layers.Embedding(input_dim=action_dim,output_dim=128,
            embeddings_regularizer=tf.keras.regularizers.l2(1e-4),name='critic_embedding')(token_int)  # (batch, 15, 128)
        # Transformer编码器
        transformer_encoder = TokenTransformerEncoder(embed_dim=128, num_heads=4, ff_dim=256, dropout=0.1)(token_embedding)

        # ==================== 跨分支注意力融合 ====================
        fused_features = CrossAttentionFusion(embed_dim=128, num_heads=2, dropout=0.1)(numeric_dense, transformer_encoder)  # (batch, 1920)

        # ==================== 价值输出层 ====================
        dense1 = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(fused_features)
        dropout1 = layers.Dropout(0.1)(dense1)
        dense2 = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(dropout1)
        dropout2 = layers.Dropout(0.1)(dense2)
        # 输出状态价值（单值）
        value = layers.Dense(1, name='critic_value')(dropout2)

        model = Model(inputs=[stock_input, token_input], outputs=value, name='advanced_critic')
        return model
 
    def _initialize_networks(self):
        """初始化网络权重"""
        dummy_stock_obs = np.zeros((1,self.date_history_window_size,1, self.obs_dim), dtype=np.float32)
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
    
    @tf.function(reduce_retracing=True)
    def _train_step(self, batch_stock_states, batch_token_states, batch_actions, batch_old_probs, batch_returns, batch_advantages):
        """优化的训练步骤，减少tf.function重追踪"""
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
        return actor_loss, critic_loss, entropy
    
        
    def _define_distributed_train_step(self):
        """定义分布式训练步骤"""
        @tf.function(reduce_retracing=True)
        def distributed_train_step(stock_states, token_states, actions, old_probs, returns, advantages):
            per_replica_losses = self.strategy.run(
                self._train_step,
                args=(stock_states, token_states, actions, old_probs, returns, advantages)
            )
            # 聚合所有replica的损失
            mean_actor_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[0], axis=None) / self.num_replicas
            mean_critic_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[1], axis=None) / self.num_replicas
            mean_entropy = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[2], axis=None) / self.num_replicas
            return mean_actor_loss, mean_critic_loss, mean_entropy
        
        self.distributed_train_step = distributed_train_step

    def learn(self):
        """训练PPO代理"""
        # 从buffer中获取批次数据
        states, actions, old_probs, vals, rewards, dones, batches = self.buffer.generate_batches()  

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
                # 确保维度正确（移除可能的额外维度）
                batch_stock_states = np.squeeze(batch_stock_states, axis=1)
                batch_token_states = np.squeeze(batch_token_states, axis=1)
                # 使用优化的训练步骤
                
                actor_loss, critic_loss, entropy = self.distributed_train_step(
                    batch_stock_states, batch_token_states, batch_actions, 
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
        

    def save_model(self):
        path_prefix = self.path_prefix
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        # 保存actor网络权重
        actor_path = f"{path_prefix}_actor"
        self.actor.save_weights(actor_path)
        # 保存critic网络权重
        critic_path = f"{path_prefix}_critic"
        self.critic.save_weights(critic_path)
        print(f"模型已保存到: {critic_path}")
        
    
    def load_model(self):
        # 加载actor网络权重
        path_prefix = self.path_prefix
        actor_path = f"{path_prefix}_actor"
        self.actor.load_weights(actor_path)
        # 加载critic网络权重
        critic_path = f"{path_prefix}_critic"
        self.critic.load_weights(critic_path)
        print(f"模型已从: {critic_path} 加载")




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