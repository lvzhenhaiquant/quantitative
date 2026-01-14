import tensorflow as tf
from keras  import layers, Model, backend as K

# ========== 工具层定义（TCN/Transformer组件） ==========
class TCNBlock(layers.Layer):
    """时间卷积块（TCN）：捕捉时序特征的局部依赖+残差连接"""
    def __init__(self, filters, kernel_size=3, dilation_rate=1, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate, activation='relu')
        self.conv2 = layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate, activation='relu')
        self.dropout = layers.Dropout(dropout)
        self.residual = layers.Conv1D(filters, 1) if filters != None else layers.Identity()
        self.activation = layers.ReLU()

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.dropout(x, training=training)
        x = self.conv2(x)
        x = self.dropout(x, training=training)
        # 残差连接
        residual = self.residual(inputs)
        x = x + residual
        return self.activation(x)

class TokenTransformerEncoder(layers.Layer):
    """Token序列Transformer编码器：捕捉Token上下文依赖"""
    def __init__(self, embed_dim=128, num_heads=2, ff_dim=64, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
            layers.Dropout(dropout)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # 残差1
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)  # 残差2

class CrossAttentionFusion(layers.Layer):
    """跨分支注意力融合：融合时序特征和Token特征"""
    def __init__(self, embed_dim=128, num_heads=2, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout)

    def call(self, numeric_features, token_features, training=False):
        # numeric_features: (batch, num_features) → 扩展为3D (batch, 1, num_features)
        numeric_expanded = K.expand_dims(numeric_features, axis=1)
        # 交叉注意力：Token特征作为Query，数值特征作为Key/Value
        attn_output = self.att(token_features, numeric_expanded, training=training)
        attn_output = self.dropout(attn_output, training=training)
        # 融合后平坦化
        fused = layers.Flatten()(self.layernorm(token_features + attn_output))
        return fused

# 第一步：定义自定义类型转换层（继承 Keras Layer，极简实现）
class TokenCastLayer(tf.keras.layers.Layer):
    def __init__(self, target_dtype=tf.int32, **kwargs):
        super().__init__(**kwargs)
        self.target_dtype = target_dtype  # 目标转换类型

    def call(self, inputs):
        """核心：仅执行类型转换，无额外开销"""
        return tf.cast(inputs, self.target_dtype)

    def compute_output_shape(self, input_shape):
        """可选：告诉 Keras 输出形状与输入一致，提升形状推断效率"""
        return input_shape

    def get_config(self):
        """可选：序列化层参数，支持模型保存/加载"""
        config = super().get_config()
        config.update({'target_dtype': self.target_dtype})
        return config

def build_tcn_subnetwork(obs_dim):
    """构建单只股票的TCN处理子网络（函数式API）"""
    # 子网络输入：单只股票的时序数据 → (None, 天数, obs_dim)
    sub_input = layers.Input(shape=(None, obs_dim))
    masked_input = layers.Masking(mask_value=0.0)(sub_input)
    # 步骤1：执行3个TCNBlock（串行）
    tcn1 = TCNBlock(filters=32, kernel_size=3, dilation_rate=1, dropout=0.1)(masked_input)
    tcn2 = TCNBlock(filters=32, kernel_size=3, dilation_rate=2, dropout=0.1)(tcn1)
    tcn3 = TCNBlock(filters=32, kernel_size=3, dilation_rate=4, dropout=0.1)(tcn2)
    # 步骤2：并行执行平均池化和最大池化（分支流程，Sequential不支持）
    avg_pool = layers.GlobalAveragePooling1D()(tcn3)  # (None, 32)
    max_pool = layers.GlobalMaxPooling1D()(tcn3)     # (None, 32)
    # 步骤3：拼接两个池化结果
    concat_features = layers.Concatenate()([avg_pool, max_pool])  # (None, 64)
    # 构建子网络并返回
    sub_model = Model(inputs=sub_input, outputs=concat_features, name='tcn_subnetwork')
    return sub_model


# def tcn_transformer_build_actor(obs_dim, token_len,ast_feature_dim, action_dim):        
#     # 输入1：个股数值特征 → shape=(None, 952, 4) (batch, 个股数, 特征数)
#     stock_input = layers.Input(shape=(None,None, obs_dim), dtype=tf.float32, name='stock_input')
#     # 输入2：全局Token序列 → shape=(None, 15) (batch, Token长度)
#     token_input = layers.Input(shape=(token_len,), dtype=tf.float32, name='token_input')
#     # 输入3：AST结构特征 → shape=(None, 9) (batch, AST特征维度)
#     ast_input = layers.Input(shape=(ast_feature_dim,), dtype=tf.float32, name='ast_input')
#     # ==================== 分支1：个股数值特征（TCN+池化） ====================
#     # 转换为：(batch, num_stocks, days, features)
#     stock_reshaped = tf.keras.layers.Permute((2, 1, 3))(stock_input)  # (batch, num_stocks, days, features)
#     # 核心修改：用函数式API构建的子网络替换Sequential（解决并行池化问题）
#     tcn_submodel = build_tcn_subnetwork(obs_dim=obs_dim)
#     stock_features = tf.keras.layers.TimeDistributed(tcn_submodel)(stock_reshaped)
#     # 在TCN处理后添加注意力层
#     attention = layers.MultiHeadAttention(num_heads=2, key_dim=32)(stock_features, stock_features)
#     attended_features = layers.Add()([stock_features, attention])
#     attended_features = layers.LayerNormalization(epsilon=1e-6)(attended_features)
#     # 全局池化获取批次级特征
#     numeric_pooled  = tf.keras.layers.GlobalAveragePooling1D()(attended_features)
#     numeric_dense = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(numeric_pooled)
#     numeric_dense = layers.Dropout(0.1)(numeric_dense)  # (batch, 64)

#     # ==================== 分支2：Token序列（Embedding+Transformer） ====================
#     token_int = TokenCastLayer(target_dtype=tf.int32, name='token_cast')(token_input)
#     # Embedding层（加入正则化防止过拟合）
#     token_embedding = layers.Embedding(input_dim=action_dim,output_dim=128,
#         embeddings_regularizer=tf.keras.regularizers.l2(1e-4),name='token_embedding')(token_int)  # (batch, 15, 128)
#     # Transformer编码器捕捉Token上下文
#     transformer_encoder = TokenTransformerEncoder(embed_dim=128, num_heads=4, ff_dim=256, dropout=0.1)(token_embedding)  # (batch, 15, 128)

#     # ==================== 跨分支注意力融合 ====================
#     fused_features = CrossAttentionFusion(embed_dim=128, num_heads=2, dropout=0.1)(numeric_dense, transformer_encoder)  # (batch, 15*128=1920)
#     # ==================== 输出层（加入正则化） ====================
#     dense1 = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(fused_features)
#     dropout1 = layers.Dropout(0.1)(dense1)
#     dense2 = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(dropout1)
#     dropout2 = layers.Dropout(0.1)(dense2)
#     # 输出动作概率（logits）
#     action_logits = layers.Dense(action_dim, name='actor_logits')(dropout2)
#     model = Model(inputs=[stock_input, token_input], outputs=action_logits, name='advanced_actor')
#     return model

# # ========== 升级后的Critic网络构建函数 ==========
# def tcn_transformer_build_critic(obs_dim, token_len, action_dim):
#     # 输入定义（和Actor保持一致）
#     stock_input = layers.Input(shape=(None, None, obs_dim), dtype=tf.float32, name='critic_stock_input')
#     token_input = layers.Input(shape=(token_len,), dtype=tf.float32, name='critic_token_input')

#     # ==================== 分支1：个股数值特征（TCN+池化） ====================
#     # 与Actor网络保持一致的TCN处理
#     stock_reshaped = tf.keras.layers.Permute((2, 1, 3))(stock_input)  # (batch, num_stocks, days, features)
#     # 使用从DeepNeuralNetworks导入的TCN子网络
#     tcn_submodel = build_tcn_subnetwork(obs_dim=obs_dim)
#     stock_features = tf.keras.layers.TimeDistributed(tcn_submodel)(stock_reshaped)  # (batch, num_stocks, 64)
#     # 在TCN处理后添加注意力层
#     attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)(stock_features, stock_features)
#     attended_features = layers.Add()([stock_features, attention])
#     attended_features = layers.LayerNormalization(epsilon=1e-6)(attended_features)
#     # 全局池化获取批次级特征
#     numeric_pooled = tf.keras.layers.GlobalAveragePooling1D()(attended_features)  # (batch, 64)
#     numeric_dense = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(numeric_pooled)
#     numeric_dense = layers.Dropout(0.1)(numeric_dense)  # (batch, 64)

#     # ==================== 分支2：Token序列（Embedding+Transformer） ====================
#     token_int = TokenCastLayer(target_dtype=tf.int32, name='critic_token_cast')(token_input)
#     token_embedding = layers.Embedding(input_dim=action_dim,output_dim=128,
#         embeddings_regularizer=tf.keras.regularizers.l2(1e-4),name='critic_embedding')(token_int)  # (batch, 15, 128)
#     # Transformer编码器
#     transformer_encoder = TokenTransformerEncoder(embed_dim=128, num_heads=4, ff_dim=256, dropout=0.1)(token_embedding)

#     # ==================== 跨分支注意力融合 ====================
#     fused_features = CrossAttentionFusion(embed_dim=128, num_heads=2, dropout=0.1)(numeric_dense, transformer_encoder)  # (batch, 1920)

#     # ==================== 价值输出层 ====================
#     dense1 = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(fused_features)
#     dropout1 = layers.Dropout(0.1)(dense1)
#     dense2 = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(dropout1)
#     dropout2 = layers.Dropout(0.1)(dense2)
#     # 输出状态价值（单值）
#     value = layers.Dense(1, name='critic_value')(dropout2)

#     model = Model(inputs=[stock_input, token_input], outputs=value, name='advanced_critic')
#     return model




def tcn_transformer_build_actor(obs_dim, token_len, ast_feature_dim,action_dim):        
    # 输入1：个股数值特征 → shape=(None, 952, 4) (batch, 个股数, 特征数)
    stock_input = layers.Input(shape=(None,None, obs_dim), dtype=tf.float32, name='stock_input')
    # 输入2：全局Token序列 → shape=(None, 15) (batch, Token长度)
    token_input = layers.Input(shape=(token_len,), dtype=tf.float32, name='token_input')
    # 输入3：AST结构特征 → shape=(None, 9) (batch, AST特征维度)
    ast_input = layers.Input(shape=(ast_feature_dim,), dtype=tf.float32, name='ast_input')
    
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

    # ==================== 分支3：AST结构特征（MLP处理） ====================
    ast_dense1 = layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(ast_input)
    ast_dense2 = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(ast_dense1)
    ast_dense2 = layers.Dropout(0.1)(ast_dense2)  # (batch, 64)

    # ==================== 跨分支注意力融合 ====================
    # 首先融合数值特征和AST特征
    combined_numeric_ast = layers.Concatenate()([numeric_dense, ast_dense2])  # (batch, 128)
    fused_features = CrossAttentionFusion(embed_dim=128, num_heads=2, dropout=0.1)(combined_numeric_ast, transformer_encoder)  # (batch, 15*128=1920)
    
    # ==================== 输出层（加入正则化） ====================
    dense1 = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(fused_features)
    dropout1 = layers.Dropout(0.1)(dense1)
    dense2 = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(dropout1)
    dropout2 = layers.Dropout(0.1)(dense2)
    # 输出动作概率（logits）
    action_logits = layers.Dense(action_dim, name='actor_logits')(dropout2)
    model = Model(inputs=[stock_input, token_input, ast_input], outputs=action_logits, name='advanced_actor')
    return model

# ========== 升级后的Critic网络构建函数 ==========
def tcn_transformer_build_critic(obs_dim, token_len,ast_feature_dim, action_dim):
    # 输入1：个股数值特征
    stock_input = layers.Input(shape=(None, None, obs_dim), dtype=tf.float32, name='critic_stock_input')
    # 输入2：全局Token序列
    token_input = layers.Input(shape=(token_len,), dtype=tf.float32, name='critic_token_input')
    # 输入3：AST结构特征
    ast_input = layers.Input(shape=(ast_feature_dim,), dtype=tf.float32, name='critic_ast_input')

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

    # ==================== 分支3：AST结构特征（MLP处理） ====================
    ast_dense1 = layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(ast_input)
    ast_dense2 = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(ast_dense1)
    ast_dense2 = layers.Dropout(0.1)(ast_dense2)  # (batch, 64)

    # ==================== 跨分支注意力融合 ====================
    # 首先融合数值特征和AST特征
    combined_numeric_ast = layers.Concatenate()([numeric_dense, ast_dense2])  # (batch, 128)
    fused_features = CrossAttentionFusion(embed_dim=128, num_heads=2, dropout=0.1)(combined_numeric_ast, transformer_encoder)  # (batch, 1920)

    # ==================== 价值输出层 ====================
    dense1 = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(fused_features)
    dropout1 = layers.Dropout(0.1)(dense1)
    dense2 = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(dropout1)
    dropout2 = layers.Dropout(0.1)(dense2)
    # 输出状态价值（单值）
    value = layers.Dense(1, name='critic_value')(dropout2)

    model = Model(inputs=[stock_input, token_input, ast_input], outputs=value, name='advanced_critic')
    return model