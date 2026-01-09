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




