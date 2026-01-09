


import hashlib
from matplotlib import pyplot as plt
from openai import APIConnectionError, APIError, RateLimitError


#大模型调用包
import os
from openai import OpenAI
from openai import APIError, APIConnectionError, RateLimitError

def show_loss_info(actor_losses, critic_losses, entropies,rewards):
    # 绘制训练损失图
    plt.figure(figsize=(60, 40))

    # 绘制Actor损失
    plt.subplot(4, 1, 1)
    plt.plot(actor_losses, label='Actor Loss', color='blue')
    plt.title('Actor Loss during Training')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 绘制Critic损失
    plt.subplot(4, 1, 2)
    plt.plot(critic_losses, label='Critic Loss', color='green')
    plt.title('Critic Loss during Training')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 绘制熵
    plt.subplot(4, 1, 3)
    plt.plot(entropies, label='Entropy', color='yellow')
    plt.title('Entropy during Training')
    plt.xlabel('Episode')
    plt.ylabel('Entropy')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(rewards, label='rewards', color='red')
    plt.title('rewards')
    plt.xlabel('Episode')
    plt.ylabel('rewards')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('./RL_Data/result.svg',bbox_inches='tight')
    plt.show()
    


def load_token_from_txt(file_path='config.txt'):
    """从txt文件读取token"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            token = f.read().strip()
        if not token:
            raise ValueError("Token文件为空")
        return token
    except FileNotFoundError:
        print(f"错误：找不到配置文件 '{file_path}'")
        raise



def test_gpu():
# 确保激活了虚拟环境，直接执行这段Python代码
    import tensorflow as tf
    import os
    # 强制打印GPU/cuDNN相关日志，方便定位问题
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ['TF_DEBUG_GPU_MEMORY_ALLOCATOR'] = '1'

    # 基础信息检查
    print("===== 环境基础信息 =====")
    print("TensorFlow版本:", tf.__version__)
    print("CUDA编译支持:", tf.test.is_built_with_cuda())
    print("cuDNN编译支持:", tf.test.is_gpu_available())

    # GPU设备检查
    print("\n===== GPU设备检查 =====")
    physical_gpus = tf.config.list_physical_devices('GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(f"物理GPU数量: {len(physical_gpus)}")
    print(f"逻辑GPU数量: {len(logical_gpus)}")

    # 打印每块GPU的详情
    if physical_gpus:
        for i, gpu in enumerate(physical_gpus):
            print(f"GPU {i} 详情: {gpu}")
            # 测试单GPU计算（验证cuDNN加速）
            with tf.device(f'/GPU:{i}'):
                # 构建简单矩阵运算（触发cuDNN加速）
                x = tf.random.normal((2048, 2048), dtype=tf.float32)
                y = tf.random.normal((2048, 2048), dtype=tf.float32)
                z = tf.matmul(x, y)
                print(f"GPU {i} 计算结果形状: {z.shape}，计算设备: {z.device}")
    else:
        print("❌ 未识别到任何GPU！")