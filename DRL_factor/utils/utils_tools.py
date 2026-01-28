
from matplotlib import pyplot as plt
import numpy as np



def show_loss_info(actor_losses, critic_losses, entropies, rewards, window_size=30):
    # 绘制训练损失图
    plt.figure(figsize=(15, 30))
    # 计算移动窗口均值
    actor_ma = moving_average(actor_losses, window_size)
    critic_ma = moving_average(critic_losses, window_size)
    entropies_ma = moving_average(entropies, window_size)
    rewards_ma = moving_average(rewards, window_size)

    # 绘制Actor损失
    plt.subplot(4, 1, 1)
    plt.plot(actor_losses, label='Actor Loss Raw', color='lightblue', alpha=0.5)
    # 确保x轴范围与移动平均数据的长度一致
    x_range = range(len(actor_ma)) if len(actor_losses) < window_size else range(window_size-1, len(actor_losses))
    plt.plot(x_range, actor_ma, label=f'Actor Loss MA ({window_size})', color='blue')
    plt.title('Actor Loss during Training')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 绘制Critic损失
    plt.subplot(4, 1, 2)
    plt.plot(critic_losses, label='Critic Loss Raw', color='lightgreen', alpha=0.5)
    # 确保x轴范围与移动平均数据的长度一致
    x_range = range(len(critic_ma)) if len(critic_losses) < window_size else range(window_size-1, len(critic_losses))
    plt.plot(x_range, critic_ma, label=f'Critic Loss MA ({window_size})', color='green')
    plt.title('Critic Loss during Training')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 绘制熵
    plt.subplot(4, 1, 3)
    plt.plot(entropies, label='Entropy Raw', color='lightyellow', alpha=0.5)
    # 确保x轴范围与移动平均数据的长度一致
    x_range = range(len(entropies_ma)) if len(entropies) < window_size else range(window_size-1, len(entropies))
    plt.plot(x_range, entropies_ma, label=f'Entropy MA ({window_size})', color='black')
    plt.title('Entropy during Training')
    plt.xlabel('Episode')
    plt.ylabel('Entropy')
    plt.legend()
    plt.grid(True)

    # 绘制奖励
    plt.subplot(4, 1, 4)
    plt.plot(rewards, label='Rewards Raw', color='lightcoral', alpha=0.5)
    # 确保x轴范围与移动平均数据的长度一致
    x_range = range(len(rewards_ma)) if len(rewards) < window_size else range(window_size-1, len(rewards))
    plt.plot(x_range, rewards_ma, label=f'Rewards MA ({window_size})', color='red')
    plt.title('Rewards during Training')
    plt.yscale('log')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('./RL_Data/result.svg', bbox_inches='tight')
    plt.show()


def show_expert_loss(all_losses, window_size=5):
    """显示专家预训练损失曲线，并添加平滑效果"""
    plt.figure(figsize=(10, 6))
    # 绘制原始损失曲线
    plt.plot(range(1, len(all_losses) + 1), all_losses, label='train loss', color='lightblue', alpha=0.5)
    # 计算并绘制平滑损失曲线
    smoothed_losses = moving_average(all_losses, window_size)
    # 确保x轴范围与移动平均数据的长度一致
    if len(all_losses) < window_size:
        x_range = range(1, len(smoothed_losses) + 1)
    else:
        x_range = range(window_size, len(all_losses) + 1)
    plt.plot(x_range, smoothed_losses, label=f'train loss（windows={window_size}）', color='blue')
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.title('expert learn train loss')
    plt.legend()
    plt.grid(True)
    # 保存损失图
    loss_plot_path = "./RL_Data/expert_loss.svg"
    plt.savefig(loss_plot_path)
    plt.close()


def moving_average(data, window_size=10):
    """计算移动窗口均值"""
    if len(data) < window_size:
        return data  # 数据点不足窗口大小时，返回原数据
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, 'valid')


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