


from matplotlib import pyplot as plt


def show_loss_info(actor_losses, critic_losses, entropies,rewards):
    # 绘制训练损失图
    plt.figure(figsize=(15, 5))

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