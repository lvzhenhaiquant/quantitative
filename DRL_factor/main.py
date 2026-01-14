from RlEnv import FactorRLEnv
from PPOAgent import PPOAgent
import utils.utils_tools as utils_tools

if __name__ == "__main__":

    env = FactorRLEnv()
    agent = PPOAgent(env, learning_rate=3e-4, batch_size=32)
    # 进行模仿学习预训练
    # agent.imitation_pretrain()
    agent.load_model()

    # 用于记录训练数据
    actor_losses = []
    critic_losses = []
    entropies = []
    rewards = []
    episodes = []
    total_steps = 0
    train_count=0
    
    # 训练配置
    num_steps = 500  # 总训练epochs
    collect_days = 1454  # 每次收集的步数,总共回测的天数
    
    while total_steps < num_steps:
        DRL_env = env.reset()
        observation = DRL_env.observation
        step_count = 1
        while step_count < collect_days: #总天数
            reward = agent.collect_experience_for_day(observation) #每天执行
            rewards.append(reward)
            if step_count % 15 == 0: # 训练代理
                loss_info = agent.learn()
                # 记录训练信息
                actor_losses.append(loss_info['actor_loss'])
                critic_losses.append(loss_info['critic_loss'])
                entropies.append(loss_info['entropy'])
                train_count += 1
                utils_tools.show_loss_info(actor_losses, critic_losses, entropies,rewards)
                print(f"days进度：{step_count}/{collect_days},总执行进度：{total_steps}/{num_steps},训练轮次：{train_count}")
            # 保存模型
            if step_count % 100 == 0:
                agent.save_model()
            step_count += 1
        total_steps += collect_days



        
    


