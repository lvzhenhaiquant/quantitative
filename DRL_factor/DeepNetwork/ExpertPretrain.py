import sys
import os



# 进行模仿学习预训练
# cd /home/yunbo/project/quantitative/DRL_factor
# ~/software/venv/bin/python3.10 /home/yunbo/project/quantitative/DRL_factor/DeepNetwork/ExpertPretrain.py
current_file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_file_dir)
sys.path.append(parent_dir)

from PPOAgent import PPOAgent
from RlEnv import FactorRLEnv

if __name__ == "__main__":

    env = FactorRLEnv()
    agent = PPOAgent(env, learning_rate=3e-4, batch_size=32)
    # 进行模仿学习预训练
    agent.imitation_pretrain()
