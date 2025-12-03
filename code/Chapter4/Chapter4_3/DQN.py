"""
Improved DQN for CartPole with better convergence
"""

import random
import math
import collections
from dataclasses import dataclass
from typing import Deque, Tuple, List
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym

# ============= Improved Hyperparameters =============
@dataclass
class Config:
    env_name: str = "CartPole-v1"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # training
    total_episodes: int = 800  # 增加训练轮次
    max_steps_per_episode: int = 500
    batch_size: int = 32  # 减小batch size
    gamma: float = 0.99
    lr: float = 1e-3  # 适当的学习率

    # exploration
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: int = 300  # 更长的探索衰减

    # replay buffer
    replay_size: int = 10000
    min_replay_size: int = 1000

    # target network update
    target_update_freq: int = 100  # 按步数更新，而不是按episode

    # logging
    eval_every: int = 20

cfg = Config()

# ============= Simple but Effective Network =============
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        # 使用更简单的网络结构
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Tuple] = collections.deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# ============= Improved Agent =============
class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: Config):
        self.device = cfg.device
        self.action_dim = action_dim
        self.cfg = cfg

        # 网络
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        # 优化器
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.replay_buffer = ReplayBuffer(cfg.replay_size)

        # 训练计数器
        self.steps_done = 0

    def get_epsilon(self):
        """线性衰减的epsilon"""
        eps = self.cfg.eps_end + (self.cfg.eps_start - self.cfg.eps_end) * \
              math.exp(-1. * self.steps_done / self.cfg.eps_decay)
        return eps

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """动作选择"""
        if training and random.random() < self.get_epsilon():
            return random.randrange(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        """训练更新"""
        if len(self.replay_buffer) < self.cfg.min_replay_size:
            return None

        # 采样
        transitions = self.replay_buffer.sample(self.cfg.batch_size)
        batch = list(zip(*transitions))

        states = torch.FloatTensor(np.array(batch[0])).to(self.device)
        actions = torch.LongTensor(np.array(batch[1])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(batch[2])).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(batch[3])).to(self.device)
        dones = torch.BoolTensor(np.array(batch[4])).unsqueeze(1).to(self.device)

        # 当前Q值
        current_q_values = self.q_net(states).gather(1, actions)

        # 目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.cfg.gamma * next_q_values * ~dones)

        # 损失计算
        loss = nn.functional.mse_loss(current_q_values, target_q_values)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps_done += 1

        # 更新目标网络
        if self.steps_done % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

# ============= Training with Better Monitoring =============
def evaluate_agent(env, agent, n_episodes=5):
    """评估智能体性能"""
    total_rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            state = next_state
            episode_reward += reward

        total_rewards.append(episode_reward)

    return np.mean(total_rewards), np.std(total_rewards)

def train():
    # 环境设置
    env = gym.make(cfg.env_name)

    # 设置随机种子
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if hasattr(env, 'reset'):
        env.reset(seed=cfg.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Training on {cfg.device}")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")

    agent = DQNAgent(state_dim, action_dim, cfg)

    # 训练统计
    episode_rewards = []
    moving_averages = []
    losses = []
    best_mean_reward = 0

    print("\nStarting training...")
    print("Phase 1: Random exploration (first 1000 steps)")

    # 初始探索阶段
    state, _ = env.reset()
    for step in range(cfg.min_replay_size):
        action = env.action_space.sample()  # 完全随机探索
        next_state, reward, done, truncated, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)

        if done or truncated:
            state, _ = env.reset()
        else:
            state = next_state

        if step % 200 == 0:
            print(f"  Collected {step}/{cfg.min_replay_size} random transitions")

    print("Phase 2: Start training with experience replay")

    # 主训练循环
    for episode in range(1, cfg.total_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = 0
        update_count = 0

        for step in range(cfg.max_steps_per_episode):
            # 选择动作
            action = agent.select_action(state)

            # 执行动作
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            # 存储经验
            agent.store_transition(state, action, reward, next_state, done)

            # 训练
            loss = agent.update()
            if loss is not None:
                episode_loss += loss
                update_count += 1

            state = next_state
            episode_reward += reward

            if done:
                break

        # 记录统计
        episode_rewards.append(episode_reward)

        if update_count > 0:
            avg_loss = episode_loss / update_count
            losses.append(avg_loss)
        else:
            losses.append(0)

        # 计算移动平均
        if len(episode_rewards) >= 10:
            moving_avg = np.mean(episode_rewards[-10:])
            moving_averages.append(moving_avg)
        else:
            moving_avg = np.mean(episode_rewards)
            moving_averages.append(moving_avg)

        # 定期评估和保存
        if episode % cfg.eval_every == 0:
            eval_mean, eval_std = evaluate_agent(env, agent)
            current_epsilon = agent.get_epsilon()

            print(f"Episode {episode:4d} | "
                  f"Reward: {episode_reward:5.1f} | "
                  f"Avg10: {moving_avg:6.2f} | "
                  f"Eval: {eval_mean:5.1f} ± {eval_std:3.1f} | "
                  f"Epsilon: {current_epsilon:.3f} | "
                  f"Loss: {avg_loss if update_count > 0 else 0:.4f}")

            # 保存最佳模型
            if eval_mean > best_mean_reward:
                best_mean_reward = eval_mean
                torch.save(agent.q_net.state_dict(), "dqn_cartpole_best.pth")
                print(f"  💾 New best model saved! (Score: {eval_mean:.1f})")

            # 提前停止条件
            if eval_mean >= 495 and episode >= 200:
                print(f"\n🎉 Training completed! Agent achieved near-perfect performance.")
                break

        # 检查是否学习太慢
        if episode == 200 and moving_avg < 100:
            print("⚠️  Training seems slow. Consider adjusting hyperparameters.")
        elif episode == 400 and moving_avg < 300:
            print("⚠️  Training progress is suboptimal. You might want to restart with different parameters.")

    env.close()

    # 最终评估
    print("\nFinal evaluation...")
    final_env = gym.make(cfg.env_name)
    final_mean, final_std = evaluate_agent(final_env, agent, n_episodes=10)
    final_env.close()

    print(f"Final performance: {final_mean:.1f} ± {final_std:.1f}")

    # 绘制训练曲线
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 10))

        # 奖励曲线
        plt.subplot(2, 2, 1)
        plt.plot(episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
        plt.plot(moving_averages, color='red', linewidth=2, label='Moving Average (10)')
        plt.axhline(y=475, color='green', linestyle='--', label='Target (475)')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)

        # 损失曲线
        plt.subplot(2, 2, 2)
        plt.plot(losses)
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)

        # epsilon衰减
        plt.subplot(2, 2, 3)
        epsilons = [agent.get_epsilon() for _ in range(len(episode_rewards))]
        plt.plot(epsilons)
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Exploration Rate Decay')
        plt.grid(True)

        # 最终性能分布
        plt.subplot(2, 2, 4)
        final_rewards = []
        test_env = gym.make(cfg.env_name)
        for _ in range(20):
            state, _ = test_env.reset()
            total_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, training=False)
                state, reward, done, truncated, _ = test_env.step(action)
                done = done or truncated
                total_reward += reward
            final_rewards.append(total_reward)
        test_env.close()

        plt.hist(final_rewards, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(final_rewards), color='red', linestyle='--', label=f'Mean: {np.mean(final_rewards):.1f}')
        plt.xlabel('Final Reward')
        plt.ylabel('Frequency')
        plt.title('Final Performance Distribution')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('training_analysis.png', dpi=150, bbox_inches='tight')
        print("Saved training_analysis.png")

    except ImportError:
        print("Matplotlib not available, skipping plots")

    return agent, episode_rewards, moving_averages

def demonstrate_agent(agent, num_episodes=3, render=True):
    """展示训练好的智能体"""
    print(f"\n{'='*60}")
    print("DEMONSTRATING TRAINED AGENT")
    print(f"{'='*60}")

    if render:
        try:
            env = gym.make(cfg.env_name, render_mode='human')
        except:
            env = gym.make(cfg.env_name)
            render = False
            print("Note: Visualization not available")
    else:
        env = gym.make(cfg.env_name)

    rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        print(f"\nEpisode {episode + 1}: ", end="")

        while not done and steps < 1000:  # 增加最大步数限制
            action = agent.select_action(state, training=False)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            state = next_state
            total_reward += reward
            steps += 1

            if render:
                env.render()
                time.sleep(0.02)  # 减慢速度便于观察

            if done:
                break

        rewards.append(total_reward)
        print(f"Steps: {steps}, Total Reward: {total_reward}")

        # 性能评价
        if total_reward >= 495:
            print("  🎉 Perfect! Agent maintains perfect balance.")
        elif total_reward >= 450:
            print("  👍 Excellent! Very stable control.")
        elif total_reward >= 400:
            print("  ✅ Good! Reliable performance.")
        elif total_reward >= 300:
            print("  🔶 Acceptable. Some room for improvement.")
        else:
            print("  🔻 Needs more training.")

    env.close()

    # 总结
    avg_reward = np.mean(rewards)
    print(f"\nSummary: Average reward over {num_episodes} episodes: {avg_reward:.1f}")

    if avg_reward >= 480:
        print("🎊 SUCCESS! The agent has successfully learned to control CartPole!")
    elif avg_reward >= 400:
        print("👍 Good results! The agent performs well.")
    else:
        print("💡 Consider training for more episodes.")

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        # 演示模式
        print("Loading pre-trained model for demonstration...")
        env = gym.make(cfg.env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        env.close()

        agent = DQNAgent(state_dim, action_dim, cfg)

        try:
            agent.q_net.load_state_dict(torch.load("dqn_cartpole_best.pth", map_location=cfg.device))
            print("Model loaded successfully!")
            demonstrate_agent(agent, num_episodes=5, render=True)
        except FileNotFoundError:
            print("No trained model found. Please run training first.")
    else:
        # 训练模式
        print("CartPole DQN Training")
        print("=" * 50)
        agent, rewards, moving_avgs = train()

        # 训练后演示
        print("\n" + "=" * 60)
        print("POST-TRAINING DEMONSTRATION")
        print("=" * 60)
        demonstrate_agent(agent, num_episodes=5, render=True)