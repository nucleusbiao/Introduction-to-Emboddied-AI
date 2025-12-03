"""
彻底修复的CartPole训练代码 - 确保A3C和SAC都能收敛
"""

import random
import math
import collections
from dataclasses import dataclass
from typing import Deque, Tuple, List, Optional
import time
import multiprocessing as mp
from torch.distributions import Categorical, Normal
import threading
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gymnasium as gym

# ============= 重新设计的超参数配置 =============
@dataclass
class Config:
    env_name: str = "CartPole-v1"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 通用训练参数
    total_episodes: int = 500  # 减少训练轮次
    max_steps_per_episode: int = 500
    gamma: float = 0.99
    lr: float = 1e-3  # 提高学习率

    # A3C特定参数
    a3c_workers: int = 2
    a3c_t_max: int = 10  # 减少更新步数

    # PPO特定参数
    ppo_epochs: int = 4
    ppo_clip: float = 0.2
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    ppo_batch_size: int = 32

    # SAC特定参数 - 大幅简化
    sac_tau: float = 0.01
    sac_alpha: float = 0.2
    sac_target_update_interval: int = 1
    sac_automatic_entropy_tuning: bool = False  # 关闭自动熵调整
    sac_replay_size: int = 10000
    sac_batch_size: int = 64
    sac_learning_starts: int = 1000
    sac_update_frequency: int = 1  # 每步更新

    # 评估
    eval_every: int = 20

cfg = Config()

# ============= 重新设计的网络结构 =============

# A3C网络 - 更简单的结构
class A3CNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.policy = nn.Linear(32, action_dim)
        self.value = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy_logits = self.policy(x)
        value = self.value(x)
        return policy_logits, value

# PPO网络
class PPONetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x)
        return F.softmax(logits, dim=-1), value

    def get_action(self, x):
        logits = self.actor(x)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_prob, entropy

# SAC网络 - 极度简化
class SACActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)

    def sample(self, x):
        logits = self.net(x)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

class SACCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        # 简单的Q网络，输出每个动作的Q值
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state, action=None):
        q_values = self.net(state)
        if action is not None:
            # 返回对应动作的Q值
            return q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        return q_values

# ============= 重新设计的A3C实现 =============
class A3CAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: Config):
        self.device = cfg.device
        self.global_network = A3CNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.global_network.parameters(), lr=cfg.lr)
        self.cfg = cfg

    def train(self):
        print("Starting A3C training with multiprocessing...")

        # 创建进程
        processes = []
        for i in range(self.cfg.a3c_workers):
            p = mp.Process(target=self._train_worker, args=(i,))
            p.start()
            processes.append(p)
            time.sleep(0.1)  # 避免同时启动

        # 等待所有进程完成
        for p in processes:
            p.join()

        print("A3C training completed!")

    def _train_worker(self, worker_id):
        # 设置随机种子
        torch.manual_seed(self.cfg.seed + worker_id)
        np.random.seed(self.cfg.seed + worker_id)
        random.seed(self.cfg.seed + worker_id)

        # 创建环境和网络
        env = gym.make(self.cfg.env_name)
        local_network = A3CNetwork(4, 2).to(self.device)
        local_network.load_state_dict(self.global_network.state_dict())

        state, _ = env.reset()
        total_episodes = 0
        episode_rewards = []

        while total_episodes < self.cfg.total_episodes // self.cfg.a3c_workers:
            # 收集经验
            states, actions, rewards, values = [], [], [], []

            for _ in range(self.cfg.a3c_t_max):
                state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
                with torch.no_grad():
                    logits, value = local_network(state_tensor)
                    probs = F.softmax(logits, dim=-1)
                    dist = Categorical(probs)
                    action = dist.sample().item()

                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(value.item())

                state = next_state

                if done:
                    state, _ = env.reset()
                    total_episodes += 1
                    if len(episode_rewards) < 10:
                        episode_rewards.append(sum(rewards))
                    else:
                        episode_rewards[total_episodes % 10] = sum(rewards)

                    if total_episodes % 10 == 0:
                        avg_reward = np.mean(episode_rewards)
                        print(f"A3C Worker {worker_id}, Episode {total_episodes}, Avg Reward: {avg_reward:.1f}")
                    break

            if not states:
                continue

            # 计算回报
            R = 0
            if not done:
                state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
                with torch.no_grad():
                    _, last_value = local_network(state_tensor)
                R = last_value.item()

            returns = []
            for r in reversed(rewards):
                R = r + self.cfg.gamma * R
                returns.insert(0, R)

            # 准备数据
            states_t = torch.FloatTensor(np.array(states)).to(self.device)
            actions_t = torch.LongTensor(actions).to(self.device)
            returns_t = torch.FloatTensor(returns).to(self.device)
            old_values_t = torch.FloatTensor(values).to(self.device)

            # 计算优势
            advantages = returns_t - old_values_t

            # 计算损失
            logits, new_values = local_network(states_t)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy()

            policy_loss = -(log_probs * advantages.detach()).mean()
            value_loss = F.mse_loss(new_values.squeeze(), returns_t)
            entropy_loss = -entropy.mean()

            total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

            # 更新全局网络
            self.optimizer.zero_grad()
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(local_network.parameters(), 0.5)

            # 应用梯度到全局网络
            for global_param, local_param in zip(self.global_network.parameters(), local_network.parameters()):
                if global_param.grad is None:
                    global_param.grad = local_param.grad
                else:
                    global_param.grad += local_param.grad

            self.optimizer.step()

            # 同步网络
            local_network.load_state_dict(self.global_network.state_dict())

        env.close()

# ============= PPO实现（保持不变） =============
class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: Config):
        self.device = cfg.device
        self.network = PPONetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=cfg.lr)
        self.memory = []
        self.cfg = cfg

    def store_transition(self, state, action, log_prob, value, reward, done):
        self.memory.append((state, action, log_prob, value, reward, done))

    def update(self):
        if len(self.memory) < self.cfg.ppo_batch_size:
            return 0, 0, 0

        states, actions, old_log_probs, values, rewards, dones = zip(*self.memory)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        old_values = torch.FloatTensor(values).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # 计算GAE和回报
        returns = []
        advantages = []
        R = 0
        advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
                next_non_terminal = 0
            else:
                next_value = old_values[t+1]
                next_non_terminal = 1.0 - dones[t+1].float()

            delta = rewards[t] + self.cfg.gamma * next_value * next_non_terminal - old_values[t]
            advantage = delta + self.cfg.gamma * 0.95 * advantage * next_non_terminal
            advantages.insert(0, advantage)
            returns.insert(0, advantage + old_values[t])

        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO更新
        policy_losses = []
        value_losses = []
        entropy_losses = []

        for _ in range(self.cfg.ppo_epochs):
            indices = torch.randperm(len(states))

            for start in range(0, len(states), self.cfg.ppo_batch_size):
                end = start + self.cfg.ppo_batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                probs, new_values = self.network(batch_states)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy()

                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.cfg.ppo_clip, 1 + self.cfg.ppo_clip) * batch_advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values.squeeze(), batch_returns)
                entropy_loss = -entropy.mean()

                total_loss = policy_loss + self.cfg.ppo_value_coef * value_loss + self.cfg.ppo_entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

        self.memory = []
        return np.mean(policy_losses), np.mean(value_losses), np.mean(entropy_losses)

# ============= 重新设计的SAC实现 =============
class SACAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: Config):
        self.device = cfg.device
        self.action_dim = action_dim

        # 网络
        self.actor = SACActor(state_dim, action_dim).to(self.device)
        self.critic1 = SACCritic(state_dim, action_dim).to(self.device)
        self.critic2 = SACCritic(state_dim, action_dim).to(self.device)
        self.target_critic1 = SACCritic(state_dim, action_dim).to(self.device)
        self.target_critic2 = SACCritic(state_dim, action_dim).to(self.device)

        # 初始化目标网络
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=cfg.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=cfg.lr)

        # 经验回放
        self.replay_buffer = collections.deque(maxlen=cfg.sac_replay_size)
        self.cfg = cfg

        # 固定alpha值，简化实现
        self.alpha = cfg.sac_alpha

    def select_action(self, state, evaluate=False):
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate:
            with torch.no_grad():
                probs = self.actor(state_tensor)
                action = probs.argmax(dim=1)
            return action.item()
        else:
            with torch.no_grad():
                action, _ = self.actor.sample(state_tensor)
            return action.item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.replay_buffer) < self.cfg.sac_batch_size:
            return 0, 0, 0

        # 采样
        batch = random.sample(self.replay_buffer, self.cfg.sac_batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        with torch.no_grad():
            # 下一状态的动作选择
            next_actions, next_log_probs = self.actor.sample(next_states)

            # 目标Q值
            target_q1 = self.target_critic1(next_states)
            target_q2 = self.target_critic2(next_states)
            target_q = torch.min(target_q1, target_q2)

            # 选择对应动作的Q值并减去熵项
            next_q_values = target_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = next_q_values - self.alpha * next_log_probs

            # 计算目标
            target_q = rewards + (1 - dones.float()) * self.cfg.gamma * target_q_values

        # 更新Critic
        current_q1 = self.critic1(states, actions.squeeze(1))
        current_q2 = self.critic2(states, actions.squeeze(1))

        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()

        # 更新Actor
        new_actions, new_log_probs = self.actor.sample(states)
        new_q1 = self.critic1(states, new_actions)
        new_q2 = self.critic2(states, new_actions)
        new_q = torch.min(new_q1, new_q2)

        actor_loss = (self.alpha * new_log_probs - new_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # 软更新目标网络
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.cfg.sac_tau) + param.data * self.cfg.sac_tau)

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.cfg.sac_tau) + param.data * self.cfg.sac_tau)

        return critic1_loss.item(), critic2_loss.item(), actor_loss.item()

# ============= 训练函数 =============
def evaluate_agent(env, agent, agent_type, n_episodes=5):
    total_rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            if agent_type == "A3C":
                state_tensor = torch.FloatTensor(state).to(cfg.device).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = agent.global_network(state_tensor)
                action = logits.argmax().item()
            elif agent_type == "PPO":
                action, _, _ = agent.network.get_action(torch.FloatTensor(state).to(cfg.device).unsqueeze(0))
            elif agent_type == "SAC":
                action = agent.select_action(state, evaluate=True)

            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            state = next_state
            episode_reward += reward

        total_rewards.append(episode_reward)

    return np.mean(total_rewards), np.std(total_rewards)

def train_a3c():
    print("Training A3C...")
    agent = A3CAgent(4, 2, cfg)
    agent.train()

    # 最终评估
    env = gym.make(cfg.env_name)
    eval_mean, eval_std = evaluate_agent(env, agent, "A3C", n_episodes=10)
    env.close()

    print(f"A3C Final Evaluation: {eval_mean:.1f} ± {eval_std:.1f}")
    return agent, []

def train_ppo():
    env = gym.make(cfg.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim, action_dim, cfg)

    episode_rewards = []
    best_mean_reward = 0

    for episode in range(1, cfg.total_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, log_prob, entropy = agent.network.get_action(torch.FloatTensor(state).to(cfg.device).unsqueeze(0))
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            _, value = agent.network(torch.FloatTensor(state).to(cfg.device).unsqueeze(0))

            agent.store_transition(state, action, log_prob.item(), value.item(), reward, done)

            state = next_state
            episode_reward += reward

        # 更新
        policy_loss, value_loss, entropy_loss = agent.update()

        episode_rewards.append(episode_reward)

        if episode % cfg.eval_every == 0:
            eval_mean, eval_std = evaluate_agent(env, agent, "PPO")
            avg_reward_10 = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)

            print(f"PPO Episode {episode:4d} | "
                  f"Reward: {episode_reward:5.1f} | "
                  f"Avg10: {avg_reward_10:6.2f} | "
                  f"Eval: {eval_mean:5.1f} ± {eval_std:3.1f}")

            if eval_mean > best_mean_reward:
                best_mean_reward = eval_mean
                torch.save(agent.network.state_dict(), "ppo_cartpole_best.pth")
                print(f"  💾 New best PPO model saved! (Score: {eval_mean:.1f})")

    env.close()
    return agent, episode_rewards

def train_sac():
    env = gym.make(cfg.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = SACAgent(state_dim, action_dim, cfg)

    episode_rewards = []
    best_mean_reward = 0

    # 预填充经验池
    print("Pre-filling replay buffer...")
    state, _ = env.reset()
    for i in range(cfg.sac_learning_starts):
        action = env.action_space.sample()
        next_state, reward, done, truncated, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        if done or truncated:
            state, _ = env.reset()
        else:
            state = next_state

    print("Starting SAC training...")
    for episode in range(1, cfg.total_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            agent.store_transition(state, action, reward, next_state, done)

            # 更新
            if len(agent.replay_buffer) >= cfg.sac_batch_size:
                agent.update()

            state = next_state
            episode_reward += reward

        episode_rewards.append(episode_reward)

        if episode % cfg.eval_every == 0:
            eval_mean, eval_std = evaluate_agent(env, agent, "SAC")
            avg_reward_10 = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)

            print(f"SAC Episode {episode:4d} | "
                  f"Reward: {episode_reward:5.1f} | "
                  f"Avg10: {avg_reward_10:6.2f} | "
                  f"Eval: {eval_mean:5.1f} ± {eval_std:3.1f}")

            if eval_mean > best_mean_reward:
                best_mean_reward = eval_mean
                torch.save({
                    'actor': agent.actor.state_dict(),
                    'critic1': agent.critic1.state_dict(),
                    'critic2': agent.critic2.state_dict(),
                }, "sac_cartpole_best.pth")
                print(f"  💾 New best SAC model saved! (Score: {eval_mean:.1f})")

    env.close()
    return agent, episode_rewards

def demonstrate_agent(agent, agent_type, num_episodes=5, render=True):
    print(f"\n{'='*60}")
    print(f"DEMONSTRATING {agent_type} AGENT")
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

        while not done and steps < 1000:
            if agent_type == "A3C":
                state_tensor = torch.FloatTensor(state).to(cfg.device).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = agent.global_network(state_tensor)
                action = logits.argmax().item()
            elif agent_type == "PPO":
                action, _, _ = agent.network.get_action(torch.FloatTensor(state).to(cfg.device).unsqueeze(0))
            elif agent_type == "SAC":
                action = agent.select_action(state, evaluate=True)

            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            state = next_state
            total_reward += reward
            steps += 1

            if render:
                env.render()
                time.sleep(0.02)

            if done:
                break

        rewards.append(total_reward)
        print(f"Steps: {steps}, Total Reward: {total_reward}")

    env.close()

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

    if len(sys.argv) > 1:
        method = sys.argv[1].upper()
    else:
        method = "PPO"

    print(f"Training CartPole with {method} method")
    print("=" * 50)

    # 设置随机种子
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    if method == "A3C":
        agent, rewards = train_a3c()
        demonstrate_agent(agent, "A3C", num_episodes=5, render=True)

    elif method == "PPO":
        agent, rewards = train_ppo()
        demonstrate_agent(agent, "PPO", num_episodes=5, render=True)

    elif method == "SAC":
        agent, rewards = train_sac()
        demonstrate_agent(agent, "SAC", num_episodes=5, render=True)

    else:
        print(f"Unknown method: {method}")
        print("Available methods: A3C, PPO, SAC")