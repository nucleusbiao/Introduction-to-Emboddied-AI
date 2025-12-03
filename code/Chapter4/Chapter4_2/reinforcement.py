import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class SimplePolicy:
    """
    简单的策略类，不使用深度神经网络
    演示一个简单的离散动作空间问题
    """

    def __init__(self, n_states=4, n_actions=2, learning_rate=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate

        # 简单的策略参数表：state -> action probabilities
        self.theta = torch.randn(n_states, n_actions, requires_grad=False)

        # 存储轨迹信息
        self.log_probs = []
        self.rewards = []

    def get_action(self, state):
        """根据状态选择动作"""
        logits = self.theta[state]
        max_logit = torch.max(logits)
        stable_logits = logits - max_logit
        exp_logits = torch.exp(stable_logits)
        probs = exp_logits / torch.sum(exp_logits)

        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[action])
        self.log_probs.append(log_prob)

        return action

    def get_action_probabilities(self, state):
        """获取状态的动作概率"""
        logits = self.theta[state]
        max_logit = torch.max(logits)
        stable_logits = logits - max_logit
        exp_logits = torch.exp(stable_logits)
        return exp_logits / torch.sum(exp_logits)

    def update_policy(self):
        """REINFORCE策略更新"""
        returns = self._calculate_returns()

        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)

        total_loss = torch.stack(policy_loss).sum()

        with torch.no_grad():
            for state in range(self.n_states):
                for t, (log_prob, G) in enumerate(zip(self.log_probs, returns)):
                    if t < len(self.log_probs):
                        grad_theta = self._compute_gradient_manual(state, t, G)
                        self.theta[state] += self.learning_rate * grad_theta

        self.log_probs = []
        self.rewards = []

        return total_loss.item()

    def _compute_gradient_manual(self, state, time_step, return_t):
        """手动计算梯度"""
        current_probs = self.get_action_probabilities(state)
        grad = torch.zeros(self.n_actions)

        if time_step < len(self.log_probs):
            action_taken = torch.argmax(self.log_probs[time_step].detach()) if len(self.log_probs) > time_step else 0
            for a in range(self.n_actions):
                if a == action_taken:
                    grad[a] = (1 - current_probs[a]) * return_t
                else:
                    grad[a] = (-current_probs[a]) * return_t

        return grad

    def _calculate_returns(self, gamma=0.99):
        """计算回报"""
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def add_reward(self, reward):
        """添加奖励"""
        self.rewards.append(reward)


def simple_environment(state, action):
    """简单的演示环境"""
    if state == 0:
        if action == 0:
            next_state = 1
            reward = 0
        else:
            next_state = 0
            reward = -1
    elif state == 1:
        if action == 0:
            next_state = 2
            reward = 0
        else:
            next_state = 0
            reward = 0
    elif state == 2:
        if action == 0:
            next_state = 3
            reward = 10
        else:
            next_state = 1
            reward = 0
    else:
        next_state = 3
        reward = 0

    done = (next_state == 3)
    return next_state, reward, done


class ReinforcementAnimator:
    """强化学习动画演示类"""

    def __init__(self):
        self.fig = plt.figure(figsize=(15, 10))
        self.setup_plots()

    def setup_plots(self):
        """设置绘图区域"""
        # 环境可视化
        self.ax_env = self.fig.add_subplot(2, 3, 1)
        self.ax_env.set_title('环境状态与智能体移动')
        self.ax_env.set_xlim(-1, 5)
        self.ax_env.set_ylim(-1, 2)

        # 策略概率可视化
        self.ax_policy = self.fig.add_subplot(2, 3, 2)
        self.ax_policy.set_title('策略概率分布')
        self.ax_policy.set_xlabel('状态')
        self.ax_policy.set_ylabel('概率')

        # 学习曲线
        self.ax_learning = self.fig.add_subplot(2, 3, 3)
        self.ax_learning.set_title('学习曲线')
        self.ax_learning.set_xlabel('Episode')
        self.ax_learning.set_ylabel('Total Reward')

        # 参数变化
        self.ax_params = self.fig.add_subplot(2, 3, 4)
        self.ax_params.set_title('策略参数变化')
        self.ax_params.set_xlabel('状态')
        self.ax_params.set_ylabel('参数值')

        # 实时轨迹
        self.ax_trajectory = self.fig.add_subplot(2, 3, 5)
        self.ax_trajectory.set_title('当前轨迹')
        self.ax_trajectory.set_xlabel('时间步')
        self.ax_trajectory.set_ylabel('状态')

        # 回报分布
        self.ax_rewards = self.fig.add_subplot(2, 3, 6)
        self.ax_rewards.set_title('即时回报')
        self.ax_rewards.set_xlabel('时间步')
        self.ax_rewards.set_ylabel('回报')

        plt.tight_layout()

    def update_animation(self, episode, policy, episode_rewards, current_trajectory, current_rewards, current_state):
        """更新动画帧"""
        self.fig.suptitle(f'REINFORCE算法演示 - Episode {episode}', fontsize=16)

        # 清空所有子图
        for ax in [self.ax_env, self.ax_policy, self.ax_learning,
                   self.ax_params, self.ax_trajectory, self.ax_rewards]:
            ax.clear()

        self.plot_environment(current_state)
        self.plot_policy(policy)
        self.plot_learning_curve(episode_rewards)
        self.plot_parameters(policy)
        self.plot_trajectory(current_trajectory)
        self.plot_rewards(current_rewards)

    def plot_environment(self, current_state):
        """绘制环境状态"""
        states = [0, 1, 2, 3]
        colors = ['lightblue' if s != current_state else 'red' for s in states]

        # 绘制状态节点
        for i, state in enumerate(states):
            circle = plt.Circle((i, 0), 0.3, color=colors[i], alpha=0.7)
            self.ax_env.add_patch(circle)
            self.ax_env.text(i, 0, f'S{state}', ha='center', va='center', fontweight='bold')

        # 绘制箭头（转移）
        arrows = [(0, 1, '0'), (1, 0, '1'), (1, 2, '0'), (2, 1, '1'), (2, 3, '0')]

        for start, end, action in arrows:
            dx = end - start
            self.ax_env.arrow(start + 0.3, 0, dx - 0.6, 0, head_width=0.1,
                              head_length=0.1, fc='gray', ec='gray', alpha=0.6)
            self.ax_env.text((start + end) / 2, 0.2, f'A{action}',
                             ha='center', va='center', fontsize=8)

        self.ax_env.set_xlim(-0.5, 3.5)
        self.ax_env.set_ylim(-0.5, 1)
        self.ax_env.set_aspect('equal')
        self.ax_env.set_title(f'当前状态: S{current_state}')
        self.ax_env.axis('off')

    def plot_policy(self, policy):
        """绘制策略概率"""
        states = range(4)
        action_names = ['向右', '向左']

        for state in states:
            probs = policy.get_action_probabilities(state).detach().numpy()
            bottom = 0
            for action, prob in enumerate(probs):
                self.ax_policy.bar(state + action * 0.3, prob, width=0.3,
                                   label=action_names[action] if state == 0 else "",
                                   alpha=0.7)

        self.ax_policy.set_xticks([0.15, 1.15, 2.15, 3.15])
        self.ax_policy.set_xticklabels(['S0', 'S1', 'S2', 'S3'])
        self.ax_policy.set_ylabel('概率')
        self.ax_policy.set_ylim(0, 1)
        self.ax_policy.set_title('各状态的动作概率')
        self.ax_policy.legend()
        self.ax_policy.grid(True, alpha=0.3)

    def plot_learning_curve(self, episode_rewards):
        """绘制学习曲线"""
        if len(episode_rewards) > 0:
            episodes = range(len(episode_rewards))
            self.ax_learning.plot(episodes, episode_rewards, 'b-', alpha=0.7, label='每回合回报')

            # 计算移动平均
            if len(episode_rewards) >= 10:
                window = min(10, len(episode_rewards))
                moving_avg = [np.mean(episode_rewards[i:i + window])
                              for i in range(len(episode_rewards) - window + 1)]
                self.ax_learning.plot(range(window - 1, len(episode_rewards)),
                                      moving_avg, 'r-', linewidth=2, label='10期移动平均')

            self.ax_learning.set_xlabel('Episode')
            self.ax_learning.set_ylabel('Total Reward')
            self.ax_learning.set_title('学习曲线')
            self.ax_learning.legend()
            self.ax_learning.grid(True, alpha=0.3)

    def plot_parameters(self, policy):
        """绘制策略参数"""
        states = range(4)
        params = policy.theta.detach().numpy()

        for action in range(2):
            self.ax_params.plot(states, params[:, action], 'o-',
                                label=f'动作{action}', alpha=0.7)

        self.ax_params.set_xlabel('状态')
        self.ax_params.set_ylabel('参数值')
        self.ax_params.set_title('策略参数 θ')
        self.ax_params.legend()
        self.ax_params.grid(True, alpha=0.3)

    def plot_trajectory(self, trajectory):
        """绘制当前轨迹"""
        if len(trajectory) > 0:
            steps = range(len(trajectory))
            self.ax_trajectory.plot(steps, trajectory, 'go-', linewidth=2, markersize=6)
            self.ax_trajectory.set_xlabel('时间步')
            self.ax_trajectory.set_ylabel('状态')
            self.ax_trajectory.set_title('当前轨迹')
            self.ax_trajectory.set_ylim(-0.5, 3.5)
            self.ax_trajectory.grid(True, alpha=0.3)

    def plot_rewards(self, rewards):
        """绘制回报"""
        if len(rewards) > 0:
            steps = range(len(rewards))
            self.ax_rewards.bar(steps, rewards, alpha=0.7, color='orange')
            self.ax_rewards.set_xlabel('时间步')
            self.ax_rewards.set_ylabel('回报')
            self.ax_rewards.set_title('即时回报')
            self.ax_rewards.grid(True, alpha=0.3)


def train_reinforce_with_animation():
    """带动画演示的REINFORCE训练"""
    policy = SimplePolicy(n_states=4, n_actions=2, learning_rate=0.1)
    animator = ReinforcementAnimator()

    episodes = 200
    episode_rewards = []

    print("开始训练REINFORCE算法（带动画演示）...")
    print("环境说明: 状态0→1→2→3，动作0=向右，1=向左")
    print("目标: 从状态0到达状态3\n")

    # 创建动画
    plt.ion()  # 开启交互模式
    plt.show()

    for episode in range(episodes):
        state = 0
        total_reward = 0
        done = False
        steps = 0

        # 存储当前episode的轨迹和回报
        current_trajectory = [state]
        current_rewards = []

        policy.log_probs = []
        policy.rewards = []

        # 运行一个episode
        while not done and steps < 20:
            action = policy.get_action(state)
            next_state, reward, done = simple_environment(state, action)

            policy.add_reward(reward)
            total_reward += reward
            state = next_state
            steps += 1

            current_trajectory.append(state)
            current_rewards.append(reward)

        # 更新策略
        loss = policy.update_policy()
        episode_rewards.append(total_reward)

        # 更新动画（每10个episode或最后几个episode）
        if episode % 10 == 0 or episode >= episodes - 5:
            animator.update_animation(episode, policy, episode_rewards,
                                      current_trajectory, current_rewards,
                                      current_trajectory[-1] if current_trajectory else 0)
            plt.pause(0.1)

        if episode % 20 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.1f}, Steps: {steps}")

    # 训练结束，保持最终画面
    print("\n训练完成！")
    animator.update_animation(episodes - 1, policy, episode_rewards,
                              current_trajectory, current_rewards,
                              current_trajectory[-1] if current_trajectory else 0)
    plt.ioff()  # 关闭交互模式
    plt.show()

    # 显示最终策略
    print("\n训练后的策略:")
    for state in range(4):
        probs = policy.get_action_probabilities(state)
        print(f"状态 {state}: 向右概率={probs[0]:.3f}, 向左概率={probs[1]:.3f}")

    return policy


def demonstrate_learned_policy(policy):
    """演示学习到的策略"""
    print("\n最终策略演示:")
    state = 0
    steps = 0
    path = [state]
    actions_taken = []

    while state != 3 and steps < 10:
        action_probs = policy.get_action_probabilities(state)
        action = torch.argmax(action_probs).item()

        state, reward, done = simple_environment(state, action)
        path.append(state)
        actions_taken.append(action)
        steps += 1

        action_str = "向右" if action == 0 else "向左"
        print(f"步骤 {steps}: 状态 {path[-2]} → {action_str} → 状态 {state} (回报: {reward})")

    print(f"\n最终路径: {' → '.join(map(str, path))}")
    print(f"动作序列: {['向右' if a == 0 else '向左' for a in actions_taken]}")


if __name__ == "__main__":
    # 训练并显示动画
    trained_policy = train_reinforce_with_animation()

    # 演示最终策略
    demonstrate_learned_policy(trained_policy)

    # 保持窗口打开
    input("按Enter键退出...")