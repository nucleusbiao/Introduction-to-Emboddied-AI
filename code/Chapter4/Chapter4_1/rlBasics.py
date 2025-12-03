import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from time import sleep

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class MazeEnvironment:
    """简单的迷宫环境"""

    def __init__(self, size=5):
        self.size = size
        self.reset()

        # 定义迷宫布局：0=空地，1=墙，2=起点，3=终点
        self.maze = np.zeros((size, size))
        # 设置墙壁 - 创建一个更有挑战性的迷宫
        for i in range(1, size - 1):
            self.maze[i, 2] = 1

        # 设置起点和终点
        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)

    def reset(self):
        """重置环境到初始状态"""
        self.agent_pos = [0, 0]
        self.steps = 0
        self.max_steps = 100
        return self.get_state()

    def get_state(self):
        """获取当前状态（位置元组）"""
        return tuple(self.agent_pos)

    def step(self, action):
        """执行动作并返回(next_state, reward, done)"""
        self.steps += 1

        # 动作映射：0=上，1=右，2=下，3=左
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        new_pos = [
            self.agent_pos[0] + moves[action][0],
            self.agent_pos[1] + moves[action][1]
        ]

        # 检查是否越界或撞墙
        if (0 <= new_pos[0] < self.size and
                0 <= new_pos[1] < self.size and
                self.maze[new_pos[0], new_pos[1]] != 1):
            self.agent_pos = new_pos

        # 计算奖励
        reward = 0
        done = False

        # 到达终点 - 大正奖励
        if self.agent_pos == list(self.goal_pos):
            reward = 10
            done = True
        # 步数限制 - 负奖励
        elif self.steps >= self.max_steps:
            reward = -1
            done = True
        # 鼓励向终点移动 - 稠密奖励
        else:
            # 基于曼哈顿距离的奖励
            current_dist = abs(self.agent_pos[0] - self.goal_pos[0]) + \
                           abs(self.agent_pos[1] - self.goal_pos[1])
            reward = -0.1  # 每步小惩罚鼓励高效路径

        return self.get_state(), reward, done

    def render(self):
        """可视化迷宫当前状态"""
        grid = np.copy(self.maze)
        grid[self.agent_pos[0], self.agent_pos[1]] = 4  # 智能体位置
        grid[self.goal_pos[0], self.goal_pos[1]] = 3  # 终点

        plt.figure(figsize=(6, 6))
        plt.imshow(grid, cmap='viridis')
        plt.title(f"Maze - Agent at {self.agent_pos}")
        plt.axis('off')
        plt.show()


class QLearningAgent:
    """Q-learning智能体 - 经典的表格法强化学习"""

    def __init__(self, state_size, action_size):
        self.action_size = action_size

        # Q表：状态-动作值函数
        self.q_table = {}

        # 强化学习参数
        self.learning_rate = 0.1  # 学习率 α
        self.discount_factor = 0.95  # 折扣因子 γ
        self.epsilon = 1.0  # 探索率 ε
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def get_q_value(self, state, action):
        """获取Q值，如果状态不存在则初始化"""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
        return self.q_table[state][action]

    def set_q_value(self, state, action, value):
        """设置Q值"""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
        self.q_table[state][action] = value

    def act(self, state):
        """ε-贪心策略选择动作"""
        # 探索：以ε概率随机选择动作
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        # 利用：选择当前状态下Q值最大的动作
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)

        # 如果多个动作有相同Q值，随机选择其中一个
        max_q = np.max(self.q_table[state])
        best_actions = [a for a, q in enumerate(self.q_table[state]) if q == max_q]
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):
        """Q-learning更新规则"""
        current_q = self.get_q_value(state, action)

        if done:
            # 如果是终止状态，目标值就是即时奖励
            target = reward
        else:
            # 否则使用贝尔曼方程更新
            next_max_q = np.max([self.get_q_value(next_state, a)
                                 for a in range(self.action_size)])
            target = reward + self.discount_factor * next_max_q

        # Q-learning更新公式
        new_q = current_q + self.learning_rate * (target - current_q)
        self.set_q_value(state, action, new_q)

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_agent():
    """训练Q-learning智能体"""
    env = MazeEnvironment(size=5)
    action_size = 4  # 动作空间大小：上下左右

    agent = QLearningAgent(None, action_size)  # 状态大小由环境决定
    episodes = 500
    scores = []
    success_rate = []

    print("开始训练Q-learning智能体...")
    print("=" * 50)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        success = False

        while True:
            # 智能体选择动作
            action = agent.act(state)

            # 执行动作，环境反馈
            next_state, reward, done = env.step(action)

            # 智能体学习
            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                if env.agent_pos == list(env.goal_pos):
                    success = True
                break

        scores.append(total_reward)
        success_rate.append(1 if success else 0)

        # 定期输出训练进度
        if episode % 50 == 0:
            recent_success = np.mean(success_rate[-50:]) * 100
            avg_score = np.mean(scores[-50:])
            print(f"回合 {episode:3d} | "
                  f"得分: {total_reward:6.2f} | "
                  f"平均得分: {avg_score:6.2f} | "
                  f"成功率: {recent_success:5.1f}% | "
                  f"探索率: {agent.epsilon:.3f} | "
                  f"步数: {steps:3d}")

    return agent, env, scores, success_rate


class MazeAnimation:
    """迷宫动画演示类"""

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.path = []

    def run_episode(self):
        """运行一个回合并记录路径"""
        state = self.env.reset()
        self.path = [self.env.agent_pos.copy()]
        total_reward = 0
        done = False

        # 保存原始探索率
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0  # 测试时关闭探索

        while not done:
            action = self.agent.act(state)
            next_state, reward, done = self.env.step(action)
            self.path.append(self.env.agent_pos.copy())
            state = next_state
            total_reward += reward

        # 恢复探索率
        self.agent.epsilon = original_epsilon
        return total_reward

    def update_frame(self, frame):
        """更新动画帧"""
        self.ax.clear()

        # 绘制迷宫
        grid = np.copy(self.env.maze)
        grid[self.env.goal_pos[0], self.env.goal_pos[1]] = 3  # 终点

        # 显示网格数值
        self.ax.imshow(grid, cmap='viridis', alpha=0.7)

        # 绘制路径（到当前帧）
        if frame < len(self.path):
            current_path = self.path[:frame + 1]
            path_array = np.array(current_path)

            # 绘制路径线
            if len(path_array) > 1:
                self.ax.plot(path_array[:, 1], path_array[:, 0], 'r-',
                             linewidth=3, alpha=0.7, label='路径')

            # 绘制所有经过的点
            self.ax.scatter(path_array[:, 1], path_array[:, 0],
                            color='yellow', s=100, alpha=0.6, marker='o')

            # 当前智能体位置
            current_pos = self.path[frame]
            self.ax.scatter(current_pos[1], current_pos[0],
                            color='red', s=300, marker='*',
                            edgecolors='black', linewidth=2, label='智能体')

        # 起点和终点
        self.ax.scatter(self.env.start_pos[1], self.env.start_pos[0],
                        color='green', s=300, marker='o',
                        edgecolors='black', linewidth=2, label='起点')
        self.ax.scatter(self.env.goal_pos[1], self.env.goal_pos[0],
                        color='blue', s=300, marker='s',
                        edgecolors='black', linewidth=2, label='终点')

        # 添加网格和标签
        self.ax.set_xticks(np.arange(-0.5, self.env.size, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.env.size, 1), minor=True)
        self.ax.grid(which="minor", color="gray", linestyle='-', linewidth=2, alpha=0.3)
        self.ax.tick_params(which="minor", size=0)

        self.ax.set_xlim(-0.5, self.env.size - 0.5)
        self.ax.set_ylim(-0.5, self.env.size - 0.5)
        self.ax.invert_yaxis()  # 让y轴从上到下增加

        # 添加步数信息
        if frame < len(self.path):
            current_pos = self.path[frame]
            self.ax.set_title(f'强化学习智能体走迷宫演示\n'
                              f'步数: {frame}/{len(self.path) - 1} | '
                              f'位置: {current_pos} | '
                              f'动作: {"↑→↓←"[frame % 4] if frame > 0 else "开始"}',
                              fontsize=14, fontweight='bold')

        self.ax.legend(loc='upper right')
        self.ax.axis('off')

    def create_animation(self, interval=500, save_gif=False):
        """创建动画"""
        print("生成动画演示...")
        total_reward = self.run_episode()

        # 创建动画
        anim = FuncAnimation(self.fig, self.update_frame,
                             frames=len(self.path),
                             interval=interval,
                             repeat=True)

        plt.tight_layout()

        # 保存为GIF
        if save_gif:
            print("保存动画为GIF文件...")
            anim.save('maze_rl_demo.gif', writer='pillow', fps=2)
            print("已保存为 maze_rl_demo.gif")

        print(f"动画生成功！总步数: {len(self.path) - 1}, 总奖励: {total_reward:.2f}")
        plt.show()

        return anim


def plot_training_progress(scores, success_rate):
    """绘制训练过程图表"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 得分曲线
    window = 20
    moving_avg_scores = [np.mean(scores[i - window:i])
                         for i in range(window, len(scores))]

    ax1.plot(scores, alpha=0.3, color='blue', label='单回合得分')
    ax1.plot(range(window, len(scores)), moving_avg_scores,
             color='red', linewidth=2, label=f'{window}回合移动平均')
    ax1.set_title('训练得分')
    ax1.set_xlabel('回合')
    ax1.set_ylabel('得分')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 成功率曲线
    moving_avg_success = [np.mean(success_rate[i - window:i]) * 100
                          for i in range(window, len(success_rate))]

    ax2.plot(range(window, len(success_rate)), moving_avg_success,
             color='green', linewidth=2)
    ax2.set_title('成功率移动平均')
    ax2.set_xlabel('回合')
    ax2.set_ylabel('成功率 (%)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.show()


def demonstrate_q_table(agent, env):
    """展示学习到的Q表"""
    print("\n" + "=" * 50)
    print("学习到的Q表示例")
    print("=" * 50)

    # 显示部分状态的Q值
    sample_states = list(agent.q_table.keys())[:10]  # 显示前10个状态

    for state in sample_states:
        q_values = agent.q_table[state]
        best_action = np.argmax(q_values)
        action_symbols = ['↑', '→', '↓', '←']

        print(f"状态 {state}: Q值 = [{q_values[0]:6.2f}, {q_values[1]:6.2f}, "
              f"{q_values[2]:6.2f}, {q_values[3]:6.2f}] | "
              f"最优动作: {action_symbols[best_action]}")

    print(f"\n总共学习了 {len(agent.q_table)} 个状态")
    print(f"最终探索率: {agent.epsilon:.4f}")


def test_agent_performance(agent, env, num_tests=10):
    """测试智能体性能"""
    print("\n" + "=" * 50)
    print("智能体性能测试")
    print("=" * 50)

    successes = 0
    total_steps = []

    for i in range(num_tests):
        state = env.reset()
        steps = 0
        original_epsilon = agent.epsilon
        agent.epsilon = 0  # 测试时关闭探索

        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            state = next_state
            steps += 1

            if done:
                if env.agent_pos == list(env.goal_pos):
                    successes += 1
                break

        agent.epsilon = original_epsilon
        total_steps.append(steps)

    success_rate = successes / num_tests * 100
    avg_steps = np.mean(total_steps)

    print(f"测试回合数: {num_tests}")
    print(f"成功率: {success_rate:.1f}%")
    print(f"平均步数: {avg_steps:.1f}")
    print(f"最佳步数: {min(total_steps)}")
    print(f"最差步数: {max(total_steps)}")


if __name__ == "__main__":
    # 训练智能体
    print("开始训练强化学习智能体...")
    agent, env, scores, success_rate = train_agent()

    # 绘制训练进度
    plot_training_progress(scores, success_rate)

    # 展示学习结果
    demonstrate_q_table(agent, env)

    # 测试智能体性能
    test_agent_performance(agent, env)

    # 创建动画演示
    print("\n" + "=" * 50)
    print("开始动画演示")
    print("=" * 50)

    animation = MazeAnimation(env, agent)
    anim = animation.create_animation(interval=800, save_gif=True)

    # 显示迷宫环境布局
    print("\n迷宫环境布局:")
    env.render()