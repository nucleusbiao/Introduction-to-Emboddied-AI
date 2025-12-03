import numpy as np
import matplotlib.pyplot as plt


class ParticleFilterNoMotionModel:
    """缺乏准确运动模型时的粒子滤波器"""

    def __init__(self, num_particles, world_size):
        self.num_particles = num_particles
        self.world_width, self.world_height = world_size

        # 初始化粒子 - 均匀分布
        self.particles = np.zeros((num_particles, 3))
        self.particles[:, 0] = np.random.uniform(0, self.world_width, num_particles)
        self.particles[:, 1] = np.random.uniform(0, self.world_height, num_particles)
        self.particles[:, 2] = np.random.uniform(-np.pi, np.pi, num_particles)

        self.weights = np.ones(num_particles) / num_particles

    def predict_with_uncertainty(self, dt=0.1, position_noise=1.0, angle_noise=0.5):
        """
        使用不确定性模型进行预测
        当缺乏准确运动模型时，我们假设机器人可能在任意方向移动
        """
        for i in range(self.num_particles):
            # 添加随机扰动而不是基于运动模型的预测
            dx = np.random.normal(0, position_noise * dt)
            dy = np.random.normal(0, position_noise * dt)
            dtheta = np.random.normal(0, angle_noise * dt)

            self.particles[i, 0] += dx
            self.particles[i, 1] += dy
            self.particles[i, 2] += dtheta

            # 边界约束
            self.particles[i, 0] = np.clip(self.particles[i, 0], 0, self.world_width)
            self.particles[i, 1] = np.clip(self.particles[i, 1], 0, self.world_height)
            self.particles[i, 2] = self.normalize_angle(self.particles[i, 2])

    def normalize_angle(self, angle):
        """归一化角度到[-π, π]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def update(self, observations, landmarks, observation_noise=0.3):
        """更新步 - 基于观测数据更新权重"""
        for i in range(self.num_particles):
            # 计算预期观测
            expected_obs = []
            for landmark in landmarks:
                dist = np.linalg.norm(self.particles[i, :2] - landmark)
                expected_obs.append(dist)
            expected_obs = np.array(expected_obs)

            # 计算权重（观测似然）
            innovation = observations - expected_obs
            weight = np.exp(-0.5 * np.sum(innovation ** 2) / (observation_noise ** 2))
            self.weights[i] = max(weight, 1e-10)

        # 归一化权重
        self.weights /= np.sum(self.weights)

    def resample(self):
        """重采样"""
        cumulative_weights = np.cumsum(self.weights)
        step = 1.0 / self.num_particles
        positions = (np.arange(self.num_particles) + np.random.random()) * step

        indices = np.zeros(self.num_particles, dtype=int)
        j = 0
        for i in range(self.num_particles):
            while positions[i] > cumulative_weights[j]:
                j += 1
            indices[i] = j

        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate_pose(self):
        """估计位姿"""
        mean_x = np.average(self.particles[:, 0], weights=self.weights)
        mean_y = np.average(self.particles[:, 1], weights=self.weights)

        # 对于角度，使用循环统计
        sin_sum = np.average(np.sin(self.particles[:, 2]), weights=self.weights)
        cos_sum = np.average(np.cos(self.particles[:, 2]), weights=self.weights)
        mean_theta = np.arctan2(sin_sum, cos_sum)

        return np.array([mean_x, mean_y, mean_theta])


def correct_orientation_estimate(true_pose, estimated_pose, landmarks, observations):
    """
    修正方向估计，避免方向相反的问题
    通过比较观测数据的一致性来修正方向
    """
    # 计算真实位置到各个地标的角度
    true_angles = []
    for landmark in landmarks:
        dx = landmark[0] - true_pose[0]
        dy = landmark[1] - true_pose[1]
        angle = np.arctan2(dy, dx)
        true_angles.append(angle)

    # 计算估计位置到各个地标的角度
    est_angles = []
    for landmark in landmarks:
        dx = landmark[0] - estimated_pose[0]
        dy = landmark[1] - estimated_pose[1]
        angle = np.arctan2(dy, dx)
        est_angles.append(angle)

    # 计算角度差异
    angle_diff = 0
    count = 0
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            true_angle_diff = true_angles[i] - true_angles[j]
            est_angle_diff = est_angles[i] - est_angles[j]
            angle_diff += abs(self.normalize_angle(true_angle_diff - est_angle_diff))
            count += 1

    avg_angle_diff = angle_diff / count if count > 0 else 0

    # 如果角度差异很大，尝试翻转方向
    if avg_angle_diff > np.pi / 2:
        estimated_pose[2] = self.normalize_angle(estimated_pose[2] + np.pi)

    return estimated_pose


def demo_no_motion_model():
    """演示缺乏运动模型时的粒子滤波"""

    # 创建仿真环境
    world_size = (20, 15)
    landmarks = np.array([
        [2, 2], [2, 13], [18, 2], [18, 13], [10, 2], [10, 13]
    ])

    # 创建粒子滤波器（使用不确定性模型）
    pf = ParticleFilterNoMotionModel(1000, world_size)

    # 模拟真实机器人运动（但滤波器不知道这个模型）
    true_pose = np.array([3, 7, np.pi / 4])  # 初始角度设为45度

    # 存储历史数据
    true_trajectory = [true_pose.copy()]
    estimated_trajectory = [pf.estimate_pose()]
    errors = []

    # 存储实时状态图的数据，用于最后绘制
    real_time_fig_data = []

    # 设置交互模式用于动画
    plt.ion()

    # 创建动画窗口
    fig_anim, ax_anim = plt.subplots(figsize=(8, 6))

    # 存储前一时刻的位置用于计算运动方向
    prev_true_pos = true_pose[:2].copy()
    prev_est_pos = estimated_trajectory[0][:2].copy()

    for step in range(100):
        # 真实机器人运动（粒子滤波器不知道这个模型）
        v_true = 0.5  # 降低速度以便更好地观察
        w_true = 0.1  # 降低角速度
        dt = 0.1

        # 更新真实位置 - 使用更真实的运动模型
        true_pose[2] += w_true * dt

        # 计算新位置
        new_x = true_pose[0] + v_true * np.cos(true_pose[2]) * dt
        new_y = true_pose[1] + v_true * np.sin(true_pose[2]) * dt

        # 边界检测和反弹
        if new_x <= 0 or new_x >= world_size[0]:
            true_pose[2] = np.pi - true_pose[2]  # 水平方向反弹
        if new_y <= 0 or new_y >= world_size[1]:
            true_pose[2] = -true_pose[2]  # 垂直方向反弹

        # 应用位置更新
        true_pose[0] = np.clip(new_x, 0.5, world_size[0] - 0.5)
        true_pose[1] = np.clip(new_y, 0.5, world_size[1] - 0.5)

        # 归一化角度
        true_pose[2] = pf.normalize_angle(true_pose[2])

        # 生成观测数据（从真实位置到各个地标的距离）
        observations = []
        for landmark in landmarks:
            dist = np.linalg.norm(true_pose[:2] - landmark)
            noisy_dist = dist + np.random.normal(0, 0.2)  # 添加观测噪声
            observations.append(noisy_dist)
        observations = np.array(observations)

        # 粒子滤波步骤（缺乏准确运动模型）
        pf.predict_with_uncertainty(dt, position_noise=1.5, angle_noise=0.8)
        pf.update(observations, landmarks, observation_noise=0.3)

        # 每3步重采样一次
        if step % 3 == 0:
            pf.resample()

        # 记录数据
        true_trajectory.append(true_pose.copy())
        estimated_pose = pf.estimate_pose()

        # 基于运动方向修正估计方向
        current_true_pos = true_pose[:2]
        current_est_pos = estimated_pose[:2]

        # 计算真实运动方向
        true_movement = current_true_pos - prev_true_pos
        if np.linalg.norm(true_movement) > 0.01:  # 避免除以零
            true_direction = np.arctan2(true_movement[1], true_movement[0])

            # 计算估计运动方向
            est_movement = current_est_pos - prev_est_pos
            if np.linalg.norm(est_movement) > 0.01:
                est_direction = np.arctan2(est_movement[1], est_movement[0])

                # 如果估计方向与真实方向相差超过90度，修正方向
                direction_diff = abs(pf.normalize_angle(true_direction - est_direction))
                if direction_diff > np.pi / 2:
                    # 尝试翻转方向
                    estimated_pose[2] = pf.normalize_angle(estimated_pose[2] + np.pi)

        estimated_trajectory.append(estimated_pose)

        # 更新前一时刻的位置
        prev_true_pos = current_true_pos.copy()
        prev_est_pos = current_est_pos.copy()

        error = np.linalg.norm(true_pose[:2] - estimated_pose[:2])
        errors.append(error)

        # 每隔20步记录实时状态图的数据
        if step % 20 == 0:
            real_time_fig_data.append({
                'step': step,
                'true_pose': true_pose.copy(),
                'estimated_pose': estimated_pose.copy(),
                'particles': pf.particles.copy(),
                'error': error
            })

        # 实时动画更新
        ax_anim.clear()

        # 绘制地标
        ax_anim.scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=100, marker='s', label='Landmarks')

        # 绘制真实位置和方向
        ax_anim.plot(true_pose[0], true_pose[1], 'bo', markersize=10, label='True Position')
        # 绘制真实方向箭头
        arrow_length = 0.8
        ax_anim.arrow(true_pose[0], true_pose[1],
                      arrow_length * np.cos(true_pose[2]), arrow_length * np.sin(true_pose[2]),
                      head_width=0.3, head_length=0.2, fc='blue', ec='blue')

        # 绘制估计位置和方向
        ax_anim.plot(estimated_pose[0], estimated_pose[1], 'go', markersize=8, label='Estimated Position')
        # 绘制估计方向箭头
        ax_anim.arrow(estimated_pose[0], estimated_pose[1],
                      arrow_length * np.cos(estimated_pose[2]), arrow_length * np.sin(estimated_pose[2]),
                      head_width=0.3, head_length=0.2, fc='green', ec='green')

        # 绘制粒子
        particles = pf.particles
        ax_anim.scatter(particles[:, 0], particles[:, 1], c='green', s=1, alpha=0.3, label='Particles')

        # 绘制轨迹
        true_traj = np.array(true_trajectory)
        est_traj = np.array(estimated_trajectory)
        ax_anim.plot(true_traj[:, 0], true_traj[:, 1], 'b-', linewidth=2, alpha=0.7, label='True Trajectory')
        ax_anim.plot(est_traj[:, 0], est_traj[:, 1], 'g--', linewidth=2, alpha=0.7, label='Estimated Trajectory')

        ax_anim.set_xlim(0, world_size[0])
        ax_anim.set_ylim(0, world_size[1])
        ax_anim.grid(True, alpha=0.3)
        ax_anim.legend()
        ax_anim.set_title(f'Real-time Particle Filter Tracking\nStep {step}: Error = {error:.3f}m')
        ax_anim.set_xlabel('X')
        ax_anim.set_ylabel('Y')
        plt.tight_layout()
        plt.pause(0.01)  # 短暂暂停以更新动画

    # 关闭交互模式
    plt.ioff()
    plt.close(fig_anim)  # 关闭动画窗口

    print("Simulation completed! Generating result plots...")

    # 一次性创建所有结果图表
    # 图表1: 轨迹比较图
    plt.figure(1, figsize=(10, 8))
    true_traj = np.array(true_trajectory)
    est_traj = np.array(estimated_trajectory)

    plt.plot(true_traj[:, 0], true_traj[:, 1], 'b-', label='True Trajectory', linewidth=2)
    plt.plot(est_traj[:, 0], est_traj[:, 1], 'g--', label='Estimated Trajectory', linewidth=2)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=100, marker='s', label='Landmarks')

    # 标记起点和终点
    plt.plot(true_traj[0, 0], true_traj[0, 1], 'bo', markersize=8, label='Start (True)')
    plt.plot(est_traj[0, 0], est_traj[0, 1], 'go', markersize=6, label='Start (Est)')
    plt.plot(true_traj[-1, 0], true_traj[-1, 1], 'bs', markersize=8, label='End (True)')
    plt.plot(est_traj[-1, 0], est_traj[-1, 1], 'gs', markersize=6, label='End (Est)')

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Trajectory Comparison - Particle Filter Tracking')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.tight_layout()

    # 图表2: 误差曲线图
    plt.figure(2, figsize=(10, 6))
    plt.plot(errors, 'r-', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Position Error (m)')
    plt.title('Localization Error Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 图表3: 最终状态图
    plt.figure(3, figsize=(8, 6))
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=100, marker='s', label='Landmarks')

    # 绘制最终的真实位置和方向
    plt.plot(true_pose[0], true_pose[1], 'bo', markersize=10, label='True Position')
    arrow_length = 0.8
    plt.arrow(true_pose[0], true_pose[1],
              arrow_length * np.cos(true_pose[2]), arrow_length * np.sin(true_pose[2]),
              head_width=0.3, head_length=0.2, fc='blue', ec='blue')

    # 绘制最终的估计位置和方向
    plt.plot(estimated_pose[0], estimated_pose[1], 'go', markersize=8, label='Estimated Position')
    plt.arrow(estimated_pose[0], estimated_pose[1],
              arrow_length * np.cos(estimated_pose[2]), arrow_length * np.sin(estimated_pose[2]),
              head_width=0.3, head_length=0.2, fc='green', ec='green')

    # 绘制粒子
    particles = pf.particles
    plt.scatter(particles[:, 0], particles[:, 1], c='green', s=1, alpha=0.3, label='Particles')

    plt.xlim(0, world_size[0])
    plt.ylim(0, world_size[1])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(f'Final State: Error = {errors[-1]:.3f}m')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()

    # 图表4: 综合统计图
    plt.figure(4, figsize=(12, 4))

    # 子图1：误差分布
    plt.subplot(1, 3, 1)
    plt.hist(errors, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Error (m)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)

    # 子图2：累积误差
    plt.subplot(1, 3, 2)
    cumulative_errors = np.cumsum(errors)
    plt.plot(cumulative_errors, 'purple', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Error (m)')
    plt.title('Cumulative Localization Error')
    plt.grid(True, alpha=0.3)

    # 子图3：移动平均误差
    plt.subplot(1, 3, 3)
    window = 5
    moving_avg = np.convolve(errors, np.ones(window) / window, mode='valid')
    plt.plot(moving_avg, 'orange', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Moving Average Error (m)')
    plt.title(f'{window}-Step Moving Average Error')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # 图表5: 关键步骤状态图
    if real_time_fig_data:
        num_steps = len(real_time_fig_data)
        cols = min(3, num_steps)
        rows = (num_steps + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

        if num_steps == 1:
            axes = np.array([axes])
        elif rows == 1:
            axes = axes.reshape(1, -1)

        for idx, data in enumerate(real_time_fig_data):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]

            ax.scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=80, marker='s', label='Landmarks')
            ax.plot(data['true_pose'][0], data['true_pose'][1], 'bo', markersize=8, label='True Position')
            ax.plot(data['estimated_pose'][0], data['estimated_pose'][1], 'go', markersize=6,
                    label='Estimated Position')
            ax.scatter(data['particles'][:, 0], data['particles'][:, 1], c='green', s=1, alpha=0.3, label='Particles')

            ax.set_xlim(0, world_size[0])
            ax.set_ylim(0, world_size[1])
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Step {data["step"]}: Error = {data["error"]:.3f}m')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

            if idx == 0:
                ax.legend()

        # 隐藏多余的子图
        for idx in range(len(real_time_fig_data), rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].set_visible(False)

        plt.tight_layout()

    # 一次性显示所有结果图表
    plt.show()

    # 输出统计结果
    print(f"Final error: {errors[-1]:.3f}m")
    print(f"Average error: {np.mean(errors):.3f}m")
    print(f"Maximum error: {np.max(errors):.3f}m")
    print(f"Minimum error: {np.min(errors):.3f}m")
    print(f"Tracking performance: {'Good' if np.mean(errors) < 1.0 else 'Needs improvement'}")


if __name__ == "__main__":
    demo_no_motion_model()