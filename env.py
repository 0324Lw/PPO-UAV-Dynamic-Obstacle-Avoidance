import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


# ==========================================
# 1. 统一参数配置类 Config
# ==========================================
class Config:
    """
    室内微观场景 (20m x 20m) 动态避障环境配置
    """
    # 1.1 环境与动力学参数
    MAP_SIZE = 20.0  # 室内地图大小 20m x 20m
    DT = 0.1  # 仿真步长 (s) - 设定为0.1s，使得200步=20s飞行时间
    MAX_V = 1.5  # 最大线速度 (m/s)
    MAX_W = 2.0  # 最大角速度 (rad/s)
    SAFE_RADIUS_START = 3.0  # 起点周围安全区 (无障碍物)
    MIN_OBS_DIST = 3.0  # 同类型障碍物之间的最小绝对间距 (m)

    # 1.2 障碍物参数
    NUM_STATIC_OBS = 5  # 静态障碍物数量
    NUM_DYN_OBS = 3  # 动态障碍物数量
    OBS_SIZE_MIN = 1.5  # 障碍物最小尺寸 (半径或半边长)
    OBS_SIZE_MAX = 1.5  # 障碍物最大尺寸
    DYN_SPEED_MIN = 0.4 # 动态障碍物最小移速 (m/s)
    DYN_SPEED_MAX = 0.5  # 动态障碍物最大移速 (m/s)，不能快于无人机

    # 1.3 训练超参数
    MAX_STEPS = 250  # 最大步数，限制在 200~250 之间
    TOTAL_TIMESTEPS = 2500000
    NUM_STEPS = 2048  # PPO 收集步数
    NUM_EPOCHS = 10
    MINIBATCH_SIZE = 64
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_COEF = 0.2
    LR = 3e-4
    ENT_COEF = 0.02

    # 1.4 奖励函数系数 (单步控制在 [-2, 2], 总分控制在 [-200, 200])
    COEFF_STEP = -0.05  # 步数惩罚 (鼓励尽快到达)
    COEFF_FORWARD = 0.8  # 前进奖励 (基于距离缩减)
    COEFF_BACKWARD = -1.5  # 后退/绕远路平方惩罚
    COEFF_DIR = 0.1  # 方向对齐奖励 (朝向终点)
    COEFF_SMOOTH = -0.05  # 动作突变惩罚 (平方惩罚)
    COEFF_DANGER = -1.0  # 危险惩罚 (距离障碍物边界<2m时触发)
    COEFF_COLLISION = -100.0  # 碰撞一次性惩罚
    COEFF_GOAL = 200.0  # 到达终点大额奖励


# ==========================================
# 2. 强化学习环境类 UAVEnv
# ==========================================
class UAVEnv(gym.Env):
    def __init__(self, cfg=Config()):
        super(UAVEnv, self).__init__()
        self.cfg = cfg

        # 2.1 动作空间：[v, w] 归一化到 [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 2.1 状态空间：7维核心状态 + 5个障碍物 * 5维 = 32维
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32)

        self.pos = np.zeros(2)
        self.yaw = 0.0
        self.v = 0.0
        self.w = 0.0
        self.prev_action = np.zeros(2)
        self.step_count = 0

        self.static_obs = []
        self.dynamic_obs = []
        self.start_pos = np.zeros(2)
        self.goal_pos = np.zeros(2)

    def _get_boundary_dist(self, pos, obs):
        """2.3 碰撞检测核心：计算无人机质心到障碍物边界的最短距离"""
        if obs['shape'] == 'circle':
            dist = np.linalg.norm(pos - obs['pos']) - obs['size']
        else:  # square (size 为半边长)
            dx = np.abs(pos[0] - obs['pos'][0]) - obs['size']
            dy = np.abs(pos[1] - obs['pos'][1]) - obs['size']
            if dx > 0 and dy > 0:
                dist = np.sqrt(dx ** 2 + dy ** 2)  # 在对角线外部
            else:
                dist = max(dx, dy)  # 在正上下或正左右
        return dist

    def _generate_scenario(self):
        """2.2 地形生成逻辑：确保起终点距离、安全区、均匀分布、相交路线与多样性"""

        # =========================================
        # 解决要求 4：保证起终点限制距离在（20, 25）之间
        # =========================================
        while True:
            # 强制在对角象限采样，以满足 20~25m 的苛刻距离要求
            if np.random.rand() > 0.5:
                s_pos = np.random.uniform([1.0, 1.0], [5.0, 5.0])
                g_pos = np.random.uniform([15.0, 15.0], [19.0, 19.0])
            else:
                s_pos = np.random.uniform([1.0, 15.0], [5.0, 19.0])
                g_pos = np.random.uniform([15.0, 1.0], [19.0, 5.0])

            dist = np.linalg.norm(g_pos - s_pos)
            if 20.0 <= dist <= 25.0:
                self.start_pos = s_pos
                self.goal_pos = g_pos
                break

        self.static_obs = []
        self.dynamic_obs = []

        # 维护一个全局位置列表，用于统一计算所有障碍物的初始间距
        existing_positions = []

        # 九宫格划分用于静态障碍物
        grid_w = self.cfg.MAP_SIZE / 3.0
        grids = [(i, j) for i in range(3) for j in range(3)]
        np.random.shuffle(grids)

        # =========================================
        # 静态障碍物生成
        # =========================================
        for i in range(self.cfg.NUM_STATIC_OBS):
            shape = 'circle' if np.random.rand() > 0.5 else 'square'
            size = np.random.uniform(self.cfg.OBS_SIZE_MIN, self.cfg.OBS_SIZE_MAX)
            placed = False

            for _ in range(100):
                gx, gy = grids[i % len(grids)]
                # 解决要求 1：限制采样范围，确保加上size后也不会越界
                cx = np.random.uniform(gx * grid_w + size, (gx + 1) * grid_w - size)
                cy = np.random.uniform(gy * grid_w + size, (gy + 1) * grid_w - size)
                pos = np.array([cx, cy])

                # 起终点安全区检测
                if np.linalg.norm(pos - self.start_pos) < self.cfg.SAFE_RADIUS_START + size: continue
                if np.linalg.norm(pos - self.goal_pos) < 2.0 + size: continue

                # 解决要求 5：障碍物之间的最小距离约束
                conflict = False
                for ex_pos, ex_size in existing_positions:
                    if np.linalg.norm(pos - ex_pos) < size + ex_size + self.cfg.MIN_OBS_DIST:
                        conflict = True;
                        break

                if not conflict:
                    self.static_obs.append(
                        {'pos': pos, 'size': size, 'shape': shape, 'type': 0 if shape == 'circle' else 1})
                    existing_positions.append((pos, size))
                    placed = True
                    break

            if not placed:  # 兜底全局随机插入
                for _ in range(100):
                    cx = np.random.uniform(size, self.cfg.MAP_SIZE - size)
                    cy = np.random.uniform(size, self.cfg.MAP_SIZE - size)
                    pos = np.array([cx, cy])
                    if np.linalg.norm(pos - self.start_pos) < self.cfg.SAFE_RADIUS_START + size: continue
                    if np.linalg.norm(pos - self.goal_pos) < 2.0 + size: continue

                    conflict = False
                    for ex_pos, ex_size in existing_positions:
                        if np.linalg.norm(pos - ex_pos) < size + ex_size + self.cfg.MIN_OBS_DIST:
                            conflict = True;
                            break
                    if not conflict:
                        self.static_obs.append(
                            {'pos': pos, 'size': size, 'shape': shape, 'type': 0 if shape == 'circle' else 1})
                        existing_positions.append((pos, size))
                        break

        # =========================================
        # 动态障碍物生成
        # =========================================
        path_vec = self.goal_pos - self.start_pos
        path_dir = path_vec / np.linalg.norm(path_vec)
        base_angle = np.arctan2(path_dir[1], path_dir[0])

        for i in range(self.cfg.NUM_DYN_OBS):
            shape = 'circle' if np.random.rand() > 0.5 else 'square'
            size = np.random.uniform(self.cfg.OBS_SIZE_MIN, self.cfg.OBS_SIZE_MAX)
            speed = np.random.uniform(self.cfg.DYN_SPEED_MIN, self.cfg.DYN_SPEED_MAX)

            placed = False
            for _ in range(100):
                # 在起终点连线上随机取一个交点
                u = np.random.uniform(0.2, 0.8)
                intersect_pt = self.start_pos + u * path_vec

                # 解决要求 3：生成随机偏转角 (30度~150度)，打破完全平行的规律
                angle_offset = np.random.uniform(np.pi / 6, 5 * np.pi / 6)
                if np.random.rand() > 0.5: angle_offset = -angle_offset

                move_angle = base_angle + angle_offset
                move_dir = np.array([np.cos(move_angle), np.sin(move_angle)])

                # 确定轨迹长度并计算端点
                sweep_len = np.random.uniform(3.0, 6.0)
                start_p = intersect_pt + move_dir * sweep_len
                end_p = intersect_pt - move_dir * sweep_len

                # 解决要求 1：轨迹端点裁剪，确保动态障碍物在运动时绝不出界
                start_p = np.clip(start_p, size, self.cfg.MAP_SIZE - size)
                end_p = np.clip(end_p, size, self.cfg.MAP_SIZE - size)

                # 安全区检测
                if np.linalg.norm(start_p - self.start_pos) < self.cfg.SAFE_RADIUS_START + size: continue
                if np.linalg.norm(start_p - self.goal_pos) < 2.0 + size: continue

                # 解决要求 2：动态障碍物的起始点也必须严格满足距离约束
                conflict = False
                for ex_pos, ex_size in existing_positions:
                    if np.linalg.norm(start_p - ex_pos) < size + ex_size + self.cfg.MIN_OBS_DIST:
                        conflict = True;
                        break

                if not conflict:
                    self.dynamic_obs.append({
                        'start': start_p, 'end': end_p, 'pos': np.copy(start_p),
                        'size': size, 'speed': speed, 'shape': shape,
                        'type': 2 if shape == 'circle' else 3
                    })
                    existing_positions.append((start_p, size))  # 加入全局记录
                    placed = True
                    break

            if not placed:  # 兜底全局随机轨迹
                for _ in range(100):
                    start_p = np.random.uniform(size, self.cfg.MAP_SIZE - size, 2)
                    end_p = np.random.uniform(size, self.cfg.MAP_SIZE - size, 2)

                    if np.linalg.norm(start_p - self.start_pos) < self.cfg.SAFE_RADIUS_START + size: continue
                    if np.linalg.norm(start_p - self.goal_pos) < 2.0 + size: continue

                    conflict = False
                    for ex_pos, ex_size in existing_positions:
                        if np.linalg.norm(start_p - ex_pos) < size + ex_size + self.cfg.MIN_OBS_DIST:
                            conflict = True;
                            break

                    if not conflict:
                        self.dynamic_obs.append({
                            'start': start_p, 'end': end_p, 'pos': np.copy(start_p),
                            'size': size, 'speed': speed, 'shape': shape,
                            'type': 2 if shape == 'circle' else 3
                        })
                        existing_positions.append((start_p, size))
                        break

    def _update_dynamic(self):
        for obs in self.dynamic_obs:
            dist = np.linalg.norm(obs['end'] - obs['start'])
            cycle_time = dist / obs['speed'] / self.cfg.DT
            phase = (self.step_count % (2 * cycle_time)) / cycle_time
            if phase <= 1.0:
                obs['pos'] = obs['start'] + phase * (obs['end'] - obs['start'])
            else:
                obs['pos'] = obs['end'] + (phase - 1.0) * (obs['start'] - obs['end'])

    def _get_obs(self):
        """2.1 提取 32 维网络输入特征"""
        # 1. 核心状态 (7维)
        dist_goal = np.linalg.norm(self.goal_pos - self.pos)
        angle_goal = np.arctan2(self.goal_pos[1] - self.pos[1], self.goal_pos[0] - self.pos[0])
        rel_angle_goal = (angle_goal - self.yaw + np.pi) % (2 * np.pi) - np.pi
        alpha_v = np.cos(rel_angle_goal)  # 速度与目标夹角余弦

        s_agent = [dist_goal / self.cfg.MAP_SIZE, rel_angle_goal / np.pi, 0.0,
                   self.v / self.cfg.MAX_V, alpha_v, 0.0, self.w / self.cfg.MAX_W]

        # 2. 障碍物状态 (收集所有并取最近的5个)
        obs_list = []
        for o in self.static_obs + self.dynamic_obs:
            dist = self._get_boundary_dist(self.pos, o)
            angle = np.arctan2(o['pos'][1] - self.pos[1], o['pos'][0] - self.pos[0])
            rel_angle = (angle - self.yaw + np.pi) % (2 * np.pi) - np.pi
            v_rel = o.get('speed', 0.0)  # 简化：由于没有朝向，使用标量速度
            obs_list.append([dist, rel_angle, 0.0, v_rel, o['type']])

        obs_list.sort(key=lambda x: x[0])  # 按距离排序
        s_obs = []
        for i in range(5):
            if i < len(obs_list):
                o = obs_list[i]
                s_obs.extend([o[0] / self.cfg.MAP_SIZE, o[1] / np.pi, o[2], o[3] / self.cfg.MAX_V, o[4]])
            else:
                s_obs.extend([1.0, 0.0, 0.0, 0.0, -1])  # 补零

        return np.array(s_agent + s_obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._generate_scenario()
        self.pos = np.copy(self.start_pos)
        self.yaw = np.arctan2(self.goal_pos[1] - self.start_pos[1], self.goal_pos[0] - self.start_pos[0])
        self.v = 0.0
        self.w = 0.0
        self.prev_action = np.zeros(2)
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        # 动作解算：a[0]映射到 [0, MAX_V], a[1]映射到 [-MAX_W, MAX_W]
        self.v = (action[0] + 1.0) / 2.0 * self.cfg.MAX_V
        self.w = action[1] * self.cfg.MAX_W

        prev_dist = np.linalg.norm(self.goal_pos - self.pos)

        # 运动学更新
        self.yaw += self.w * self.cfg.DT
        self.yaw = (self.yaw + np.pi) % (2 * np.pi) - np.pi
        self.pos[0] += self.v * np.cos(self.yaw) * self.cfg.DT
        self.pos[1] += self.v * np.sin(self.yaw) * self.cfg.DT

        self.step_count += 1
        self._update_dynamic()

        # 2.4 奖励函数组件计算
        r_step = self.cfg.COEFF_STEP

        current_dist = np.linalg.norm(self.goal_pos - self.pos)
        dist_diff = prev_dist - current_dist
        max_possible_diff = self.cfg.MAX_V * self.cfg.DT
        norm_diff = dist_diff / max_possible_diff

        r_forward = self.cfg.COEFF_FORWARD * norm_diff if norm_diff > 0 else 0.0
        r_backward = self.cfg.COEFF_BACKWARD * (norm_diff ** 2) if norm_diff <= 0 else 0.0

        target_yaw = np.arctan2(self.goal_pos[1] - self.pos[1], self.goal_pos[0] - self.pos[0])
        yaw_diff = (target_yaw - self.yaw + np.pi) % (2 * np.pi) - np.pi
        r_dir = self.cfg.COEFF_DIR * (1.0 - np.abs(yaw_diff) / np.pi)

        act_diff = action - self.prev_action
        r_smooth = self.cfg.COEFF_SMOOTH * np.sum(act_diff ** 2)
        self.prev_action = np.copy(action)

        # 碰撞与危险检测
        r_danger = 0.0
        r_collision = 0.0
        terminated = False
        info = {'is_success': 0, 'is_collision': 0}

        min_obs_dist = float('inf')
        for o in self.static_obs + self.dynamic_obs:
            d = self._get_boundary_dist(self.pos, o)
            min_obs_dist = min(min_obs_dist, d)
            if d <= 0:
                r_collision = self.cfg.COEFF_COLLISION
                terminated = True
                info['is_collision'] = 1
                info['reason'] = 'collision'
                break
            elif d < 1.5:
                # 危险惩罚，距离越近惩罚越大，且裁剪上限防止单步爆炸
                penalty = self.cfg.COEFF_DANGER * (2.0 - d) ** 2
                r_danger += max(penalty, -2.0)

        # 越界检测
        if self.pos[0] < 0 or self.pos[0] > self.cfg.MAP_SIZE or self.pos[1] < 0 or self.pos[1] > self.cfg.MAP_SIZE:
            r_collision = self.cfg.COEFF_COLLISION
            terminated = True
            info['is_collision'] = 1
            info['reason'] = 'out_of_bounds'

        # 到达目标
        r_goal = 0.0
        if current_dist < 1.0:
            r_goal = self.cfg.COEFF_GOAL
            terminated = True
            info['is_success'] = 1
            info['reason'] = 'goal_reached'

        truncated = self.step_count >= self.cfg.MAX_STEPS

        # 汇总奖励并裁剪单步
        step_reward_unclipped = r_step + r_forward + r_backward + r_dir + r_smooth + r_danger
        step_reward = np.clip(step_reward_unclipped, -2.0, 2.0)

        # 终止状态大奖励不裁剪
        total_reward = step_reward + r_collision + r_goal

        info['reward_comps'] = {
            'r_step': r_step, 'r_forward': r_forward, 'r_backward': r_backward,
            'r_dir': r_dir, 'r_smooth': r_smooth, 'r_danger': r_danger,
            'r_collision': r_collision, 'r_goal': r_goal
        }
        info['min_obs_dist'] = min_obs_dist

        return self._get_obs(), total_reward, terminated, truncated, info


# ==========================================
# 3. 统一数据记录与绘图类 DataLogger
# ==========================================
class DataLogger:
    """
    负责将训练数据保存为 CSV 并绘制 SCI 级别趋势图
    """

    def __init__(self, save_dir="Results"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.data_list = []

    def append(self, episode_data):
        """传入字典格式的单轮数据"""
        self.data_list.append(episode_data)

    def log_and_plot(self, exp_name="Experiment"):
        if not self.data_list:
            return

        df = pd.DataFrame(self.data_list)
        csv_path = os.path.join(self.save_dir, f"{exp_name}_log.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"[*] 数据已保存至: {csv_path}")

        # 绘图逻辑
        metrics = [col for col in df.columns if col != 'episode']
        n_metrics = len(metrics)
        fig, axs = plt.subplots((n_metrics + 1) // 2, 2, figsize=(12, 4 * ((n_metrics + 1) // 2)))
        axs = axs.flatten()

        for i, metric in enumerate(metrics):
            ax = axs[i]
            y_raw = df[metric].values

            if metric in ['is_success', 'is_collision']:
                y_raw = y_raw * 100
                ylabel = f"{metric.title()} Rate (%)"
            else:
                ylabel = metric.title()

            # 零相位双重指数平滑生成趋势线 (参考上一篇文章优化)
            s = pd.Series(y_raw).interpolate(method='linear').bfill().ffill()
            s1 = s.ewm(alpha=0.01, adjust=True).mean()
            y_trend = s1[::-1].ewm(alpha=0.01, adjust=True).mean()[::-1].values

            ax.plot(df['episode'], y_raw, alpha=0.2, color='gray', label='Raw')
            ax.plot(df['episode'], y_trend, alpha=1.0, color='#D62728', linewidth=2.5, label='Trend')

            ax.set_xlabel('Episode')
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()

        for j in range(n_metrics, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        img_path = os.path.join(self.save_dir, f"{exp_name}_trends.png")
        plt.savefig(img_path, dpi=300)
        plt.close()
        print(f"[*] 趋势图已保存至: {img_path}")