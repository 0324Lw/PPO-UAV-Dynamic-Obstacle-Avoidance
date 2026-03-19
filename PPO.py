import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import time

# 導入我們剛剛編寫好的微觀環境與統一配置、數據日誌類
from env import UAVEnv, Config, DataLogger


# ==========================================
# 訓練技巧: 運行狀態歸一化模塊
# ==========================================
class RunningMeanStd:
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.n + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.n
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.n * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.n = tot_count


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# ==========================================
# 網絡結構: 傳統純 MLP 基線網絡
# (無圖網絡、無時序記憶、無物理約束)
# ==========================================
class MLP_ActorCritic(nn.Module):
    def __init__(self, obs_dim=32, act_dim=2):
        super(MLP_ActorCritic, self).__init__()

        # 簡單的 Actor 分支 (直接處理 32 维扁平特徵)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

        # 簡單的 Critic 分支
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


# ==========================================
# 主訓練循環 (純 MLP + PPO)
# ==========================================
# ==========================================
# MLP 主訓練循環 (包含全量數據攔截記錄)
# ==========================================
def train():
    cfg = Config()
    env = UAVEnv(cfg=cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = MLP_ActorCritic(obs_dim, act_dim).to(device)
    total_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)

    optimizer = optim.Adam(agent.parameters(), lr=cfg.LR, eps=1e-5)
    obs_rms = RunningMeanStd(shape=(obs_dim,))

    logger = DataLogger(save_dir="MLP_Results")
    recent_rewards = deque(maxlen=50)
    recent_success = deque(maxlen=50)
    recent_collisions = deque(maxlen=50)

    obs_buf = torch.zeros((cfg.NUM_STEPS, obs_dim)).to(device)
    act_buf = torch.zeros((cfg.NUM_STEPS, act_dim)).to(device)
    logprob_buf = torch.zeros((cfg.NUM_STEPS,)).to(device)
    rew_buf = torch.zeros((cfg.NUM_STEPS,)).to(device)
    val_buf = torch.zeros((cfg.NUM_STEPS,)).to(device)
    done_buf = torch.zeros((cfg.NUM_STEPS,)).to(device)

    global_step = 0
    episode_count = 0
    num_updates = cfg.TOTAL_TIMESTEPS // cfg.NUM_STEPS

    raw_obs, _ = env.reset()
    obs_rms.update(np.array([raw_obs]))
    obs_norm = np.clip((raw_obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0)

    # 用於記錄每輪詳細狀態的緩存列表
    ep_reward, ep_steps = 0, 0
    ep_v_list = []
    ep_w_list = []
    ep_min_dist_list = []
    current_v_loss, current_pg_loss = 0.0, 0.0

    start_time = time.time()

    for update in range(1, num_updates + 1):
        frac = 1.0 - (update - 1.0) / num_updates
        optimizer.param_groups[0]["lr"] = frac * cfg.LR

        for step in range(cfg.NUM_STEPS):
            global_step += 1
            obs_tensor = torch.FloatTensor(obs_norm).to(device)

            obs_buf[step] = obs_tensor
            done_buf[step] = 0

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_tensor.unsqueeze(0))

            val_buf[step] = value.flatten()
            act_buf[step] = action.flatten()
            logprob_buf[step] = logprob.flatten()

            cpu_action = action.cpu().numpy().flatten()
            mapped_action = np.tanh(cpu_action)

            next_raw_obs, reward, terminated, truncated, info = env.step(mapped_action)
            ep_reward += reward
            ep_steps += 1
            rew_buf[step] = reward

            # 收集單步動力學數據
            ep_v_list.append(env.v)
            ep_w_list.append(env.w)
            ep_min_dist_list.append(info.get('min_obs_dist', 0.0))
            obs_rms.update(np.array([next_raw_obs]))
            obs_norm = np.clip((next_raw_obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0)

            # 輪次結束，觸發全量數據計算與記錄
            if terminated or truncated:
                episode_count += 1
                is_success = 1 if info.get('is_success') == 1 else 0
                is_collision = 1 if info.get('is_collision') == 1 else 0

                # 計算動力學統計特徵
                avg_v = np.mean(ep_v_list) if ep_v_list else 0.0
                var_v = np.var(ep_v_list) if ep_v_list else 0.0
                avg_w = np.mean(np.abs(ep_w_list)) if ep_w_list else 0.0
                var_w = np.var(np.abs(ep_w_list)) if ep_w_list else 0.0
                path_curvature = np.mean(np.abs(ep_w_list)) if ep_w_list else 0.0
                avg_obs_dist = np.mean(ep_min_dist_list) if ep_min_dist_list else 0.0
                # 打包 11 項核心數據
                ep_data = {
                    'episode': episode_count,
                    'reward': ep_reward,
                    'steps': ep_steps,
                    'is_success': is_success,
                    'is_collision': is_collision,
                    'path_curvature': path_curvature,
                    'avg_v': avg_v,
                    'var_v': var_v,
                    'avg_w': avg_w,
                    'var_w': var_w,
                    'avg_obs_dist': avg_obs_dist,
                    'v_loss': current_v_loss,
                    'pg_loss': current_pg_loss
                }
                logger.append(ep_data)

                recent_rewards.append(ep_reward)
                recent_success.append(is_success)
                recent_collisions.append(is_collision)

                # 清空緩存
                ep_reward, ep_steps = 0, 0
                ep_v_list.clear()
                ep_w_list.clear()
                ep_min_dist_list.clear()
                raw_obs, _ = env.reset()
                obs_norm = np.clip((raw_obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0)
                done_buf[step] = 1

        # GAE 計算
        with torch.no_grad():
            next_value = agent.get_value(torch.FloatTensor(obs_norm).unsqueeze(0).to(device)).flatten()
            advantages = torch.zeros_like(rew_buf).to(device)
            lastgaelam = 0
            for t in reversed(range(cfg.NUM_STEPS)):
                nextnonterminal = 1.0 - (0 if t == cfg.NUM_STEPS - 1 else done_buf[t])
                nextvalues = next_value if t == cfg.NUM_STEPS - 1 else val_buf[t + 1]
                delta = rew_buf[t] + cfg.GAMMA * nextvalues * nextnonterminal - val_buf[t]
                advantages[t] = lastgaelam = delta + cfg.GAMMA * cfg.GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + val_buf

        # PPO 網絡更新
        b_inds = np.arange(cfg.NUM_STEPS)
        for epoch in range(cfg.NUM_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, cfg.NUM_STEPS, cfg.MINIBATCH_SIZE):
                end = start + cfg.MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs_buf[mb_inds], act_buf[mb_inds])
                logratio = newlogprob - logprob_buf[mb_inds]
                ratio = logratio.exp()
                mb_advantages = advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.CLIP_COEF, 1 + cfg.CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue.flatten() - returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ENT_COEF * entropy_loss + v_loss * 0.5

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

                current_pg_loss = pg_loss.item()
                current_v_loss = v_loss.item()

        if update % 5 == 0 and len(recent_rewards) > 0:
            elapsed = time.time() - start_time
            print(f"Upd: {update:04d} | Rew: {np.mean(recent_rewards):6.2f} | "
                  f"Succ: {np.mean(recent_success) * 100:5.1f}% | Coll: {np.mean(recent_collisions) * 100:5.1f}% | "
                  f"Time: {elapsed:.1f}s")
            start_time = time.time()

    logger.log_and_plot(exp_name="MLP")


if __name__ == "__main__":
    train()