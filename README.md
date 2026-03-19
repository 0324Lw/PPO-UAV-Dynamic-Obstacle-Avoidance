# 🚀 PPO-UAV-Dynamic-Obstacle-Avoidance

> 基于 Proximal Policy Optimization (PPO) 深度强化学习的无人机室内微观场景动态避障与路径规划算法。
> 本项目构建了一个高度自定义的 `gymnasium` 强化学习环境，支持混合动静态障碍物避障，并内置了完整的 PPO 训练管线与全自动的数据记录和 SCI 级别绘图系统。

## ✨ 核心特性 (Features)

* 🌍 **定制化微观环境**：构建了 20m x 20m 的室内地图，支持连续动作空间（线速度与角速度控制）。
* 🛸 **高复杂度混合避障**：环境内置 5 个静态障碍物和 3 个动态障碍物，障碍物形状随机生成（圆形或方形）。环境生成逻辑严格保证了起终点距离限制（20m~25m）以及障碍物之间的最小安全间距。
* 🧠 **PPO 强化学习底座**：采用 MLP 结构的 Actor-Critic 网络。集成了广义优势估计 (GAE)、梯度截断 (Clip)、熵正则化 (Entropy Coef) 以及动态学习率衰减等 PPO 核心机制。
* 📊 **运行状态归一化**：内置 `RunningMeanStd` 模块，对无人机获取的 32 维环境观测状态（包含目标距离、相对角度、各障碍物距离与速度等）进行实时在线均值方差归一化处理，极大提升训练稳定性。
* 📈 **全自动数据分析与可视化**：提供完备的 `DataLogger` 接口。训练过程中记录 11 项核心数据，结束后自动保存为 CSV 文件，并利用零相位双重指数平滑算法生成包含成功率、奖励值、损失函数等指标的高清 SCI 级别趋势图表。

## 🛠️ 环境依赖 (Requirements)

本项目基于 Python 开发，核心依赖库如下：

* `torch` (PyTorch)
* `gymnasium`
* `numpy`
* `pandas`
* `matplotlib`
* `scipy`

**快速安装指南 (Conda)：**
```bash
conda create -n uav_ppo python=3.10 -y
conda activate uav_ppo
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
conda install numpy pandas matplotlib scipy gymnasium -c conda-forge -y
```

![env_2_dynamic_simulation](https://github.com/user-attachments/assets/7f8dacae-38f3-4259-87c0-d4dfb472e321)
![env_1_dynamic_simulation](https://github.com/user-attachments/assets/e64b0169-48d8-4158-9d29-823e2ccfb6b3)
