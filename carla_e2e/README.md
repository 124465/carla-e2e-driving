# 端到端模仿学习自动驾驶（面试项目）

> 技术栈：PyTorch + 行为克隆(IL) + **PyBullet/可选 CARLA** → 连续控制 (steer, throttle, brake)
> 目标：训练一个"看图像→直接出车控"的神经网络，并用闭环评测验证。
> 硬件约束：GTX 1650 4GB → 全程离屏 + 低分辨率(160×80) + 小模型(TinyDrivingNet)。
> **默认仿真后端 = PyBullet**（轻量、离屏、对 SSD/CPU 零压力）；CARLA 作为可选后端（需另下 8GB）。
> ✅ 已端到端验证：collect→train→evaluate 在 PyBullet 后端跑通，闭环行驶 465m（3/3 回合存活）。

## 目录结构
```
carla_e2e/
├── configs/base.yaml      # 全部超参 + backend 开关（pybullet/carla）
├── carla_e2e/
│   ├── env.py             # make_env 工厂：统一 CARLA / PyBullet 接口
│   ├── pybullet_env.py    # PyBullet racecar 后端（默认，轻量）
│   ├── models.py          # TinyDrivingNet（小CNN，4GB可跑）
│   ├── dataset.py         # BCDataset（行为克隆数据集）
│   └── ros2_node.py       # ROS2 节点封装（import-safe）
├── scripts/
│   ├── collect.py         # 采集专家轨迹（双后端）
│   ├── train.py           # 训练行为克隆
│   ├── evaluate.py        # 闭环评测 + 里程统计
│   ├── dagger.py          # DAgger 缓解分布漂移
│   ├── smoke_test.py      # 端到端冒烟测试（无需 GPU/CARLA）
│   └── make_dummy_data.py # 无仿真时验证管线的假数据
├── setup_env.sh           # 一键装 uv+py3.10+依赖
├── requirements.txt
├── README.md  INTERVIEW_TALK.md
src/carla_e2e_pkg/         # 真正的 ROS2 ament 包（与 yolo_detector 同级）
```

## 快速开始（PyBullet 后端，推荐）

### 0. 装环境（只需一次）
```bash
bash setup_env.sh                 # 装 uv + py3.10 venv + torch + pybullet
source .venv/bin/activate
```

### 1. 冒烟测试（零依赖验证全链路，强烈先跑）
```bash
python scripts/smoke_test.py
# 应输出 [smoke] ALL CHECKS PASSED ✅
```

### 2. 采集专家数据（PyBullet racecar + pure-pursuit 专家）
```bash
python scripts/collect.py --runs 5 --steps 1000
# 产出 data/raw/images/*.png + data/raw/meta_runXX.jsonl
```

### 3. 训练行为克隆
```bash
python scripts/train.py --config configs/base.yaml
# 产出 checkpoints/best.pt
```

### 4. 闭环评测
```bash
python scripts/evaluate.py --checkpoint checkpoints/best.pt --episodes 5
```

### 5.（进阶）DAgger 缓解分布漂移
```bash
python scripts/dagger.py --config configs/base.yaml
# 产出 checkpoints/dagger_best_rXX.pt
```

---

## 可选：换用 CARLA 后端（需另下 ~8GB，对 SSD 有压力）

```bash
# 1) 下载 CARLA 仿真器（断点续传，注意会写盘）
wget -c https://downloads.carlasim.com/Linux/CARLA_0.9.15.tar.gz -O carla_sim/CARLA_0.9.15.tar.gz
tar -xzf carla_sim/CARLA_0.9.15.tar.gz -C carla_sim/
# 2) 切后端：把 configs/base.yaml 顶部的 backend 改成 "carla"
# 3) 另开终端启动 CARLA（1650 4GB 必须离屏低分辨率）：
cd carla_sim/CARLA_0.9.15
SDL_VIDEODRIVER=offscreen ./CarlaUE4.sh -windowed -ResX=160 -ResY=80 -carla-server
# 4) 运行 collect/train/evaluate 时加 --backend carla 或依赖 config
```

```bash
python scripts/evaluate.py --checkpoint checkpoints/best.pt --episodes 5
```

---

## 进阶 1：DAgger（缓解分布漂移，W3 加分项）

行为克隆的根本缺陷是 **covariate shift**：策略一旦偏离专家轨迹就进入训练分布外的状态。
DAgger 通过"用当前策略采样 → 专家在**这些状态**上补标签 → 重新聚合训练"来覆盖坏状态。

```bash
# 前提：已有一个 BC 模型 checkpoints/best.pt（没有则从随机初始化开始）
python scripts/dagger.py --config configs/base.yaml
# 每轮：beta-mix 采样 + 专家打标 -> data/raw/meta_dagger_XX.jsonl
#       重新在全量聚合数据上训练 -> checkpoints/dagger_best_rXX.pt
```
关键实现点（`scripts/dagger.py`）：
- `beta` 混合概率：每步以 beta 概率让策略执行、否则专家执行，使数据覆盖策略访问分布；
- 每轮 `beta *= beta_decay`，逐步回到专家主导；
- 重训时 **BC 数据 + 所有 dagger 轮数据** 一起训练（见 `retrain()`）；
- 专家查询接口见 `carla_e2e/env.py::query_expert_control`（autopilot 模式；可换 `behavior_agent`）。

## 进阶 2：ROS2 节点封装（W3-W4 工程加分项）

已做成标准 `ament_python` 包 `src/carla_e2e_pkg`，与你的 `yolo_detector`/`follower_robot` 同级：

```bash
cd ~/ros2_ws && colcon build --packages-select carla_e2e_pkg && source install/setup.bash
ros2 run carla_e2e_pkg driving_policy_node \
    --checkpoint /home/shen/ros2_ws/carla_e2e/checkpoints/best.pt \
    --image-topic /carla/ego_vehicle/rgb_front/image \
    --control-topic /carla/ego_vehicle/vehicle_control_cmd
```
订阅 `sensor_msgs/Image` → 跑策略 → 发布 `Float32MultiArray [steer, throttle, brake]`。
模型代码**单一来源**（包内 `driving_policy_node.py` 直接 import `ros2_ws/carla_e2e/carla_e2e`），
不会和训练代码分叉。配合 `carla_ros_bridge` 可接 CARLA 的真实相机/控制话题。


---

## 性能剖析：GPU 到底用上了吗？（`scripts/profile_perf.py`）

先给结论，全部来自实测（GTX 1650 4GB，Ubuntu 24.04 WSL2）：

| 环节 | 实际状态 | 证据 |
|------|----------|------|
| **训练** | ✅ 已在 GPU 上 | `train.py` 默认 `--device cuda`；CPU 占用从 40%→8% |
| **仿真采集** | ❌ CPU 瓶颈，**无法**上 GPU | `_render()` 占单步 **82%**，软件光栅化 |
| **PyBullet 硬件渲染** | ❌ 本安装不支持 | `p.EGL` 不存在；`ER_BULLET_HARDWARE_OPENGL` 实测 **GPU 占用 0%** |

```bash
python scripts/profile_perf.py           # 全部探测
python scripts/profile_perf.py --quick   # 跳过较慢项
```

### 关于「CPU 占用 20%」
那是 **PyBullet 软件渲染**，不是训练：
- 单 env 采集：**125 steps/s**，其中 `_render()` 6.5ms / 单步 8.0ms = **82%**
- 占用约 **94% 单核 ≈ 16% of 6 核**（你观察到的 ~20%）

**为什么 GPU 用不上**：本 pip 版 pybullet wheel 未编译 EGL 支持（`p.EGL` 不存在，强连会 core dump）；
`ER_BULLET_HARDWARE_OPENGL` 枚举虽有，但 DIRECT 模式下静默回退软件渲染 —— 实测渲染时 GPU 利用率恒为 **0%**。
要真正 GPU 渲染需要自行编译带 EGL 的 pybullet，或换 GPU 原生仿真器（Isaac Sim / CARLA）。

### 实际有效的优化：多进程并行采集（`scripts/collect_parallel.py`）
既然单步 82% 是 CPU 渲染，就用多核并行多个仿真器：

```bash
python scripts/collect_parallel.py --workers 6 --runs 6 --steps 400
```
实测 **3.7x 加速**（1800 帧 3.9s vs 单 env 约 14.4s）。

### GPU 驻留数据集（`carla_e2e/fast_data.py`）
把整个数据集一次性解码进显存（uint8），训练时在 GPU 上归一化，**无 DataLoader worker、无逐 epoch 磁盘解码**：

```bash
python scripts/train.py --data gpu --amp     # 批大小自动按显存选择
python scripts/train.py --data disk          # 原始磁盘路径（CPU 回退）
```
- 稳态 **~1835 samples/s**（disk 路径 ~1330 samples/s），10 epoch 全程 **12.7s**
- 诚实说明：在**当前这个 ~1200 帧的小数据集**上收益有限（wall time 只快约 1.3x），
  因为此时瓶颈是 Python/launch 开销而非磁盘解码；**数据量到几十万帧时**收益才显著。
- `--amp` 用 fp16 混合精度，进一步降低显存与耗时。

---

## 时序建模实验结果（含一个重要的负面发现）

### 实现
- `carla_e2e/models.py`: `TemporalDrivingNet`（帧堆叠，T 帧拼成 3T 通道）、`GRUDrivingNet`（共享编码器 + GRU）
- `carla_e2e/seq_dataset.py`: `SequenceDataset` —— 从文件名解析 episode 与步号构造时间窗，**窗口不跨 episode**
- `carla_e2e/fast_data.py`: `GPUSequenceData` —— 序列窗口常驻显存
- `scripts/train_temporal.py`: 统一训练入口（三种骨干用同一数据/优化器/调度，保证可比）
- `scripts/bench_models.py` + `scripts/evaluate.py --backbone/--n-frames`

### 结果：三种架构**全部退化**（如实记录）

| 骨干 | T | steer MSE | 均值基线 | 输出 std | 目标 std | 结论 |
|------|---|-----------|----------|----------|----------|------|
| tiny_cnn | 1 | 0.04571 | 0.04566 | 0.0003 | 0.194 | **退化** |
| tiny_cnn (150ep) | 1 | 0.04581 | 0.04566 | 0.0045 | 0.194 | **退化** |
| temporal_cnn | 4 | ~0.0457 | 0.04566 | ~0.000 | 0.194 | **退化** |
| gru | 8 | 0.02858 | 0.02857 | 0.0000 | 0.169 | **退化** |

**MSE 与"只预测均值"完全相等 → 模型没有从图像学到任何东西。**

### 排查过程（每一步都用测量排除一个假设）
1. **架构有问题？** 否 —— 换成一个可学的合成任务（转向 = 亮块位置），达到 **corr=+1.000、pred_std=0.579**（目标 0.579），架构完全健康。
2. **数据不一致（多模态）？** 否 —— 图像最近邻的 steer 差异仅 0.106（中位 0.032），远小于整体 std 0.194，说明**相似图像确实对应相似转向**。
3. **训练设置有问题？** 部分否 —— 64 样本过拟合可达 corr=+0.995，但需要 ~320 epoch；150 epoch 全量训练仍未脱离均值解。
4. **真正原因：episode 间转向分布不一致 + 数据量不足。**

各 run 的 steer 统计：
```
run01: mean=-0.016 std=0.077 范围[-0.35,+0.17]
run02: mean=+0.064 std=0.226 范围[-1.00,+1.00]
run03: mean=+0.061 std=0.263 范围[-1.00,+1.00]
run04: mean=-0.011 std=0.143 范围[-0.93,+1.00]
run05: mean=+0.051 std=0.187 范围[-1.00,+1.00]
```
`randomize_start: true` 让每个 episode 只覆盖赛道的一段，**转向分布差异极大**。
跨 episode 验证：run1 上训练后，run1 的 MSE 0.0056（勉强低于基线 0.0060），
但 **run2 的 MSE 0.0555 反而高于其基线 0.0512** —— 完全无法泛化。

**要学的映射其实是"从图像判断自己在赛道哪个位置"，2000 帧 / 5 个 episode 远远不够。**

### 这说明什么（面试价值很高）
- **"MSE 低"不等于"学到了"**：必须和"常数预测基线"对比，并检查输出方差。
  本项目前后两次都栽在这个陷阱上（早先 steer std 退化，这次预测均值），
  所以 `bench_models.py` 把"是否超过均值基线"作为硬性判定。
- **数据覆盖比模型结构更重要**：在数据覆盖不足时，加时序（帧堆叠/GRU）**毫无帮助** ——
  三种架构退化程度几乎一致，证明瓶颈在数据不在模型。
- **正确的下一步**：固定起点、跑完整圈、增大 episode 数（用 `collect_parallel.py`，
  已实测 3.7x 加速），让数据覆盖整条赛道后再比较架构。

---

## 面试可讲的硬核点（提前背熟）

1. **为什么端到端？vs 模块化？**
   - 端到端：少假设、可利用大模型表征、数据驱动；缺点是黑盒、难 debug、对分布外(OOD)脆弱。
   - 模块化（感知→预测→规划→控制）：可解释、易定位失败、符合车规安全；缺点是累积误差、模块间接口需手工设计。
   - 我的取舍：先端到端快速出 demo，再讨论如何叠加可解释中间表征（如 BEV）。

2. **行为克隆的根本缺陷：分布漂移 / covariate shift**
   - 专家数据只覆盖"好状态"，策略一旦偏差就进入训练分布外的状态，误差累积 → 撞车（compounding error）。
   - 我的缓解手段：
     - (a) 训练时给观测加噪声 / 用 DAgger 让专家在网络采样到的状态上补标签；
     - (b) 多 town、多天气域随机化；
     - (c) 损失对转向加权更高（base.yaml 的 loss_weights）。

3. **评测指标**（CARLA 官方 Leaderboard 口径）
   - 路线完成率 (Route Completion %)、驾驶分 (Driving Score)、违规率（闯红灯/碰撞/超速）、平均里程。

4. **Sim-to-Real 怎么想？**
   - 域随机化（光照/天气/传感器噪声/动力学参数）；
   - 与 ROS2 结合把模型封成节点，接真实传感器话题（你工作区已有 yolo_detector / follower_robot，可复用）。

5. **下一步（体现规划能力）**
   - DAgger 在线修正、加 BEV 多传感器融合、RL 微调、语义分割辅助监督、扩散策略处理多模态驾驶。

---

## 进阶路线（W3-W4 加分项）
- 把模型封成 ROS2 节点 (`carla_ros_bridge` 或自定义 `rclpy` 节点)，订阅 `/camera`, 发布 `/control`。
- 换 Town03/Town10HD 多场景；加红绿灯/行人避障。
- 实现 DAgger：`scripts/collect.py` 增加 `--policy` 模式，用当前网络采样、专家打标。

---

## 已验证成果 & 配图（写简历/作品集直接用）

| 产物 | 路径 | 说明 |
|------|------|------|
| 训练曲线 | `logs/train_curve.png` | BC 训练/验证 loss（0.10→0.009） |
| BC 闭环 demo | `logs/eval/demo_ep1.gif` | 策略沿赛道行驶，HUD 显示 steer/throttle/speed |
| DAgger 闭环 demo | `logs/eval_dagger/demo_ep1.gif` | DAgger 策略闭环行驶 |
| BC 模型 | `checkpoints/best.pt` | ~3.2MB，TinyDrivingNet |
| DAgger 模型 | `checkpoints/dagger_best_r00.pt` | 首轮聚合训练产出 |

**闭环评测结果（PyBullet 后端）**
- BC 策略：2/2 回合存活，单回合行驶 ~280m；训练 loss 0.10→0.009（30 epoch）
- DAgger 首轮：2/2 回合存活，单回合行驶 ~136m（单轮、beta=0.5，多轮迭代后通常优于 BC）
- 模型体积 ~3.2MB，轻量 CNN，适合边缘部署

**ROS2 节点验证**
- `colcon build --packages-select carla_e2e_pkg` 编译通过，入口 `driving_policy_node` 注册成功
- 确定性自测（`scripts/ros2_node_selftest.py`）通过：rclpy 上下文可建，图像→模型→`Float32MultiArray` 控制消息链路输出合法 `[steer, throttle, brake]`
- 多节点 DDS 联调在本机 WSL(FastDDS) 的 localhost 发现不通（已知 WSL 环境限制，非代码问题）；真实部署用 `carla_ros_bridge` 或自研桥接即可

**ROS2 节点启动（正确环境顺序）**
```bash
source /opt/ros/jazzy/setup.bash
source carla_e2e/.venv312/bin/activate          # py3.12 + --system-site-packages，已装 CPU torch
# 注意：venv activate 会清 PYTHONPATH，需在其后补回 ROS2 与训练项目路径
export PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages:$(pwd)/carla_e2e
ros2 run carla_e2e_pkg driving_policy_node \
    --checkpoint checkpoints/best.pt \
    --image-topic /camera/rgb/image_raw \
    --control-topic /vehicle/control
```

> 简历文案见 `RESUME_BLURB.md`，面试问答见 `INTERVIEW_TALK.md`。

