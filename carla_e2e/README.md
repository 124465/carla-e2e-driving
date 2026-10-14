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

