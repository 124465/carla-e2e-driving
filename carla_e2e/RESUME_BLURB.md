# 简历项目描述 + 成果（可直接粘贴）

## 项目标题建议
**端到端自动驾驶模仿学习（行为克隆 + DAgger）** —— 基于 PyTorch / PyBullet / ROS2

## 一句话摘要（放简历 bullet）
- 用 PyTorch 实现端到端驾驶策略（图像→转向/油门/刹车连续控制），在 PyBullet 仿真中以行为克隆(BC)训练，闭环评测车辆稳定行驶；用 DAgger 缓解模仿学习的分布漂移，并封装为 ROS2 节点接入机器人系统。

## 详细 bullet（按 STAR，挑 3-4 条）
1. **端到端网络设计**：设计轻量 CNN（TinyDrivingNet，~5 层卷积+两层 MLP），从单目前视图像直接回归 (steer, throttle, brake)；针对 4GB 显存约束采用 160×80 低分辨率 + 离屏渲染，单卡可训。
2. **专家数据采集与行为克隆**：在 PyBullet 中用纯追踪(pure-pursuit)控制器生成专家轨迹，采集 1200+ 帧 (图像, 动作) 对，用带权 MSE 训练；**闭环评测车辆 2/2 回合存活、单回合行驶 ~280m**，证明策略学到沿赛道行驶而非拟合退化解。
3. **DAgger 缓解分布漂移**：实现数据集聚合(DAgger)，用当前策略采样、专家在策略访问到的状态上补标签、全量重训；完成首轮聚合训练产出 `dagger_best_r00.pt`，闭环验证可部署。
4. **工程落地（ROS2）**：将策略封装为标准 `ament_python` ROS2 包（`carla_e2e_pkg`），订阅 `sensor_msgs/Image`、发布 `Float32MultiArray` 控制指令；通过 rclpy 上下文自测验证“图像→推理→控制消息”链路，可与 `carla_ros_bridge` 或自研 PyBullet 桥接对接真机/仿真。

## 技术栈
PyTorch · PyBullet（仿真后端，可切 CARLA）· ROS2 Jazzy (rclpy/cv_bridge) · 行为克隆 / DAgger · 纯追踪专家 · 离屏低分辨率渲染

## 可量化成果（面试/简历用）
- 闭环存活率 2/2、单回合行驶 ~280m（BC）/ ~136m（DAgger 首轮）
- 训练 loss 0.10 → 0.009（BC，30 epoch）
- 模型体积 ~3.2MB，推理为轻量 CNN，适合边缘部署
- 仿真后端默认 PyBullet（安装包 98MB，对 SSD/CPU 零压力）；CARLA 作为可选后端（需另下 8GB）

## 面试延伸（体现深度，见 INTERVIEW_TALK.md）
端到端 vs 模块化取舍、covariate shift 与 DAgger 数学直觉、多模态动作(扩散策略)、Sim-to-Real 域随机化、BEV 多传感器融合、时序建模(GRU/Transformer)。

## 配图（放项目 README / 作品集）
- `logs/train_curve.png` —— 训练/验证 loss 曲线
- `logs/eval/demo_ep1.gif` —— BC 策略闭环驾驶 demo（带 HUD：steer/throttle/speed）
- `logs/eval_dagger/demo_ep1.gif` —— DAgger 策略闭环 demo
