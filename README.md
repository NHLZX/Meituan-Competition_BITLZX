# 美团第三届低空经济智能飞行管理挑战赛 (性能赛) - 视觉语言导航 (VLN) 解决方案

![ROS](https://img.shields.io/badge/ROS-Noetic-green) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![C++](https://img.shields.io/badge/C++-14-purple) ![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow)

本项目是针对** 美团第三届低空经济智能飞行管理挑战赛（性能赛）**的完整开源解决方案。本方案基于 ROS 框架，结合了基于规则的自然语言指令解析、YOLOv8 与 3D LiDAR 的多模态感知融合、鲁棒的有限状态机（FSM）决策以及高精度的 ArUco 终点对齐技术，实现了地面小车在复杂仿真环境下的自主视觉语言导航。

## 📖 赛题背景与任务要求

在无人机起飞机场中，我们希望通过自动化设备的交互来减少操作人员的机械重复劳动、提升工作效率和体验。

**核心任务：**
地面小车将载着无人机在取餐点完成取餐，然后根据工作人员的英文自然语言指令（如 *"move forward to the tree, turn right, go straight and stop at the traffic cone"*）以最短的路径到达目标点位，从而放飞无人机。

**任务规则与评分：**
* **起点与终点**：任务开始时，地面小车将被初始化在 `start_pose`，接收到指令后开始移动。小车的最终目的地是指定的 marker 板上方（尺寸为 1m * 1m）。
* **任务结束**：地面小车移动到指定 marker 板中心时，需发送 `/status` 指令表示任务结束。
* **成功判定**：发送状态后，系统将自动判定小车位置是否处于以 marker 板为中心的 2m 半径内，如是则任务成功，否则失败。移动过程中发生碰撞或超时均视为失败。
* **核心指标 (SPL)**：采用路径加权的成功率 (Success weighted by Path Length, SPL) 作为核心评分标准。

---

## 💻 平台与环境配置

比赛平台基于 UE 与 Gazebo 构建，高保真还原了地面小车搭载 RGB 相机、LiDAR 点云、里程计的传感与运动特性。

### 传感器与控制接口
* **语言指令**: `/instruction` (String，发送一次)
* **RGB 图像**: `/magv/camera/image_compressed/compressed` (10Hz)
* **雷达点云 (Livox MID360)**: `/magv/scan/3d` (10Hz)
* **里程计**: `/magv/odometry/gt` (25Hz)
* **控制方式**: 支持速度控制 (`/magv/omni_drive_controller/cmd_vel`) 或位姿控制 (`/magv/planning/pos_cmd`)。

### 环境要求
* **操作系统**: 推荐 Ubuntu 20.04 (方便本地使用 ROS Noetic 进行调试)。线上运行环境为 Ubuntu 22.04。
* **ROS**: ROS Noetic Desktop Full。
* **Docker**: 平台要求使用 Docker 容器化部署，并需安装 Nvidia Container Toolkit 以支持 GPU 加速。

---

## 🚀 核心解决方案 (Core Solution)

本方案采用**模块化**架构设计，摒弃了高延迟且难以解释的端到端黑盒模型，极大提升了系统的实时性和鲁棒性。主要包含以下四个核心节点：

### 1. 指令解析引擎 (Keyword Extractor)
我们实现了一个轻量级的 NLP 处理器 (`keyword_extractor_node.py`)，用于精准提取和优化指令：
* **词汇映射**：精确提取 8 种目标物体（如 `trash`, `bench`, `traffic cone` 等）和运动方向（`front`, `left`, `pass` 等）。
* **优先级与逻辑重构**：
  * **第一优先级**：预处理 `front` 命令，仅保留序列首个 `front`，去除后续冗余的前进指令，防止过度移动。
  * **第二优先级**：处理绕行 (`pass` + Object) 逻辑，当识别到绕过某物体时，自动将该物体及其动作从等待目标队列中移除，避免小车误停。
  * **第三优先级**：处理动作与物体的语序倒装（例如将 `["left", "right", "tree"]` 智能重排为 `["left", "tree", "right"]`），极大增强了对复杂自然语言语法的泛化能力。

### 2. 多模态感知系统 (YOLOv8 + 3D LiDAR Fusion)
位于 `yolo_detector.py` 的视觉融合节点：
* **2D/3D 联合检测**：使用 YOLOv8 (`best.pt`) 进行 2D 目标检测，并通过 `message_filters.ApproximateTimeSynchronizer` 严格同步压缩图像与 3D 点云数据。
* **点云深度估计**：将 3D 点云通过相机内参矩阵逆投影至 2D 边界框内，筛选距离包围盒中心最近的 7 个点，计算鲁棒的平均距离与 3D 质心。
* **🌟 核心创新：基于射线投射的 Mark 板定位**：
  针对地面平铺的终点 Mark 板，LiDAR 点云往往因掠射角问题而十分稀疏。我们针对类别 ID 9 (`mark`) 独立设计了**射线投射 (Ray-casting) 算法**。将相机光学中心的 3D 射线通过 TF 变换至全局 `map` 坐标系，并直接与地平面 (Z=0) 求交，从而计算出极其精准的地面 3D 坐标，完美解决了低角度下地面标志物测距不准的痛点。

### 3. 智能有限状态机 (FSM State Machine)
位于 `state_machine.cpp`，负责将解析出的指令队列转化为物理动作：
* **状态切换**：维护 `IDLE`（空闲）、`EXECUTING_ACTION`（执行动作）和 `WAITING_FOR_OBJECT`（等待物体）三种核心状态。
* **动态距离阈值**：对不同体积的物体采取不同的到达判定阈值（例如汽油桶 `barrel` 为 2.0m，广告牌 `billboard` 放宽至 4.5m，默认 3.5m），确保小车在最佳距离执行转弯或停止。
* **动作执行机制**：通过封装好的 ROS Service (`goalAdv/go_straight`, `turn_left` 等) 安全下发运动指令。

### 4. 终点精确停靠 (ArUco Alignment)
* 比赛指定的 marker 板为 OpenCV `DICT_4X4_50` 字典的标准 ArUco 码。
* **延迟激活策略**：为节省前端算力并防止误触，状态机仅在执行到**最后一条指令**时，才激活 ArUco 检测，并启用亚像素级角点精炼 (`CORNER_REFINE_SUBPIX`)。
* **高精度坐标系转换**：检测到 ArUco 后，结合预设的相机外参（包含 0.314 弧度的俯仰角补偿），将坐标解算至 `base_link`，再通过 `tf2_ros` 映射至 `map` 全局坐标系，累积 5 帧平滑后锁定最终目标位置，确保小车完美停靠在 2m 半径内。

---

## 🛠️ 快速启动 (Quick Start)

1. **拉取比赛基础 SDK 镜像**：
   ```bash
   docker pull uav-challenge.tencentcloudcr.com/uav_challenge_2025/uav_challenge:sdk
   ```

2. **编译本工作空间**：
   将本仓库代码克隆至工作空间的 `src` 目录下，并使用 `catkin_make` 编译：
   ```bash
   catkin_make
   source devel/setup.bash
   ```

3. **运行完整节点栈**：
   启动所有相关的感知与规控节点（请根据实际 launch 文件名进行调整）：
   ```bash
   roslaunch local_planner local_planner.launch
   roslaunch keyword_extractor keyword_extractor.launch
   roslaunch yolo_detect yolo_detect.launch
   ```

4. **提交至评测平台**：
   将你的镜像打包，并通过官方 `submit.sh` 脚本提交测试任务：
   ```bash
   ./submit.sh submit uav-challenge.tencentcloudcr.com/uav_challenge_2025/appkey:tag test.json 2
   ```

## 📜 许可证

本项目遵循 MIT 许可证开源。感谢美团低空经济挑战赛组委会提供的优秀仿真平台与支持！
