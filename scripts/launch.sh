#!/bin/bash

# ===================================================================================
# 智航2025 仿真与测试任务一键启动脚本
# 功能：自动为每个主要模块打开新终端并执行相应命令。
# 使用：
# 1. 保存为 launch_all.sh
# 2. chmod +x launch_all.sh
# 3. ./launch_all.sh
# 注意：此脚本依赖 gnome-terminal。如果您的系统使用不同的终端，
#      例如 Konsole 或 xterm，您需要相应地修改命令。
# ===================================================================================

echo "🚀 [主进程] 欢迎使用智航2025一键启动脚本！"
echo "🕒 将在5秒后开始启动所有进程..."
sleep 5

# --- 终端 1: 启动 Gazebo 仿真环境和 PX4 ---
echo "➡️ [1/12] 正在新终端中启动 Gazebo 仿真环境 (roslaunch)..."
gnome-terminal -- bash -c "echo '✅ [终端1] Gazebo & PX4 启动中...'; roslaunch px4 zhihang2025.launch; exec bash"
echo "🕒 等待15秒，确保 Gazebo 和 PX4 完全启动..."
sleep 30

# --- 终端 2 & 3: 通信脚本 (根据您的注释，这些可能不是必需的) ---
echo "➡️ [2/12] 正在新终端中启动 VTOL 通信脚本..."
# 注意：根据您的说明，以下通信脚本可能不是必需的，如果不需要可以注释掉下一行
gnome-terminal -- bash -c "echo '✅ [终端2] vtol_communication 启动中...'; cd ~/XTDrone/communication/; python3 vtol_communication.py standard_vtol 0; exec bash"
sleep 5

echo "➡️ [3/12] 正在新终端中启动 Iris 通信脚本..."
# 注意：根据您的说明，以下通信脚本可能不是必需的，如果不需要可以注释掉下一行
gnome-terminal -- bash -c "echo '✅ [终端3] multirotor_communication 启动中...'; cd ~/XTDrone/communication/; python3 multirotor_communication.py iris 0; exec bash"
sleep 5

# --- 终端 4 & 5: 获取位姿真值 ---
echo "➡️ [4/12] 正在新终端中获取 VTOL 位姿真值..."
gnome-terminal -- bash -c "echo '✅ [终端4] VTOL Pose Ground Truth 启动中...'; cd ~/XTDrone/sensing/pose_ground_truth/; python3 get_local_pose.py standard_vtol 1; exec bash"
sleep 5

echo "➡️ [5/12] 正在新终端中获取 Iris 位姿真值..."
gnome-terminal -- bash -c "echo '✅ [终端5] Iris Pose Ground Truth 启动中...'; cd ~/XTDrone/sensing/pose_ground_truth/; python3 get_local_pose.py iris 1; exec bash"
sleep 5

# --- 终端 6: 待救援目标移动 ---
echo "➡️ [6/12] 正在新终端中启动目标移动控制脚本..."
gnome-terminal -- bash -c "echo '✅ [终端6] 目标移动脚本 (zhihang_control_targets.py) 启动中...'; cd ~/XTDrone/zhihang2025; python3 zhihang_control_targets.py; exec bash"
sleep 1

# --- 手动步骤: 启动 QGC ---
echo "手动步骤: 请现在手动启动 QGroundControl 地面站。"
echo "➡️ [7/12] cd ~/Downloads; ./QGroundControl.AppImage"
echo "🕒 等待10秒，假设您已启动QGC..."
sleep 15


# --- 终端 7: 启动数据记录 ---
#echo "➡️ [8/12] 正在新终端中启动 rosbag 数据记录..."
#gnome-terminal -- bash -c "echo '✅ [终端7] Rosbag 记录中...'; cd ~/XTDrone/zhihang2025; rosbag record -O score1 /standard_vtol_0/mavros/state /iris_0/mavros/state /gazebo/model_states /xtdrone/standard_vtol_0/cmd /xtdrone/iris_0/cmd /zhihang/first_point /zhihang2025/first_man/pose /zhihang2025/second_man/pose /zhihang2025/third_man/pose /zhihang2025/iris_healthy_man/pose /zhihang2025/iris_bad_man/pose /zhihang/downtown; exec bash"
sleep 2

# --- 终端 8: 阶段一固定翼控制 ---
echo "➡️ [9/12] 正在新终端中启动阶段一 VTOL 控制脚本..."
gnome-terminal -- bash -c "echo '✅ [终端8] VTOL 控制脚本 (vtol_commander.py) 启动中...'; cd ~/catkin_ws/src/offboard_run/scripts; python3 vtol_commander.py; exec bash"
sleep 2

# --- 终端 9: 阶段一识别 ---
echo "➡️ [10/12] 正在新终端中启动阶段一识别脚本..."
gnome-terminal -- bash -c "echo '✅ [终端9] 阶段一识别脚本 (detect_stage1.py) 启动中...'; source ~/catkin_ws/devel/setup.bash; rosrun offboard_run detect_stage1.py; exec bash"
sleep 2

# --- 终端 10: 阶段二四旋翼控制 ---
echo "➡️ [11/12] 正在新终端中启动阶段二 Iris 控制脚本..."
gnome-terminal -- bash -c "echo '✅ [终端10] Iris 控制脚本 (iris_commander1.py) 启动中...'; cd ~/catkin_ws/src/offboard_run/scripts; python3 iris_commander1.py; exec bash"
sleep 2

# --- 终端 11 & 12: 阶段二识别 ---
echo "➡️ [12/12] 正在新终端中启动阶段二识别脚本 (红 & 白)..."
gnome-terminal -- bash -c "echo '✅ [终端11] 阶段二红色目标识别脚本 (detect_red.py) 启动中...'; source ~/catkin_ws/devel/setup.bash; rosrun offboard_run detect_red.py; exec bash"
sleep 2
gnome-terminal -- bash -c "echo '✅ [终端12] 阶段二白色目标识别脚本 (detect_white.py) 启动中...'; source ~/catkin_ws/devel/setup.bash; rosrun offboard_run detect_white.py; exec bash"


echo ""
echo "✅ [主进程] 所有脚本已在新的终端窗口中启动！请检查各个窗口的输出。"
echo "任务流程已全部启动，此主终端可以关闭。"
