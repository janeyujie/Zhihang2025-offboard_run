# README

需要把该目录放在catkin_ws/src下

（虚拟机连不上github/复制不了整个工作区到windows下面，不然应该传整个工作区上来的

```bash
# 使用前需要先
catkin build
```



完整的代码启动流程：

```bash
# 如果小人没加载完全的话要重新启动一次
roslaunch px4 zhihang2025.launch

# 运行通信脚本
cd ~/XTDrone/communication/
python3 vtol_communication.py standard_vtol 0

cd ~/XTDrone/communication/
python3 multirotor_communication.py iris 0

# 开启gazebo位姿真值
cd ~/XTDrone/sensing/pose_ground_truth/
python3 get_local_pose.py standard_vtol 1

cd ~/XTDrone/sensing/pose_ground_truth/
python3 get_local_pose.py iris 1

# 开启待救援目标移动代码
cd ~/XTDrone/zhihang2025
python3 zhihang_control_targets.py

# 启动 QGC 地面站
cd ~/Downloads
./QGroundControl.AppImage

# 阶段一固定翼无人机控制
cd ~/catkin_ws/src/offboard_run/scripts
python3 vtol_commander.py

# 阶段一识别待救援人员坐标脚本
source ~/catkin_ws/devel/setup.bash
rosrun offboard_run detect_stage1.py

# 阶段二四旋翼无人机控制
cd ~/catkin_ws/src/offboard_run/scripts
python3 iris_commander1.py

# 阶段二识别重伤人员和健康人员详细坐标脚本
source ~/catkin_ws/devel/setup.bash
rosrun offboard_run detect.py

```



订阅的相关话题（调试用

```bash
# 无人机自身位置话题
rostopic echo standard_vtol_0/mavros/local_position/pose
rostopic echo standard_vtol_0/mavros/vision_pose/pose
rostopic echo iris_0/mavros/local_position/pose
rostopic echo iris_0/mavros/vision_pose/pose

# 阶段一识别到的三个待救援人员的坐标位置
rostopic echo /zhihang2025/first_man/pose
rostopic echo /zhihang2025/second_man/pose
rostopic echo /zhihang2025/third_man/pose

# 阶段二识别到的重伤人员和健康人员的坐标位置
rostopic echo /zhihang2025/first_man/pose
rostopic echo /zhihang2025/third_man/pose

rostopic pub -1 /zhihang2025/third_man/pose geometry_msgs/Pose '{position: {x: 1495.0, y: -111.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}'
rostopic pub -1 /zhihang2025/first_man/pose geometry_msgs/Pose '{position: {x: 1495.0, y: -105.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}'
rostopic pub -1 /part1_completed std_msgs/Bool "data: true"
rostopic pub -1 standard_vtol_0/waypoint_reached std_msgs/Bool "data: true"
rostopic pub -1 /zhihang2025/first_man/reached std_msgs/Bool "data: true"
rostopic pub -1 /zhihang2025/third_man/reached std_msgs/Bool "data: true"
```

