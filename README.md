# README

### 环境配置：

```
pip install ultralytics
```

需要把该目录放在catkin_ws/src下

（虚拟机连不上github/复制不了整个工作区到windows下面，不然应该传整个工作区上来的

```bash
# 使用前需要
cd ~/catkin_ws
catkin build
```



```
# 四旋翼
MPC_XY_CRUISE 巡航速度
MPC_VEL_MANUAL 最大水平速度限制
# 固定翼
FW_AIRSPD_TRIM 巡航速度
```



### 完整的代码启动流程：

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


# 启动数据记录
cd ~/XTDrone/zhihang2025
rosbag record -O score1 /standard_vtol_0/mavros/state /iris_0/mavros/state  /gazebo/model_states /xtdrone/standard_vtol_0/cmd /xtdrone/iris_0/cmd /zhihang/first_point /zhihang2025/first_man/pose /zhihang2025/second_man/pose /zhihang2025/third_man/pose /zhihang2025/iris_healthy_man/pose  /zhihang2025/iris_bad_man/pose  /zhihang/downtown


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
rosrun offboard_run detect_red.py

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



目前还存在的问题：

阶段一：

- 识别上已经没太大问题，后续考虑把图像中的位置转换成坐标的算法是否还能调优
- 尝试动态调整飞行速度但没成功，可能考虑直接在QGC中调慢飞行速度

阶段二：

- 根据坐标定位粗略寻找已经基本实现，精确降落还在调试中
- 新的模型权重刚刚得到，还在测试识别效果如何
- 暂时还没考虑降落时树和人的因素
