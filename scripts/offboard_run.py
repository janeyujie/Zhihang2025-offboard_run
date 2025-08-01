#!/usr/bin/env python

import rospy
import math
from geometry_msgs.msg import PoseStamped, Pose, Twist
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class DroneController:
    def __init__(self):
        # --- 参数定义 ---
        # 目标途径点
        self.half_x = 1200.0
        self.half_y = -250.0
        self.half_z = 30.0
        
        # 初始的最终目标点
        self.target_x = 1500.0
        self.target_y = 0.0
        self.target_z = 30.0

        # --- 状态变量 ---
        self.current_state = State()
        self.current_pose = Pose()
        self.arrived_half = False
        self.fixed_wing_mode = False
        self.gps_info_received = False
        self.position_received = False

        # CV Bridge for image processing
        self.bridge = CvBridge()

        # --- ROS 节点初始化 ---
        rospy.init_node('offb_node_py', anonymous=True)

        # --- ROS 通信 ---
        # 订阅者
        rospy.Subscriber("standard_vtol_0/mavros/state", State, self._state_cb)
        rospy.Subscriber("standard_vtol_0/mavros/local_position/pose", PoseStamped, self._pose_callback)
        rospy.Subscriber("/zhihang/first_point", Pose, self._gps_callback)
        rospy.Subscriber("standard_vtol_0/camera/image_raw", Image, self._image_callback)
        
        # 发布者
        self.setpoint_pos_pub = rospy.Publisher("standard_vtol_0/mavros/setpoint_position/local", PoseStamped, queue_size=10)
        self.xtdrone_state_pub = rospy.Publisher("standard_vtol_0/cmd", String, queue_size=10)

        # 服务客户端
        rospy.wait_for_service("standard_vtol_0/mavros/cmd/arming")
        self.arming_client = rospy.ServiceProxy("standard_vtol_0/mavros/cmd/arming", CommandBool)
        rospy.wait_for_service("standard_vtol_0/mavros/set_mode")
        self.set_mode_client = rospy.ServiceProxy("standard_vtol_0/mavros/set_mode", SetMode)
        
        rospy.loginfo("Python drone controller node initialized.")

    # --- 回调函数 ---
    def _state_cb(self, msg):
        self.current_state = msg

    def _pose_callback(self, msg):
        self.current_pose = msg.pose
        self.position_received = True

    def _gps_callback(self, msg):
        self.target_x = msg.position.x
        self.target_y = msg.position.y
        self.gps_info_received = True

    def _image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imshow("camera_image", cv_image)
            cv2.waitKey(1)
        except CvBridgeError as e:
            rospy.logerr(e)
            
    # --- 辅助函数 ---
    def _distance(self, x1, y1, x2, y2):
        return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

    # --- 主控制逻辑 ---
    def run(self):
        rate = rospy.Rate(20.0)

        # 等待飞控连接
        rospy.loginfo("Waiting for FCU connection...")
        while not rospy.is_shutdown() and not self.current_state.connected:
            rate.sleep()
        rospy.loginfo("FCU connected.")

        # 在进入OFFBOARD模式前，发送一些设定点
        pose = PoseStamped()
        pose.pose.position.z = self.target_z

        rospy.loginfo("Sending a few setpoints before starting...")
        for i in range(100):
            if rospy.is_shutdown():
                break
            self.setpoint_pos_pub.publish(pose)
            rate.sleep()

        # 请求进入OFFBOARD模式并解锁
        last_request = rospy.Time.now()
        rospy.loginfo("Attempting to enter OFFBOARD mode and arm...")
        while not rospy.is_shutdown() and not self.current_state.armed:
            current_time = rospy.Time.now()
            if self.current_state.mode != "OFFBOARD" and (current_time - last_request > rospy.Duration(5.0)):
                try:
                    if self.set_mode_client(custom_mode='OFFBOARD').mode_sent:
                        rospy.loginfo("OFFBOARD mode enabled")
                except rospy.ServiceException as e:
                    rospy.logerr("Service call failed: %s" % e)
                last_request = current_time
            elif not self.current_state.armed and (current_time - last_request > rospy.Duration(5.0)):
                try:
                    if self.arming_client(True).success:
                        rospy.loginfo("Vehicle armed")
                except rospy.ServiceException as e:
                    rospy.logerr("Service call failed: %s" % e)
                last_request = current_time

            self.setpoint_pos_pub.publish(pose)
            rate.sleep()

        # --- 主任务循环 ---
        rospy.loginfo("Starting main mission loop.")
        while not rospy.is_shutdown():
            # 阶段1: 垂直起飞
            if self.current_pose.position.z < self.target_z - 0.5:
                pose.pose.position.z = self.target_z
            else:
                # 阶段2: 达到高度后，切换为固定翼模式 (仅一次)
                if not self.fixed_wing_mode:
                    mode_msg = String()
                    mode_msg.data = "plane"
                    self.xtdrone_state_pub.publish(mode_msg)
                    rospy.loginfo("Switched to fixed-wing mode")
                    self.fixed_wing_mode = True
            
            # 阶段3: 导航
            '''if self.fixed_wing_mode:
                # 如果还未到达途径点，则飞向途径点
                if not self.arrived_half:
                    pose.pose.position.x = self.half_x
                    pose.pose.position.y = self.half_y
                    pose.pose.position.z = self.half_z
                else:
                    # 如果已到达途径点，则飞向最终目标
                    pose.pose.position.x = self.target_x
                    pose.pose.position.y = self.target_y
                    pose.pose.position.z = self.target_z'''

            # 检查是否到达途径点
            dist_to_half = self._distance(self.current_pose.position.x, self.current_pose.position.y, self.half_x, self.half_y)
            if not self.arrived_half and dist_to_half < 5.0: # 稍微放宽距离判断
                rospy.loginfo("Reached halfway point: [x: %.2f, y: %.2f]", self.current_pose.position.x, self.current_pose.position.y)
                self.arrived_half = True
            
            # 检查是否到达最终目标点
            if self.arrived_half:
                dist_to_target = self._distance(self.current_pose.position.x, self.current_pose.position.y, self.target_x, self.target_y)
                if dist_to_target < 5.0:
                    rospy.loginfo("Reached final target: [x: %.2f, y: %.2f]", self.current_pose.position.x, self.current_pose.position.y)
                    break # 任务完成，退出循环

            # 持续发布目标位置
            self.setpoint_pos_pub.publish(pose)
            rate.sleep()

        rospy.loginfo("Mission finished.")
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        controller = DroneController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
