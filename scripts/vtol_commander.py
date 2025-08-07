#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import tf
import math
from geometry_msgs.msg import PoseStamped, Pose, Quaternion, Twist, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge, CvBridgeError
import cv2
import time

class Commander:
    def __init__(self):
        rospy.init_node("commander_node")
        
        self.current_state = State()
        self.current_pose = Pose()
        self.current_yaw = 0.0
        self.position_received = False
        self.bridge = CvBridge()
        
        self.is_fixed_wing = False
        self.fixed_wing_start_point = None
        self.rotor_zone_radius = 100.0
        self.searched = False

        # --- 订阅者与发布者 ---
        rospy.Subscriber("standard_vtol_0/mavros/state", State, self._state_cb)
        rospy.Subscriber("standard_vtol_0/mavros/local_position/pose", PoseStamped, self._pose_cb)
        rospy.Subscriber("standard_vtol_0/camera/image_raw", Image, self._image_callback)
        
        self.setpoint_pos_pub = rospy.Publisher("standard_vtol_0/mavros/setpoint_position/local", PoseStamped, queue_size=10)
        self.xtdrone_vel_pub = rospy.Publisher("xtdrone/standard_vtol_0/cmd_vel_flu", Twist, queue_size=10)
        self.xtdrone_cmd_pub = rospy.Publisher("xtdrone/standard_vtol_0/cmd", String, queue_size=10)
        self.waypoint_pub = rospy.Publisher("standard_vtol_0/waypoint_reached", Bool, queue_size=10)
        self.ending_pub = rospy.Publisher("standard_vtol_0/search_completed", Bool, queue_size=10)
        self.part1_completed_pub = rospy.Publisher("/part1_completed", Bool, queue_size=10)
        
        # --- 服务客户端 ---
        rospy.wait_for_service("standard_vtol_0/mavros/cmd/arming")
        self.arming_client = rospy.ServiceProxy("standard_vtol_0/mavros/cmd/arming", CommandBool)
        rospy.wait_for_service("standard_vtol_0/mavros/set_mode")
        self.set_mode_client = rospy.ServiceProxy("standard_vtol_0/mavros/set_mode", SetMode)
        
        rospy.loginfo("Commander node initialized.")

    def _switch_to_fixed_wing(self):
        if not self.is_fixed_wing:
            rospy.loginfo("Switching to FIXED-WING mode...")
            cmd = String()
            cmd.data = "plane"
            # 多发布几次确保指令被接收
            for _ in range(5):
                self.xtdrone_cmd_pub.publish(cmd)
                rospy.sleep(0.1)
            self.is_fixed_wing = True
            # 记录切换点作为旋翼模式区的中心
            if self.fixed_wing_start_point is None:
                self.fixed_wing_start_point = self.current_pose.position
                rospy.loginfo("Rotor mode zone center set at (%.2f, %.2f)" % (self.fixed_wing_start_point.x, self.fixed_wing_start_point.y))

    def _switch_to_multirotor(self):
        if self.is_fixed_wing:
            rospy.loginfo("Switching to MULTIROTOR mode...")
            cmd = String()
            cmd.data = "multirotor"
            for _ in range(5):
                self.xtdrone_cmd_pub.publish(cmd)
                rospy.sleep(0.1)
            self.is_fixed_wing = False
    
    # --- 回调函数 ---
    def _state_cb(self, msg):
        self.current_state = msg

    def _pose_cb(self, msg):
        self.current_pose = msg.pose
        self.position_received = True
        
        orientation_q = self.current_pose.orientation
        _, _, self.current_yaw = tf.transformations.euler_from_quaternion(
            [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])

    def _image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            scale = 0.5
            resized_image = cv2.resize(cv_image, (0, 0), fx=scale, fy=scale)
            cv2.imshow("Standard_vtol Camera", resized_image)
            cv2.waitKey(1)
        except CvBridgeError as e:
            rospy.logerr(e)
            
    def _distance(self, p1, p2):
        return math.sqrt(math.pow(p1.x - p2.x, 2) + math.pow(p1.y - p2.y, 2))

    def activate_offboard_and_arm(self):
        """
        激活 OFFBOARD 模式并解锁无人机。
        这是一个阻塞方法，直到成功或ROS关闭。
        @return: 成功返回 True，否则返回 False。
        """
        rate = rospy.Rate(20.0)
        
        rospy.loginfo("Attempting to activate OFFBOARD mode and arm...")
        
        pose = PoseStamped()
        
        # 等待位置信息
        while not rospy.is_shutdown() and not self.position_received:
            rospy.logwarn_throttle(2, "waiting for position information...")
            rate.sleep()
            
        pose.pose = self.current_pose

        last_request = rospy.Time.now()
        
        while not rospy.is_shutdown():
            current_time = rospy.Time.now()
            
            # 检查并设置模式
            if self.current_state.mode != "OFFBOARD" and (current_time - last_request > rospy.Duration(5.0)):
                try:
                    if self.set_mode_client(custom_mode='OFFBOARD').mode_sent:
                        rospy.loginfo("OFFBOARD mode enabled")
                except rospy.ServiceException as e:
                    rospy.logerr("Service call for OFFBOARD failed: %s" % e)
                last_request = current_time
            
            # 检查并解锁
            elif not self.current_state.armed and (current_time - last_request > rospy.Duration(5.0)):
                try:
                    if self.arming_client(True).success:
                        rospy.loginfo("Vehicle armed")
                except rospy.ServiceException as e:
                    rospy.logerr("Service call for arming failed: %s" % e)
                last_request = current_time

            # 检查是否成功
            if self.current_state.mode == "OFFBOARD" and self.current_state.armed:
                rospy.loginfo("Vehicle is in OFFBOARD mode and armed.")
                return True

            # 持续发布设定点以保持连接
            pose.header.stamp = rospy.Time.now()
            pose.pose = self.current_pose 
            self.setpoint_pos_pub.publish(pose)
            rate.sleep()
            
        return False
    
    def disarm(self):
        rospy.loginfo("Commanding DISARM")
        try:
            if self.arming_client(False).success:
                rospy.loginfo("Vehicle disarmed successfully.")
        except rospy.ServiceException as e:
            rospy.logerr("Service call for disarming failed: %s" % e)
    
    # --- `takeoff` 函数只负责起飞 ---
    def takeoff(self, height):
        rate = rospy.Rate(20.0)
        
        rospy.loginfo("Waiting for first position message...")
        while not rospy.is_shutdown() and not self.position_received: rate.sleep()
        rospy.loginfo("Position received.")
        rospy.loginfo("Waiting for FCU connection...")
        while not rospy.is_shutdown() and not self.current_state.connected: rate.sleep()
        rospy.loginfo("FCU connected.")
        
        if not self.activate_offboard_and_arm():
            rospy.logfatal("Failed to activate OFFBOARD and arm. Aborting takeoff.")
            return

        rospy.loginfo("Takeoff to height: %.2f" % height)
        takeoff_pose = PoseStamped()
        takeoff_pose.pose.position.x = self.current_pose.position.x
        takeoff_pose.pose.position.y = self.current_pose.position.y
        takeoff_pose.pose.position.z = height

        while not rospy.is_shutdown():
            takeoff_pose.header.stamp = rospy.Time.now()
            self.setpoint_pos_pub.publish(takeoff_pose)

            if abs(self.current_pose.position.z - height) < 0.5:
                rospy.loginfo("Takeoff complete. Hovering at %.2f meters." % self.current_pose.position.z)
                # 起飞完成后，继续发送指令以保持悬停，然后返回
                for _ in range(60):
                    takeoff_pose.header.stamp = rospy.Time.now()
                    self.setpoint_pos_pub.publish(takeoff_pose)
                    rate.sleep()
                return
            rate.sleep()
            
    # --- 改变高度的函数 ---
    def change_altitude(self, height):
        rospy.loginfo("change to altitude: %.2f" % height)
        rate = rospy.Rate(20.0)
        
        target_pose = PoseStamped()
        target_pose.pose.position.x = self.current_pose.position.x
        target_pose.pose.position.y = self.current_pose.position.y
        target_pose.pose.position.z = height

        while not rospy.is_shutdown():
            target_pose.header.stamp = rospy.Time.now()
            self.setpoint_pos_pub.publish(target_pose)

            if abs(self.current_pose.position.z - height) < 0.5:
                rospy.loginfo("Changing altitude to %.2f meters." % self.current_pose.position.z)
                # 继续发送指令以保持悬停，然后返回
                for _ in range(100): # 悬停5秒
                    target_pose.header.stamp = rospy.Time.now()
                    self.setpoint_pos_pub.publish(target_pose)
                    rate.sleep()
                return
            rate.sleep()
            
    # --- `move` 函数控制移动 ---
    def move(self, x, y, z, frame="world"):
        if not (self.current_state.mode == "OFFBOARD" and self.current_state.armed):
            rospy.loginfo("Attempting to reactivate OFFBOARD and arm...")
            if not self.activate_offboard_and_arm():
                rospy.logfatal("Failed to activate OFFBOARD and arm. Aborting move.")
                return

        target_pose = PoseStamped()
        target_pose.pose.position.x = x
        target_pose.pose.position.y = y
        target_pose.pose.position.z = z
        
        delta_x = target_pose.pose.position.x - self.current_pose.position.x
        delta_y = target_pose.pose.position.y - self.current_pose.position.y

        if math.sqrt(delta_x**2 + delta_y**2) > 1.0: # 避免在原地改变高度时改变偏航
            target_yaw_rad = math.atan2(delta_y, delta_x)
            target_quaternion = tf.transformations.quaternion_from_euler(0, 0, target_yaw_rad)
            target_pose.pose.orientation = Quaternion(*target_quaternion)
        else:
            # 如果是垂直移动，保持当前偏航角
            target_pose.pose.orientation = self.current_pose.orientation
        
        rospy.loginfo("Moving to the target position (%.2f, %.2f, %.2f)" % (x, y, z))
        rate = rospy.Rate(20.0) 

        completion_radius = 15.0 if self.is_fixed_wing else 1.5

        while not rospy.is_shutdown():
            target_pose.header.stamp = rospy.Time.now()
            self.setpoint_pos_pub.publish(target_pose)
            
            # 只检查水平距离
            if self._distance(self.current_pose.position, target_pose.pose.position) < completion_radius:
                start_time = rospy.Time.now()
                while rospy.Time.now() - start_time < rospy.Duration(2.0):
                    self.setpoint_pos_pub.publish(target_pose)
                    rate.sleep()

                rospy.loginfo("Move command finished for target (%.2f, %.2f, %.2f)." % (self.current_pose.position.x, self.current_pose.position.y, self.current_pose.position.z))
                return 
            rate.sleep()
        
        self.hold()
        rospy.loginfo("Move command finished.")
    
    def move_with_velocity(self, x, y, z, target_speed):
        """
        使用速度控制向目标点移动
        @param x, y, z: 全局目标坐标
        @param target_speed: 飞向目标的期望速度 (m/s)
        """
        rospy.loginfo(f"Moving to ({x:.2f}, {y:.2f}, {z:.2f}) with speed {target_speed:.2f} m/s...")
        rate = rospy.Rate(20.0)
        
        completion_radius = 15.0 if self.is_fixed_wing else 3.0

        while not rospy.is_shutdown():
            if not self.position_received:
                rate.sleep()
                continue
            
            if self.current_pose.position.x >= 1400 and self.searched == False: 
                rospy.loginfo("Reaching waypoint, start to search")
                self.searched = True
                msg = Bool()
                msg.data = True
                self.waypoint_pub.publish(msg)
            
            # --- 核心控制逻辑 ---
            dx = x - self.current_pose.position.x
            dy = y - self.current_pose.position.y
            dz = z - self.current_pose.position.z
            
            distance = math.sqrt(dx**2 + dy**2 + dz**2)

            # 检查是否到达
            if distance < completion_radius:
                rospy.loginfo("Target waypoint reached.")
                # 到达后发送悬停指令（零速度）
                self.xtdrone_vel_pub.publish(Twist()) 
                self.xtdrone_cmd_pub.publish(String("HOVER"))
                rospy.sleep(2) # 悬停稳定
                return

            # 1. 计算全局坐标系下的速度向量
            global_vel_x = (dx / distance) * target_speed
            global_vel_y = (dy / distance) * target_speed
            global_vel_z = (dz / distance) * target_speed
            
            # 2. 将全局速度转换为机体坐标系下的速度 (前、左、上)
            forward_vel = global_vel_x * math.cos(self.current_yaw) + global_vel_y * math.sin(self.current_yaw)
            leftward_vel = -global_vel_x * math.sin(self.current_yaw) + global_vel_y * math.cos(self.current_yaw)
            upward_vel = global_vel_z

            # 3. 发布Twist消息
            twist_msg = Twist()
            twist_msg.linear.x = forward_vel
            twist_msg.linear.y = leftward_vel
            twist_msg.linear.z = upward_vel
            
            self.xtdrone_vel_pub.publish(twist_msg)
            rate.sleep()
        
    # --- 返航并在旋翼模式区降落 ---
    def return_and_land(self):
        """
        命令无人机返回旋翼模式区中心，切换回多旋翼模式，然后降落。
        """
        if self.fixed_wing_start_point is None:
            rospy.logerr("Cannot return, fixed-wing start point was never set.")
            return

        rospy.loginfo("--- Starting Return, Switch and Land Sequence ---")

        # 步骤1: 以固定翼模式返回旋翼区中心点附近（保持巡航高度）
        rospy.loginfo("Step 1: Returning to rotor zone center in fixed-wing mode.")
        self.move(self.fixed_wing_start_point.x, self.fixed_wing_start_point.y, self.fixed_wing_start_point.z)

        # 步骤2: 切换回多旋翼模式
        rospy.loginfo("Step 2: Arrived at zone center. Switching to multirotor mode.")
        self._switch_to_multirotor()
        rospy.sleep(5) # 给予充足时间让飞行状态稳定下来

        # 步骤3: 稳定悬停
        rospy.loginfo("Step 3: Commanding stable hold.")
        self.hold()
        rospy.sleep(3)

        # 步骤4: 执行自动降落
        rospy.loginfo("Step 4: Commanding automatic land.")
        self.land()
        
        rospy.loginfo("Waiting for landing and disarm...")
        landing_start_time = rospy.Time.now()
        while not rospy.is_shutdown() and self.current_state.armed:
            if rospy.Time.now() - landing_start_time > rospy.Duration(60.0):
                self.disarm()
                break 
            rospy.sleep(1)

        if not self.current_state.armed:
            rospy.loginfo("Landed and disarmed successfully. Mission complete.")
            msg = Bool()
            msg.data = True
            self.part1_completed_pub.publish(msg)
        else:
            rospy.logerr("Mission finished, but failed to disarm the vehicle.")
        

    def land(self):
        rospy.loginfo("Commanding LAND")
        try:
            if self.set_mode_client(custom_mode='AUTO.LAND').mode_sent:
                rospy.loginfo("Land command sent successfully.")
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
            
    def hold(self):
        rospy.loginfo("Switching to Position Hold mode (AUTO.LOITER)")
        try:
            if self.set_mode_client(custom_mode='AUTO.LOITER').mode_sent:
                rospy.loginfo("Switched to LOITER mode successfully.")
        except rospy.ServiceException as e:
            rospy.logerr("Service call for LOITER mode failed: %s" % e)
            
    def start_to_search(self):
        if self.searched == False: 
            rospy.loginfo("Reaching waypoint, start to search")
            self.searched = True
            msg = Bool()
            msg.data = True
            self.waypoint_pub.publish(msg)
            
    def end_searching(self):
        rospy.loginfo("Searching mission ended...")
        msg = Bool()
        msg.data = True
        self.ending_pub.publish(msg)


if __name__ == "__main__":
    try:
        con = Commander()
        
        con.takeoff(20.0)
        
        con._switch_to_fixed_wing()
        rospy.loginfo("switching to fixed wing mode...")
        rospy.sleep(1)
        
        # 3. 在固定翼模式下执行所有巡航任务
        rospy.loginfo("--- Starting Cruise Mission  ---")
        con.change_altitude(40.0)
        con.move(1200, -250, 40.0)
        con.move(1495, -250, 40.0)
        #con.move_with_velocity(1200, -250, 40.0, 25)
        #con.move_with_velocity(1495, -250, 40.0, 25)
        
        rospy.loginfo("--- Starting Search Mission ---")
        con.change_altitude(10.0)
        con.start_to_search()
        '''con.move(1450, 250, 12)
        con.move(1460, 250, 12)
        con.move(1460, -250, 12)
        con.move(1470, -250, 12)
        con.move(1470, 250, 12)
        con.move(1480, 250, 12)
        con.move(1480, -250, 12)
        con.move(1490, -250, 12)
        con.move(1490, 250, 12)
        con.move(1500, 250, 12)
        con.move(1500, -250, 12)
        con.move(1510, -250, 12)
        con.move(1510, 250, 12)
        con.move(1520, 250, 12)
        con.move(1520, -250, 12)
        con.move(1530, -250, 12)
        con.move(1530, 250, 12)
        con.move(1540, 250, 12)
        con.move(1540, -250, 12)
        con.move(1550, -250, 12)
        con.move(1550, 250, 12)'''
        #con.move(1495, -250, 10)
        con.move(1495, 250, 10)
        con.end_searching()
        #con.move_with_velocity(1495, 250, 10.0, 4)
        
        con.change_altitude(40.0)
        con.move(1450, 250, 40)
        # 任务完成后，返回、切换模式并降落
        con.move(1200, 250, 40.0)
        con.return_and_land()

    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
        rospy.loginfo("Commander node terminated.")
