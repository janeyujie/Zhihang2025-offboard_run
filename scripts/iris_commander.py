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
import threading

class QuadcopterCommander:
    def __init__(self):
        rospy.init_node("quadcopter_commander_node")
        
        self.current_state = State()
        self.current_pose = Pose()
        self.position_received = False
        self.bridge = CvBridge()

        self.start_point = None
        self.can_start = True
        self.rotor_zone_radius = 100.0
        

        # --- 订阅者与发布者 ---
        rospy.Subscriber("/iris_0/mavros/state", State, self._state_cb)
        rospy.Subscriber("/iris_0/mavros/local_position/pose", PoseStamped, self._pose_cb)
        rospy.Subscriber("/iris_0/camera/image_raw", Image, self._image_callback)
        rospy.Subscriber("/part1_completed", Bool, self._start_cb)
        self.cmd_pub = rospy.Publisher("/xtdrone/iris_0/cmd", String, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher("/xtdrone/iris_0/cmd_vel_flu", Twist, queue_size=1)
        
        self.setpoint_pos_pub = rospy.Publisher("/iris_0/mavros/setpoint_position/local", PoseStamped, queue_size=10)
        self.part2_completed_pub = rospy.Publisher("/part2_completed", Bool, queue_size=10)
        
        rospy.Subscriber("/zhihang2025/first_man/pose", PoseStamped, self._critical_man_cb)
        rospy.Subscriber("/zhihang2025/third_man/pose", PoseStamped, self._healthy_man_cb)
        
        self.critical_man_reached_pub = rospy.Publisher("/zhihang2025/first_man/reached", Bool, queue_size=1)
        self.healthy_man_reached_pub = rospy.Publisher("/zhihang2025/third_man/reached", Bool, queue_size=1)
        
        # --- 服务客户端 ---
        rospy.wait_for_service("/iris_0/mavros/cmd/arming")
        self.arming_client = rospy.ServiceProxy("/iris_0/mavros/cmd/arming", CommandBool)
        rospy.wait_for_service("/iris_0/mavros/set_mode")
        self.set_mode_client = rospy.ServiceProxy("/iris_0/mavros/set_mode", SetMode)
        
        self.rate = rospy.Rate(20)
        rospy.loginfo("Commander node initialized.")
    
    # --- 回调函数 ---
    def _state_cb(self, msg):
        self.current_state = msg

    def _pose_cb(self, msg):
        self.current_pose = msg.pose
        self.position_received = True

    def _image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imshow("Iris Camera", cv_image)
            cv2.waitKey(1)
        except CvBridgeError as e:
            rospy.logerr(e)
            
    def _start_cb(self, msg):
        if msg.data:
            self.can_start = True
            rospy.loginfo("Mission Part1 completed, Part2 start")
    
    def _critical_man_cb(self, msg):
        if self.critical_man_pose is None:
            rospy.loginfo("Received pose for CRITICAL person.")
        self.critical_man_pose = msg

    def _healthy_man_cb(self, msg):
        if self.healthy_man_pose is None:
            rospy.loginfo("Received pose for HEALTHY person.")
        self.healthy_man_pose = msg
            
    def _distance(self, p1, p2):
        return math.sqrt(math.pow(p1.x - p2.x, 2) + math.pow(p1.y - p2.y, 2))
    
    def publish_command(self, command):
        """发布一个字符串命令，例如 'ARM', 'AUTO.TAKEOFF'"""
        cmd_msg = String()
        cmd_msg.data = command
        self.cmd_pub.publish(cmd_msg)
        rospy.loginfo(f"Published command: {command}")

    def publish_velocity(self, forward=0.0, leftward=0.0, upward=0.0, angular_z=0.0):
        """发布速度指令 (机体坐标系：前左上)"""
        twist_msg = Twist()
        twist_msg.linear.x = forward
        twist_msg.linear.y = leftward
        twist_msg.linear.z = upward
        twist_msg.angular.z = angular_z
        self.cmd_vel_pub.publish(twist_msg)

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
    
    def offboard(self):
        self.publish_command('OFFBOARD')
        rospy.loginfo("Commanding offboard...")
        rospy.sleep(1)
    
    def disarm(self):
        rospy.loginfo("Commanding DISARM")
        try:
            if self.arming_client(False).success:
                rospy.loginfo("Vehicle disarmed successfully.")
        except rospy.ServiceException as e:
            rospy.logerr("Service call for disarming failed: %s" % e)
    
    # --- `takeoff` 函数只负责起飞 ---
    def takeoff(self):
        # 这里的takeoff飞到2m自动悬停
        self.publish_command('AUTO.TAKEOFF')
        rospy.loginfo("Taking off to 2m...")
        rospy.sleep(5)
            
    # --- 改变高度的函数 ---
    def change_altitude(self, height):
        self.offboard()

        rospy.loginfo("change to altitude: %.2f" % height)
        rate = rospy.Rate(20.0)
        
        target_pose = PoseStamped()
        target_pose.pose.position.x = self.current_pose.position.x
        target_pose.pose.position.y = self.current_pose.position.y
        target_pose.pose.position.z = height

        while not rospy.is_shutdown():
            target_pose.header.stamp = rospy.Time.now()
            self.setpoint_pos_pub.publish(target_pose)
            self.publish_velocity(upward=4)
            
            if abs(self.current_pose.position.z - height) < 0.2:
                rospy.loginfo("Changing altitude to %.2f meters." % self.current_pose.position.z)
                return
            
            rate.sleep()
            
    # --- `move` 函数控制移动 ---
    def move(self, x, y, z, frame="world", completion_radius=0.5):
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
        target_yaw_rad = math.atan2(delta_y, delta_x)
        target_quaternion = tf.transformations.quaternion_from_euler(0, 0, target_yaw_rad)
        target_pose.pose.orientation = Quaternion(*target_quaternion)
        
        rospy.loginfo("Moving to the target position (%.2f, %.2f, %.2f)" % (x, y, z))
        rate = rospy.Rate(20.0) 

        while not rospy.is_shutdown():
            target_pose.header.stamp = rospy.Time.now()
            self.setpoint_pos_pub.publish(target_pose)
            
            # 只检查水平距离
            if self._distance(self.current_pose.position, target_pose.pose.position) < completion_radius:
                
                start_time = rospy.Time.now()
                while rospy.Time.now() - start_time < rospy.Duration(3.0):
                    self.setpoint_pos_pub.publish(target_pose)
                    rate.sleep()

                rospy.loginfo("Move command finished for target (%.2f, %.2f, %.2f)." % (self.current_pose.position.x, self.current_pose.position.y, self.current_pose.position.z))
                return 
            rate.sleep()
        
        rospy.loginfo("Move command finished.")
        
    def moving_to_health(self):
        rospy.loginfo("Waiting for HEALTHY person's location...")
        while not rospy.is_shutdown() and self.healthy_man_pose is None:
            rospy.sleep(1)

        target = self.healthy_man_pose.pose.position
        rospy.loginfo("Navigating to HEALTHY person at (%.2f, %.2f)" % (target.x, target.y))
        self.move(target.x, target.y, self.current_pose.position.z, completion_radius=3.0)

        rospy.loginfo("HEALTHY person reached. Publishing signal and hovering.")
        self.healthy_man_reached_pub.publish(Bool(True))
        self.hover(5)

    def moving_to_critical(self):
        rospy.loginfo("Waiting for CRITICAL person's location...")
        while not rospy.is_shutdown() and self.critical_man_pose is None:
            rospy.sleep(1)
        
        target = self.critical_man_pose.pose.position
        rospy.loginfo("Navigating to CRITICAL person at (%.2f, %.2f)" % (target.x, target.y))
        self.move(target.x, target.y, self.current_pose.position.z, completion_radius=3.0)
        
        rospy.loginfo("CRITICAL person reached. Publishing signal and hovering.")
        self.critical_man_reached_pub.publish(Bool(True))
        self.hover(5)
        
    # --- 返航并在旋翼模式区降落 ---
    def return_and_land(self):
        """
        命令无人机返回旋翼模式区中心，切换回多旋翼模式，然后降落。
        """
        rospy.loginfo("--- Starting Return, Switch and Land Sequence ---")

        # 步骤1: 返回旋翼区中心点附近（保持巡航高度）
        rospy.loginfo("Step 1: Returning to rotor zone center.")
        self.move(0, 0, 0)

        # 步骤2: 稳定悬停
        rospy.loginfo("Step 2: Commanding stable hold.")
        self.hold()
        rospy.sleep(3)

        # 步骤3: 执行自动降落
        rospy.loginfo("Step 3: Commanding automatic land.")
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
            self.part2_completed_pub.publish(msg)
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
            
    def hover(self, duration):
        rospy.loginfo("Hovering for %.1f seconds." % duration)
        rate = rospy.Rate(20.0)

        hover_pose = PoseStamped()
        hover_pose.header.stamp = rospy.Time.now()
        hover_pose.pose = self.current_pose
        
        start_time = rospy.Time.now()
        while not rospy.is_shutdown() and (rospy.Time.now() - start_time < rospy.Duration(duration)):
            self.setpoint_pos_pub.publish(hover_pose)
            rate.sleep()


if __name__ == "__main__":
    try:
        con = QuadcopterCommander()
        
        rospy.loginfo("Quadcopter is waiting for part1 completed...")
        rate = rospy.Rate(1)
        while not rospy.is_shutdown() and not con.can_start:
            rospy.loginfo_throttle(10, "Waiting for signal...")
            rate.sleep()
        rospy.loginfo("Signal received! Continuing to part2")
        
        con.takeoff()
        con.change_altitude(40.0)
        con.move(1200, -250, 40.0)
        con.move(1450, -250, 40.0)
        con.change_altitude(12.0)
        
        con.moving_to_health()
        con.moving_to_critical()
        
        con.change_altitude(40.0)
        con.move(1450, 250, 40.0)
        con.move(1200, 250, 40.0)
        con.return_and_land()

    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
        rospy.loginfo("Commander node terminated.")
