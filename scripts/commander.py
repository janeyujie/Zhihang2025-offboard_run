#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import tf
import math
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import time

class Commander:
    def __init__(self):
        rospy.init_node("commander_node")
        
        self.current_state = State()
        self.current_pose = Pose()
        self.position_received = False
        self.bridge = CvBridge()
        
        self.is_fixed_wing = False
        self.fixed_wing_start_point = None
        self.rotor_zone_radius = 100.0

        # --- Subcribers and Publishers ---
        rospy.Subscriber("standard_vtol_0/mavros/state", State, self._state_cb)
        rospy.Subscriber("standard_vtol_0/mavros/local_position/pose", PoseStamped, self._pose_cb)
        rospy.Subscriber("standard_vtol_0/camera/image_raw", Image, self._image_callback)
        
        self.setpoint_pos_pub = rospy.Publisher("standard_vtol_0/mavros/setpoint_position/local", PoseStamped, queue_size=10)
        self.xtdrone_cmd_pub = rospy.Publisher("xtdrone/standard_vtol_0/cmd", String, queue_size=10)
        
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
            self.xtdrone_cmd_pub.publish(cmd)
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
            self.xtdrone_cmd_pub.publish(cmd)
            self.is_fixed_wing = False
    
    def _state_cb(self, msg):
        self.current_state = msg

    def _pose_cb(self, msg):
        self.current_pose = msg.pose
        self.position_received = True

    def _image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            scale = 0.5  # 缩小一半
            resized_image = cv2.resize(cv_image, (0, 0), fx=scale, fy=scale)
            cv2.imshow("Drone Camera", resized_image)
            cv2.waitKey(1)
        except CvBridgeError as e:
            rospy.logerr(e)
            
    def _distance(self, x1, y1, z1, x2, y2, z2):
        return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2) + math.pow(z1 - z2, 2))

    # --- 核心API方法 ---
    def activate_offboard_and_arm(self):
        """
        激活 OFFBOARD 模式并解锁无人机。
        这是一个阻塞方法，直到成功或ROS关闭。
        @return: True if successful, False otherwise.
        """
        rate = rospy.Rate(20.0)
        
        rospy.loginfo("Attempting to activate OFFBOARD mode and arm...")
        
        pose = PoseStamped()
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

            # 持续发布设定点
            pose.pose = self.current_pose # 持续更新以保持当前位置
            self.setpoint_pos_pub.publish(pose)
            rate.sleep()
            
        return False
    
    def takeoff(self, height):
        """
        在当前位置垂直起飞到指定高度并悬停。
        """
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
            
            if abs(self.current_pose.position.z - height) < 0.1:
                rospy.loginfo("Takeoff complete. Hovering at %.2f meters." % height)
                break
            rate.sleep()
            
        # 起飞完成后，切换到固定翼模式
        self._switch_to_fixed_wing()
        # 悬停
        self.hold()
            
    def move(self, x, y, frame="world"):
        """
        命令无人机先转向目标方向，然后再向目标点水平移动。
        这确保了在移动过程中机头方向始终对准目标。
        """
        if not self.is_fixed_wing:
            rospy.logwarn("Move command called, but vehicle is not in fixed-wing mode. Switching now.")
            self._switch_to_fixed_wing()
            rospy.sleep(5)
        
        if not self.activate_offboard_and_arm():
            rospy.logfatal("Failed to activate OFFBOARD and arm. Aborting move.")
            return

        rospy.loginfo("Calculating target position and orientation...")
        target_pose = PoseStamped()
        target_pose.header.stamp = rospy.Time.now()

        # 计算世界坐标系下的最终目标点位置
        if frame == "body":
            q_current = [self.current_pose.orientation.x, self.current_pose.orientation.y, self.current_pose.orientation.z, self.current_pose.orientation.w]
            offset_rotated = tf.transformations.quaternion_multiply(
                tf.transformations.quaternion_multiply(q_current, [x, y, 0, 0]),
                tf.transformations.quaternion_conjugate(q_current)
            )
            target_pose.pose.position.x = self.current_pose.position.x + offset_rotated[0]
            target_pose.pose.position.y = self.current_pose.position.y + offset_rotated[1]
        elif frame == "world":
            target_pose.pose.position.x = x
            target_pose.pose.position.y = y
        
        target_pose.pose.position.z = self.current_pose.position.z

        # 计算从当前位置到目标点的方向向量，并转换为偏航角(Yaw)
        delta_x = target_pose.pose.position.x - self.current_pose.position.x
        delta_y = target_pose.pose.position.y - self.current_pose.position.y
        target_yaw_rad = math.atan2(delta_y, delta_x)
        target_yaw_deg = math.degrees(target_yaw_rad)

        rospy.loginfo("Step 1: Turning to face the target direction (%.2f degrees)..." % target_yaw_deg)
        self.turn(target_yaw_deg)
        time.sleep(2)
        
        rospy.loginfo("Step 2: Moving to the target position...")
        
        # 将计算出的目标朝向设置到最终的指令中
        target_quaternion = tf.transformations.quaternion_from_euler(0, 0, target_yaw_rad)
        target_pose.pose.orientation = Quaternion(*target_quaternion)
        
        # 以一定频率持续发布目标点，直到无人机到达
        rate = rospy.Rate(20.0)
        
        while not rospy.is_shutdown():
            target_pose.header.stamp = rospy.Time.now()
            
            if self._distance(self.current_pose.position.x, self.current_pose.position.y, 0, 
                            target_pose.pose.position.x, target_pose.pose.position.y, 0) < 0.2:
                rospy.loginfo("Reached target position.")
                break
            self.setpoint_pos_pub.publish(target_pose)
            rate.sleep()
        
        rospy.loginfo("Move command finished.")
        
        self.hold()
        
    def return_to_zone_and_land(self):
        """
        命令无人机返回旋翼模式区中心，切换回多旋翼模式，然后降落。
        """
        if self.fixed_wing_start_point is None:
            rospy.logerr("Cannot return, fixed-wing start point was never set.")
            return

        rospy.loginfo("--- Starting Return, Switch and Land Sequence ---")

        # 以固定翼模式返回旋翼区中心点
        rospy.loginfo("Step 1: Returning to rotor zone center in fixed-wing mode.")
        self.move(self.fixed_wing_start_point.x, self.fixed_wing_start_point.y, frame="world")

        # 检查是否在旋翼区内，然后切换模式
        rospy.loginfo("Step 2: Arrived at zone center. Switching to multirotor mode.")
        self._switch_to_multirotor()
        rospy.sleep(5)

        # 执行自动降落并锁桨
        rospy.loginfo("Step 3: Commanding automatic land.")
        self.land()
        
        # 等待降落完成后锁桨
        rospy.loginfo("Waiting for landing and disarm...")
        while not rospy.is_shutdown() and self.current_state.armed:
            rospy.sleep(1)
        rospy.loginfo("Landed and disarmed successfully. Mission complete.")
        
    def turn(self, yaw_degree):
        if not self.activate_offboard_and_arm():
            rospy.logfatal("Failed to activate OFFBOARD and arm. Aborting takeoff.")
            return
        
        rospy.loginfo("Commanding turn to %.2f degrees" % yaw_degree)
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.pose.position = self.current_pose.position
        yaw_rad = math.radians(yaw_degree)
        quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw_rad)
        pose.pose.orientation = Quaternion(*quaternion)
        self.setpoint_pos_pub.publish(pose)

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


if __name__ == "__main__":
    try:
        con = Commander()
        con.takeoff(40.0)
        time.sleep(3)
        
        con.move(1200, -250, frame="world")
        time.sleep(5)
        
        con.move(1495, -150, frame="world")
        time.sleep(5)
        
        con.takeoff(10.0)
        time.sleep(3)
        
        rospy.loginfo("--- Starting Search Mission ---")
        con.move(1495, 150, frame="world")
        time.sleep(5)
        
        con.return_to_zone_and_land()

    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
        rospy.loginfo("Commander node terminated.")

