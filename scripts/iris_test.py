#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rospy
import tf
import math
from geometry_msgs.msg import PoseStamped, Pose, Quaternion, Twist, TwistStamped
from mavros_msgs.msg import State, ExtendedState 
from mavros_msgs.srv import CommandBool, SetMode
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge, CvBridgeError
import cv2

class XTDroneController:
    def __init__(self):
        """
        初始化自主控制器节点
        """
        rospy.init_node("quadcopter_commander_node")

        self.current_state = State()
        self.current_pose = Pose()
        self.current_yaw = 0.0
        self.position_received = False
        self.bridge = CvBridge()
        self.can_start = False
        self.landed_state = ExtendedState.LANDED_STATE_UNDEFINED
        
        # subscriber
        rospy.Subscriber("/iris_0/mavros/state", State, self._state_cb)
        rospy.Subscriber("/iris_0/mavros/local_position/pose", PoseStamped, self._pose_cb)
        rospy.Subscriber("/iris_0/mavros/extended_state", ExtendedState, self._extended_state_cb)
        rospy.Subscriber("/iris_0/camera/image_raw", Image, self._image_callback)
        rospy.Subscriber("/part1_completed", Bool, self._start_cb)
        # publisher
        self.cmd_pub = rospy.Publisher("/xtdrone/iris_0/cmd", String, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher("/xtdrone/iris_0/cmd_vel_flu", Twist, queue_size=1)
        #self.setpoint_pos_pub = rospy.Publisher("/iris_0/mavros/setpoint_position/local", PoseStamped, queue_size=10)
        self.waypoint_pub = rospy.Publisher("/iris_0/waypoint_reached", Bool, queue_size=10)
        self.part2_completed_pub = rospy.Publisher("/part2_completed", Bool, queue_size=10)
        
        rospy.wait_for_service("/iris_0/mavros/cmd/arming")
        self.arming_client = rospy.ServiceProxy("/iris_0/mavros/cmd/arming", CommandBool)
        rospy.wait_for_service("/iris_0/mavros/set_mode")
        self.set_mode_client = rospy.ServiceProxy("/iris_0/mavros/set_mode", SetMode)
        
        self.rate = rospy.Rate(20)
        rospy.loginfo("XTDrone Controller Initialized. Waiting for connection...")

    def _extended_state_cb(self, msg):
        self.landed_state = msg.landed_state
    
    def _state_cb(self, msg):
        self.current_state = msg

    def _pose_cb(self, msg):
        self.current_pose = msg.pose
        self.position_received = True
        
        # --- 从四元数中提取偏航角 ---
        orientation_q = self.current_pose.orientation
        _, _, self.current_yaw = tf.transformations.euler_from_quaternion(
            [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])

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
            
    def _distance(self, x1, y1, x2, y2):
        return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    
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
        #self.offboard()

    def offboard(self):
        self.publish_command('OFFBOARD')
        rospy.loginfo("Commanding offboard...")
        rospy.sleep(1)
        
    def arm(self):
        self.publish_command('ARM')
        rospy.loginfo("Commanding arm...")
        rospy.sleep(3)
        
    def takeoff(self):
        # 这里的takeoff飞到2m自动悬停
        self.publish_command('AUTO.TAKEOFF')
        rospy.loginfo("Taking off to 2m...")
        rospy.sleep(5)
    
    def change_altitude(self, height):
        # 实现飞行高度的控制
        rospy.loginfo(f"Changing altitude to {height}m...")
        last_request = rospy.Time.now()
        while abs(height - self.current_pose.position.z) > 0.2:
            if rospy.is_shutdown():
                break
            current_time = rospy.Time.now()
            if self.current_state.mode != "OFFBOARD" and (current_time - last_request > rospy.Duration(5.0)):
                try:
                    if self.set_mode_client(custom_mode='OFFBOARD').mode_sent:
                        rospy.loginfo("OFFBOARD mode enabled")
                except rospy.ServiceException as e:
                    rospy.logerr("Service call for OFFBOARD failed: %s" % e)
                last_request = current_time
            
            dz = height - self.current_pose.position.z
            upward_vel = 1.0 * dz
            
            self.publish_velocity(upward=5)
            self.rate.sleep()
        
        rospy.loginfo(f"Altitude {self.current_pose.position.z}m reached.")    
        self.hover()
    
    def move(self, x, y, vel):
        # 以vel大小的速度水平移动到坐标(x,y)
        rospy.loginfo(f"Moving to ({x}, {y}) at {vel} m/s...")
        last_request = rospy.Time.now()
        h = self.current_pose.position.z
        initial_yaw = self.current_yaw
        
        while self._distance(self.current_pose.position.x, self.current_pose.position.y, x, y) > 0.2:
            if rospy.is_shutdown():
                break
            current_time = rospy.Time.now()
            if self.current_state.mode != "OFFBOARD" and (current_time - last_request > rospy.Duration(5.0)):
                try:
                    if self.set_mode_client(custom_mode='OFFBOARD').mode_sent:
                        rospy.loginfo("OFFBOARD mode enabled")
                except rospy.ServiceException as e:
                    rospy.logerr("Service call for OFFBOARD failed: %s" % e)
                last_request = current_time
            
            dx = x - self.current_pose.position.x
            dy = y - self.current_pose.position.y
            dz = h - self.current_pose.position.z
            dis = self._distance(self.current_pose.position.x, self.current_pose.position.y, x, y)
            
            vel_x = (dx / dis) * vel
            vel_y = (dy / dis) * vel
            
            #target_yaw = math.atan2(dy, dx)
            #d_yaw = target_yaw - self.current_yaw
            d_yaw = initial_yaw - self.current_yaw
            if d_yaw > math.pi:
                d_yaw -= 2 * math.pi
            elif d_yaw < -math.pi:
                d_yaw += 2 * math.pi
            angular_vel_z = 1.0 * d_yaw
            
            forward_vel = vel_x * math.cos(self.current_yaw) + vel_y * math.sin(self.current_yaw)
            leftward_vel = -vel_x * math.sin(self.current_yaw) + vel_y * math.cos(self.current_yaw)
            
            self.publish_velocity(forward=forward_vel, leftward=leftward_vel, upward=0.0, angular_z=angular_vel_z)
            self.rate.sleep()
            
        rospy.loginfo(f"Position ({self.current_pose.position.x}, {self.current_pose.position.y}, {self.current_pose.position.z}) reached.")
        self.hover()
    
    def hover(self):
        self.publish_command('HOVER')
        self.publish_velocity()
    
    def return_home(self):
        self.publish_command('AUTO.RTL')
        rospy.loginfo("Returning home...")
        
    def disarm(self):
        rospy.loginfo("Waiting for drone to land...")
        while self.landed_state != ExtendedState.LANDED_STATE_ON_GROUND:
            if rospy.is_shutdown():
                break
            rospy.loginfo_throttle(2, "Drone is still in the air...")
            self.rate.sleep()
        rospy.loginfo("Drone has landed.")
        self.publish_command('DISARM')
        rospy.loginfo("Commanding disarm...")
    

if __name__ == '__main__':
    try:
        con = XTDroneController()
        while not con.position_received:
            con.rate.sleep()
        
        rospy.loginfo("--- Starting Mission ---")
        
        con.offboard()
        con.arm()
        con.takeoff()
        
        con.change_altitude(20)
        con.move(120,-25,10)
        con.move(145,-25,10)
        con.move(145,25,10)
        con.return_home()
        con.disarm()
        
    except rospy.ROSInterruptException:
        pass
