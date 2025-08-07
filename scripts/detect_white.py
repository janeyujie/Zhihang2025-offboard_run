from collections import deque
from unittest import result
import rospkg
import os
import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Pose, Quaternion, Twist, TwistStamped
from mavros_msgs.msg import State, ExtendedState 
from mavros_msgs.srv import CommandBool, SetMode
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge, CvBridgeError
import cv2
from ultralytics import YOLO
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_matrix
from scipy.spatial.transform import Rotation as R
# from offboard_run.msg import Object3DStamped # 注释掉自定义消息，使用标准消息
import threading 
from queue import Queue, Empty
import queue
from message_filters import Subscriber, ApproximateTimeSynchronizer
import tf
import math
from simple_pid import PID


class ObjectTrackingHoverNode:
    def __init__(self):
        rospy.init_node('object_tracking_hover_node', anonymous=True)
        
        # 队列和锁
        self.image_queue = Queue(maxsize=3)
        self.result_queue = Queue()
        self.camera_info = None
        self.drone_pose = None
        self.camera_info_lock = threading.Lock()
        self.drone_pose_lock = threading.Lock()
        self.target_lock = threading.Lock()
        self.vehicle_id = "iris_0"
        
        # 状态变量
        self.current_state = State()
        self.current_yaw = 0.0
        self.start_tracking = False
        self.target = None
        self.target_found = False
        self.healthy_pose = Pose()
        self.target_pose_published = False
        self.current_pos = Pose()
        
        # CV和YOLO
        self.bridge = CvBridge()
        
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('offboard_run')
        model_path = os.path.join(pkg_path, 'models', 'stage2.pt')
        self.model = YOLO(model_path)
        
        # 速率
        self.image_rate = rospy.Rate(20)
        self.control_rate = rospy.Rate(20)
        self.rate = rospy.Rate(20)
        
        self.target_3d_queues = {
            "red": deque(maxlen=5),
            "yellow": deque(maxlen=5),
            "white": deque(maxlen=5)
        }
        
        self.landing_state = "SEARCHING"  # Initial state: SEARCHING, ALIGNING, DESCENDING
        
        self.motion_data_points = []  # 用于存储目标运动轨迹点 (timestamp, x, y)
        self.motion_model = None      # 存储计算出的运动模型
        self.modeling_min_points = 40 # 开始建模所需的最少数据点
        self.intercept_point = None       # 存储计算出的拦截点
        
        self.descend_speed_ms = -0.4      # 下降速度 (m/s)
        self.final_altitude_m = 0.7       # 最终触发任务完成的高度 (m)
        self.sync_altitude_m = 10.0       # 开始同步下降的高度
        self.landing_commit_alt = 1.5   # 切换到最终降落阶段的高度
        self.prediction_time_s = 0.2    # 预测未来多少秒的位置，用于补偿延迟
        self.final_align_tolerance_px = 15 # 中心对准的像素容忍度

        # Camera parameters (will be updated from CameraInfo)
        self.camera_center_x = None
        self.camera_center_y = None

        self._init_subscribers()
        self._init_publishers()
        self._init_services()
        self._init_threads()
        self._init_tf()

        rospy.loginfo("Object Tracking Hover Node Initialized.")

    def _init_tf(self):
        # 相机朝下安装，定义相机到机体坐标变换矩阵
        self.T_cam2body = np.eye(4)
        self.T_cam2body[:3, :3] = np.array([
            [ 0, -1,  0],
            [-1,  0,  0],
            [ 0,  0, -1]
        ])
        self.T_cam2body[:3, 3] = [0.0, 0.0, -0.03]

    def _init_subscribers(self):
        # 修改话题名称以适配标准PX4+MAVROS设置
        vehicle_id = self.vehicle_id  # 或者根据你的设置修改
        
        # 状态订阅 - 使用标准MAVROS话题
        rospy.Subscriber(f'/{vehicle_id}/mavros/state', State, self._state_cb)
        rospy.Subscriber(f'/{vehicle_id}/mavros/local_position/pose', PoseStamped, self._pose_cb)
        rospy.Subscriber(f'/{vehicle_id}/mavros/vision_pose/pose', PoseStamped, self._position_cb)
        
        # 图像和相机信息订阅 - 修改为标准Gazebo相机话题
        self.image_sub = Subscriber(f'/{vehicle_id}/camera/image_raw', Image)
        self.camera_info_sub = Subscriber(f'/{vehicle_id}/camera/camera_info', CameraInfo)
        self.pose_sub = Subscriber(f'/{vehicle_id}/mavros/local_position/pose', PoseStamped)

        # 同步订阅
        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.camera_info_sub, self.pose_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.synchronized_callback)
        
        # 开始信号订阅
        #rospy.Subscriber('/zhihang2025/first_man/reached', Bool, self._start_tracking_cb)
        rospy.Subscriber('/zhihang2025/third_man/reached', Bool, self._start_tracking_cb)
        #rospy.Subscriber('/start_tracking', Bool, self._start_tracking_cb)
        rospy.Subscriber('/zhihang2025/third_man/pose', Pose, self._target_pose_cb)

    def _init_publishers(self):
        vehicle_id = self.vehicle_id  # 修改为你的机体ID
        
        # 使用标准MAVROS速度控制话题
        self.velocity_pub = rospy.Publisher(f'/{vehicle_id}/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=10)
        rospy.loginfo("Velocity publisher initialized")
        # 备用发布器（如果需要）
        self.cmd_pub = rospy.Publisher(f'/xtdrone/{vehicle_id}/cmd', String, queue_size=1)
        self.mavros_vel_pub = rospy.Publisher(f'/xtdrone/{vehicle_id}/cmd_vel_flu', Twist, queue_size=10)
        rospy.loginfo("MAVROS velocity publisher initialized")
        # 移除xtdrone相关发布器，因为你使用的是标准PX4+MAVROS
        # self.cmd_pub = rospy.Publisher('/xtdrone/standard_vtol_0/cmd', String, queue_size=10)
        
        # 使用标准消息类型替代自定义消息
        self.object_pose_pub = rospy.Publisher('/detected_object_pose', PoseStamped, queue_size=10)
        self.pose_publishers = {
            "white": rospy.Publisher('/zhihang2025/iris_healthy_man/pose', Pose, queue_size=10),
            "red": rospy.Publisher('/zhihang2025/iris_bad_man/pose', Pose, queue_size=10)
        }
        rospy.loginfo("Publishers initialized")

    def _init_services(self):
        vehicle_id = self.vehicle_id  # 修改为你的机体ID
        
        try:
            # 等待MAVROS服务
            rospy.wait_for_service(f'{vehicle_id}/mavros/cmd/arming', timeout=5.0)
            rospy.wait_for_service(f'{vehicle_id}/mavros/set_mode', timeout=5.0)
            self.arming_client = rospy.ServiceProxy(f'{vehicle_id}/mavros/cmd/arming', CommandBool)
            self.set_mode_client = rospy.ServiceProxy(f'{vehicle_id}/mavros/set_mode', SetMode)
            self.use_mavros_services = True
            rospy.loginfo("MAVROS services available")
        except rospy.ROSException:
            rospy.logerr("MAVROS services not available! Please check your PX4+MAVROS setup")
            self.use_mavros_services = False

    def _init_threads(self):
        # 图像处理线程
        self.detection_thread = threading.Thread(target=self._detection_worker)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        # 控制线程
        self.control_thread = threading.Thread(target=self._control_worker)
        self.control_thread.daemon = True
        self.control_thread.start()

    def _state_cb(self, msg):
        self.current_state = msg

    def _pose_cb(self, msg):
        with self.drone_pose_lock:
            self.drone_pose = msg
            # 提取偏航角
            orientation_q = msg.pose.orientation
            _, _, self.current_yaw = tf.transformations.euler_from_quaternion(
                [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])

    def _start_tracking_cb(self, msg):
        if msg.data:
            self.start_tracking = True
            rospy.loginfo("Received start tracking signal - beginning target tracking mode")
            
    def _target_pose_cb(self, msg):
        self.healthy_pose = msg
        
    def _position_cb(self, msg):
        self.current_pos = msg.pose

    def synchronized_callback(self, image_msg, camera_info_msg, pose_msg):
        """同步回调函数"""
        with self.camera_info_lock:
            if self.camera_info is None: # 只在第一次接收时赋值
                self.camera_info = camera_info_msg
                self.camera_matrix = np.array(camera_info_msg.K).reshape((3, 3))
                self.dist_coeffs = np.array(camera_info_msg.D)
                # ADDED: Get camera center from camera info
                self.camera_center_x = self.camera_matrix[0, 2]
                self.camera_center_y = self.camera_matrix[1, 2]
                rospy.loginfo(f"Camera center initialized to ({self.camera_center_x}, {self.camera_center_y})")


        with self.drone_pose_lock:
            self.drone_pose = pose_msg

        if not self.image_queue.full():
            self.image_queue.put(image_msg)
        else:
            try:
                self.image_queue.get_nowait()
                self.image_queue.put(image_msg)
            except:
                pass
            
    def _publish_worker(self):
        while not rospy.is_shutdown():
            try:
                result = self.result_queue.get(timeout=1.0)
                object_type = result['type']
                world_coords = result['world_coords']

                if object_type in self.target_3d_queues:
                    self.target_3d_queues[object_type].append(world_coords)
                    
                    target_q = self.target_3d_queues[object_type]
                    if len(target_q) > 0:
                        smoothed_position = np.mean(target_q, axis=0)
                        # 平滑后的目标信息，包含位置和类型
                        self.smoothed_target_info = {
                            'position': smoothed_position,
                            'type': object_type
                        }
            except Empty:
                continue

    def _detection_worker(self):
        """图像处理工作线程"""
        while not rospy.is_shutdown():
            try:
                if not self.start_tracking:
                    rospy.sleep(0.1)
                    continue
                    
                msg = self.image_queue.get(timeout=1.0)
                
                # 检查必要数据
                with self.camera_info_lock:
                    if self.camera_info is None:
                        continue
                with self.drone_pose_lock:
                    if self.drone_pose is None:
                        continue
                
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                results = self.model(source=cv_image, verbose=False)
                
                self.target_found = False
                best_target = None
                best_conf = 0.0

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf.item())
                        cls = int(box.cls.item())
                        
                        if conf > best_conf and conf > 0.7:  # 添加置信度阈值
                            best_conf = conf
                            u_pixel = (x1 + x2) / 2
                            v_pixel = (y1 + y2) / 2
                            world_coords = self.get_target_position(u_pixel, v_pixel)
                            
                            if world_coords is not None:
                                best_target = {
                                    'center': (u_pixel, v_pixel),
                                    'world_coords': world_coords,
                                    'confidence': conf,
                                    'bbox': (x1, y1, x2, y2),
                                    'type': self.model.names[cls],
                                    'u_pixel': u_pixel,
                                    'v_pixel': v_pixel
                                }
                                self.target_found = True

                        # 绘制边界框
                        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(cv_image, f"{self.model.names[cls]} {conf:.2f}", 
                                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if best_target:
                    self.target = best_target
                    
                    # 发布检测到的目标位置（使用标准消息）
                    target_pose = PoseStamped()
                    target_pose.header.stamp = rospy.Time.now()
                    target_pose.header.frame_id = "map"
                    target_pose.pose.position.x = best_target['world_coords'][0]
                    target_pose.pose.position.y = best_target['world_coords'][1]
                    target_pose.pose.position.z = best_target['world_coords'][2]
                    self.object_pose_pub.publish(target_pose)
                    
                    rospy.loginfo(f"Target found at pixel: {self.target}, world: {best_target['world_coords']}")
                else:
                    self.target = None
                    self.target_found = False
                    rospy.loginfo("No target found")

                # 显示图像
                scale = 0.5
                resized_image = cv2.resize(cv_image, (0, 0), fx=scale, fy=scale)
                cv2.imshow("Target Tracking", resized_image)
                cv2.waitKey(1)
                
            except queue.Empty:
                continue
            except Exception as e:
                rospy.logerr(f"Detection error: {e}")
                
            self.image_rate.sleep()

    def _control_worker(self):
        """
        控制工作线程，实现“对准-下降” 的精确降落逻辑。
        """
        rospy.loginfo("Control worker started with Intercept-and-Wait logic.")
        
        while not rospy.is_shutdown():
            
            if not self.start_tracking:
                rospy.sleep(0.1)
                continue

            if self.drone_pose is None or self.camera_info is None:
                #self._publish_velocity_command(0.0, 0.0, 0.0) # 保持悬停
                self.hover()
                self.landing_state = "SEARCHING"
                rospy.sleep(0.1)
                continue

            '''with self.target_lock:
                target_found_local = self.target_found
                target_local = self.target'''

            # ------------------ 状态机逻辑 ------------------
            self._ensure_offboard_mode()
            current_altitude = self.drone_pose.pose.position.z
            vel = 1.0
            
            # 阶段 1：搜索
            if self.landing_state == "SEARCHING":
                if self.target is None:
                    rospy.loginfo_throttle(2, "State: SEARCHING - No target. Moving to initial search pose.")
                    #self.move(self.healthy_pose.position.x, self.healthy_pose.position.y, 1.5)
                    dx = self.healthy_pose.position.x - self.drone_pose.pose.position.x
                    dy = self.healthy_pose.position.y - self.drone_pose.pose.position.y
                    dist = math.sqrt(dx**2 + dy**2)
                    if dist > 0.5:
                        rospy.loginfo_throttle(2, f"State: SEARCHING - Moving to search pose, distance: {dist:.2f}m")
                        vel_x = (dx / dist) * vel
                        vel_y = (dy / dist) * vel
                        f_vel = vel_x * math.cos(self.current_yaw) + vel_y * math.sin(self.current_yaw)
                        l_vel = -vel_x * math.sin(self.current_yaw) + vel_y * math.cos(self.current_yaw)
                        self._publish_velocity_command(f_vel, l_vel, 0.0)
                    else:
                        rospy.loginfo_throttle(5, "State: SEARCHING - Arrived at search pose. Hovering.")
                        self.hover()
                else:
                    rospy.loginfo("Target acquired. Switching to MODELING.")
                    self.landing_state = "MODELING"
                    self.motion_data_points = [] # 清空旧数据
                    self.hover()
            
            # 阶段 2: 运动建模
            elif self.landing_state == "MODELING":
                if self.target is None:
                    rospy.logwarn("Target lost during modeling. Returning to SEARCHING.")
                    self.landing_state = "SEARCHING"
                    continue
                
                # 收集数据点
                current_time = rospy.Time.now().to_sec()
                world_coords = self.target['world_coords']
                self.motion_data_points.append((current_time, world_coords[0], world_coords[1]))
                rospy.loginfo_throttle(1, f"State: MODELING - Collecting data points ({len(self.motion_data_points)}/{self.modeling_min_points}).")
                
                # 悬停在同步高度，以获得稳定的观测视角
                self.hover()

                # 数据足够时，尝试计算模型
                if len(self.motion_data_points) >= self.modeling_min_points:
                    if self._calculate_motion_model():
                        rospy.loginfo("Motion model created. Switching to SYNCHRONIZING.")
                        self.landing_state = "INTERCEPTING"
                        self.intercept_point = None
            
            elif self.landing_state == "INTERCEPTING":
                if self.motion_model is None:
                    rospy.logerr("In INTERCEPTING state but no motion model available. Returning to SEARCHING.")
                    self.landing_state = "SEARCHING"
                    continue
                
                # 只计算一次拦截点
                if self.intercept_point is None:
                    # 简单策略：选择圆上离无人机当前位置最近的点作为拦截点
                    center = self.motion_model['center']
                    radius = self.motion_model['radius']
                    drone_pos_xy = np.array([self.drone_pose.pose.position.x, self.drone_pose.pose.position.y])
                    
                    vec_to_drone = drone_pos_xy - center
                    vec_on_circle = (vec_to_drone / np.linalg.norm(vec_to_drone)) * radius
                    self.intercept_point = center + vec_on_circle
                    rospy.loginfo(f"Calculated intercept point: ({self.intercept_point[0]:.2f}, {self.intercept_point[1]:.2f})")

                    dx = self.intercept_point[0] - self.drone_pose.pose.position.x
                    dy = self.intercept_point[1] - self.drone_pose.pose.position.y
                    dist = math.sqrt(dx**2 + dy**2)
                    if dist > 0.5:
                        rospy.loginfo_throttle(1, f"State: INTERCEPTING - Moving to intercept point, distance: {dist:.2f}m")
                        vel_x = (dx / dist) * vel
                        vel_y = (dy / dist) * vel
                        f_vel = vel_x * math.cos(self.current_yaw) + vel_y * math.sin(self.current_yaw)
                        l_vel = -vel_x * math.sin(self.current_yaw) + vel_y * math.cos(self.current_yaw)
                        self._publish_velocity_command(f_vel, l_vel, 0.0)
                    else:
                        rospy.loginfo("Reached intercept point. Switching to DESCENDING_TO_WAIT.")
                        self.landing_state = "DESCENDING_TO_WAIT"
                        self.hover()
                '''# 移动到拦截点上空
                self.move(self.intercept_point[0], self.intercept_point[1], 1.0)

                # 检查是否到达
                dist_to_intercept = self._distance(self.drone_pose.pose.position.x, self.drone_pose.pose.position.y, self.intercept_point[0], self.intercept_point[1])
                if dist_to_intercept < 0.5: # 到达容忍度0.5米
                    rospy.loginfo("Reached intercept point. Switching to DESCENDING_TO_WAIT.")
                    self.landing_state = "DESCENDING_TO_WAIT"
                    #self.pid_z.setpoint = self.final_altitude_m'''
                    
            # 阶段 4: 下降至等待高度
            elif self.landing_state == "DESCENDING_TO_WAIT":
                rospy.loginfo_throttle(1, f"State: DESCENDING_TO_WAIT - Descending to {self.final_altitude_m}m. Current: {current_altitude:.2f}m")
                # 使用PID控制器进行平滑垂直下降
                #vz_command = self.pid_z(current_altitude)
                # 保持水平位置不动
                self._publish_velocity_command(vx=0.0, vy=0.0, vz=-0.6)

                # 检查是否到达
                if abs(current_altitude - self.final_altitude_m) < 0.05: # 到达容忍度0.1米
                    rospy.loginfo(f"Reached waiting altitude. Switching to WAITING_FOR_TARGET.")
                    self.landing_state = "WAITING_FOR_TARGET"
                    self.hover()
            
            # 阶段 5: 等待目标通过
            elif self.landing_state == "WAITING_FOR_TARGET":
                rospy.loginfo_throttle(2, "State: WAITING_FOR_TARGET - Hovering at low altitude, waiting for target.")
                # 保持悬停
                self.hover()
                
                # 检查目标是否在摄像头中心
                if self.target is not None:
                    error_x_px = self.target['u_pixel'] - self.camera_center_x
                    error_y_px = self.target['v_pixel'] - self.camera_center_y
                    if (abs(error_x_px) < self.final_align_tolerance_px and 
                        abs(error_y_px) < self.final_align_tolerance_px):
                        
                        rospy.loginfo("Target is in camera center! Switching to FINALIZING.")
                        self.landing_state = "FINALIZING"

            # 阶段 6: 任务完成
            elif self.landing_state == "FINALIZING":
                rospy.loginfo("State: FINALIZING - Publishing success topic.")
                self.hover() # 最后发布前也保持悬停

                # 发布任务完成话题
                if self.target is not None and self.target['type'] == 'white':
                    pose_msg = Pose()
                    #pose_msg.position.x = self.target['world_coords'][0]
                    pose_msg.position.x = self.current_pos.position.x
                    #pose_msg.position.y = self.target['world_coords'][1]
                    pose_msg.position.y = self.current_pos.position.y
                    pose_msg.position.z = self.target['world_coords'][2]
                    self.pose_publishers['white'].publish(pose_msg)
                    rospy.loginfo(f"SUCCESS: Target '{self.target['type']}' pose published.")
                    self.target_pose_published = True # 触发主线程退出
                else:
                    rospy.logwarn("Finalizing but target is lost or not white. Cannot publish.")
            
            self.control_rate.sleep()
    
    def _ensure_offboard_mode(self):
        """确保飞行器处于OFFBOARD模式"""
        '''if self.use_mavros_services:
            if (self.current_state.mode != "OFFBOARD" and 
                hasattr(self, '_last_mode_request') and 
                (rospy.Time.now() - self._last_mode_request).to_sec() > 2.0):
                
                try:
                    offb_set_mode = SetMode()
                    offb_set_mode.custom_mode = 'OFFBOARD'
                    offb_set_mode.base_mode = 'OFFBOARD'
                    if self.set_mode_client(offb_set_mode).mode_sent:
                        rospy.loginfo("OFFBOARD mode enabled via MAVROS")
                    self._last_mode_request = rospy.Time.now()
                except rospy.ServiceException as e:
                    rospy.logerr(f"MAVROS service call failed: {e}")
        
        if not hasattr(self, '_last_mode_request'):
            self._last_mode_request = rospy.Time.now()'''
        self.publish_command('OFFBOARD')
        #rospy.loginfo("Commanding offboard...")

    def _distance(self, x1, y1, x2, y2):
        return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    
    def publish_command(self, command):
        """发布一个字符串命令，例如 'ARM', 'AUTO.TAKEOFF'"""
        cmd_msg = String()
        cmd_msg.data = command
        self.cmd_pub.publish(cmd_msg)
        rospy.loginfo(f"Published command: {command}")

    def _publish_velocity_command(self, vx=0.0, vy=0.0, vz=0.0, az=0.0):
        """发布速度指令到MAVROS,发布速度指令 (机体坐标系：前左上)"""
        # 使用TwistStamped消息
        twist_stamped = TwistStamped()
        twist_stamped.header.stamp = rospy.Time.now()
        twist_stamped.header.frame_id = "base_link"
        twist_stamped.twist.linear.x = vx
        twist_stamped.twist.linear.y = vy
        twist_stamped.twist.linear.z = vz
        twist_stamped.twist.angular.x = 0.0
        twist_stamped.twist.angular.y = 0.0
        twist_stamped.twist.angular.z = az
        self.velocity_pub.publish(twist_stamped)
        
        # 备用：也发布Twist消息
        twist = Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        twist.linear.z = vz
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = az
        self.mavros_vel_pub.publish(twist)
        #rospy.loginfo(f"Published twist: {twist}")

    def move(self, x, y, vel):
        # 以vel大小的速度水平移动到坐标(x,y)
        rospy.loginfo(f"Moving to ({x}, {y}) at {vel} m/s...")
        last_request = rospy.Time.now()
        h = self.drone_pose.pose.position.z
        initial_yaw = self.current_yaw
        
        while self._distance(self.drone_pose.pose.position.x, self.drone_pose.pose.position.y, x, y) > 0.5:
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
            
            dx = x - self.drone_pose.pose.position.x
            dy = y - self.drone_pose.pose.position.y
            dz = h - self.drone_pose.pose.position.z
            dis = self._distance(self.drone_pose.pose.position.x, self.drone_pose.pose.position.y, x, y)
            
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
            
            self._publish_velocity_command(forward_vel, leftward_vel, 0.0, az=angular_vel_z)
            self.rate.sleep()
            
        rospy.loginfo(f"Position ({self.drone_pose.pose.position.x}, {self.drone_pose.pose.position.y}, {self.drone_pose.pose.position.z}) reached.")

    def hover(self):
        self.publish_command('HOVER')
        self._publish_velocity_command()
    
    # ============ pose calculate ==========
    def normalize(self, v):
        return v / np.linalg.norm(v)
    
    def pixel_to_camera_ray(self, u, v):
        pixel_h = np.array([u, v, 1.0])
        ray_cam = np.linalg.inv(self.camera_matrix) @ pixel_h
        return self.normalize(ray_cam)

    def camera_ray_to_world(self, ray_cam, orientation):
        # 相机坐标 → 机体坐标
        ray_body = self.T_cam2body[:3, :3] @ ray_cam
        # 机体坐标 → 世界坐标
        q = orientation
        R_world_body = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        ray_world = R_world_body @ ray_body
        return self.normalize(ray_world)

    def get_target_position(self, u, v, ground_z=0.0):
        if self.camera_info is None or self.drone_pose is None:
            return None
            
        # 像素坐标转相机方向向量
        ray_cam = self.pixel_to_camera_ray(u, v)

        # 相机方向转世界坐标方向向量
        pose = self.drone_pose.pose
        ray_world = self.camera_ray_to_world(ray_cam, pose.orientation)

        # 计算相机在世界坐标系中的位置
        cam_offset = self.T_cam2body[:3, 3]
        R_world_body = quaternion_matrix([pose.orientation.x,
                                          pose.orientation.y,
                                          pose.orientation.z,
                                          pose.orientation.w])[:3, :3]
        drone_position = np.array([pose.position.x, pose.position.y, pose.position.z])
        v1 = np.array([2.52, 2.67, 0.1])
        v2 = np.array([0, 0, 0])
        #drone_position = drone_position + v1 + v2
        cam_position = drone_position + R_world_body @ cam_offset

        # 计算射线与地面的交点
        t = (ground_z - cam_position[2]) / ray_world[2]
        if t < 0:
            return None

        target_world = cam_position + t * ray_world
        return target_world
    
    # 将这两个新函数添加到你的 QuadcopterCommander 类中

    def _calculate_motion_model(self):
        """
        使用最小二乘法拟合圆形轨迹，并计算角速度。
        """
        if len(self.motion_data_points) < self.modeling_min_points:
            return False

        # 提取x, y坐标
        points = np.array([p[1:] for p in self.motion_data_points])
        x = points[:, 0]
        y = points[:, 1]
        
        # 最小二乘法拟合圆
        # x^2 + y^2 + a*x + b*y + c = 0
        A = np.vstack([x, y, np.ones(len(x))]).T
        B = -(x**2 + y**2)
        
        try:
            # 求解 a, b, c
            coeffs, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
            a, b, c = coeffs
            
            # 计算圆心和半径
            cx = -a / 2
            cy = -b / 2
            radius = np.sqrt(cx**2 + cy**2 - c)
        except np.linalg.LinAlgError:
            rospy.logwarn("Circle fitting failed. Not enough unique points.")
            self.motion_data_points = [] # 清空数据重新收集
            return False


        # 计算角速度
        angles = [math.atan2(p[2] - cy, p[1] - cx) for p in self.motion_data_points]
        times = [p[0] for p in self.motion_data_points]
        
        # 使用差分计算角速度，并取平均值
        angular_velocities = []
        for i in range(1, len(angles)):
            delta_angle = angles[i] - angles[i-1]
            # 处理角度跳跃 (从 +pi 到 -pi 或反之)
            if delta_angle > math.pi:
                delta_angle -= 2 * math.pi
            elif delta_angle < -math.pi:
                delta_angle += 2 * math.pi
            
            delta_time = times[i] - times[i-1]
            if delta_time > 1e-6: # 避免除以零
                angular_velocities.append(delta_angle / delta_time)

        if not angular_velocities:
            rospy.logwarn("Could not calculate angular velocity.")
            return False
            
        omega = np.mean(angular_velocities)
        
        # 半径必须在合理范围内 (例如 3m +/- 1.5m)
        if not (1.5 < radius < 4.5):
            rospy.logwarn(f"Calculated radius {radius:.2f}m is out of expected range. Retrying.")
            self.motion_data_points = [] # 清空数据重新收集
            return False

        # 保存模型
        self.motion_model = {
            'center': np.array([cx, cy]),
            'radius': radius,
            'omega': omega,
            'last_angle': angles[-1],
            'last_time': times[-1]
        }
        rospy.loginfo(f"Motion model UPDATED: Center({cx:.2f}, {cy:.2f}), Radius={radius:.2f}, Omega={omega:.2f} rad/s")
        return True

    def _predict_target_position(self):
        """
        根据已建立的运动模型，预测目标在未来某个时刻的位置。
        """
        if self.motion_model is None:
            return None
        
        model = self.motion_model
        
        # 计算从上次更新模型到现在的时间差
        time_since_last_update = rospy.Time.now().to_sec() - model['last_time']
        
        # 加上我们想要预测的未来时间
        total_time_delta = time_since_last_update + self.prediction_time_s

        # 预测新的角度
        predicted_angle = model['last_angle'] + model['omega'] * total_time_delta
        
        # 计算预测的世界坐标
        cx, cy = model['center']
        R = model['radius']
        predicted_x = cx + R * math.cos(predicted_angle)
        predicted_y = cy + R * math.sin(predicted_angle)
        
        return np.array([predicted_x, predicted_y])

    def run(self):
        rospy.loginfo("Object Tracking Hover Node is running...")
        rospy.loginfo("Waiting for /start_tracking signal to begin target tracking...")
        rospy.spin()


if __name__ == '__main__':
    try:
        node = ObjectTrackingHoverNode()
        #node.run()
        rate = rospy.Rate(1)
        rospy.loginfo("Object Tracking Hover Node is running...")
        rospy.loginfo("Waiting for /start_tracking signal to begin target tracking...")
        while not rospy.is_shutdown():
            if node.target_pose_published:
                rospy.loginfo("Ending signal received. Shutting down the node.")
                rospy.signal_shutdown("Mission part 1 completed, shutdown signal received")
                break 
                
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    finally:
        rospy.loginfo("Object Tracking Hover node terminated.")
        cv2.destroyAllWindows()
        exit(0)


