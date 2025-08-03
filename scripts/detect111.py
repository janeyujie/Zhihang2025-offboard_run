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
        self.vehicle_id = "iris_0"
        
        # 状态变量
        self.current_state = State()
        self.current_yaw = 0.0
        self.start_tracking = False
        self.target = None
        self.target_found = False
        self.healthy_man_pose_published = False
        
        # CV和YOLO
        self.bridge = CvBridge()
        
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('offboard_run')
        model_path = os.path.join(pkg_path, 'models', 'stage2.pt')
        self.model = YOLO(model_path)
        
        # PID控制器参数调整
        self.pid_x = PID(Kp=2.5, Ki=0.3, Kd=0.8, setpoint=0)
        self.pid_y = PID(Kp=2.5, Ki=0.3, Kd=0.8, setpoint=0)
        self.pid_z = PID(Kp=1.5, Ki=0.2, Kd=0.5, setpoint=0)
        self.pid_x.output_limits = (-2.0, 2.0)
        self.pid_y.output_limits = (-2.0, 2.0)
        self.pid_z.output_limits = (-1.0, 1.0)
        
        # 速率
        self.image_rate = rospy.Rate(20)
        self.control_rate = rospy.Rate(20)
        self.rate = rospy.Rate(20)
        
        self.target_3d_queues = {
            "red": deque(maxlen=5),
            "yellow": deque(maxlen=5),
            "white": deque(maxlen=5)
        }

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
        rospy.Subscriber('/zhihang2025/first_man/reached', Bool, self._start_tracking_cb)
        rospy.Subscriber('/zhihang2025/third_man/reached', Bool, self._start_tracking_cb)
        #rospy.Subscriber('/start_tracking', Bool, self._start_tracking_cb)

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

    def synchronized_callback(self, image_msg, camera_info_msg, pose_msg):
        """同步回调函数"""
        with self.camera_info_lock:
            self.camera_info = camera_info_msg
            self.camera_matrix = np.array(camera_info_msg.K).reshape((3, 3))
            self.dist_coeffs = np.array(camera_info_msg.D)

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
                        
                        if conf > best_conf and conf > 0.5:  # 添加置信度阈值
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
                                    'type': self.model.names[cls]
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
        """控制工作线程"""
        target_altitude = 0.5
        position_tolerance = 0.3
        altitude_tolerance = 1.5
        
        # 确保飞行器处于OFFBOARD模式
        self._ensure_offboard_mode()
        rospy.loginfo("Control worker started - waiting for target tracking to begin")
        
        last_request = rospy.Time.now()
        flag = 0
        pos3_distance = None
        
        # 初始化变量用于记录上一次的XY轴速度
        last_control_x = 0.0
        last_control_y = 0.0
        
        while not rospy.is_shutdown():
            if not self.start_tracking:
                rospy.sleep(0.1)
                continue
                
            if self.drone_pose is None or self.camera_info is None:
                rospy.sleep(0.1)
                continue
                
            try:
                # 定期检查并设置OFFBOARD模式
                # self._ensure_offboard_mode()
                rospy.loginfo("Control worker active - tracking target")
                
                current_altitude = self.drone_pose.pose.position.z
                
                if self.target_found and self.target is not None:
                    # 计算目标世界坐标
                    world_coords = self.target['world_coords']
                    object_type = self.target['type']
                    rospy.loginfo(f"worker:::::Target world coordinates: {world_coords}")
                    
                    if world_coords is not None:
                        # 计算水平位置误差
                        error_x = world_coords[0] - self.drone_pose.pose.position.x
                        error_y = world_coords[1] - self.drone_pose.pose.position.y
                        rospy.loginfo(f"error_x: {error_x}; error_y: {error_y}")
                        
                        # 计算到目标的水平距离
                        horizontal_distance = np.sqrt(error_x**2 + error_y**2)
                        if horizontal_distance > 2.0:
                            self.move(world_coords[0], world_coords[1], 2)
                            continue
                        
                        # 计算高度误差
                        altitude_error = current_altitude - target_altitude
                        
                        '''# 水平位置PID控制
                        abs_error_x = abs(error_x)
                        abs_error_y = abs(error_y)
                        
                        control_x = self.pid_x(abs_error_x)
                        control_y = self.pid_y(abs_error_y)
                        
                        # 根据误差符号确定方向
                        if error_x < 0:
                            control_x = -control_x
                        if error_y < 0:
                            control_y = -control_y'''
                        control_x = self.pid_x(error_x)
                        control_y = self.pid_y(error_y)
                            
                        # 动态调整控制速度
                        '''if horizontal_distance > 2.0:
                            speed_factor = min(2.0, horizontal_distance * 0.8)
                            control_x = np.clip(control_x * speed_factor, -6.0, 6.0)
                            control_y = np.clip(control_y * speed_factor, -6.0, 6.0)
                        elif horizontal_distance > 1.0:
                            speed_factor = 1.5
                            control_x = np.clip(control_x * speed_factor, -4.0, 4.0)
                            control_y = np.clip(control_y * speed_factor, -4.0, 4.0)
                        else:
                            speed_factor = 0.8
                            control_x = np.clip(control_x * speed_factor, -2.0, 2.0)
                            control_y = np.clip(control_y * speed_factor, -2.0, 2.0)
                        
                        # 高度控制
                        if abs(altitude_error) > altitude_tolerance:
                            if altitude_error > 0:
                                control_z = -min(1.5, abs(altitude_error) * 1.0)
                            else:
                                control_z = min(1.5, abs(altitude_error) * 1.0)
                        else:
                            control_z = 0.0'''
                        # 根据距离选择控制策略
                        if horizontal_distance > 1.5:
                            speed_factor = 1.5 
                            control_x = self.pid_x(error_x) * speed_factor
                            control_y = self.pid_y(error_y) * speed_factor
                            control_z = self.pid_z(-altitude_error)
                        else:
                            speed_factor = 0.8
                            control_x = self.pid_x(error_x) * speed_factor
                            control_y = self.pid_y(error_y) * speed_factor                            
                            # 在近距离时，采用缓慢下降的策略
                            if abs(altitude_error) > altitude_tolerance:
                                if altitude_error > 0:
                                    control_z = -min(1.5, abs(altitude_error) * 1.5)
                                else:
                                    control_z = min(1.5, abs(altitude_error) * 1.5)
                            else:
                                control_z = 0.0
                        
                        forward_vel = control_x * math.cos(self.current_yaw) + control_y * math.sin(self.current_yaw)
                        leftward_vel = -control_x * math.sin(self.current_yaw) + control_y * math.cos(self.current_yaw)
                        
                        # 检查是否已到达目标位置并悬停
                        if (horizontal_distance < position_tolerance and 
                            abs(altitude_error) < altitude_tolerance):
                            self._publish_velocity_command(0.0, 0.0, 0.0)
                            rospy.loginfo(f"TARGET REACHED! Hovering at {current_altitude:.2f}m above target")
                            pose_msg = Pose()
                            pose_msg.position.x = world_coords[0]
                            pose_msg.position.y = world_coords[1]
                            pose_msg.position.z = world_coords[2]
                            self.pose_publishers[object_type].publish(pose_msg)
                        else:
                            self._publish_velocity_command(forward_vel, leftward_vel, control_z)
                            
                            rospy.loginfo(f"Tracking target - Control: x={control_x:.2f}, y={control_y:.2f}, z={control_z:.2f}, "
                                        f"h_dist={horizontal_distance:.2f}m, alt={current_altitude:.2f}m, "
                                        f"alt_error={altitude_error:.2f}m, mode={self.current_state.mode}")
                    
                else:
                    # 没有找到目标时悬停在当前位置
                    self._publish_velocity_command(0.0, 0.0, 0.0)
                    rospy.loginfo_throttle(2, f"No target detected - hovering and searching... mode={self.current_state.mode}")
                    
            except Exception as e:
                rospy.logerr(f"Control error: {e}")
                
            self.control_rate.sleep()
    
    def _ensure_offboard_mode(self):
        """确保飞行器处于OFFBOARD模式"""
        if self.use_mavros_services:
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
            self._last_mode_request = rospy.Time.now()

    def _distance(self, x1, y1, x2, y2):
        return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    
    def publish_command(self, command):
        """发布一个字符串命令，例如 'ARM', 'AUTO.TAKEOFF'"""
        cmd_msg = String()
        cmd_msg.data = command
        self.cmd_pub.publish(cmd_msg)
        rospy.loginfo(f"Published command: {command}")

    def _publish_velocity_command(self, vx, vy, vz, az=0.0):
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
        rospy.loginfo(f"Published twist: {twist}")

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

    def run(self):
        rospy.loginfo("Object Tracking Hover Node is running...")
        rospy.loginfo("Waiting for /start_tracking signal to begin target tracking...")
        rospy.spin()


if __name__ == '__main__':
    try:
        node = ObjectTrackingHoverNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        rospy.loginfo("Object Tracking Hover node terminated.")
        cv2.destroyAllWindows()
        exit(0)