import rospkg
import os
import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Pose
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
from tf.transformations import quaternion_matrix
from scipy.spatial.transform import Rotation as R
from offboard_run.msg import Object3DStamped # 导入自定义消息类型 (已定义并编译)
from std_msgs.msg import String, Bool
from collections import deque

class ObjectLocalizationNode:
    def __init__(self):
        rospy.init_node('object_localization_node_stage_1', anonymous=True)

        self.bridge = CvBridge()
        
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('offboard_run')
        model_path = os.path.join(pkg_path, 'models', 'stage1.pt')
        self.model = YOLO(model_path)

        self.camera_info = None
        self.drone_pose = None
        self.can_start = False
        self.ending = False

        # 队列保存检测结果
        self.target_queues = {
            "red": deque(maxlen=10),
            "yellow": deque(maxlen=10),
            "white": deque(maxlen=10)
        }

        # 发布平滑后的目标位置, 显然，目标会移动，yellow & white坐标是错的
        self.pose_publishers = {
            "red": rospy.Publisher('/zhihang2025/first_man/pose', Pose, queue_size=10),
            "yellow": rospy.Publisher('/zhihang2025/second_man/pose', Pose, queue_size=10),
            "white": rospy.Publisher('/zhihang2025/third_man/pose', Pose, queue_size=10)
        }

        # 订阅相机图像和信息
        rospy.Subscriber('/standard_vtol_0/camera/image_raw', Image, self.image_callback, queue_size=1, buff_size=2**24)
        rospy.Subscriber('/standard_vtol_0/camera/camera_info', CameraInfo, self.camera_info_callback, queue_size=1)
        # 订阅无人机姿态
        rospy.Subscriber('/standard_vtol_0/mavros/local_position/pose', PoseStamped, self.drone_pose_callback, queue_size=1)
        rospy.Subscriber('/standard_vtol_0/waypoint_reached', Bool, self._reached_cb)
        rospy.Subscriber("/standard_vtol_0/search_completed", Bool, self._ending_cb)
        self._init_tf()  # 初始化相机到机体坐标变换矩阵

        # 发布3D目标位置和种类
        self.object_3d_pub = rospy.Publisher('/detected_objects_3d', Object3DStamped, queue_size=10)  # 10Hz发布一次

        rospy.loginfo("Object Localization Node Initialized.")
    def _init_tf(self):
        self.T_cam2body = np.eye(4)
        self.T_cam2body[:3, :3] = np.array([
            [ 0, -1,  0],
            [-1,  0,  0],
            [ 0,  0, -1]
        ])
        self.T_cam2body[:3, 3] = [0.0, 0.0, -0.03]  # 相机在无人机下方 3cm, check .sdf

    def camera_info_callback(self, msg):
        self.camera_info = msg
        # 提取相机内参矩阵 K 和畸变系数 D
        self.camera_matrix = np.array(msg.K).reshape((3, 3))
        self.dist_coeffs = np.array(msg.D)

    def drone_pose_callback(self, msg):
        self.drone_pose = msg
    
    def _reached_cb(self, msg):
        if msg.data:
            self.can_start = True
            rospy.loginfo("Reached waypoint, starting to detect")
            
    def _ending_cb(self, msg):
        if msg.data:
            self.ending = True
            rospy.loginfo("Mission end...")
    
    def image_callback(self, msg):
        if not self.can_start:
            return
        
        if self.camera_info is None or self.drone_pose is None:
            rospy.logdebug("Waiting for camera info and drone pose...")
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            XTDrone_pose = self.drone_pose # 同步读取数据
        except Exception as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # 执行YOLOv8检测
        results = self.model(source=cv_image, verbose=False)

        found_target = False

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # 获取2D边界框和类别信息
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf.item())
                cls = int(box.cls.item())
                object_type = self.model.names[cls]

                # 获取边界框中心点 (u, v)
                u_pixel = (x1 + x2) / 2
                v_pixel = (y1 + y2) / 2

                target_position = self.get_target_position(u_pixel, v_pixel, XTDrone_pose.pose)
                
                if target_position is not None and conf > 0.8:  # 置信度阈值
                    found_target = True
                    rospy.loginfo(f"Detected {object_type} at 3D position (World): {target_position}")

                    # 分类保存到队列
                    if object_type in self.target_queues:
                        self.target_queues[object_type].append(target_position)
                        self.publish_smoothed_position(object_type, target_position)

                    # 发布消息
                    obj_msg = Object3DStamped()
                    obj_msg.header.stamp = rospy.Time.now()
                    obj_msg.header.frame_id = "world"
                    obj_msg.object_type = object_type
                    obj_msg.position.x = float(target_position[0])
                    obj_msg.position.y = float(target_position[1])
                    obj_msg.position.z = float(target_position[2])
                    obj_msg.confidence = float(conf)
                    self.object_3d_pub.publish(obj_msg)

                # 绘制边界框
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(cv_image, f"{object_type} {conf:.2f} pose:{target_position}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示图像（仅在检测到目标时）
        if found_target:
            scale = 0.5  # 缩小一半
            resized_image = cv2.resize(cv_image, (0, 0), fx=scale, fy=scale)
            cv2.imshow("YOLOv8 Detection", resized_image)
            cv2.waitKey(1)

    def publish_smoothed_position(self, object_type, target_position):
        """对目标位置进行滤波平滑处理并发布"""
        queue = self.target_queues[object_type]
        if object_type == "red" and len(queue) > 5:
            smoothed_position = np.mean(queue, axis=0)
            pose_msg = Pose()
            pose_msg.position.x = smoothed_position[0]
            pose_msg.position.y = smoothed_position[1]
            pose_msg.position.z = smoothed_position[2]
            self.pose_publishers[object_type].publish(pose_msg)
            rospy.loginfo(f"Published smoothed position for {object_type}: {smoothed_position}")
        if object_type == "yellow":
            pose_msg = Pose()
            pose_msg.position.x = target_position[0]
            pose_msg.position.y = target_position[1]
            pose_msg.position.z = target_position[2]
            self.pose_publishers[object_type].publish(pose_msg)
            rospy.loginfo(f"Published position for {object_type}: {target_position}")

        if object_type == "white" and len(queue) > 3:
            pose_msg = Pose()
            pose_msg.position.x = target_position[0]
            pose_msg.position.y = target_position[1]
            pose_msg.position.z = target_position[2]
            self.pose_publishers[object_type].publish(pose_msg)
            rospy.loginfo(f"Published position for {object_type}: {target_position}")

    def normalize(self, v):
        return v / np.linalg.norm(v)
    
    def pixel_to_camera_ray(self, u, v):
        pixel_h = np.array([u, v, 1.0])  # 齐次像素
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

    def get_target_position(self, u, v, pose, ground_z=0.0):
        if self.camera_info is None or self.drone_pose is None:
            rospy.logwarn("Waiting for camera info and drone pose...")
            return None
        # Step 1: 像素坐标转相机方向向量
        ray_cam = self.pixel_to_camera_ray(u, v)

        # Step 2: 相机方向转世界坐标方向向量
        # pose = self.drone_pose.pose
        ray_world = self.camera_ray_to_world(ray_cam, pose.orientation)

        # Step 3: 相机位置（考虑相机相对无人机机体的偏移）
        cam_offset = self.T_cam2body[:3, 3]  # 平移向量
        R_world_body = quaternion_matrix([pose.orientation.x,
                                          pose.orientation.y,
                                          pose.orientation.z,
                                          pose.orientation.w])[:3, :3]
        drone_position = np.array([pose.position.x, pose.position.y, pose.position.z])
        v1 = np.array([2.3, 0.4, 1.3]) # local相对World的偏移
        v2 = np.array([0, 0, -0.05]) # 相机相对于机体的偏移
        drone_position = drone_position + v1 + v2  # 假设无人机位置偏移
        # 计算相机在世界坐标系中的位置
        cam_position = drone_position + R_world_body @ cam_offset
        # rospy.loginfo(f"{self.drone_pose}, {self.T_cam2body}, {self.camera_info}, ")

        # Step 4: 计算 ray 与地面 Z=ground_z 的交点 t = -z / dz
        t = (ground_z - cam_position[2]) / ray_world[2]
        if t < 0:
            rospy.logwarn("Ray does not intersect ground plane.")
            return None

        target_world = cam_position + t * ray_world
        v3 = np.array([5.2, -0.5, 0])
        target_world -= v3 # 人物相对于靶心的偏移
        return target_world

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = ObjectLocalizationNode()
        
        rospy.loginfo("Waiting for waypoint reached...")
        rate = rospy.Rate(1)
        while not rospy.is_shutdown() and not node.can_start:
            rospy.loginfo_throttle(10, "Waiting for signal...")
            rate.sleep()
        rospy.loginfo("Signal received! Starting to detect...")
        node.run()
        while not rospy.is_shutdown():
            if node.ending:
                rospy.loginfo("Ending signal received. Shutting down the node.")
                rospy.signal_shutdown("Mission part 1 completed, shutdown signal received")
                break 
                
            rate.sleep()

    except rospy.ROSInterruptException:
        rospy.loginfo("Node interrupted by user")
    except Exception as e:
        rospy.logfatal(f"An unhandled error occurred: {e}", exc_info=True)
    finally:
        rospy.loginfo("Object Localization node is shutting down.")
        cv2.destroyAllWindows()

