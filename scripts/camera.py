#!/usr/bin/env python
# -*- coding: UTF-8 -*-

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

class CameraViewer:
    def __init__(self):
        """
        初始化节点，仅用于显示摄像头画面
        """
        rospy.init_node("camera_viewer_node")

        self.bridge = CvBridge()
        
        # --- 订阅摄像头话题 ---
        # 只需要这一个订阅者
        rospy.Subscriber("/iris_0/camera/image_raw", Image, self._image_callback)
        #rospy.Subscriber("/iris_0/realsense/depth_camera/color/image_raw", Image, self._image_callback)
        #rospy.Subscriber("/iris_0/realsense/depth_camera/depth/image_raw", Image, self._image_callback)
        #rospy.Subscriber("/iris_0/stereo_camera/left/image_raw", Image, self._image_callback)
        #rospy.Subscriber("/iris_0/stereo_camera/right/image_raw", Image, self._image_callback)
        
        rospy.loginfo("Camera viewer initialized. Waiting for image stream...")
        rospy.loginfo("Press Ctrl+C in this terminal to close the window.")

    def _image_callback(self, msg):
        """
        这个回调函数会在每接收到一帧图像时被调用
        """
        try:
            # 将ROS图像消息转换为OpenCV图像格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 在一个名为"Iris Camera"的窗口中显示图像
            cv2.imshow("Iris Camera", cv_image)
            
            # 刷新GUI窗口，这是cv2.imshow()正常工作所必需的
            cv2.waitKey(1)
            
        except CvBridgeError as e:
            rospy.logerr(e)

    def shutdown_hook(self):
        """
        在节点关闭时被调用，用于清理
        """
        rospy.loginfo("Shutting down. Closing OpenCV windows...")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        # 1. 创建CameraViewer类的实例
        viewer = CameraViewer()
        
        # 2. 设置一个关闭时的回调函数 (好习惯)
        rospy.on_shutdown(viewer.shutdown_hook)
        
        # 3. 让脚本持续运行
        # rospy.spin()会阻塞主程序，防止它退出，同时允许所有订阅者回调函数在后台工作。
        # 这是保持ROS节点持续运行的标准方法。
        rospy.spin()
        
    except rospy.ROSInterruptException:
        pass
