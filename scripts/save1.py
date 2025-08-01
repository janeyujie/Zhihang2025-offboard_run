#!/usr/bin/env python
import rospy
import cv2
import os
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class ImageCollector:
    def __init__(self):
        rospy.loginfo("Initializing ImageCollector...")
        
        self.save_dir = "./img_save"
        os.makedirs(self.save_dir, exist_ok=True)
        rospy.loginfo(f"Images will be saved to: {os.path.abspath(self.save_dir)}")
        
        self.bridge = CvBridge()
        self.count = 0
        self.latest_image = None
        self.latest_stamp = None
        
        self.image_sub = rospy.Subscriber("/iris_0/camera/image_raw", Image, self.callback) # 这里需要根据所录的摄像头修改对应的话题，先用rqt --> visual --> image_view查看当前话题，或者rostopic list | grep carm查看摄像机话题
        rospy.loginfo("Subscribed to /standard_vtol_0/mavros/camera/image_raw. Waiting for images...")

    def callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_image = cv_image
            self.latest_stamp = msg.header.stamp.to_nsec()
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {str(e)}")
        except Exception as e:
            rospy.logerr(f"Unexpected error in callback: {str(e)}")

    def run(self):
        rospy.loginfo("Press 's' to save image, 'q' to quit.")
        while not rospy.is_shutdown():
            if self.latest_image is not None:
                cv2.imshow("Live Camera Feed", self.latest_image)
                key = cv2.waitKey(1) & 0xFF  # Mask to get lower 8 bits
                if key == ord('s'):
                    filename = f"{self.save_dir}/img_{self.count+1}_{self.latest_stamp}.jpg"
                    cv2.imwrite(filename, self.latest_image)
                    self.count += 1
                    rospy.loginfo(f"Saved image #{self.count}: {filename}")
                elif key == ord('q'):
                    rospy.loginfo("Quit signal received.")
                    break
        cv2.destroyAllWindows()

if __name__ == '__main__':
    rospy.init_node('image_collector', log_level=rospy.INFO)
    rospy.loginfo("Starting image_collector node")

    try:
        collector = ImageCollector()
        collector.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node interrupted by user")
    except Exception as e:
        rospy.logerr(f"Fatal error: {str(e)}")
    finally:
        rospy.loginfo(f"Node shutdown. Total images saved: {collector.count if 'collector' in locals() else 0}")
