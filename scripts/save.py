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
        
        # --- MODIFICATION START ---
        # 1. Define the save interval. Here it's set to 5.0 seconds.
        self.save_interval = rospy.Duration(1.0) 
        # 2. Initialize the time of the last save to the current time.
        self.last_save_time = rospy.Time.now()
        # --- MODIFICATION END ---
        
        # This log message in your original code was incorrect, I've fixed it.
        self.image_sub = rospy.Subscriber("/iris_0/camera/image_raw", Image, self.callback)
        rospy.loginfo("Subscribed to /iris_0/camera/image_raw. Waiting for images...")

    def callback(self, msg):
        # --- MODIFICATION START ---
        # 3. Check if the time since the last save is less than the desired interval.
        
        rospy.logdebug(f"Received image to be saved.")
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
        # --- MODIFICATION START ---
            # 4. Update the last save time to now, because we are about to save.
        self.last_save_time = rospy.Time.now()
            # --- MODIFICATION END ---

            #timestamp = msg.header.stamp.to_nsec()
            # Let's save with a more human-readable timestamp
        filename = f"{self.save_dir}/img_{self.count+1}_{rospy.Time.now().to_sec()}.jpg"
        cv2.imwrite(filename, cv_image)
        self.count += 1
            
        rospy.loginfo(f"Saved image #{self.count} to {filename}")
               

if __name__ == '__main__':
    rospy.init_node('image_collector', log_level=rospy.INFO) # Changed to INFO for cleaner output
    rospy.loginfo("Starting image_collector node")
    
    try:
        collector = ImageCollector()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node interrupted by user")
    finally:
        rospy.loginfo(f"Node shutdown.")
