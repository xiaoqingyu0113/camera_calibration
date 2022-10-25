#!/usr/bin/env python3
# license removed for brevity
import cv2
import rospy
from sensor_msgs.msg import CompressedImage
import numpy as np
import message_filters

class MultiImgSync:
    def __init__(self, debug=False, init_ros_node=False):

        self.img1 = None
        self.img2 = None        
        self.img3 = None

        self.iter = 0


        rospy.init_node('image_sync', anonymous=True)

        self.image_sub1 = message_filters.Subscriber("/camera_1/image_color/compressed", CompressedImage)
        self.image_sub2 = message_filters.Subscriber("/camera_2/image_color/compressed",CompressedImage)
        self.image_sub3 = message_filters.Subscriber("/camera_3/image_color/compressed", CompressedImage)

        self.tss = message_filters.ApproximateTimeSynchronizer([self.image_sub1, self.image_sub2,self.image_sub3],
                                                               queue_size=3, slop=0.010)
        self.tss.registerCallback(self.callback)

    def callback(self,msg1,msg2,msg3):
        if self.iter >0:
            return
            
        np_arr = np.fromstring(msg1.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv2.imwrite(f"data/multiview_pingpong/cam1/{self.iter:05d}.jpg",image_np)
      
        np_arr = np.fromstring(msg2.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv2.imwrite(f"data/multiview_pingpong/cam2/{self.iter:05d}.jpg",image_np)

        np_arr = np.fromstring(msg3.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv2.imwrite(f"data/multiview_pingpong/cam3/{self.iter:05d}.jpg",image_np)

        self.iter = self.iter + 1
        print(f"saved {self.iter:05d}.jpg")


if __name__ == '__main__':
    try:
        mis = MultiImgSync()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass