#!/usr/bin/env python3
import rospy
from std_msgs.msg import Bool
from yolo_detect.msg import FullDetectionArray

class EvasiveAccelerator:
    def __init__(self):
        rospy.init_node('evasive_accelerator', anonymous=True)

        # --- MODIFIED ---: 参数已根据新逻辑更新
        self.trigger_distance = rospy.get_param("~trigger_distance", 7.0)
        self.acceleration_duration = rospy.get_param("~acceleration_duration", 7.0) # 加速持续时间（秒）
        self.human_class_name = "human"

        # --- MODIFIED ---: 状态变量已更新，现在只有一个永久锁
        self.action_completed = False # 永久状态锁，确保整个流程只执行一次

        # --- MODIFIED ---: 发布器已更改为新的加速话题
        self.accelerate_pub = rospy.Publisher('/robot_control/accelerate', Bool, queue_size=1)

        # --- MODIFIED ---: 订阅者依然是 /yolo/full_detections
        rospy.Subscriber('/yolo/full_detections', FullDetectionArray, self.detection_callback)
        
        rospy.loginfo("Evasive Accelerator node initialized.")
        rospy.loginfo(f"--> Trigger distance: < {self.trigger_distance}m")
        rospy.loginfo(f"--> Acceleration duration: {self.acceleration_duration}s")
        rospy.loginfo("--> Publishing commands on: /robot_control/accelerate")

    def detection_callback(self, msg):
        # 1. 检查永久状态锁：如果已经完成了一次加速规避，则此节点不再执行任何操作
        if self.action_completed:
            return

        # 2. 检查是否满足触发条件
        for detection in msg.detections:
            # 找到行人且距离小于触发距离
            if detection.class_name == self.human_class_name and detection.distance < self.trigger_distance:
                
                # 满足条件：立即启动一次性加速流程
                rospy.logwarn(f"Human detected at {detection.distance:.2f}m (less than {self.trigger_distance}m).")
                rospy.logwarn(f"Triggering ONE-TIME evasive acceleration for {self.acceleration_duration} seconds.")
                
                # 立即上锁，防止重复触发
                self.action_completed = True
                
                self.start_acceleration()
                
                # 找到第一个满足条件的行人后，立即退出循环
                break

    def start_acceleration(self):
        # 1. 发布“开始加速”指令 (True)
        self.accelerate_pub.publish(Bool(data=True))
        
        # 2. 启动一个严格的定时器，时间一到就调用 stop_acceleration
        rospy.Timer(rospy.Duration(self.acceleration_duration), self.stop_acceleration, oneshot=True)

    def stop_acceleration(self, event):
        # 定时器触发此函数
        rospy.logwarn(f"Acceleration duration of {self.acceleration_duration}s has ended.")
        
        # 1. 发布“停止加速”指令 (False)
        self.accelerate_pub.publish(Bool(data=False))
        
        rospy.logwarn("Evasive action is COMPLETE. Resuming normal speed. This node will no longer trigger acceleration.")

if __name__ == '__main__':
    try:
        EvasiveAccelerator()
        # 初始发布一个不加速的指令，确保机器人启动时处于正常状态
        # 等待发布器建立连接
        rospy.sleep(1.0) 
        rospy.Publisher('/robot_control/accelerate', Bool, queue_size=1, latch=True).publish(Bool(data=False))
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
