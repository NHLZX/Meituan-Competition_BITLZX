#!/usr/bin/env python3
"""
用于执行一次性规避加速的ROS节点。

该节点监视YOLO检测结果中的行人信息。如果在一个预设的距离内检测到行人，
它会触发一次持续时间可配置的临时加速。在这次规避行为完成后，
节点将在剩余的运行时间内自动禁用，以防止重复触发。

订阅的话题:
  - /yolo/full_detections (yolo_detect.msg.FullDetectionArray): 接收详细的检测结果。

发布的话题:
  - /robot_control/accelerate (std_msgs.msg.Bool): 发布True以开始加速，发布False以停止。
"""

import rospy
from std_msgs.msg import Bool
from yolo_detect.msg import FullDetectionArray

class EvasiveAccelerator:
    """管理一次性规避加速逻辑的类。"""
    def __init__(self):
        """初始化节点、参数、发布者和订阅者。"""
        rospy.init_node('evasive_accelerator', anonymous=True)

        # --- 参数 ---
        # 触发距离（米）：当行人距离小于此值时，触发加速行为。
        self.trigger_distance = rospy.get_param("~trigger_distance", 7.0)
        # 加速持续时间（秒）：机器人将保持加速状态的时间。
        self.acceleration_duration = rospy.get_param("~acceleration_duration", 7.0) 
        # 需要检测的目标类别名称。
        self.human_class_name = "human"

        # --- 状态变量 ---
        # 一个永久性的状态锁，确保整个加速流程只执行一次。
        self.action_completed = False 

        # --- ROS 通信接口 ---
        self.accelerate_pub = rospy.Publisher('/robot_control/accelerate', Bool, queue_size=1)
        rospy.Subscriber('/yolo/full_detections', FullDetectionArray, self.detection_callback)
        
        rospy.loginfo("规避加速节点 (Evasive Accelerator) 已初始化。")
        rospy.loginfo(f"--> 触发距离: < {self.trigger_distance} 米")
        rospy.loginfo(f"--> 加速持续时间: {self.acceleration_duration} 秒")
        rospy.loginfo("--> 在话题 /robot_control/accelerate 上发布指令")

    def detection_callback(self, msg):
        """
        YOLO检测结果的回调函数。检查是否满足触发条件。
        
        Args:
            msg (FullDetectionArray): 包含当前所有检测结果的消息。
        """
        # 1. 检查永久状态锁。如果规避行为已经完成，则此节点不再执行任何操作。
        if self.action_completed:
            return

        # 2. 遍历检测结果，检查是否满足触发条件。
        for detection in msg.detections:
            # 条件：检测到类别为"human"的目标，并且其距离小于触发距离。
            if detection.class_name == self.human_class_name and detection.distance < self.trigger_distance:
                
                # --- 触发一次性的规避行为 ---
                rospy.logwarn(f"在 {detection.distance:.2f} 米处检测到行人 (小于触发距离 {self.trigger_distance} 米)。")
                rospy.logwarn(f"触发一次性规避加速，持续 {self.acceleration_duration} 秒。")
                
                # 立即上锁，防止此代码块被重复执行。
                self.action_completed = True
                
                # 启动加速流程。
                self.start_acceleration()
                
                # 既然已经找到了触发条件，无需再检查其他检测结果，立即退出循环。
                break

    def start_acceleration(self):
        """启动加速阶段。"""
        # 1. 发布一个布尔值为 True 的消息，命令机器人开始加速。
        self.accelerate_pub.publish(Bool(data=True))
        
        # 2. 启动一个一次性的定时器。在 `acceleration_duration` 秒后，
        #    它将自动调用 `stop_acceleration` 方法来结束加速。
        rospy.Timer(rospy.Duration(self.acceleration_duration), self.stop_acceleration, oneshot=True)

    def stop_acceleration(self, event):
        """
        结束加速阶段。此方法由 rospy.Timer 定时器自动调用。
        
        Args:
            event: 定时器事件对象 (此处未使用)。
        """
        rospy.logwarn(f"{self.acceleration_duration} 秒的加速时间已结束。")
        
        # 1. 发布一个布尔值为 False 的消息，命令机器人停止加速，恢复正常速度。
        self.accelerate_pub.publish(Bool(data=False))
        
        rospy.logwarn("规避行为已完成。恢复正常速度。本节点将不再触发加速。")

if __name__ == '__main__':
    try:
        EvasiveAccelerator()
        # 这个代码块确保机器人在启动时处于非加速的正常状态。
        # 等待一小段时间，以确保发布者有足够的时间建立连接。
        rospy.sleep(1.0) 
        # 创建一个“锁存(latch)”的发布者来发送一个初始的 'False' 指令。
        # 'latch=True' 确保任何后来连接到这个话题的订阅者都能立刻收到这条最后发布的消息。
        rospy.Publisher('/robot_control/accelerate', Bool, queue_size=1, latch=True).publish(Bool(data=False))
        rospy.spin()
    except rospy.ROSInterruptException:
        pass