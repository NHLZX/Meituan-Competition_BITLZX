#!/usr/bin/env python3
import rospy
import numpy as np
import math
import cv2
import cv2.aruco as aruco
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped

class ArucoDetector:
    def __init__(self):
        rospy.init_node('aruco_detector', anonymous=True)
        
        # 初始化CvBridge
        self.bridge = CvBridge()
        
        # 设置Aruco参数（新版本API）
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # 摄像头内参 (根据实际情况调整)
        image_width = 1080
        image_height = 720
        horizontal_fov = 2.0  # 弧度
        
        # 计算焦距
        fx = (image_width / 2.0) / math.tan(horizontal_fov / 2.0)
        fy = fx  # 假设像素是正方形
        cx = image_width / 2.0  # 主点x坐标
        cy = image_height / 2.0  # 主点y坐标
        
        self.camera_matrix = np.array([[fx, 0.0, cx],
                                      [0.0, fy, cy],
                                      [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.aruco_size = 1  # Aruco标记的实际大小(米)
        
        # 相机外参 (相机相对于机器人的位置和姿态)
        self.camera_translation = np.array([0.5, -0.04, 0.57])  # 相机位置
        self.camera_pitch = 0.314  # 相机俯仰角 (弧度)
        
        # 机器人位置和朝向
        self.robot_pos = None
        self.robot_yaw = 0.0
        
        # 创建发布器，用于发布检测到的Aruco标记位置
        self.aruco_position_pub = rospy.Publisher('/aruco_position', PointStamped, queue_size=1)
        
        # 订阅摄像头图像和里程计
        rospy.Subscriber('/magv/camera/image_compressed/compressed', CompressedImage, self.image_callback)
        rospy.Subscriber('/magv/odometry/gt', Odometry, self.odom_callback)
        
        rospy.loginfo("Aruco检测器已启动，等待图像和里程计数据...")

    def odom_callback(self, msg):
        """更新机器人位置和朝向"""
        self.robot_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        
        # 提取机器人朝向（yaw角）
        orientation = msg.pose.pose.orientation
        self.robot_yaw = self.quaternion_to_yaw(orientation)

    def image_callback(self, msg):
        """处理摄像头图像，检测Aruco标记"""
        try:
            # 转换ROS图像到OpenCV格式
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            
            # 检测Aruco标记（新版本API）
            corners, ids, rejected = self.detector.detectMarkers(cv_image)
            
            if ids is not None:
                rospy.loginfo(f"检测到 {len(ids)} 个Aruco标记，ID: {ids.flatten()}")
                
                # 估计姿态
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    corners, self.aruco_size, self.camera_matrix, self.dist_coeffs
                )
                
                # 处理所有检测到的标记
                for i in range(len(ids)):
                    marker_id = ids[i][0]
                    tvec = tvecs[i]
                    rvec = rvecs[i]
                    
                    # 计算并发布标记位置
                    self.calculate_and_publish_aruco_position(tvec, rvec, marker_id)
                    
                    # 在图像上绘制标记和坐标轴
                    aruco.drawDetectedMarkers(cv_image, corners, ids)
                    cv_image = cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)
            
            # 可选：显示检测结果
            # cv2.imshow("Aruco Detection", cv_image)
            # cv2.waitKey(1)
            
        except Exception as e:
            rospy.logwarn(f"图像处理出错: {e}")

    def calculate_and_publish_aruco_position(self, tvec, rvec, marker_id):
        """计算Aruco标记位置并发布到话题"""
        if self.robot_pos is None:
            rospy.logwarn("无法计算标记位置：机器人位置未知")
            return
            
        # tvec是相对于摄像头的位置 (x, y, z)
        camera_x, camera_y, camera_z = tvec.flatten()
        
        # 考虑相机的俯仰角进行坐标变换
        cos_pitch = math.cos(self.camera_pitch)
        sin_pitch = math.sin(self.camera_pitch)
        
        # 摄像头坐标系到机器人坐标系的转换
        robot_relative_x = camera_z * cos_pitch + camera_y * sin_pitch
        robot_relative_y = -camera_x  # 摄像头的x轴对应机器人的y轴(左)，取负号
        robot_relative_z = -camera_z * sin_pitch + camera_y * cos_pitch
        
        # 加上相机相对于机器人的位置偏移
        robot_relative_x += self.camera_translation[0]
        robot_relative_y += self.camera_translation[1]
        robot_relative_z += self.camera_translation[2]
        
        # 考虑机器人的朝向，转换到世界坐标系
        cos_yaw = math.cos(self.robot_yaw)
        sin_yaw = math.sin(self.robot_yaw)
        
        world_x = self.robot_pos[0] + robot_relative_x * cos_yaw - robot_relative_y * sin_yaw
        world_y = self.robot_pos[1] + robot_relative_x * sin_yaw + robot_relative_y * cos_yaw
        world_z = self.robot_pos[2] + robot_relative_z
        
        # 创建点消息并发布
        point_msg = PointStamped()
        point_msg.header.stamp = rospy.Time.now()
        point_msg.header.frame_id = "map"
        point_msg.point.x = world_x
        point_msg.point.y = world_y
        point_msg.point.z = world_z
        
        self.aruco_position_pub.publish(point_msg)
        
        # 记录检测信息
        distance = math.sqrt(robot_relative_x**2 + robot_relative_y**2)
        rospy.loginfo(f"标记 ID {marker_id} 位置: ({world_x:.2f}, {world_y:.2f}, {world_z:.2f})")
        rospy.loginfo(f"相对机器人距离: {distance:.2f}m, 方位角: {math.degrees(math.atan2(robot_relative_y, robot_relative_x)):.1f}°")

    def quaternion_to_yaw(self, orientation):
        """将四元数转换为yaw角"""
        x = orientation.x
        y = orientation.y
        z = orientation.z
        w = orientation.w
        
        # 计算yaw角
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw

if __name__ == '__main__':
    try:
        ArucoDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
