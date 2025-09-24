#!/usr/bin/env python3
"""
使用YOLOv8和点云进行2D/3D目标检测的ROS节点。

该节点在一个压缩的相机图像流上进行实时目标检测。它将图像与3D点云进行同步，
以估计每个检测到的物体的距离和3D位置。节点对一个特定的"mark"物体有特殊的处理逻辑：
通过射线投射和地平面相交的方法来计算其精确的地面位置。
处理后的结果会以多种格式发布，供其他节点使用，包括带标注的图像、详细的检测消息以及用于RViz可视化的标记。

订阅的话题:
  - /magv/camera/image_compressed/compressed (sensor_msgs.msg.CompressedImage): 输入的图像流。
  - /filtered_cloud (sensor_msgs.msg.PointCloud2): 已同步的3D点云数据。

发布的话题:
  - /yolo/detected_image/compressed (sensor_msgs.msg.CompressedImage): 绘制了检测框和标签的输入图像。
  - /yolo/full_detections (yolo_detect.msg.FullDetectionArray): 为每个检测目标发布的详细消息，包括类别、置信度、距离、3D质心和关联的点云簇。
  - /yolo/mark_position_ground (geometry_msgs.msg.PoseStamped): "mark"物体在'map'坐标系下，地面上的估计3D位置。
  - /yolo/markers (visualization_msgs.msg.MarkerArray): 用于在RViz中可视化检测目标的标记。
  - ... 以及其他几个用于调试的话题。
"""

import rospy
import cv2
import numpy as np
import tf2_ros
import tf
import tf2_geometry_msgs
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, PointCloud2, Image, CameraInfo
from sensor_msgs import point_cloud2
from geometry_msgs.msg import PointStamped, TransformStamped, PoseArray, Pose, Point, PoseStamped, Vector3Stamped
from ultralytics import YOLO
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float32MultiArray, String, Header, ColorRGBA, Int32MultiArray
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from tf.transformations import euler_from_quaternion, quaternion_matrix
from yolo_detect.msg import Detection, DetectionArray, FullDetection, FullDetectionArray
import os
import rospkg

class YOLODetector:
    """管理YOLO检测、数据融合和发布流程的类。"""
    def __init__(self):
        """初始化YOLO模型、ROS节点、参数和通信接口。"""
        rospy.init_node('yolo_detector', anonymous=True)
        
        # --- 模型和ROS工具初始化 ---
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('yolo_detect')
        model_path = os.path.join(pkg_path, 'models', 'best.pt')
        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        
        # 将模型输出的类别ID映射到可读的名称。
        self.class_map = {
            0: "trash", 1: "bench", 2: "billboard", 3: "tree", 
            4: "tractor_trailer", 5: "barrel", 6: "fire_hydrant", 
            7: "traffic_cone", 8: "human", 9: "mark"
        }
        
        # --- 相机内参计算 ---
        # 这些参数用于将3D点投影到2D图像平面。
        self.image_width = 1080
        self.image_height = 720
        self.horizontal_fov = 2.0  # 水平视场角 (弧度)
        self.fx = self.image_width / (2 * np.tan(self.horizontal_fov / 2))
        self.fy = self.fx
        self.cx = self.image_width / 2
        self.cy = self.image_height / 2
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        self.K_inv = np.linalg.inv(self.K) # 逆矩阵用于反向投影 (从2D像素生成3D射线)
        
        # --- TF2 和坐标变换初始化 ---
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.camera_to_base_mat = self.create_static_transform_matrix()
        
        # --- ROS 订阅者和发布者 ---
        # 使用 ApproximateTimeSynchronizer 根据时间戳同步图像和点云消息。
        image_sub = message_filters.Subscriber('/magv/camera/image_compressed/compressed', CompressedImage)
        cloud_sub = message_filters.Subscriber('/filtered_cloud', PointCloud2)
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, cloud_sub], queue_size=10, slop=0.5)
        ts.registerCallback(self.sync_callback)
        
        # 用于发布各种处理结果的话题
        self.image_pub = rospy.Publisher('/yolo/detected_image/compressed', CompressedImage, queue_size=1)
        self.marker_pub = rospy.Publisher('/yolo/markers', MarkerArray, queue_size=1)
        self.detection_pub = rospy.Publisher('/yolo/detections', DetectionArray, queue_size=1)
        self.centroid_debug_pub = rospy.Publisher('/yolo/debug/centroids', PoseArray, queue_size=1)
        self.class_debug_pub = rospy.Publisher('/yolo/debug/detected_classes', String, queue_size=1)
        self.detection_status_pub = rospy.Publisher('/yolo/debug/detection_status', String, queue_size=1)
        self.camera_info_pub = rospy.Publisher('/yolo/debug/camera_info', CameraInfo, queue_size=1, latch=True)

        self.mark_pose_pub = rospy.Publisher('/yolo/mark_position_ground', PoseStamped, queue_size=1)
        self.mark_marker_pub = rospy.Publisher('/yolo/mark_ground_marker', Marker, queue_size=1)

        self.full_detection_pub = rospy.Publisher('/yolo/full_detections', FullDetectionArray, queue_size=1)

        # 用于在ROS标准的相机坐标系(物理系)和OpenCV标准的相机坐标系(光学系)之间转换的旋转矩阵。
        self.physical_to_optical_rot = np.array([
            [ 0, -1,  0],
            [ 0,  0, -1],
            [ 1,  0,  0]
        ])
        self.optical_to_physical_rot = self.physical_to_optical_rot.T
        
        rospy.loginfo(f"YOLOv8 检测器已初始化。")

    def _get_color_from_distance(self, distance):
        """辅助函数，根据距离生成一个从红色(近)到蓝色(远)的颜色梯度。"""
        if distance is None:
            return (0, 165, 255) # 对于没有距离信息的物体，使用默认的橙色
        color_factor = min(1.0, distance / 8.0) # 将距离归一化到8米范围
        blue = int(color_factor * 200 + 55)
        red = int((1 - color_factor) * 200 + 55)
        return (blue, 150, red)

    def calculate_distance_and_cloud(self, bbox, camera_points):
        """
        通过查找2D边界框内的点云点来估计物体距离。
        
        该方法将所有3D点投影到2D图像上。它收集落在给定边界框内的点，并选择
        离边界框中心最近的N个点来稳健地计算平均距离和3D质心。

        Args:
            bbox (list): 边界框 [x1, y1, x2, y2]。
            camera_points (np.array): 在相机光学坐标系下的3D点云。

        Returns:
            tuple: (平均距离, 质心在base_link下的坐标, 点的数量, 在base_link下的点簇)
        """
        x1, y1, x2, y2 = map(int, bbox)
        center_u = (x1 + x2) / 2.0
        center_v = (y1 + y2) / 2.0
        candidate_points = []
        if camera_points is None: return None, None, 0, []
        # 找到所有投影到2D边界框内的3D点
        for point in camera_points:
            pixel = self.project_point(point)
            if pixel and x1 <= pixel[0] <= x2 and y1 <= pixel[1] <= y2:
                point_base = self.transform_point_to_base(point)
                if point_base is not None:
                    distance = np.linalg.norm(point_base)
                    dist_to_center_sq = (pixel[0] - center_u)**2 + (pixel[1] - center_v)**2
                    candidate_points.append((dist_to_center_sq, point_base, distance))
        if not candidate_points: return None, None, 0, []
        # 按与边界框中心的接近程度对点进行排序
        candidate_points.sort(key=lambda x: x[0])
        # 只使用最接近中心的N个点进行计算，以获得更稳健的估计
        NUM_POINTS_TO_KEEP = 7
        num_to_use = min(NUM_POINTS_TO_KEEP, len(candidate_points))
        closest_points = candidate_points[:num_to_use]
        if not closest_points: return None, None, 0, []
        # 从这个子集中计算平均距离和质心
        distances = [item[2] for item in closest_points]
        centroid_points = [item[1] for item in closest_points]
        mean_distance = np.mean(distances)
        centroid_base = np.mean(centroid_points, axis=0)
        points_in_box_base = [item[1] for item in closest_points]
        return mean_distance, centroid_base, len(candidate_points), points_in_box_base

    def sync_callback(self, image_msg, cloud_msg):
        """用于处理已同步的图像和点云数据的主回调函数。"""
        try:
            self.publish_camera_info()
            # 在图像中预测物体
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            results = self.model.predict(cv_image, conf=0.45, verbose=False)
            
            detections = results[0]
            annotated_image = cv_image.copy()

            boxes = detections.boxes.xyxy.cpu().numpy()
            classes = detections.boxes.cls.cpu().numpy()
            confidences = detections.boxes.conf.cpu().numpy()
            
            # 预处理点云：将其转换到机器人的基座(base_link)坐标系
            camera_points = None
            try:
                camera_points_physical = self.transform_cloud(cloud_msg, "car/base_link")
                if camera_points_physical is not None and len(camera_points_physical) > 0:
                    camera_points = (self.physical_to_optical_rot @ camera_points_physical.T).T
            except Exception as e:
                rospy.logwarn(f"点云处理错误: {str(e)}")

            full_detection_array = FullDetectionArray()
            full_detection_array.header.stamp = image_msg.header.stamp
            full_detection_array.header.frame_id = "car/base_link"
            candidate_marks = []
            render_infos = []
            det_info_for_markers = []
            detection_results_legacy = []

            # --- 阶段一：数据收集与初步处理 ---
            # 这个阶段运行YOLO检测，并为所有检测到的物体进行初步的数据关联。
            for i, (bbox, class_id, conf) in enumerate(zip(boxes, classes, confidences)):
                int_class_id = int(class_id)
                
                # 对 "mark" 物体的特殊处理：计算其地面位置
                if int_class_id == 9:
                    ground_pos = self.calculate_mark_ground_position(bbox, image_msg.header.stamp)
                    if ground_pos is not None:
                        candidate_marks.append({'position': ground_pos, 'original_index': i})
                
                # 对所有物体的标准距离/质心计算
                distance, centroid, _, points_in_box = self.calculate_distance_and_cloud(bbox, camera_points)
                
                # 创建详细的 FullDetection 消息
                full_det = FullDetection()
                full_det.class_id = int_class_id
                full_det.class_name = self.class_map[int_class_id]
                full_det.confidence = float(conf)
                full_det.bbox = bbox.tolist()

                if distance is not None:
                    full_det.distance = float(distance)
                    if centroid is not None:
                        full_det.centroid_base.x, full_det.centroid_base.y, full_det.centroid_base.z = centroid
                    if points_in_box:
                        cloud_header = Header(stamp=image_msg.header.stamp, frame_id="car/base_link")
                        full_det.in_box_cloud = point_cloud2.create_cloud_xyz32(cloud_header, points_in_box)
                    
                    det_info_for_markers.append((class_id, distance, centroid))
                    detection_results_legacy.append({ "class_id": int_class_id, "distance": float(distance) })

                full_detection_array.detections.append(full_det)

                # 存储稍后用于渲染的信息
                label = f"{self.class_map[int_class_id]}"
                if distance is not None:
                    label += f": {distance:.1f}m (cloud)"
                
                render_infos.append({
                    'bbox': bbox,
                    'label': label,
                    'color': self._get_color_from_distance(distance)
                })

            # --- 阶段二：Mark板处理与数据修正 ---
            # 这个阶段使用地面投影的结果来精炼 "mark" 物体的数据。
            if candidate_marks:
                best_mark_for_publish = None
                try:
                    transform = self.tf_buffer.lookup_transform("map", "car/base_link", image_msg.header.stamp, rospy.Duration(0.1))
                    robot_pos_np = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
                    _, _, robot_yaw = euler_from_quaternion([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w])
                    min_angle_diff = float('inf')
                    
                    for mark in candidate_marks:
                        mark_pos = mark['position']
                        distance_map = np.linalg.norm(mark_pos - robot_pos_np)
                        
                        vec_to_mark = mark_pos - robot_pos_np
                        angle_to_mark = np.arctan2(vec_to_mark[1], vec_to_mark[0])
                        angle_diff = abs(angle_to_mark - robot_yaw)
                        if angle_diff > np.pi: angle_diff = 2 * np.pi - angle_diff
                        
                        angle_diff_deg = np.degrees(angle_diff)
                        # 过滤掉角度过大且距离过远的mark板，为了应对多mark板的情况下
                        if angle_diff_deg > 37.0 and distance_map > 6.0:
                            continue
                        
                        original_index = mark['original_index']
                        
                        # 使用更精确的地面投影距离覆盖掉原始的点云估算距离
                        full_detection_array.detections[original_index].distance = distance_map
                        
                        # 在标签中加入角度信息 (A代表Angle)
                        new_label = f"{self.class_map[9]}: {distance_map:.1f}m A:{angle_diff_deg:.1f}deg"
                        render_infos[original_index]['label'] = new_label
                        render_infos[original_index]['color'] = self._get_color_from_distance(distance_map)
                        
                        # 选出与机器人朝向最接近的mark板作为最佳目标
                        if angle_diff < min_angle_diff:
                            min_angle_diff = angle_diff
                            best_mark_for_publish = mark
                
                except Exception as e:
                    rospy.logwarn(f"在mark板筛选过程中发生TF错误: {e}")

                if best_mark_for_publish:
                    # 发布最佳 "mark" 的位姿和标记
                    best_pos = best_mark_for_publish['position']
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = image_msg.header.stamp
                    pose_msg.header.frame_id = "map"
                    pose_msg.pose.position = Point(*best_pos)
                    pose_msg.pose.orientation.w = 1.0
                    self.mark_pose_pub.publish(pose_msg)
                    self.publish_mark_marker(best_pos, image_msg.header.stamp)
            
            # --- 阶段三：最终渲染 ---
            # 将所有的检测框和标签绘制到图像上。
            for info in render_infos:
                bbox = info['bbox']
                label = info['label']
                color = info['color']
                xy1 = (int(bbox[0]), int(bbox[1]))
                xy2 = (int(bbox[2]), int(bbox[3]))
                cv2.rectangle(annotated_image, xy1, xy2, color, 3)
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated_image, xy1, (xy1[0] + w + 5, xy1[1] - h - 5), color, -1)
                cv2.putText(annotated_image, label, (xy1[0] + 3, xy1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # --- 阶段四：发布 ---
            # 发布所有处理完成的数据。
            compress_out = self.bridge.cv2_to_compressed_imgmsg(annotated_image)
            self.image_pub.publish(compress_out)

            if full_detection_array.detections:
                self.full_detection_pub.publish(full_detection_array)

            if detection_results_legacy:
                detection_array = DetectionArray()
                for result in detection_results_legacy:
                    detection_msg = Detection(class_id=result["class_id"], distance=result["distance"])
                    detection_array.detections.append(detection_msg)
                self.detection_pub.publish(detection_array)

            if det_info_for_markers:
                self.publish_detection_markers(det_info_for_markers)
                self.publish_centroid_debug(det_info_for_markers)

            detected_classes_str = [self.class_map[int(c)] for c in classes]
            class_msg = "Detected classes: " + ", ".join(detected_classes_str)
            self.class_debug_pub.publish(String(data=class_msg))

            status_msg = f"Processed {len(boxes)} detections, {len(det_info_for_markers)} with distance"
            self.detection_status_pub.publish(String(data=status_msg))

        except Exception as e:
            rospy.logerr(f"在 sync_callback 中发生处理错误: {e}")
    
    def transform_point_to_base(self, point_camera_optical):
        """将点从相机的光学坐标系转换到机器人的基座坐标系。"""
        point_physical = self.optical_to_physical_rot @ point_camera_optical
        point_hom = np.append(point_physical, 1.0)
        point_base = np.dot(self.camera_to_base_mat, point_hom)
        return point_base[:3]
    
    def create_static_transform_matrix(self):
        """设置一个从相机物理坐标系到机器人基座坐标系的硬编码变换矩阵。"""
        translation = [0.500, -0.040, 0.570]
        rotation_quaternion = [0.000, 0.156, 0.000, 0.988]
        T = np.eye(4)
        T[0:3, 3] = translation
        R = tf.transformations.quaternion_matrix(rotation_quaternion)
        T[:3, :3] = R[:3, :3]
        return T

    def transform_cloud(self, cloud_msg, target_frame):
        """使用TF将整个点云转换到目标坐标系。"""
        try:
            if cloud_msg.width == 0 or cloud_msg.height == 0: return None
            transform = self.tf_buffer.lookup_transform(target_frame, cloud_msg.header.frame_id, cloud_msg.header.stamp, rospy.Duration(0.1))
            transformed_cloud = do_transform_cloud(cloud_msg, transform)
            points_gen = point_cloud2.read_points(transformed_cloud, field_names=("x", "y", "z"), skip_nans=True)
            if not points_gen: return None
            return np.array([p for p in points_gen if p[2] > 0.1])
        except Exception as e:
            rospy.logwarn(f"在 transform_cloud 中发生TF/点云错误: {str(e)}")
            return None

    def project_point(self, point_camera):
        """使用相机内参矩阵(K)将相机光学坐标系中的3D点投影到2D图像平面。"""
        if point_camera[2] <= 0.1: return None # 避免投影相机后方的点
        try:
            u = int((self.fx * point_camera[0] / point_camera[2]) + self.cx)
            v = int((self.fy * point_camera[1] / point_camera[2]) + self.cy)
        except ZeroDivisionError: return None
        if 0 <= u < self.image_width and 0 <= v < self.image_height: return (u, v)
        return None

    def create_marker(self, detection, marker_id):
        """为单个检测结果创建用于RViz显示的Marker（球体+文字）。"""
        marker = Marker()
        class_id, distance, centroid = detection
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "car/base_link"
        marker.ns = "yolo_objects"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.pose.position.x = centroid[0] if centroid is not None else 0.0
        marker.pose.position.y = centroid[1] if centroid is not None else 0.0
        marker.pose.position.z = centroid[2] if centroid is not None else 0.0
        marker.pose.orientation.w = 1.0
        scale = 0.3
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.7
        marker.lifetime = rospy.Duration(1.0)
        
        text_marker = Marker()
        text_marker.header = marker.header
        text_marker.ns = "yolo_labels"
        text_marker.id = marker_id
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.pose = marker.pose
        text_marker.pose.position.z += scale * 1.5
        text_marker.scale.z = 0.2
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        text_marker.text = f"{self.class_map[int(class_id)]} {distance:.1f}m" if distance is not None else self.class_map[int(class_id)]
        text_marker.lifetime = marker.lifetime
        return marker, text_marker

    def publish_detection_markers(self, detections):
        """发布所有检测结果的RViz Marker。"""
        marker_array = MarkerArray()
        for midx, detection in enumerate(detections):
            shape_marker, text_marker = self.create_marker(detection, midx)
            marker_array.markers.append(shape_marker)
            marker_array.markers.append(text_marker)
        self.marker_pub.publish(marker_array)

    def publish_centroid_debug(self, detections):
        """发布所有质心位置用于调试。"""
        pose_array = PoseArray()
        pose_array.header.frame_id = "car/base_link"
        pose_array.header.stamp = rospy.Time.now()
        for detection in detections:
            _, _, centroid = detection
            if centroid is not None:
                pose = Pose()
                pose.position.x = centroid[0]
                pose.position.y = centroid[1]
                pose.position.z = centroid[2]
                pose.orientation.w = 1.0
                pose_array.poses.append(pose)
        self.centroid_debug_pub.publish(pose_array)

    def calculate_mark_ground_position(self, bbox, stamp):
        """
        通过将射线投影到地平面上来计算 "mark" 的3D位置。

        此方法执行射线投射：
        1. 从相机的光学中心，穿过2D边界框的中心，创建一条3D射线。
        2. 使用TF将这条射线转换到全局的 'map' 坐标系。
        3. 计算转换后的射线与地平面 (在map系中 Z=0) 的交点。
        
        Args:
            bbox (list): mark物体的边界框。
            stamp (rospy.Time): 用于TF查找的时间戳。

        Returns:
            np.array: mark在'map'坐标系下地面上的3D位置 [x, y, z]，如果失败则返回 None。
        """
        try:
            # 1. 将2D像素反向投影为相机光学坐标系中的3D射线
            u = (bbox[0] + bbox[2]) / 2.0
            v = (bbox[1] + bbox[3]) / 2.0
            ray_optical = self.K_inv @ np.array([u, v, 1.0])
            ray_physical = self.optical_to_physical_rot @ ray_optical
            
            # 2. 获取从相机到map坐标系的变换
            transform = self.tf_buffer.lookup_transform("map", "car/camera_link", stamp, rospy.Duration(0.2))
            
            # 3. 将相机的原点和射线的方向向量转换到map坐标系
            C_map = transform.transform.translation
            camera_origin_map = np.array([C_map.x, C_map.y, C_map.z])

            vec_stamped = Vector3Stamped()
            vec_stamped.header.frame_id = "car/camera_link"
            vec_stamped.header.stamp = stamp
            vec_stamped.vector.x, vec_stamped.vector.y, vec_stamped.vector.z = ray_physical
            
            V_map_stamped = tf2_geometry_msgs.do_transform_vector3(vec_stamped, transform)
            V_map = np.array([V_map_stamped.vector.x, V_map_stamped.vector.y, V_map_stamped.vector.z])
            V_map /= np.linalg.norm(V_map) # 归一化方向向量
            
            # 4. 计算与地平面 (Z=0) 的交点
            # 射线方程为: P(t) = camera_origin_map + t * V_map
            # 我们需要z分量为0: camera_origin_map.z + t * V_map.z = 0
            if abs(V_map[2]) < 1e-6: return None # 射线与地平面平行
            
            t = -camera_origin_map[2] / V_map[2]
            if t < 0: return None # 交点在相机后面

            return camera_origin_map + t * V_map
        except Exception as e:
            rospy.logwarn(f"在 calculate_mark_ground_position 中发生错误: {e}")
            return None

    def publish_mark_marker(self, position_map, stamp):
        """发布一个箭头Marker来可视化Mark板的地面位置。"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = stamp
        marker.ns = "mark_ground_position"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.points.append(Point(position_map[0], position_map[1], position_map[2] + 0.5))
        marker.points.append(Point(position_map[0], position_map[1], position_map[2]))
        marker.scale.x = 0.05
        marker.scale.y = 0.1
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration(1.0)
        self.mark_marker_pub.publish(marker)
        
    def publish_camera_info(self):
        """发布相机的内参信息，供其他ROS节点（如RViz）使用。"""
        try:
            info = CameraInfo()
            info.header.stamp = rospy.Time.now()
            info.header.frame_id = "car/camera_link"
            info.height = self.image_height
            info.width = self.image_width
            info.K = list(self.K.flatten())
            info.distortion_model = "plumb_bob"
            info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
            P_flat = np.zeros(12)
            P_flat[0:3] = self.K[0,:]
            P_flat[4:7] = self.K[1,:]
            P_flat[8:11] = self.K[2,:]
            info.P = P_flat.tolist()
            self.camera_info_pub.publish(info)
        except Exception as e:
            rospy.logerr(f"发布相机信息失败: {str(e)}")

if __name__ == '__main__':
    try:
        detector = YOLODetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass