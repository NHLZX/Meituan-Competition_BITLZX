#!/usr/bin/env python3
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
    def __init__(self):
        rospy.init_node('yolo_detector', anonymous=True)
        
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('yolo_detect')
        model_path = os.path.join(pkg_path, 'models', 'best.pt')
        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        
        self.class_map = {
            0: "trash", 1: "bench", 2: "billboard", 3: "tree", 
            4: "tractor_trailer", 5: "barrel", 6: "fire_hydrant", 
            7: "traffic_cone", 8: "human", 9: "mark"
        }
        
        self.image_width = 1080
        self.image_height = 720
        self.horizontal_fov = 2.0
        self.fx = self.image_width / (2 * np.tan(self.horizontal_fov / 2))
        self.fy = self.fx
        self.cx = self.image_width / 2
        self.cy = self.image_height / 2
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        self.K_inv = np.linalg.inv(self.K)
        
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.camera_to_base_mat = self.create_static_transform_matrix()
        
        image_sub = message_filters.Subscriber('/magv/camera/image_compressed/compressed', CompressedImage)
        cloud_sub = message_filters.Subscriber('/filtered_cloud', PointCloud2)
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, cloud_sub], queue_size=10, slop=0.5)
        ts.registerCallback(self.sync_callback)
        
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

        self.physical_to_optical_rot = np.array([
            [ 0, -1,  0],
            [ 0,  0, -1],
            [ 1,  0,  0]
        ])
        self.optical_to_physical_rot = self.physical_to_optical_rot.T
        
        rospy.loginfo(f"YOLOv8 Detector initialized with Full Detection and Mark Filtering.")

    def _get_color_from_distance(self, distance):
        if distance is None:
            return (0, 165, 255)
        color_factor = min(1.0, distance / 8.0)
        blue = int(color_factor * 200 + 55)
        red = int((1 - color_factor) * 200 + 55)
        return (blue, 150, red)

    def calculate_distance_and_cloud(self, bbox, camera_points):
        x1, y1, x2, y2 = map(int, bbox)
        center_u = (x1 + x2) / 2.0
        center_v = (y1 + y2) / 2.0
        candidate_points = []
        if camera_points is None: return None, None, 0, []
        for point in camera_points:
            pixel = self.project_point(point)
            if pixel and x1 <= pixel[0] <= x2 and y1 <= pixel[1] <= y2:
                point_base = self.transform_point_to_base(point)
                if point_base is not None:
                    distance = np.linalg.norm(point_base)
                    dist_to_center_sq = (pixel[0] - center_u)**2 + (pixel[1] - center_v)**2
                    candidate_points.append((dist_to_center_sq, point_base, distance))
        if not candidate_points: return None, None, 0, []
        candidate_points.sort(key=lambda x: x[0])
        NUM_POINTS_TO_KEEP = 7
        num_to_use = min(NUM_POINTS_TO_KEEP, len(candidate_points))
        closest_points = candidate_points[:num_to_use]
        if not closest_points: return None, None, 0, []
        distances = [item[2] for item in closest_points]
        centroid_points = [item[1] for item in closest_points]
        mean_distance = np.mean(distances)
        centroid_base = np.mean(centroid_points, axis=0)
        points_in_box_base = [item[1] for item in closest_points]
        return mean_distance, centroid_base, len(candidate_points), points_in_box_base

    def sync_callback(self, image_msg, cloud_msg):
        try:
            self.publish_camera_info()
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            results = self.model.predict(cv_image, conf=0.45, verbose=False)
            
            detections = results[0]
            annotated_image = cv_image.copy()

            boxes = detections.boxes.xyxy.cpu().numpy()
            classes = detections.boxes.cls.cpu().numpy()
            confidences = detections.boxes.conf.cpu().numpy()
            
            camera_points = None
            try:
                camera_points_physical = self.transform_cloud(cloud_msg, "car/base_link")
                if camera_points_physical is not None and len(camera_points_physical) > 0:
                    camera_points = (self.physical_to_optical_rot @ camera_points_physical.T).T
            except Exception as e:
                rospy.logwarn(f"Point cloud error: {str(e)}")

            full_detection_array = FullDetectionArray()
            full_detection_array.header.stamp = image_msg.header.stamp
            full_detection_array.header.frame_id = "car/base_link"
            candidate_marks = []
            render_infos = []
            det_info_for_markers = []
            detection_results_legacy = []

            # --- STAGE 1: Data Collection and Initial Processing ---
            for i, (bbox, class_id, conf) in enumerate(zip(boxes, classes, confidences)):
                int_class_id = int(class_id)
                
                if int_class_id == 9:
                    ground_pos = self.calculate_mark_ground_position(bbox, image_msg.header.stamp)
                    if ground_pos is not None:
                        candidate_marks.append({'position': ground_pos, 'original_index': i})
                
                distance, centroid, _, points_in_box = self.calculate_distance_and_cloud(bbox, camera_points)
                
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

                label = f"{self.class_map[int_class_id]}"
                if distance is not None:
                    label += f": {distance:.1f}m (cloud)"
                
                render_infos.append({
                    'bbox': bbox,
                    'label': label,
                    'color': self._get_color_from_distance(distance)
                })

            # --- STAGE 2: Mark Processing and Data Correction ---
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
                        if angle_diff_deg > 37.0 and distance_map > 6.0:
                            continue
                        
                        original_index = mark['original_index']
                        
                        full_detection_array.detections[original_index].distance = distance_map
                        
                        # <<< 新增修改: 在标签中加入角度信息 (A代表Angle)
                        new_label = f"{self.class_map[9]}: {distance_map:.1f}m A:{angle_diff_deg:.1f}deg"
                        render_infos[original_index]['label'] = new_label
                        render_infos[original_index]['color'] = self._get_color_from_distance(distance_map)
                        
                        if angle_diff < min_angle_diff:
                            min_angle_diff = angle_diff
                            best_mark_for_publish = mark
                
                except Exception as e:
                    rospy.logwarn(f"TF error during mark selection: {e}")

                if best_mark_for_publish:
                    best_pos = best_mark_for_publish['position']
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = image_msg.header.stamp
                    pose_msg.header.frame_id = "map"
                    pose_msg.pose.position = Point(*best_pos)
                    pose_msg.pose.orientation.w = 1.0
                    self.mark_pose_pub.publish(pose_msg)
                    self.publish_mark_marker(best_pos, image_msg.header.stamp)
            
            # --- STAGE 3: Final Rendering ---
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

            # --- STAGE 4: Publishing ---
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
            rospy.logerr(f"Processing Error in sync_callback: {e}")
    
    def transform_point_to_base(self, point_camera_optical):
        point_physical = self.optical_to_physical_rot @ point_camera_optical
        point_hom = np.append(point_physical, 1.0)
        point_base = np.dot(self.camera_to_base_mat, point_hom)
        return point_base[:3]
    
    def create_static_transform_matrix(self):
        translation = [0.500, -0.040, 0.570]
        rotation_quaternion = [0.000, 0.156, 0.000, 0.988]
        T = np.eye(4)
        T[0:3, 3] = translation
        R = tf.transformations.quaternion_matrix(rotation_quaternion)
        T[:3, :3] = R[:3, :3]
        return T

    def transform_cloud(self, cloud_msg, target_frame):
        try:
            if cloud_msg.width == 0 or cloud_msg.height == 0: return None
            transform = self.tf_buffer.lookup_transform(target_frame, cloud_msg.header.frame_id, cloud_msg.header.stamp, rospy.Duration(0.1))
            transformed_cloud = do_transform_cloud(cloud_msg, transform)
            points_gen = point_cloud2.read_points(transformed_cloud, field_names=("x", "y", "z"), skip_nans=True)
            if not points_gen: return None
            return np.array([p for p in points_gen if p[2] > 0.1])
        except Exception as e:
            rospy.logwarn(f"TF/Cloud Error in transform_cloud: {str(e)}")
            return None

    def project_point(self, point_camera):
        if point_camera[2] <= 0.1: return None
        try:
            u = int((self.fx * point_camera[0] / point_camera[2]) + self.cx)
            v = int((self.fy * point_camera[1] / point_camera[2]) + self.cy)
        except ZeroDivisionError: return None
        if 0 <= u < self.image_width and 0 <= v < self.image_height: return (u, v)
        return None

    def create_marker(self, detection, marker_id):
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
        marker_array = MarkerArray()
        for midx, detection in enumerate(detections):
            shape_marker, text_marker = self.create_marker(detection, midx)
            marker_array.markers.append(shape_marker)
            marker_array.markers.append(text_marker)
        self.marker_pub.publish(marker_array)

    def publish_centroid_debug(self, detections):
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
        try:
            u = (bbox[0] + bbox[2]) / 2.0
            v = (bbox[1] + bbox[3]) / 2.0
            ray_optical = self.K_inv @ np.array([u, v, 1.0])
            ray_physical = self.optical_to_physical_rot @ ray_optical
            
            transform = self.tf_buffer.lookup_transform("map", "car/camera_link", stamp, rospy.Duration(0.2))
            
            C_map = transform.transform.translation
            camera_origin_map = np.array([C_map.x, C_map.y, C_map.z])

            vec_stamped = Vector3Stamped()
            vec_stamped.header.frame_id = "car/camera_link"
            vec_stamped.header.stamp = stamp
            vec_stamped.vector.x, vec_stamped.vector.y, vec_stamped.vector.z = ray_physical
            
            V_map_stamped = tf2_geometry_msgs.do_transform_vector3(vec_stamped, transform)
            V_map = np.array([V_map_stamped.vector.x, V_map_stamped.vector.y, V_map_stamped.vector.z])
            V_map /= np.linalg.norm(V_map)
            
            if abs(V_map[2]) < 1e-6: return None
            
            t = -camera_origin_map[2] / V_map[2]
            if t < 0: return None

            return camera_origin_map + t * V_map
        except Exception as e:
            rospy.logwarn(f"Error in calculate_mark_ground_position: {e}")
            return None

    def publish_mark_marker(self, position_map, stamp):
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
            rospy.logerr(f"Failed to publish camera info: {str(e)}")

if __name__ == '__main__':
    try:
        detector = YOLODetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
