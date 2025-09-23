#include <ros/ros.h>
#include <std_srvs/Empty.h>
#include <std_msgs/Int32.h>
#include <queue>
#include <map>
#include <string>
#include <algorithm>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <keyword_extractor/KeywordsResult.h> 
#include <yolo_detect/DetectionArray.h>       
#include <cmath>
#include <boost/bind/bind.hpp>
#include <image_transport/transport_hints.h>
#include <std_msgs/Bool.h>

using namespace boost::placeholders; // 用于 _1

class StateMachine {
public:
    StateMachine() : 
        current_state_(IDLE), 
        is_final_command_(false),
        aruco_detection_active_(false),
        aruco_target_found_(false),
        tf_listener_(tf_buffer_),
        aruco_detection_count_(0),
        aruco_size_(1.0)  // 设置为1米×1米
    {
        ros::NodeHandle nh;
        ros::NodeHandle private_nh("~");
        
        // 建立物体名称到ID的映射（共8种物体）
        object_id_map_ = {
            {"trash", 0},
            {"bench", 1},
            {"billboard", 2},
            {"tree", 3},
            {"tractor trailer", 4},   // 拖车
            {"barrel", 5},
            {"fire hydrant", 6},
            {"traffic cone", 7}   // 注意：现在只有8种物体
        };

        // 创建服务客户端
        go_straight_client_ = nh.serviceClient<std_srvs::Empty>("goalAdv/go_straight");
        turn_left_client_ = nh.serviceClient<std_srvs::Empty>("goalAdv/turn_left");
        turn_right_client_ = nh.serviceClient<std_srvs::Empty>("goalAdv/turn_right");
        turn_back_client_ = nh.serviceClient<std_srvs::Empty>("goalAdv/turn_back");

        // 订阅关键词话题
        keywords_sub_ = nh.subscribe("/extracted_keywords", 1, &StateMachine::keywordsCallback, this);
        
        // 订阅检测结果话题
        detections_sub_ = nh.subscribe("/yolo/detections", 1, &StateMachine::detectionsCallback, this);

        // Aruco标记位置发布器
        aruco_position_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/aruco_position_in_map", 1, true);//ture:如果该话题不再发布，订阅者依旧可以收到旧数据
        int_pub_ = nh.advertise<std_msgs::Int32>("/int_topic", 1);
        final_command_pub_ = nh.advertise<std_msgs::Bool>("/final_command_active", 1, true); // latch=true

        std_msgs::Bool initial_msg;
        initial_msg.data = false;
        final_command_pub_.publish(initial_msg);

        // 图像订阅器（用于Aruco检测）
        image_sub_ = nh.subscribe(
        "/magv/camera/image_compressed/compressed", 
        1, 
        &StateMachine::imageCallback, 
        this
    );
        
        // 加载相机参数 外参  后续可以使用tf转换，由于目前参数较少，直接加载比较方便
        private_nh.param<double>("camera_pitch", camera_pitch_, 0.314); // 相机俯仰角弧度
        private_nh.param<double>("camera_translation_x", camera_translation_[0], 0.5);
        private_nh.param<double>("camera_translation_y", camera_translation_[1], -0.04);
        private_nh.param<double>("camera_translation_z", camera_translation_[2], 0.57);
        
        // 根据提供的参数计算相机内参矩阵
        double image_width = 1080.0;  // 图像宽度
        double image_height = 720.0;  // 图像高度
        double horizontal_fov = 2.0;  // 水平视场角（弧度）
        
        // 计算焦点距离（焦距）
        double fx = (image_width / 2.0) / std::tan(horizontal_fov / 2.0);
        double fy = fx; // 假设垂直方向焦距与水平相同
        double cx = image_width / 2.0;  // 主点x坐标
        double cy = image_height / 2.0; // 主点y坐标
        
        // 构建相机内参矩阵
        camera_matrix_ = (cv::Mat_<double>(3, 3) << 
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0);
            
        // 畸变系数 - 初始化为0
        dist_coeffs_ = (cv::Mat_<double>(1, 5) << 0.0, 0.0, 0.0, 0.0, 0.0);
        
        ROS_INFO("Calculated camera matrix:");
        ROS_INFO("  fx: %.2f, fy: %.2f", fx, fy);
        ROS_INFO("  cx: %.2f, cy: %.2f", cx, cy);
        
        // 创建Aruco字典
        aruco_dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
        aruco_params_ = cv::aruco::DetectorParameters::create();
	aruco_params_->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX; // 使用亚像素角点精炼
        aruco_params_->adaptiveThreshWinSizeMin = 5; // 调整自适应阈值窗口大小
        aruco_params_->adaptiveThreshWinSizeMax = 25;
        aruco_params_->adaptiveThreshWinSizeStep = 5;
        aruco_params_->errorCorrectionRate = 0.8; // 提高错误修正率
        
        ROS_INFO("State Machine initialized with 8 object types and Aruco detection");
    }

    void run() {
        ros::spin();
    }

private:
    enum State { IDLE, EXECUTING_ACTION, WAITING_FOR_OBJECT };
    State current_state_;
    bool is_final_command_;  // 标记当前指令是否是最后一条
    bool aruco_detection_active_; // Aruco检测是否激活
    bool aruco_target_found_; // Aruco标记是否已找到
    int aruco_detection_count_;  // Aruco检测计数器
    std::deque<geometry_msgs::PoseStamped> aruco_position_history_;  // 存储最近五次位置
    
    // TF2相关
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    
    // ROS 通信对象
    ros::Subscriber keywords_sub_;
    ros::Subscriber detections_sub_;
    ros::Subscriber image_sub_;
    ros::Publisher aruco_position_pub_;
    ros::Publisher int_pub_;
    ros::ServiceClient go_straight_client_;
    ros::ServiceClient turn_left_client_;
    ros::ServiceClient turn_right_client_;
    ros::ServiceClient turn_back_client_;
    ros::Publisher final_command_pub_; //是否到达最后一条指令的判断

    // Aruco检测相关
    cv::Ptr<cv::aruco::Dictionary> aruco_dict_;
    cv::Ptr<cv::aruco::DetectorParameters> aruco_params_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    const double aruco_size_; // Aruco标记尺寸（米），固定为1.0
    double camera_pitch_; // 相机俯仰角（弧度）
    double camera_translation_[3]; // 相机相对于机器人中心的偏移
    
    // 其他成员变量
    std::queue<std::string> command_queue_;
    std::map<std::string, int> object_id_map_;
    std::string current_object_to_wait_;
    int current_object_id_;

    void keywordsCallback(const keyword_extractor::KeywordsResult::ConstPtr& msg) {
        // 重置状态
        while (!command_queue_.empty()) command_queue_.pop();
        is_final_command_ = false;
        aruco_detection_active_ = false;
        aruco_target_found_ = false;
        
        // 装载新的指令序列
        ROS_INFO("Received %zu commands:", msg->keywords.size());
        for (const auto& keyword : msg->keywords) {
            command_queue_.push(keyword);
            ROS_INFO("- %s", keyword.c_str());
        }
        
        // 重置状态机
        current_state_ = EXECUTING_ACTION;
        processNextCommand();
    }

    void detectionsCallback(const yolo_detect::DetectionArray::ConstPtr& msg) {
        if (current_state_ != WAITING_FOR_OBJECT) return;

        float min_distance = std::numeric_limits<float>::max();
        bool object_found = false;

        // 查找目标物体的最小距离
        for (const auto& detection : msg->detections) {
            if (detection.class_id == current_object_id_) {
                object_found = true;
                if (detection.distance < min_distance) {
                    min_distance = detection.distance;
                }
            }
        }


        float distance_threshold = 3.5f; // 默认阈值
        if (current_object_to_wait_ == "barrel") {
            distance_threshold = 2.0f;
        } else if (current_object_to_wait_ == "billboard") {
            distance_threshold = 4.5f;
        } 
        else{
            distance_threshold = 3.5f;
        }

        // 检查是否达到距离阈值
        if (object_found && min_distance < distance_threshold) {
            ROS_INFO("Reached object: %s (distance: %.2fm)", current_object_to_wait_.c_str(), min_distance);
	
            
            if (is_final_command_) {
                ROS_WARN(">>>>>>> FINAL COMMAND COMPLETED: %s <<<<<<<", current_object_to_wait_.c_str());
                ROS_WARN(">>>>>>> ACTIVATING ARUCO DETECTION <<<<<<<");
                aruco_detection_active_ = true;//到达最终目标后激活Aruco检测，实际上在指令层，早已激活检测，因为只要到了最后一条指令就可以开始检测了，防止错误识别距离的情况
                aruco_target_found_ = false;//该变量无用，会导致无法持续发布位置
            } else {
                current_state_ = EXECUTING_ACTION;
                processNextCommand();
            }
        }
    }

    void processNextCommand() {
        // 检查是否是最后一条指令
        is_final_command_ = command_queue_.size() == 1;
        
        if (command_queue_.empty()) {
            current_state_ = IDLE;
            ROS_INFO("All commands processed. Waiting for new instructions.");
            return;
        }

        std::string cmd = command_queue_.front();
        command_queue_.pop();
        
	if (cmd == "pass") {
            ROS_INFO("Processing 'pass' command...");
            // 检查队列中是否还有下一个指令
            if (!command_queue_.empty()) {
                std::string next_cmd = command_queue_.front();
                // 检查下一个指令是否是物体关键词
                if (object_id_map_.count(next_cmd)) {
                    ROS_INFO("Next command is an object ('%s'). Skipping it.", next_cmd.c_str());
                    command_queue_.pop(); // 将物体指令也弹出
                }
            }
            // 无论pass后面是什么，都立即处理再下一条指令
            processNextCommand();
            return; // 结束本次函数调用
        }

        if (is_final_command_) {
            std_msgs::Bool active_msg;
            active_msg.data = true;
            final_command_pub_.publish(active_msg);
            ROS_WARN("Processing FINAL command: %s", cmd.c_str());
        }

        // 处理动作指令
        if (cmd == "front" || cmd == "left" || cmd == "right" || cmd == "back") {
            // 如果是最后一条指令且是动作指令
            if (is_final_command_) {
                executeAction(cmd);  // 先执行动作指令
                ROS_WARN(">>>>>>> FINAL ACTION COMMAND: %s <<<<<<<", cmd.c_str());
                ROS_WARN(">>>>>>> ACTIVATING ARUCO DETECTION <<<<<<<");
                aruco_detection_active_ = true;
                aruco_target_found_ = false;
                return;
            }
            
            executeAction(cmd);
            
            // 检查是否需要忽略后续的"front"指令
            if (!command_queue_.empty()) {
                std::string next_cmd = command_queue_.front();
                if ((cmd == "left" || cmd == "right" || cmd == "back"|| cmd == "front") && 
                    next_cmd == "front") {
                    ROS_INFO("Skipping redundant 'front' command after turn");
                    command_queue_.pop(); // 丢弃多余的front指令
                }
            }
             
            // 准备处理下一条指令
            if (!command_queue_.empty()) {

                current_state_ = EXECUTING_ACTION;
		        ros::Duration(2.5).sleep();
                processNextCommand();
            } else {
                current_state_ = IDLE;
            }
        }
        // 处理物体指令
        else if (object_id_map_.find(cmd) != object_id_map_.end()) {
            current_object_to_wait_ = cmd;
            current_object_id_ = object_id_map_[cmd];
            
            // 如果是最后一条指令（物体）
            if (is_final_command_) {
                ROS_WARN(">>>>>>> FINAL OBJECT COMMAND: %s <<<<<<<", cmd.c_str());
                current_state_ = WAITING_FOR_OBJECT;

                aruco_detection_active_ = true;
                ROS_INFO("Waiting for object: %s (ID: %d)", cmd.c_str(), current_object_id_);
            } 
            // 如果不是最后一条指令
            else {
                current_state_ = WAITING_FOR_OBJECT;
                ROS_INFO("Waiting for object: %s (ID: %d)", cmd.c_str(), current_object_id_);
            }
        }
        else {
            ROS_WARN("Unknown command: %s", cmd.c_str());
            processNextCommand(); // 跳过未知命令
        }
    }

    void executeAction(const std::string& action) {
        std_srvs::Empty srv;
        bool success = false;

        if (action == "front") {
            success = go_straight_client_.call(srv);
            ROS_INFO("Executing FRONT action");
        }
        else if (action == "left") {
            success = turn_left_client_.call(srv);
            ROS_INFO("Executing LEFT action");
        }
        else if (action == "right") {
            success = turn_right_client_.call(srv);
            ROS_INFO("Executing RIGHT action");
        }
        else if (action == "back") {
            success = turn_back_client_.call(srv);
            ROS_INFO("Executing BACK action");
        }

        if (!success) {
            ROS_ERROR("Failed to execute %s action", action.c_str());
        }
    }
    
    void imageCallback(const sensor_msgs::CompressedImageConstPtr& msg) {
        if (!aruco_detection_active_ ||aruco_detection_count_ >= 5) {
            return;
        }
        
        try {
            // 转换ROS图像到OpenCV格式
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
            cv::Mat image = cv_ptr->image;
            
            // 检测Aruco标记
            std::vector<int> markerIds;
            std::vector<std::vector<cv::Point2f>> markerCorners;
            cv::aruco::detectMarkers(image, aruco_dict_, markerCorners, markerIds, aruco_params_);
                        std_msgs::Int32 id_msg1;
                        id_msg1.data = 2;
                        int_pub_.publish(id_msg1);
            if (!markerIds.empty()) {
                ROS_WARN("Detected Aruco markers: %zu", markerIds.size());
                // 估计标记姿态（使用1米的标记尺寸）
                std::vector<cv::Vec3d> rvecs, tvecs;
                cv::aruco::estimatePoseSingleMarkers(markerCorners, aruco_size_, 
                                                   camera_matrix_, dist_coeffs_, 
                                                   rvecs, tvecs);
                
                // 遍历所有检测到的标记（这里只处理第一个）
                if (!tvecs.empty()) {
                    // 计算标记在机器人坐标系中的位置
                    geometry_msgs::PoseStamped marker_in_base_link;
                    calculateMarkerPosition(tvecs[0], rvecs[0], marker_in_base_link);
                    
                    // 转换到map坐标系
                    geometry_msgs::PoseStamped marker_in_map;
                    if (transformToMap(marker_in_base_link, marker_in_map)) {
                        aruco_detection_count_++;
                        aruco_position_history_.push_back(marker_in_map);
                        if (aruco_detection_count_ >= 5) {
                        
                            geometry_msgs::PoseStamped final_position = aruco_position_history_.back();
                            aruco_position_pub_.publish(final_position);
                        // 发布位置
                        // aruco_position_pub_.publish(marker_in_map);
                        // ROS_WARN(">>>>>>> ARUCO POSITION IN MAP: (%.2f, %.2f, %.2f) <<<<<<<",
                        //          marker_in_map.pose.position.x, 
                        //          marker_in_map.pose.position.y, 
                        //          marker_in_map.pose.position.z);
                        aruco_target_found_ = true;
                        ROS_WARN(">>>>>>> ARUCO TARGET FOUND. WAITING FOR USER INPUT <<<<<<<");
                        }
                    }
                }
            }
        }
        catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
        catch (cv::Exception& e) {
            ROS_ERROR("OpenCV exception: %s", e.what());
        }
        catch (tf2::TransformException& ex) {
            ROS_WARN("TF exception: %s", ex.what());
        }
    }
    
    void calculateMarkerPosition(const cv::Vec3d& tvec, const cv::Vec3d& rvec, 
                                geometry_msgs::PoseStamped& result) {
        // tvec: 相机坐标系中的位置 (x, y, z)
        double camera_x = tvec[0];
        double camera_y = tvec[1];
        double camera_z = tvec[2];
        
        // 考虑相机的俯仰角进行坐标变换
        double cos_pitch = cos(camera_pitch_);
        double sin_pitch = sin(camera_pitch_);
        
        // 转换到机器人坐标系 (base_link)
        // 相机坐标系: x-右, y-下, z-前
        // 机器人坐标系: x-前, y-左, z-上
        double robot_relative_x = camera_z * cos_pitch + camera_y * sin_pitch;
        double robot_relative_y = -camera_x; // 右到左，所以取负
        double robot_relative_z = -camera_z * sin_pitch + camera_y * cos_pitch;
        
        // 加上相机相对于机器人的位置偏移
        robot_relative_x += camera_translation_[0];
        robot_relative_y += camera_translation_[1];
        robot_relative_z += camera_translation_[2];
        
        // 设置结果
        result.header.stamp = ros::Time::now();
        result.header.frame_id = "car/base_link";
        result.pose.position.x = robot_relative_x;
        result.pose.position.y = robot_relative_y;
        result.pose.position.z = robot_relative_z;
        
        // 姿态暂时不设置，因为我们只关心位置
        result.pose.orientation.w = 1.0;
        
        ROS_INFO("Aruco position in base_link: (%.2f, %.2f, %.2f)", 
                 robot_relative_x, robot_relative_y, robot_relative_z);
    }
    
    bool transformToMap(const geometry_msgs::PoseStamped& input, 
                       geometry_msgs::PoseStamped& output) {
        try {
            // 获取从base_link到map的变换
            geometry_msgs::TransformStamped transform = 
                tf_buffer_.lookupTransform("map", input.header.frame_id, 
                                          input.header.stamp, ros::Duration(0.1));
            
            // 应用变换
            tf2::doTransform(input, output, transform);
            return true;
        }
        catch (tf2::TransformException& ex) {
            ROS_WARN("Failed to transform point to map frame: %s", ex.what());
            return false;
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "state_machine");
    StateMachine state_machine;
    state_machine.run();
    return 0;
}


