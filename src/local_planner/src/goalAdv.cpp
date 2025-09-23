#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <std_srvs/Empty.h>
#include <cmath>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <std_msgs/Int8.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Bool.h>

class LocalPathPlanner {
public:
    LocalPathPlanner() : path_length_(100.0), point_spacing_(0.5), 
                         initial_yaw_set_(false), current_command_yaw_(0.0), service_call_count_(0),
                         aruco_received_(false), yolo_received_(false) // --- MODIFIED ---
    {
        ros::NodeHandle nh;
        ros::NodeHandle private_nh("~");
        
        // 参数配置
        private_nh.param("path_length", path_length_, 100.0);
        private_nh.param("path_frame", path_frame_, std::string("map"));
        
        // 创建服务
        turn_right_srv_ = nh.advertiseService("goalAdv/turn_right", &LocalPathPlanner::turnRightCallback, this);
        turn_left_srv_ = nh.advertiseService("goalAdv/turn_left", &LocalPathPlanner::turnLeftCallback, this);
        turn_back_srv_ = nh.advertiseService("goalAdv/turn_back", &LocalPathPlanner::turnBackCallback, this);
        go_straight_srv_ = nh.advertiseService("goalAdv/go_straight", &LocalPathPlanner::goStraightCallback, this);
        
        // 延迟订阅里程计信息
        wait_timer_ = nh.createTimer(ros::Duration(1.5), 
                                     &LocalPathPlanner::startOdomSubscription, this, true);
        
        // 创建目标点发布器
        target_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/move_base_simple/goal", 1, true);
        direction_pub_ = nh.advertise<std_msgs::Int8>("/direction", 1, true);

        // 订阅视觉定位话题
        aruco_position_sub_ = nh.subscribe("/aruco_position_in_map", 1, 
                                           &LocalPathPlanner::arucoPositionCallback, this);
        yolo_position_sub_ = nh.subscribe("/yolo/mark_position_ground", 1, 
                                          &LocalPathPlanner::yoloPositionCallback, this);
        final_command_sub_ = nh.subscribe("/final_command_active", 1, 
                                          &LocalPathPlanner::finalCommandCallback, this);                                  

        status_pub_ = nh.advertise<std_msgs::Int32>("/status", 10, true);
    }
    
    void run() {
        ros::spin();
    }

private:
    // ... (服务回调函数 turnRightCallback, turnLeftCallback 等保持不变)
    bool turnRightCallback(std_srvs::Empty::Request&, std_srvs::Empty::Response&) {
        turnRight();
        service_call_count_++;
        return true;
    }
    bool turnLeftCallback(std_srvs::Empty::Request&, std_srvs::Empty::Response&) {
        turnLeft();
        service_call_count_++;
        return true;
    }
    bool turnBackCallback(std_srvs::Empty::Request&, std_srvs::Empty::Response&) {
        turnBack();
        service_call_count_++;
        return true;
    }
    bool goStraightCallback(std_srvs::Empty::Request&, std_srvs::Empty::Response&) {
        goStraight();
        service_call_count_++;
        return true;
    }
    void turnRight() {
        boost::mutex::scoped_lock lock(odom_mutex_);
        if (!initial_yaw_set_) return;
        current_command_yaw_ -= M_PI/2;
        normalizeYaw(current_command_yaw_);
        generateTargetPoint();
        std_msgs::Int8 msg; msg.data = 1;
        direction_pub_.publish(msg);
        ROS_INFO("Turned RIGHT. New command yaw: %.2f deg", current_command_yaw_ * 180/M_PI);
    }
    void turnLeft() {
        boost::mutex::scoped_lock lock(odom_mutex_);
        if (!initial_yaw_set_) return;
        current_command_yaw_ += M_PI/2;
        normalizeYaw(current_command_yaw_);
        generateTargetPoint();
        std_msgs::Int8 msg; msg.data = 2;
        direction_pub_.publish(msg);
        ROS_INFO("Turned LEFT. New command yaw: %.2f deg", current_command_yaw_ * 180/M_PI);
    }
    void turnBack() {
        boost::mutex::scoped_lock lock(odom_mutex_);
        if (!initial_yaw_set_) return;
        current_command_yaw_ += M_PI;
        normalizeYaw(current_command_yaw_);
        generateTargetPoint();
        std_msgs::Int8 msg; msg.data = 3;
        direction_pub_.publish(msg);
        ROS_INFO("Turned BACK. New command yaw: %.2f deg", current_command_yaw_ * 180/M_PI);
    }
    void goStraight() {
        boost::mutex::scoped_lock lock(odom_mutex_);
        if (!initial_yaw_set_) return;
        generateTargetPoint();
        std_msgs::Int8 msg; msg.data = 0;
        direction_pub_.publish(msg);
        ROS_INFO("Go STRAIGHT. Command yaw: %.2f deg", current_command_yaw_ * 180/M_PI);
    }
    void normalizeYaw(double& yaw) {
        yaw = fmod(yaw, 2*M_PI);
        if (yaw > M_PI) yaw -= 2*M_PI;
        else if (yaw < -M_PI) yaw += 2*M_PI;
    }
    void startOdomSubscription(const ros::TimerEvent&) {
        ros::NodeHandle nh;
        odom_sub_ = nh.subscribe("/magv/odometry/gt", 1, &LocalPathPlanner::odomCallback, this);
        ROS_INFO("Started odometry subscription.");
    }

    void finalCommandCallback(const std_msgs::Bool::ConstPtr& msg) {
        boost::mutex::scoped_lock lock(odom_mutex_);
        final_command_active_ = msg->data;
        if (final_command_active_) {
            ROS_INFO("GoalAdv: Received activation signal. YOLO goals are now ENABLED.");
        } else {
            ROS_INFO("GoalAdv: Received deactivation signal. Resetting and disabling YOLO goals.");
            // 当收到 deactivation 信号时，重置YOLO状态，为下一轮任务做准备
            yolo_received_ = false;
        }
    }
    /**
     * @brief 接收YOLO发布的Mark位置信息的回调函数
     */
    void yoloPositionCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
        boost::mutex::scoped_lock lock(odom_mutex_);

        if (!final_command_active_) { //只有在最后一条指令是才激活检测
            return; // 忽略此消息
        }

        if (!yolo_received_) {
             ROS_INFO("First YOLO mark position received.");
             yolo_received_ = true;
        }
        yolo_point_ = *msg;
    }

    // --- MODIFIED ---
    /**
     * @brief 接收Aruco发布的Mark位置信息的回调函数
     */
    void arucoPositionCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
        boost::mutex::scoped_lock lock(odom_mutex_);
        if (!aruco_received_) {
            ROS_WARN("First ARUCO mark position received. It will be used as the permanent goal.");
            aruco_received_ = true; // --- MODIFIED ---: 设置永久标志位
        }
        aruco_point_ = *msg;
    }

    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        boost::mutex::scoped_lock lock(odom_mutex_);
        
        current_position_ = msg->pose.pose.position;
        
        if (!initial_yaw_set_) {
            tf2::Quaternion q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
            tf2::Matrix3x3 m(q);
            double roll, pitch;
            m.getRPY(roll, pitch, initial_yaw_);
            current_command_yaw_ = 0.0;
            initial_yaw_set_ = true;
            ROS_INFO("Initial yaw set to %.2f degrees", initial_yaw_ * 180/M_PI);
        }

        // 检查是否到达目标的逻辑
        geometry_msgs::PoseStamped current_target;
        if (aruco_received_) {
            current_target = aruco_point_;
        } else if (yolo_received_) {
            current_target = yolo_point_;
        } else {
            // 如果视觉目标都无效，则不进行到达判断
            if(service_call_count_ > 0 ){ 
                publishTarget();
            } 
            return;
        }

        double vx = msg->twist.twist.linear.x;
        double vy = msg->twist.twist.linear.y;
        double v = std::sqrt(vx*vx + vy*vy);
        double dx = current_position_.x - current_target.pose.position.x;
        double dy = current_position_.y - current_target.pose.position.y;
        double distance = std::sqrt(dx*dx + dy*dy);

        if (distance < 0.4 ) {
            if (!reached_marker_) {
                ROS_INFO("Reached visual marker! Distance: %.3f m", distance);
                reached_marker_ = true;
            }
            std_msgs::Int32 status_msg;
            status_msg.data = 0;
            status_pub_.publish(status_msg);
        } else {
            reached_marker_ = false;
        }
        
        if(service_call_count_ > 0 ){ 
            publishTarget();
        } 
    }
    
    void generateTargetPoint() {
        if (!initial_yaw_set_) return;
        double target_yaw = initial_yaw_ + current_command_yaw_;
        target_point_.header.stamp = ros::Time::now();
        target_point_.header.frame_id = path_frame_;
        target_point_.pose.position.x = current_position_.x + path_length_ * cos(target_yaw);
        target_point_.pose.position.y = current_position_.y + path_length_ * sin(target_yaw);
        target_point_.pose.position.z = current_position_.z;
        tf2::Quaternion q;
        q.setRPY(0, 0, target_yaw);
        target_point_.pose.orientation = tf2::toMsg(q);
    }
    
    // --- MODIFIED ---: 重写目标发布逻辑为纯粹的优先级判断
    void publishTarget() {
        if (!initial_yaw_set_) return;

        if (aruco_received_) {
            // 优先级1: Aruco已收到，永远使用Aruco目标
            target_pub_.publish(aruco_point_);
            ROS_INFO_THROTTLE(1.0, "Publishing permanent ARUCO goal (%.2f, %.2f)",
                aruco_point_.pose.position.x, aruco_point_.pose.position.y);
        } else if (yolo_received_) {
            // 优先级2: Aruco未收到，但YOLO已收到，使用YOLO目标
            target_pub_.publish(yolo_point_);
            ROS_INFO_THROTTLE(1.0, "ARUCO not available. Publishing fallback YOLO goal (%.2f, %.2f)",
                yolo_point_.pose.position.x, yolo_point_.pose.position.y);
        } else {
            // 优先级3: 视觉目标都未收到，使用指令生成的目标
            target_pub_.publish(target_point_);
            ROS_INFO_THROTTLE(1.0, "No visual goal. Publishing command-based goal (%.2f, %.2f)",
                target_point_.pose.position.x, target_point_.pose.position.y);
        }
    }

    // ROS 通信对象
    ros::Subscriber odom_sub_;
    ros::Subscriber aruco_position_sub_;
    ros::Subscriber yolo_position_sub_;
    ros::Publisher target_pub_;
    ros::Publisher direction_pub_;
    ros::Publisher status_pub_; 
    ros::Timer wait_timer_;
    ros::ServiceServer turn_right_srv_;
    ros::ServiceServer turn_left_srv_;
    ros::ServiceServer turn_back_srv_;
    ros::ServiceServer go_straight_srv_;
    ros::Subscriber final_command_sub_;
    
    // 同步机制
    boost::mutex odom_mutex_;
    
    // 状态变量
    geometry_msgs::Point current_position_;
    geometry_msgs::PoseStamped target_point_;
    geometry_msgs::PoseStamped aruco_point_;
    geometry_msgs::PoseStamped yolo_point_;
    bool initial_yaw_set_;
    bool aruco_received_;   // --- MODIFIED ---: 从 initial_aruco 重命名而来
    bool yolo_received_;
    double initial_yaw_;
    double current_command_yaw_;
    bool reached_marker_;   // --- MODIFIED ---: 重命名
    bool final_command_active_; 
    
    // 配置参数
    double path_length_;
    double point_spacing_;
    std::string path_frame_;

    int service_call_count_; 
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "goalAdv");
    LocalPathPlanner planner;
    planner.run();
    return 0;
}
