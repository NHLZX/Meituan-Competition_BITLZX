/**
 * @file goalAdv.cpp
 * @author BITLZX
 * @date 2025-09-23
 * @brief 该节点实现了一个目标点发布器(Goal Publisher)，它能够根据外部服务指令或视觉定位结果来发布目标点。
 *
 * @details
 * 该节点主要有三种工作模式：
 * 1. **指令模式**: 接收来自外部服务的指令（如左转、右转、直行），并根据当前机器人的位置和朝向，
 * 计算出一个远方的目标点并发布到 `/move_base_simple/goal` 话题，引导机器人移动。
 * 2. **YOLO视觉引导模式**: 当接收到 `/final_command_active` 信号后，节点会开始接收并使用
 * 来自 `/yolo/mark_position_ground` 话题的视觉定位结果作为目标点。
 * 3. **Aruco视觉引导模式**: 节点会持续监听 `/aruco_position_in_map` 话题。一旦接收到Aruco码的位置，
 * 它将永久锁定该位置作为最终目标点，此模式拥有最高优先级。
 *
 * 根据优先级（Aruco > YOLO > 指令）选择并发布当前最合适的目标点。
 */

#include <ros/ros.h>
#include <nav_msgs/Path.hh>
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

/**
 * @class LocalPathPlanner    该类名应该称作goalAdv 这里命名不太规范
 * @brief 管理目标点的生成、选择和发布的类。
 */
class LocalPathPlanner {
public:
    /**
     * @brief LocalPathPlanner类的构造函数。
     * @details 初始化ROS节点、参数、服务、发布者和订阅者。
     */
    LocalPathPlanner() : path_length_(100.0), point_spacing_(0.5), 
                         initial_yaw_set_(false), current_command_yaw_(0.0), service_call_count_(0),
                         aruco_received_(false), yolo_received_(false)
    {
        ros::NodeHandle nh;
        ros::NodeHandle private_nh("~");
        
        // 从参数服务器加载配置参数
        private_nh.param("path_length", path_length_, 100.0);
        private_nh.param("path_frame", path_frame_, std::string("map"));
        
        // 初始化ROS服务，用于接收外部运动指令
        turn_right_srv_ = nh.advertiseService("goalAdv/turn_right", &LocalPathPlanner::turnRightCallback, this);
        turn_left_srv_ = nh.advertiseService("goalAdv/turn_left", &LocalPathPlanner::turnLeftCallback, this);
        turn_back_srv_ = nh.advertiseService("goalAdv/turn_back", &LocalPathPlanner::turnBackCallback, this);
        go_straight_srv_ = nh.advertiseService("goalAdv/go_straight", &LocalPathPlanner::goStraightCallback, this);
        
        // 使用单次定时器来延迟里程计的订阅，确保TF树等其他组件已准备就绪，主要是确保里程计信息稳定，里程计在初始时不稳定
        wait_timer_ = nh.createTimer(ros::Duration(1.5), 
                                     &LocalPathPlanner::startOdomSubscription, this, true);
        
        // 初始化目标点发布器
        target_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/move_base_simple/goal", 1, true);
        direction_pub_ = nh.advertise<std_msgs::Int8>("/direction", 1, true);

        // 初始化视觉定位结果的订阅者
        aruco_position_sub_ = nh.subscribe("/aruco_position_in_map", 1, 
                                           &LocalPathPlanner::arucoPositionCallback, this);
        yolo_position_sub_ = nh.subscribe("/yolo/mark_position_ground", 1, 
                                          &LocalPathPlanner::yoloPositionCallback, this);
        // 订阅最终指令激活信号，用于控制YOLO模式的开启
        final_command_sub_ = nh.subscribe("/final_command_active", 1, 
                                          &LocalPathPlanner::finalCommandCallback, this);                                  
        
        // 初始化状态发布器，用于通知外部系统机器人是否到达目标点
        status_pub_ = nh.advertise<std_msgs::Int32>("/status", 10, true);
    }
    
    /**
     * @brief 启动ROS节点的主循环。
     */
    void run() {
        ros::spin();
    }

private:
    // --- 服务回调函数 ---

    /** @brief "右转"服务的处理回调。*/
    bool turnRightCallback(std_srvs::Empty::Request&, std_srvs::Empty::Response&) {
        turnRight();
        service_call_count_++;
        return true;
    }
    /** @brief "左转"服务的处理回调。*/
    bool turnLeftCallback(std_srvs::Empty::Request&, std_srvs::Empty::Response&) {
        turnLeft();
        service_call_count_++;
        return true;
    }
    /** @brief "掉头"服务的处理回调。*/
    bool turnBackCallback(std_srvs::Empty::Request&, std_srvs::Empty::Response&) {
        turnBack();
        service_call_count_++;
        return true;
    }
    /** @brief "直行"服务的处理回调。*/
    bool goStraightCallback(std_srvs::Empty::Request&, std_srvs::Empty::Response&) {
        goStraight();
        service_call_count_++;
        return true;
    }

    // --- 指令执行函数 ---

    /** @brief 执行右转逻辑，更新目标偏航角并生成新目标点。*/
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
    /** @brief 执行左转逻辑，更新目标偏航角并生成新目标点。*/
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
    /** @brief 执行掉头逻辑，更新目标偏航角并生成新目标点。*/
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
    /** @brief 执行直行逻辑，根据当前偏航角生成新目标点。*/
    void goStraight() {
        boost::mutex::scoped_lock lock(odom_mutex_);
        if (!initial_yaw_set_) return;
        generateTargetPoint();
        std_msgs::Int8 msg; msg.data = 0;
        direction_pub_.publish(msg);
        ROS_INFO("Go STRAIGHT. Command yaw: %.2f deg", current_command_yaw_ * 180/M_PI);
    }

    // --- 核心逻辑与回调 ---

    /**
     * @brief 将偏航角归一化到[-PI, PI]的区间内。
     * @param yaw 需要归一化的偏航角引用。
     */
    void normalizeYaw(double& yaw) {
        yaw = fmod(yaw, 2*M_PI);
        if (yaw > M_PI) yaw -= 2*M_PI;
        else if (yaw < -M_PI) yaw += 2*M_PI;
    }

    /**
     * @brief 定时器回调，用于正式启动里程计的订阅。
     * @param event 定时器事件（未使用）。
     */
    void startOdomSubscription(const ros::TimerEvent&) {
        ros::NodeHandle nh;
        odom_sub_ = nh.subscribe("/magv/odometry/gt", 1, &LocalPathPlanner::odomCallback, this);
        ROS_INFO("Started odometry subscription.");
    }

    /**
     * @brief 订阅`/final_command_active`话题的回调函数。
     * @details 此函数用于激活或关闭YOLO视觉引导模式。
     * @param msg Bool类型的消息，true为激活，false为关闭。
     */
    void finalCommandCallback(const std_msgs::Bool::ConstPtr& msg) {
        boost::mutex::scoped_lock lock(odom_mutex_);
        final_command_active_ = msg->data;
        if (final_command_active_) {
            ROS_INFO("GoalAdv: Received activation signal. YOLO goals are now ENABLED.");
        } else {
            ROS_INFO("GoalAdv: Received deactivation signal. Resetting and disabling YOLO goals.");
            // 当收到关闭信号时，重置YOLO状态，为下一轮任务做准备
            yolo_received_ = false;
        }
    }

    /**
     * @brief 订阅YOLO发布的Mark板位置信息的回调函数。
     * @param msg 包含Mark板位姿的PoseStamped消息。
     */
    void yoloPositionCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
        boost::mutex::scoped_lock lock(odom_mutex_);

        // 仅当最终指令激活时，才处理YOLO数据
        if (!final_command_active_) {
            return;
        }

        if (!yolo_received_) {
             ROS_INFO("First YOLO mark position received.");
             yolo_received_ = true;
        }
        yolo_point_ = *msg;
    }

    /**
     * @brief 订阅Aruco发布的Mark板位置信息的回调函数。
     * @details 一旦收到Aruco信息，它将被设为永久目标。
     * @param msg 包含Aruco码位姿的PoseStamped消息。
     */
    void arucoPositionCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
        boost::mutex::scoped_lock lock(odom_mutex_);
        if (!aruco_received_) {
            ROS_WARN("First ARUCO mark position received. It will be used as the permanent goal.");
            aruco_received_ = true; // 设置永久标志位
        }
        aruco_point_ = *msg;
    }

    /**
     * @brief 里程计信息回调函数，是节点的主处理循环。
     * @details
     * 1. 更新机器人当前位置。
     * 2. 在第一次接收到消息时，初始化机器人的起始偏航角。
     * 3. 检查是否到达视觉目标点（Aruco或YOLO）。
     * 4. 如果到达，发布状态信息。
     * 5. 持续调用 `publishTarget()` 来发布当前有效的目标。
     * @param msg 里程计消息。
     */
    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        boost::mutex::scoped_lock lock(odom_mutex_);
        
        current_position_ = msg->pose.pose.position;
        
        // 首次运行时，从里程计中获取并存储初始偏航角
        if (!initial_yaw_set_) {
            tf2::Quaternion q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
            tf2::Matrix3x3 m(q);
            double roll, pitch;
            m.getRPY(roll, pitch, initial_yaw_);
            current_command_yaw_ = 0.0;
            initial_yaw_set_ = true;
            ROS_INFO("Initial yaw set to %.2f degrees", initial_yaw_ * 180/M_PI);
        }

        // --- 检查是否到达视觉引导的目标点 ---
        geometry_msgs::PoseStamped current_target;
        if (aruco_received_) {
            current_target = aruco_point_;
        } else if (yolo_received_) {
            current_target = yolo_point_;
        } else {
            // 如果两个视觉目标都无效，则不进行到达判断，仅在有指令时发布目标
            if(service_call_count_ > 0 ){ 
                publishTarget();
            } 
            return;
        }

        // 计算当前速度和到目标的距离
        double vx = msg->twist.twist.linear.x;
        double vy = msg->twist.twist.linear.y;
        double v = std::sqrt(vx*vx + vy*vy);
        double dx = current_position_.x - current_target.pose.position.x;
        double dy = current_position_.y - current_target.pose.position.y;
        double distance = std::sqrt(dx*dx + dy*dy);

        // 如果距离小于阈值，则认为已到达，并发布状态0
        if (distance < 0.4 ) {
            if (!reached_marker_) {
                ROS_INFO("Reached visual marker! Distance: %.3f m", distance);
                reached_marker_ = true;
            }
            std_msgs::Int32 status_msg;
            status_msg.data = 0; // 状态0: 到达目标
            status_pub_.publish(status_msg);
        } else {
            reached_marker_ = false;
        }
        
        // 只要接收过服务指令，就持续发布目标点
        if(service_call_count_ > 0 ){ 
            publishTarget();
        } 
    }
    
    /**
     * @brief 根据指令模式下的偏航角，生成一个远方的目标点。
     */
    void generateTargetPoint() {
        if (!initial_yaw_set_) return;
        
        // 计算在map坐标系下的绝对目标偏航角
        double target_yaw = initial_yaw_ + current_command_yaw_;
        
        // 填充目标点消息
        target_point_.header.stamp = ros::Time::now();
        target_point_.header.frame_id = path_frame_;
        target_point_.pose.position.x = current_position_.x + path_length_ * cos(target_yaw);
        target_point_.pose.position.y = current_position_.y + path_length_ * sin(target_yaw);
        target_point_.pose.position.z = current_position_.z;
        
        // 设置目标姿态
        tf2::Quaternion q;
        q.setRPY(0, 0, target_yaw);
        target_point_.pose.orientation = tf2::toMsg(q);
    }
    
    /**
     * @brief 根据优先级发布目标点。
     * @details
     * 这是节点的核心决策逻辑。优先级顺序如下：
     * 1. **Aruco目标**：如果接收到Aruco位置，则永久使用它作为目标。
     * 2. **YOLO目标**：如果未接收到Aruco，但接收到YOLO位置，则使用YOLO位置。
     * 3. **指令目标**：如果以上两者都未接收到，则使用服务指令生成的目标点。
     */
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

    // --- ROS 通信对象 ---
    ros::Subscriber odom_sub_;                  ///< 里程计订阅者
    ros::Subscriber aruco_position_sub_;        ///< Aruco位置订阅者
    ros::Subscriber yolo_position_sub_;         ///< YOLO位置订阅者
    ros::Subscriber final_command_sub_;         ///< YOLO模式激活信号订阅者
    ros::Publisher target_pub_;                 ///< 最终目标点发布者
    ros::Publisher direction_pub_;              ///< 运动指令方向发布者
    ros::Publisher status_pub_;                 ///< 到达状态发布者
    ros::Timer wait_timer_;                     ///< 延迟订阅用的定时器
    ros::ServiceServer turn_right_srv_;         ///< 右转服务
    ros::ServiceServer turn_left_srv_;          ///< 左转服务
    ros::ServiceServer turn_back_srv_;          ///< 掉头服务
    ros::ServiceServer go_straight_srv_;        ///< 直行服务
    
    // --- 同步与状态变量 ---
    boost::mutex odom_mutex_;                   ///< 用于保护共享数据（如位置、偏航角）的互斥锁
    
    geometry_msgs::Point current_position_;     ///< 从里程计获取的当前机器人位置
    geometry_msgs::PoseStamped target_point_;   ///< 基于指令计算出的目标点
    geometry_msgs::PoseStamped aruco_point_;    ///< 从Aruco检测到的目标点（最高优先级）
    geometry_msgs::PoseStamped yolo_point_;     ///< 从YOLO检测到的目标点（次高优先级）

    bool initial_yaw_set_;                      ///< 标志位：是否已获取并设置了初始偏航角
    bool aruco_received_;                       ///< 标志位：是否已接收到Aruco目标
    bool yolo_received_;                        ///< 标志位：是否已接收到YOLO目标
    bool reached_marker_;                       ///< 标志位：是否已到达视觉标记点
    bool final_command_active_;                 ///< 标志位：YOLO引导模式是否被激活
    
    double initial_yaw_;                        ///< 机器人启动时的初始绝对偏航角（map系下）
    double current_command_yaw_;                ///< 相对于初始偏航角的累计指令偏航角
    
    int service_call_count_;                    ///< 服务调用次数计数器，用于判断是否开始发布目标

    // --- 配置参数 ---
    double path_length_;                        ///< 生成指令目标点时，与当前位置的距离
    double point_spacing_;                      ///< (未使用) 路径点间距
    std::string path_frame_;                    ///< 发布路径和目标点的坐标系名称
};

/**
 * @brief 主函数
 * @param argc 参数数量
 * @param argv 参数列表
 * @return int 退出码
 */
int main(int argc, char** argv) {
    ros::init(argc, argv, "goalAdv");
    LocalPathPlanner planner;
    planner.run();
    return 0;
}