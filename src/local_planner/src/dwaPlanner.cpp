/**
 * @file dwaPlanner.cpp
 * @author BITLZX
 * @date 2025-09-23
 * @brief 该文件实现了一个基于动态窗口方法(DWA)的局部路径规划器ROS节点。
 *
 * @details
 * 此节点订阅一个全局路径(`/local_plan`)、一个局部代价地图(`/local_costmap_demo`)和里程计信息。
 * 它通过在速度空间中采样，模拟多条候选轨迹，并根据一个综合评分函数（考虑目标距离、
 * 路径朝向、与障碍物的距离、速度等）来评估这些轨迹。最终，它会选择得分最高的轨迹所对应的
 * 速度指令，并将其发布到 `/magv/omni_drive_controller/cmd_vel` 话题，以引导机器人
 * 安全、高效地沿着全局路径移动。
 */

#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2_ros/transform_listener.hh>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/utils.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <nav_msgs/Path.h>
#include <boost/thread/mutex.hpp>
#include <std_msgs/Bool.h>
#include <tf2/LinearMath/Quaternion.h>

/**
 * @class DWAPlanner
 * @brief 实现动态窗口方法(DWA)的核心逻辑类。
 */
class DWAPlanner {
public:
    /**
     * @brief DWAPlanner的构造函数。
     * @details 初始化ROS节点，从参数服务器加载所有DWA相关的参数，并设置发布者和订阅者。
     */
    DWAPlanner() : tf_listener_(tf_buffer_) {
        ros::NodeHandle private_nh("~");
        
        // --- 从参数服务器加载参数 ---
        // 机器人运动学和动力学约束
        private_nh.param("max_vel_x", max_vel_x_, 1.5);
        private_nh.param("min_vel_x", min_vel_x_, -1.1);
        private_nh.param("max_vel_y", max_vel_y_, 0.4);
        private_nh.param("min_vel_y", min_vel_y_, -0.4);
        private_nh.param("max_rot_vel", max_rot_vel_, 1.7);
        private_nh.param("min_rot_vel", min_rot_vel_, -1.7);
        private_nh.param("acc_lim_x", acc_lim_x_, 0.8);
        private_nh.param("acc_lim_y", acc_lim_y_, 0.8);
        private_nh.param("rot_acc_lim", rot_acc_lim_, 3.0);
        
        // 轨迹模拟参数
        private_nh.param("sim_time", sim_time_, 2.0);
        normal_sim_time_ = sim_time_; // 存储正常的模拟时间，用于恢复
        private_nh.param("step_size", step_size_, 0.08);
        private_nh.param("control_frequency", control_frequency_, 20.0);
        
        // 目标容忍度
        private_nh.param("goal_pos_tolerance", goal_pos_tolerance_, 0.15);
        private_nh.param("goal_rot_tolerance", goal_rot_tolerance_, 0.2);

        // DWA评分函数权重
        private_nh.param("goal_cost_weight", goal_cost_weight_, 60.0);
        private_nh.param("goal_orientation_weight", goal_orientation_weight_, 10.0);
        private_nh.param("velocity_cost_weight", velocity_cost_weight_, 8.0);
        private_nh.param("obstacle_cost_weight", obstacle_cost_weight_, 5.0);
        private_nh.param("heading_weight", heading_weight_, 18.0);
        private_nh.param("clearance_weight", clearance_weight_, 0.05);
        
        // 机器人和路径跟随参数
        private_nh.param("robot_radius", robot_radius_, 0.6);
        private_nh.param("safety_margin", safety_margin_, 0.1);
        private_nh.param("base_frame", base_frame_, std::string("car/base_link"));
        private_nh.param("odom_topic", odom_topic_, std::string("/magv/odometry/gt"));
        private_nh.param("min_obstacle_dist", min_obstacle_dist_, 0.1);
        private_nh.param("lookahead_distance", lookahead_distance_, 1.0);

        // --- 初始化ROS通信接口 ---
        costmap_sub_ = nh_.subscribe("/local_costmap_demo", 1, &DWAPlanner::costmapCallback, this);
        path_sub_ = nh_.subscribe("/local_plan", 1, &DWAPlanner::pathCallback, this);
        odom_sub_ = nh_.subscribe(odom_topic_, 1, &DWAPlanner::odomCallback, this);
        stop_sub_ = nh_.subscribe("/robot_control/stop", 1, &DWAPlanner::stopCallback, this);
        accelerate_sub_ = nh_.subscribe("/robot_control/accelerate", 1, &DWAPlanner::accelerateCallback, this);
        
        cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/magv/omni_drive_controller/cmd_vel", 1);
        trajectory_pub_ = nh_.advertise<nav_msgs::Path>("/dwa_trajectory", 1);
        current_goal_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("current_goal", 1);

        // --- 初始化状态变量 ---
        current_goal_.header.frame_id = "map";
        current_goal_.pose.orientation.w = 1.0;
        path_received_ = false;
        costmap_received_ = false;
        is_stopped_ = false;
        current_vel_x_ = 0.0;
        current_vel_y_ = 0.0;
        current_rot_vel_ = 0.0;
        
        ROS_INFO("Enhanced DWA Path Follower initialized");
    }

    /**
     * @brief 启动DWA规划器的主循环。
     */
    void run() {
        ros::Rate rate(control_frequency_);
        while (ros::ok()) {
            ros::spinOnce();
            
            bool should_stop;
            {
                boost::mutex::scoped_lock lock(stop_mutex_);
                should_stop = is_stopped_;
            }

            if (should_stop) {
                // 如果收到停止指令，则发布零速并跳过所有规划
                publishZeroVelocity();
                ROS_WARN_THROTTLE(1.0, "DWA: Planning paused due to external stop command.");
            } else {
                // 否则，执行正常的DWA规划逻辑
                if (path_received_ && costmap_received_) {
                    if (updateCurrentGoal()) {
                        geometry_msgs::Twist cmd_vel = computeVelocityCommands();
                        cmd_vel_pub_.publish(cmd_vel);
                    } else {
                        publishZeroVelocity();
                        ROS_INFO_THROTTLE(1.0, "DWA: Path completed or invalid.");
                    }
                }
            }
            
            rate.sleep();
        }
    }

private:
    /**
     * @struct Trajectory
     * @brief 用于存储和评估一条模拟轨迹的数据结构。
     */
    struct Trajectory {
        std::vector<geometry_msgs::PoseStamped> path; ///< 轨迹上的路径点序列
        double vx;                                  ///< 产生此轨迹的x方向线速度
        double vy;                                  ///< 产生此轨迹的y方向线速度
        double w;                                   ///< 产生此轨迹的角速度
        double score;                               ///< 轨迹的综合评分
        double obstacle_dist;                       ///< 轨迹距离最近障碍物的距离
        double goal_dist;                           ///< 轨迹终点与当前目标的距离
    };
    
    // --- ROS回调函数 ---

    /**
     * @brief 接收停止指令的回调函数。
     * @param msg Bool消息，true表示停止，false表示恢复。
     */
    void stopCallback(const std_msgs::Bool::ConstPtr& msg) {
        boost::mutex::scoped_lock lock(stop_mutex_);
        is_stopped_ = msg->data;
        if (is_stopped_) {
            ROS_WARN("DWA: Received STOP command from human_tracker.");
        } else {
            ROS_INFO("DWA: Received GO command. Resuming normal operation.");
        }
    }

    /**
     * @brief 接收局部代价地图的回调函数。
     * @param msg 代价地图消息。
     */
    void costmapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
        boost::mutex::scoped_lock lock(costmap_mutex_);
        costmap_ = *msg;
        costmap_received_ = true;
        costmap_resolution_ = costmap_.info.resolution;
        costmap_origin_x_ = costmap_.info.origin.position.x;
        costmap_origin_y_ = costmap_.info.origin.position.y;
    }

    /**
     * @brief 接收加速/减速指令的回调函数。
     * @details 通过修改`sim_time_`参数来调整机器人的前瞻性，从而影响其速度。
     * @param msg Bool消息，true表示加速，false表示恢复正常速度。
     */
    void accelerateCallback(const std_msgs::Bool::ConstPtr& msg) {
        boost::mutex::scoped_lock lock(sim_time_mutex_);
        if (msg->data) { // 加速指令
            sim_time_ = 0.5;
            ROS_INFO("DWA: Received ACCELERATE command. Setting sim_time to %.2f", sim_time_);
        } else { // 恢复正常速度指令
            sim_time_ = normal_sim_time_;
            ROS_INFO("DWA: Received NORMAL speed command. Resetting sim_time to %.2f", sim_time_);
        }
    }

    /**
     * @brief 接收全局路径的回调函数。
     * @param msg 路径消息。
     */
    void pathCallback(const nav_msgs::Path::ConstPtr& msg) {
        boost::mutex::scoped_lock lock(path_mutex_);
        if (msg->poses.empty()) {
            ROS_WARN("Received empty path!");
            return;
        }
        
        // (可选的路径点过滤逻辑已被注释掉，当前直接使用原始路径)
        global_path_ = *msg;
        
        path_received_ = true;
        path_index_ = 0; // 每次收到新路径时，重置路径索引
    }

    /**
     * @brief 接收里程计信息的回调函数。
     * @param msg 里程计消息，主要用于获取当前速度。
     */
    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        boost::mutex::scoped_lock lock(odom_mutex_);
        current_vel_x_ = msg->twist.twist.linear.x;
        current_vel_y_ = msg->twist.twist.linear.y;
        current_rot_vel_ = msg->twist.twist.angular.z;
    }

    // --- 核心DWA逻辑 ---

    /**
     * @brief 更新当前DWA要跟随的目标点。
     * @details 采用"前视距离(lookahead)"策略。首先在路径上找到离机器人最近的点，
     * 然后从该点开始向前搜索，找到第一个超出`lookahead_distance_`的点作为当前目标。
     * @return bool 如果成功找到有效目标点则返回true，如果路径完成或无效则返回false。
     */
    bool updateCurrentGoal() {
        boost::mutex::scoped_lock lock(path_mutex_);
        if (global_path_.poses.empty()) return false;

        geometry_msgs::TransformStamped transform;
        try {
            transform = tf_buffer_.lookupTransform("map", base_frame_, ros::Time(0), ros::Duration(0.1));
        } catch (tf2::TransformException &ex) {
            ROS_WARN_DELAYED_THROTTLE(1.0, "TF lookup failed: %s", ex.what());
            return false;
        }

        double robot_x = transform.transform.translation.x;
        double robot_y = transform.transform.translation.y;

        // 查找路径上离机器人最近的点，作为搜索前视点的起点
        size_t closest_index = 0;
        double min_dist = std::numeric_limits<double>::max();
        for (size_t i = path_index_; i < global_path_.poses.size(); ++i) {
            double dx = global_path_.poses[i].pose.position.x - robot_x;
            double dy = global_path_.poses[i].pose.position.y - robot_y;
            double dist = dx*dx + dy*dy;
            if (dist < min_dist) {
                min_dist = dist;
                closest_index = i;
            }
        }
        path_index_ = closest_index;

        // 从最近点开始，查找第一个满足前视距离的点
        double lookahead_sq = lookahead_distance_ * lookahead_distance_;
        size_t goal_index = path_index_;
        for (size_t i = path_index_; i < global_path_.poses.size(); ++i) {
            double dx = global_path_.poses[i].pose.position.x - robot_x;
            double dy = global_path_.poses[i].pose.position.y - robot_y;
            double dist_sq = dx*dx + dy*dy;
            if (dist_sq >= lookahead_sq) {
                goal_index = i;
                break;
            }
        }

        // 如果搜索完整个路径都未找到满足前视距离的点（说明已接近终点），则直接使用路径的最后一个点
        if (goal_index >= global_path_.poses.size() - 1) {
            goal_index = global_path_.poses.size() - 1;
            
            // 检查是否已到达最终目标
            double dx = global_path_.poses.back().pose.position.x - robot_x;
            double dy = global_path_.poses.back().pose.position.y - robot_y;
            if (sqrt(dx*dx + dy*dy) < goal_pos_tolerance_) {
                return false; // 路径完成
            }
        }

        current_goal_ = global_path_.poses[goal_index];
        current_goal_pub_.publish(current_goal_); // 发布当前目标点以供调试
        return true;
    }

    /**
     * @brief 计算最佳速度指令的核心函数。
     * @details 这是DWA算法的实现。它按顺序执行：
     * 1. 在动态窗口内对(vx, vy, w)进行采样。
     * 2. 对每个速度样本，生成一条模拟轨迹。
     * 3. 对每条有效的轨迹进行评分。
     * 4. 选择得分最高的轨迹，返回其对应的速度指令。
     * @return geometry_msgs::Twist 计算出的最佳速度指令。
     */
    geometry_msgs::Twist computeVelocityCommands() {
        geometry_msgs::Twist best_cmd;
        best_cmd.linear.x = 0;
        best_cmd.linear.y = 0;
        best_cmd.angular.z = 0;

        geometry_msgs::TransformStamped transform;
        try {
            transform = tf_buffer_.lookupTransform("map", base_frame_, ros::Time(0), ros::Duration(0.1));
        } catch (tf2::TransformException &ex) {
            ROS_WARN_DELAYED_THROTTLE(1.0, "TF lookup failed: %s", ex.what());
            return best_cmd;
        }

        double dt = 1.0 / control_frequency_;
        std::vector<double> vx_samples, vy_samples, w_samples;
        {
            boost::mutex::scoped_lock lock(odom_mutex_);
            vx_samples = sampleVelocities(min_vel_x_, max_vel_x_, acc_lim_x_, current_vel_x_, dt, 15);
            vy_samples = sampleVelocities(min_vel_y_, max_vel_y_, acc_lim_y_, current_vel_y_, dt, 15);
            w_samples = sampleVelocities(min_rot_vel_, max_rot_vel_, rot_acc_lim_, current_rot_vel_, dt, 15);
        }

        double best_score = -std::numeric_limits<double>::infinity();
        Trajectory best_traj;
        bool valid_traj_found = false;

        // 遍历所有速度样本组合
        for (double vx : vx_samples) {
            for (double vy : vy_samples) {
                for (double w : w_samples) {
                    Trajectory traj = generateTrajectory(transform, vx, vy, w);
                    if (!traj.path.empty()) { // 仅对有效轨迹评分
                        scoreTrajectory(traj);
                        
                        if (traj.score > best_score) {
                            best_score = traj.score;
                            best_cmd.linear.x = vx;
                            best_cmd.linear.y = vy;
                            best_cmd.angular.z = w;
                            best_traj = traj;
                            valid_traj_found = true;
                        }
                    }
                }
            }
        }

        // 如果没有找到任何有效轨迹，执行简单的恢复行为（原地旋转朝向目标）
        if (!valid_traj_found) {
            ROS_WARN("No valid trajectory found! Trying recovery behavior...");
            best_cmd.linear.x = 0;
            best_cmd.linear.y = 0;
            
            double goal_yaw = tf2::getYaw(current_goal_.pose.orientation);
            double current_yaw = tf2::getYaw(transform.transform.rotation);
            double angle_diff = atan2(sin(goal_yaw - current_yaw), cos(goal_yaw - current_yaw));
            best_cmd.angular.z = (angle_diff > 0) ? 0.07 : -0.07;
        } else {
            publishTrajectory(best_traj); // 发布最佳轨迹以供可视化
        }
        
        ROS_INFO("Best cmd_vel: vx=%.2f, vy=%.2f, w=%.2f", 
                 best_cmd.linear.x, best_cmd.linear.y, best_cmd.angular.z);
        
        return best_cmd;
    }

    /**
     * @brief 根据动力学约束在速度空间中进行采样。
     * @param min_vel 最小速度。
     * @param max_vel 最大速度。
     * @param acc_lim 加速度限制。
     * @param current_vel 当前速度。
     * @param dt 时间间隔。
     * @param num_samples 采样数量。
     * @return std::vector<double> 采样得到的速度集合。
     */
    std::vector<double> sampleVelocities(double min_vel, double max_vel, double acc_lim, double current_vel, double dt, int num_samples) {
        std::vector<double> samples;
        // 计算动态窗口的边界
        double min_sample = std::max(min_vel, current_vel - acc_lim * dt);
        double max_sample = std::min(max_vel, current_vel + acc_lim * dt);
        
        if (max_sample - min_sample < 0.01) {
            min_sample = min_vel;
            max_sample = max_vel;
        }
        
        // 在动态窗口内均匀采样
        double step = (max_sample - min_sample) / (num_samples - 1);
        for (int i = 0; i < num_samples; i++) {
            samples.push_back(min_sample + i * step);
        }
        
        // 确保包含一些关键速度，如0和当前速度
        if (min_sample <= 0.0 && max_sample >= 0.0 && std::find(samples.begin(), samples.end(), 0.0) == samples.end()) {
            samples.push_back(0.0);
        }
        if (std::find(samples.begin(), samples.end(), current_vel) == samples.end()) {
            samples.push_back(current_vel);
        }
        
        // 添加一些额外的启发式采样点
        samples.push_back(0.8 * max_vel);
        samples.push_back(0.5 * max_vel);
        samples.push_back(-0.5 * min_vel);
        
        // 去重和排序
        std::sort(samples.begin(), samples.end());
        samples.erase(std::unique(samples.begin(), samples.end()), samples.end());
        
        return samples;
    }

    /**
     * @brief 根据给定的速度指令，生成一条模拟轨迹。
     * @param robot_tf 机器人当前在map坐标系下的位姿变换。
     * @param vx x方向线速度。
     * @param vy y方向线速度。
     * @param w 角速度。
     * @return Trajectory 生成的轨迹。如果轨迹与障碍物碰撞，则返回的轨迹path为空。
     */
    Trajectory generateTrajectory(const geometry_msgs::TransformStamped& robot_tf, double vx, double vy, double w) {
        Trajectory traj;
        traj.vx = vx;
        traj.vy = vy;
        traj.w = w;
        traj.obstacle_dist = std::numeric_limits<double>::max();
        
        double map_x = robot_tf.transform.translation.x;
        double map_y = robot_tf.transform.translation.y;
        double map_theta = tf2::getYaw(robot_tf.transform.rotation);

        double current_sim_time;
        {
            boost::mutex::scoped_lock lock(sim_time_mutex_);
            current_sim_time = sim_time_;
        }

        const double dt = step_size_;
        ros::Time start_time = ros::Time::now();
        bool collision = false;

        // 在sim_time内进行前向模拟
        for (double time = 0; time <= current_sim_time; time += dt) {
            // 将机器人本体速度转换到世界坐标系
            double vx_world = vx * cos(map_theta) - vy * sin(map_theta);
            double vy_world = vx * sin(map_theta) + vy * cos(map_theta);
            map_x += vx_world * dt;
            map_y += vy_world * dt;
            map_theta += w * dt;
            
            // 创建路径点
            geometry_msgs::PoseStamped global_pose;
            global_pose.header.frame_id = "map";
            global_pose.header.stamp = start_time + ros::Duration(time);
            global_pose.pose.position.x = map_x;
            global_pose.pose.position.y = map_y;
            tf2::Quaternion q;
            q.setRPY(0, 0, map_theta);
            global_pose.pose.orientation = tf2::toMsg(q);

            // 检查障碍物距离
            double dist = getNearestObstacleDistance(global_pose.pose);
            if (dist < traj.obstacle_dist) {
                traj.obstacle_dist = dist;
            }
            if (dist < min_obstacle_dist_) {
                collision = true;
            }
            
            traj.path.push_back(global_pose);
        }
        
        // 计算轨迹终点与目标的距离
        if (!traj.path.empty()) {
            const geometry_msgs::Pose& end_pose = traj.path.back().pose;
            double dx = current_goal_.pose.position.x - end_pose.position.x;
            double dy = current_goal_.pose.position.y - end_pose.position.y;
            traj.goal_dist = sqrt(dx*dx + dy*dy);
        } else {
            traj.goal_dist = std::numeric_limits<double>::max();
        }
        
        // 如果发生严重碰撞，则认为此轨迹无效
        if (collision && traj.obstacle_dist < min_obstacle_dist_ * 0.7) {
            traj.path.clear();
        }
        
        return traj;
    }
    
    /**
     * @brief 对给定的轨迹进行评分。
     * @details 这是DWA的目标函数。它将多个评价标准通过加权求和的方式组合成一个综合分数。
     * - **目标距离评分**: 轨迹终点离目标越近，得分越高。
     * - **航向评分**: 轨迹终点的朝向与“轨迹终点->目标点”的方向越一致，得分越高。
     * - **最终朝向评分**: 轨迹终点的朝向与目标点的最终朝向越一致，得分越高。
     * - **速度评分**: 前进速度越大，得分越高；后退则扣分。
     * - **障碍物评分**: 离障碍物越远，得分越高。如果太近则严重扣分。
     * - **路径清除度评分**: 鼓励轨迹在更开阔的区域。
     * @param traj 需要评分的轨迹（引用传递，评分结果会写回其中）。
     */
    void scoreTrajectory(Trajectory& traj) {
        if (traj.path.empty()) {
            traj.score = -std::numeric_limits<double>::infinity();
            return;
        }
        
        const geometry_msgs::Pose& end_pose = traj.path.back().pose;
        double traj_end_yaw = tf2::getYaw(end_pose.orientation);

        // 目标距离评分: 负相关，距离越小分数越大
        double goal_score = -goal_cost_weight_ * traj.goal_dist;
        
        // 航向评分: 评估轨迹终点朝向与目标方向的偏差
        double dx = current_goal_.pose.position.x - end_pose.position.x;
        double dy = current_goal_.pose.position.y - end_pose.position.y;
        double goal_dir = atan2(dy, dx);
        double heading_diff = std::abs(atan2(sin(goal_dir - traj_end_yaw), cos(goal_dir - traj_end_yaw)));
        double heading_score = -heading_weight_ * heading_diff;
        
        // 最终朝向评分: 评估轨迹终点姿态与目标姿态的偏差
        double goal_yaw = tf2::getYaw(current_goal_.pose.orientation);
        double yaw_diff = std::abs(atan2(sin(goal_yaw - traj_end_yaw), cos(goal_yaw - traj_end_yaw)));
        double orientation_score = -goal_orientation_weight_ * yaw_diff;
        
        // 速度评分: 鼓励前进
        double vel_score = 0.0;
        if (traj.vx > 0) {
            vel_score = velocity_cost_weight_ * traj.vx;
        } else if (traj.vx < 0) { // 对后退施加更大惩罚
            vel_score = 2 * velocity_cost_weight_ * traj.vx;
        }
        
        // 障碍物评分: 离障碍物越远越好
        double obstacle_score = 0.0;
        if (traj.obstacle_dist < min_obstacle_dist_) {
            obstacle_score = -obstacle_cost_weight_ * (min_obstacle_dist_ - traj.obstacle_dist) / min_obstacle_dist_;
        } else {
            obstacle_score = clearance_weight_ * std::min(1.0, (traj.obstacle_dist - min_obstacle_dist_) / min_obstacle_dist_);
        }
        
        // 路径清除度评分: 鼓励在开阔区域行驶
        double clearance_score = clearance_weight_ * std::min(traj.obstacle_dist / (min_obstacle_dist_ * 2), 1.0);
        
        // 综合评分
        traj.score = goal_score + heading_score + orientation_score + 
                    vel_score + obstacle_score + clearance_score;
        
        // 如果轨迹非常接近目标点，给予额外奖励，使其更容易被选中
        if (traj.goal_dist < goal_pos_tolerance_) {
            traj.score += 50.0;
        }
        
        ROS_DEBUG("Scoring: tot=%.2f (g=%.2f, h=%.2f, o=%.2f, v=%.2f, obs=%.2f, clr=%.2f)", 
                 traj.score, goal_score, heading_score, orientation_score, 
                 vel_score, obstacle_score, clearance_score);
    }

    /**
     * @brief 获取给定姿态距离最近障碍物的距离。
     * @param pose 需要检查的姿态。
     * @return double 到最近障碍物的距离（米）。如果没有障碍物，则返回一个较大的值。
     */
    double getNearestObstacleDistance(const geometry_msgs::Pose& pose) {
        boost::mutex::scoped_lock lock(costmap_mutex_);
        if (!costmap_received_) return 0.0;
        
        double min_dist = std::numeric_limits<double>::max();
        double mx = (pose.position.x - costmap_origin_x_) / costmap_resolution_;
        double my = (pose.position.y - costmap_origin_y_) / costmap_resolution_;
        
        // 定义一个搜索窗口
        int search_radius = static_cast<int>((robot_radius_ + safety_margin_) / costmap_resolution_);
        search_radius = std::min(std::max(search_radius, 5), 30);
        
        // 在窗口内搜索障碍物点
        for (int dx = -search_radius; dx <= search_radius; dx++) {
            for (int dy = -search_radius; dy <= search_radius; dy++) {
                int nx = static_cast<int>(mx) + dx;
                int ny = static_cast<int>(my) + dy;
                
                if (nx >= 0 && nx < costmap_.info.width && ny >= 0 && ny < costmap_.info.height) {
                    int index = ny * costmap_.info.width + nx;
                    if (costmap_.data[index] > 50) { // 阈值50表示障碍物
                        double dist = sqrt(dx*dx + dy*dy) * costmap_resolution_;
                        if (dist < min_dist) min_dist = dist;
                    }
                }
            }
        }
        // 如果在搜索半径内未找到障碍物，则返回一个安全的大距离值
        return (min_dist == std::numeric_limits<double>::max()) ? 
               (search_radius * costmap_resolution_) : min_dist;
    }

    // --- 辅助函数 ---

    /** @brief 发布零速指令。*/
    void publishZeroVelocity() {
        geometry_msgs::Twist cmd_vel;
        cmd_vel.linear.x = 0;
        cmd_vel.linear.y = 0;
        cmd_vel.angular.z = 0;
        cmd_vel_pub_.publish(cmd_vel);
    }
    
    /** @brief 检查一个点是否在障碍物内。 (当前未使用) */
    bool isPointInObstacle(const geometry_msgs::Point& point) {
        double mx = (point.x - costmap_origin_x_) / costmap_resolution_;
        double my = (point.y - costmap_origin_y_) / costmap_resolution_;
        
        if (mx >= 0 && mx < costmap_.info.width && my >= 0 && my < costmap_.info.height) {
            int index = static_cast<int>(my) * costmap_.info.width + static_cast<int>(mx);
            return costmap_.data[index] > 70;
        }
        return false;
    }

    /** @brief 发布最佳轨迹以供Rviz可视化。*/
    void publishTrajectory(const Trajectory& traj) {
        nav_msgs::Path path_msg;
        path_msg.header.frame_id = "map";
        path_msg.header.stamp = ros::Time::now();
        path_msg.poses = traj.path;
        trajectory_pub_.publish(path_msg);
    }

    // --- ROS和TF对象 ---
    ros::NodeHandle nh_;                                  ///< ROS节点句柄
    tf2_ros::Buffer tf_buffer_;                           ///< TF缓冲区
    tf2_ros::TransformListener tf_listener_;              ///< TF监听器
    ros::Subscriber costmap_sub_, path_sub_, odom_sub_;   ///< 订阅者：代价地图、路径、里程计
    ros::Subscriber stop_sub_;                            ///< 订阅者：外部停止信号
    ros::Subscriber accelerate_sub_;                      ///< 订阅者：外部加速信号
    ros::Publisher cmd_vel_pub_, trajectory_pub_, current_goal_pub_; ///< 发布者：速度指令、可视化轨迹、当前目标点
    
    // --- 互斥锁 ---
    boost::mutex costmap_mutex_, path_mutex_, odom_mutex_; ///< 用于保护代价地图、路径和里程计数据的互斥锁
    boost::mutex stop_mutex_;                             ///< 用于保护停止标志位的互斥锁
    boost::mutex sim_time_mutex_;                         ///< 用于保护模拟时间参数的互斥锁

    // --- DWA参数 ---
    double max_vel_x_, min_vel_x_, max_vel_y_, min_vel_y_, max_rot_vel_, min_rot_vel_; ///< 速度限制
    double acc_lim_x_, acc_lim_y_, rot_acc_lim_;          ///< 加速度限制
    double sim_time_, step_size_, control_frequency_, normal_sim_time_; ///< 模拟和控制参数
    double goal_pos_tolerance_, goal_rot_tolerance_;     ///< 目标容忍度
    double obstacle_cost_weight_, goal_cost_weight_, velocity_cost_weight_; ///< 评分权重
    double goal_orientation_weight_, heading_weight_, clearance_weight_;   ///< 评分权重
    double robot_radius_, safety_margin_, min_obstacle_dist_; ///< 机器人几何与安全参数
    double lookahead_distance_;                          ///< 路径跟随的前视距离
    std::string base_frame_, odom_topic_;                 ///< TF坐标系和话题名称
    
    // --- 状态变量 ---
    bool is_stopped_;                                     ///< 标志位：是否被外部指令停止
    nav_msgs::OccupancyGrid costmap_;                     ///< 存储的局部代价地图
    nav_msgs::Path global_path_;                          ///< 存储的全局路径
    geometry_msgs::PoseStamped current_goal_;             ///< DWA当前跟随的目标点
    bool path_received_, costmap_received_;               ///< 标志位：是否已收到路径和代价地图
    size_t path_index_;                                   ///< 当前在全局路径上的大致索引
    double current_vel_x_, current_vel_y_, current_rot_vel_; ///< 从里程计获取的当前速度
    double costmap_resolution_, costmap_origin_x_, costmap_origin_y_; ///< 代价地图元数据
};

/**
 * @brief 主函数
 * @param argc 参数数量
 * @param argv 参数列表
 * @return int 退出码
 */
int main(int argc, char** argv) {
    ros::init(argc, argv, "dwaPlanner");
    DWAPlanner planner;
    planner.run();
    return 0;
}