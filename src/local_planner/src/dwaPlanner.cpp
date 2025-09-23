#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/utils.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <nav_msgs/Path.h>
#include <boost/thread/mutex.hpp>
#include <std_msgs/Bool.h>

// 在文件顶部添加这些头文件
#include <tf2/LinearMath/Quaternion.h>
class DWAPlanner {
public:
    DWAPlanner() : tf_listener_(tf_buffer_) {
        ros::NodeHandle private_nh("~");
        private_nh.param("max_vel_x", max_vel_x_, 1.5);
        private_nh.param("min_vel_x", min_vel_x_, -1.1);
        private_nh.param("max_vel_y", max_vel_y_, 0.4);
        private_nh.param("min_vel_y", min_vel_y_, -0.4);
        private_nh.param("max_rot_vel", max_rot_vel_, 1.7);
        private_nh.param("min_rot_vel", min_rot_vel_, -1.7);
        
        private_nh.param("acc_lim_x", acc_lim_x_, 0.8);
        private_nh.param("acc_lim_y", acc_lim_y_, 0.8);
        private_nh.param("rot_acc_lim", rot_acc_lim_, 3.0);
        
        private_nh.param("sim_time", sim_time_, 2.0);
        normal_sim_time_ = sim_time_;  //通过调整sim_time_来恢复正常速度
        private_nh.param("step_size", step_size_, 0.08);
        private_nh.param("control_frequency", control_frequency_, 20.0);
        private_nh.param("goal_pos_tolerance", goal_pos_tolerance_, 0.15);
        private_nh.param("goal_rot_tolerance", goal_rot_tolerance_, 0.2);

        private_nh.param("goal_cost_weight", goal_cost_weight_, 60.0);
        private_nh.param("goal_orientation_weight", goal_orientation_weight_, 10.0);
        private_nh.param("velocity_cost_weight", velocity_cost_weight_, 8.0);
        private_nh.param("obstacle_cost_weight", obstacle_cost_weight_, 5.0);
        private_nh.param("heading_weight", heading_weight_, 18.0);
        private_nh.param("clearance_weight", clearance_weight_, 0.05);
        
        private_nh.param("robot_radius", robot_radius_, 0.6);
        private_nh.param("safety_margin", safety_margin_, 0.1);
        private_nh.param("base_frame", base_frame_, std::string("car/base_link"));
        private_nh.param("odom_topic", odom_topic_, std::string("/magv/odometry/gt"));
        private_nh.param("min_obstacle_dist", min_obstacle_dist_, 0.1);
        private_nh.param("lookahead_distance", lookahead_distance_, 1.0);

        costmap_sub_ = nh_.subscribe("/local_costmap_demo", 1, &DWAPlanner::costmapCallback, this);
        path_sub_ = nh_.subscribe("/local_plan", 1, &DWAPlanner::pathCallback, this);  // 修改为路径订阅
        odom_sub_ = nh_.subscribe(odom_topic_, 1, &DWAPlanner::odomCallback, this);
        cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/magv/omni_drive_controller/cmd_vel", 1);
        trajectory_pub_ = nh_.advertise<nav_msgs::Path>("/dwa_trajectory", 1);
        current_goal_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("current_goal", 1);  // 新增当前目标点发布
        stop_sub_ = nh_.subscribe("/robot_control/stop", 1, &DWAPlanner::stopCallback, this);
        accelerate_sub_ = nh_.subscribe("/robot_control/accelerate", 1, &DWAPlanner::accelerateCallback, this);

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

    void stopCallback(const std_msgs::Bool::ConstPtr& msg) {
        boost::mutex::scoped_lock lock(stop_mutex_); // 保护共享变量
        is_stopped_ = msg->data;
        if (is_stopped_) {
            ROS_WARN("DWA: Received STOP command from human_tracker.");
        } else {
            ROS_INFO("DWA: Received GO command. Resuming normal operation.");
        }
    }

    void costmapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
        boost::mutex::scoped_lock lock(costmap_mutex_);
        costmap_ = *msg;
        costmap_received_ = true;
        costmap_resolution_ = costmap_.info.resolution;
        costmap_origin_x_ = costmap_.info.origin.position.x;
        costmap_origin_y_ = costmap_.info.origin.position.y;
    }

    void accelerateCallback(const std_msgs::Bool::ConstPtr& msg) {
        boost::mutex::scoped_lock lock(sim_time_mutex_); // 保护sim_time_
        if (msg->data) { // 加速指令
            sim_time_ = 0.5;
            ROS_INFO("DWA: Received ACCELERATE command. Setting sim_time to %.2f", sim_time_);
        } else { // 恢复正常速度指令
            sim_time_ = normal_sim_time_;
            ROS_INFO("DWA: Received NORMAL speed command. Resetting sim_time to %.2f", sim_time_);
        }
    }

    void pathCallback(const nav_msgs::Path::ConstPtr& msg) {
        boost::mutex::scoped_lock lock(path_mutex_);
        if (msg->poses.empty()) {
            ROS_WARN("Received empty path!");
            return;
        }
        
        nav_msgs::Path filtered_path;
        filtered_path.header = msg->header;
        
        bool costmap_available = false;
        {
            boost::mutex::scoped_lock costmap_lock(costmap_mutex_);
            costmap_available = costmap_received_;
        }
        
        if (!costmap_available) {
            ROS_WARN("Costmap not available, storing unfiltered path.");
            global_path_ = *msg;
            path_received_ = true;
            path_index_ = 0;
            return;
        }
        
        // 遍历路径点进行障碍物检测
        // for (const auto& pose : msg->poses) {
        //     bool in_obstacle = false;
        //     {
        //         boost::mutex::scoped_lock costmap_lock(costmap_mutex_);
        //         in_obstacle = isPointInObstacle(pose.pose.position);
        //     }
        //     //将接近障碍物的点过滤掉,只保留安全的point
        //     if (!in_obstacle) {
        //         filtered_path.poses.push_back(pose);
        //     } else {
        //         ROS_DEBUG("Removed path point (%.2f, %.2f) due to obstacle", 
        //                  pose.pose.position.x, pose.pose.position.y);
        //     }
        // }
        
        // if (filtered_path.poses.empty()) {
        //     ROS_WARN("Filtered path is empty after obstacle removal! Using original path.");
        //     global_path_ = *msg;
        // } else {
        //     global_path_ = filtered_path;
        //     ROS_INFO("Path filtered: %lu points removed -> %lu points remain",
        //              msg->poses.size() - filtered_path.poses.size(),
        //              filtered_path.poses.size());
        // }
        global_path_ = *msg;
        
        path_received_ = true;
        path_index_ = 0;
        
        // boost::mutex::scoped_lock lock(path_mutex_);
        // if (msg->poses.empty()) {
        //     ROS_WARN("Received empty path!");
        //     return;
        // }
        
        // global_path_ = *msg;
        // path_received_ = true;
        // path_index_ = 0;  // 重置路径索引
        // ROS_INFO("New path received with %lu points", global_path_.poses.size());
    }

    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        boost::mutex::scoped_lock lock(odom_mutex_);
        current_vel_x_ = msg->twist.twist.linear.x;
        current_vel_y_ = msg->twist.twist.linear.y;
        current_rot_vel_ = msg->twist.twist.angular.z;
    }


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

    // void run() {
    //     ros::Rate rate(control_frequency_);
    //     while (ros::ok()) {
    //         ros::spinOnce();
            
    //         if (path_received_ && costmap_received_) {
    //             if (updateCurrentGoal()) {  // 更新当前目标点
    //                 geometry_msgs::Twist cmd_vel = computeVelocityCommands();
    //                 cmd_vel_pub_.publish(cmd_vel);
    //             } else {
    //                 publishZeroVelocity();
    //                 ROS_WARN("Path completed!");
    //             }
    //         }
    //         rate.sleep();
    //     }
    // }

private:
    struct Trajectory {
        std::vector<geometry_msgs::PoseStamped> path;
        double vx;
        double vy;
        double w;
        double score;
        double obstacle_dist;
        double goal_dist;
    };
    
    bool isPointInObstacle(const geometry_msgs::Point& point) {
        // 计算点在costmap中的坐标
        double mx = (point.x - costmap_origin_x_) / costmap_resolution_;
        double my = (point.y - costmap_origin_y_) / costmap_resolution_;
        
        // 只处理在代价地图范围内的点
        if (mx >= 0 && mx < costmap_.info.width && my >= 0 && my < costmap_.info.height) {
            int index = static_cast<int>(my) * costmap_.info.width + static_cast<int>(mx);
            int8_t cost = costmap_.data[index];

            // 障碍物判断标准：cost值大于50（可调整）
            return cost > 70;
        }
        
        // 代价地图范围外的点不视为障碍物
        return false;
    }

    void publishZeroVelocity() {
        geometry_msgs::Twist cmd_vel;
        cmd_vel.linear.x = 0;
        cmd_vel.linear.y = 0;
        cmd_vel.angular.z = 0;
        cmd_vel_pub_.publish(cmd_vel);
    }

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

        // 查找路径上最近点
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

        // 查找前视目标点
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

        // 如果已经接近终点，使用最后一个点
        if (goal_index >= global_path_.poses.size() - 1) {
            goal_index = global_path_.poses.size() - 1;
            
            // 检查是否到达最终目标
            double dx = global_path_.poses.back().pose.position.x - robot_x;
            double dy = global_path_.poses.back().pose.position.y - robot_y;
            if (sqrt(dx*dx + dy*dy) < goal_pos_tolerance_) {
                return false; // 路径完成
            }
        }

        current_goal_ = global_path_.poses[goal_index];
        current_goal_pub_.publish(current_goal_);  // 发布当前目标点
        return true;
    }

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

        for (double vx : vx_samples) {
            for (double vy : vy_samples) {
                for (double w : w_samples) {
                    Trajectory traj = generateTrajectory(transform, vx, vy, w);
                    if (!traj.path.empty()) {
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

        if (!valid_traj_found) {
            ROS_WARN("No valid trajectory found! Trying recovery behavior...");
            best_cmd.linear.x = 0;
            best_cmd.linear.y = 0;
            
            // 获取目标方向
            double goal_yaw = tf2::getYaw(current_goal_.pose.orientation);
            double current_yaw = tf2::getYaw(transform.transform.rotation);
            double angle_diff = atan2(sin(goal_yaw - current_yaw), cos(goal_yaw - current_yaw));
            best_cmd.angular.z = (angle_diff > 0) ? 0.07 : -0.07;
        } else {
            publishTrajectory(best_traj);
        }
        
        ROS_INFO("Best cmd_vel: vx=%.2f, vy=%.2f, w=%.2f", 
                 best_cmd.linear.x, best_cmd.linear.y, best_cmd.angular.z);
        
        return best_cmd;
    }

    std::vector<double> sampleVelocities(double min_vel, double max_vel, double acc_lim, double current_vel, double dt, int num_samples) {
        std::vector<double> samples;
        double min_sample = std::max(min_vel, current_vel - acc_lim * dt);
        double max_sample = std::min(max_vel, current_vel + acc_lim * dt);
        
        if (max_sample - min_sample < 0.01) {
            min_sample = min_vel;
            max_sample = max_vel;
        }
        
        double step = (max_sample - min_sample) / (num_samples - 1);
        for (int i = 0; i < num_samples; i++) {
            samples.push_back(min_sample + i * step);
        }
        
        if (min_sample <= 0.0 && max_sample >= 0.0 && 
            std::find(samples.begin(), samples.end(), 0.0) == samples.end()) {
            samples.push_back(0.0);
        }
        
        if (std::find(samples.begin(), samples.end(), current_vel) == samples.end()) {
            samples.push_back(current_vel);
        }
        
        samples.push_back(0.8 * max_vel);
        samples.push_back(0.5 * max_vel);
        samples.push_back(-0.5 * min_vel);
        
        std::sort(samples.begin(), samples.end());
        samples.erase(std::unique(samples.begin(), samples.end()), samples.end());
        
        return samples;
    }

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

        for (double time = 0; time <= current_sim_time; time += dt) {
            double vx_world = vx * cos(map_theta) - vy * sin(map_theta);
            double vy_world = vx * sin(map_theta) + vy * cos(map_theta);
            map_x += vx_world * dt;
            map_y += vy_world * dt;
            map_theta += w * dt;
            
            geometry_msgs::PoseStamped global_pose;
            global_pose.header.frame_id = "map";
            global_pose.header.stamp = start_time + ros::Duration(time);
            global_pose.pose.position.x = map_x;
            global_pose.pose.position.y = map_y;
            tf2::Quaternion q;
            q.setRPY(0, 0, map_theta);
            global_pose.pose.orientation = tf2::toMsg(q);

            double dist = getNearestObstacleDistance(global_pose.pose);
            if (dist < traj.obstacle_dist) {
                traj.obstacle_dist = dist;
            }
            
            if (dist < min_obstacle_dist_) {
                collision = true;
            }
            
            traj.path.push_back(global_pose);
        }
        
        if (!traj.path.empty()) {
            const geometry_msgs::Pose& end_pose = traj.path.back().pose;
            double dx = current_goal_.pose.position.x - end_pose.position.x;
            double dy = current_goal_.pose.position.y - end_pose.position.y;
            traj.goal_dist = sqrt(dx*dx + dy*dy);
        } else {
            traj.goal_dist = std::numeric_limits<double>::max();
        }
        
        if (collision && traj.obstacle_dist < min_obstacle_dist_ * 0.7) {
            traj.path.clear();
        }
        
        return traj;
    }
    
    void scoreTrajectory(Trajectory& traj) {
        if (traj.path.empty()) {
            traj.score = -std::numeric_limits<double>::infinity();
            return;
        }
        
        // 目标距离评分
        double goal_score = -goal_cost_weight_ * traj.goal_dist;
        
        // 航向评分
        const geometry_msgs::Pose& end_pose = traj.path.back().pose;
        double dx = current_goal_.pose.position.x - end_pose.position.x;
        double dy = current_goal_.pose.position.y - end_pose.position.y;
        double goal_dir = atan2(dy, dx);
        double traj_end_yaw = tf2::getYaw(end_pose.orientation);
        double heading_diff = std::abs(atan2(sin(goal_dir - traj_end_yaw), cos(goal_dir - traj_end_yaw)));
        double heading_score = -heading_weight_ * heading_diff;
        
        // 最终朝向评分
        double goal_yaw = tf2::getYaw(current_goal_.pose.orientation);
        double yaw_diff = std::abs(atan2(sin(goal_yaw - traj_end_yaw), cos(goal_yaw - traj_end_yaw)));
        double orientation_score = -goal_orientation_weight_ * yaw_diff;
        
        // 速度评分
        double vel_score = 0.0;
        if (traj.vx > 0) {
            vel_score = velocity_cost_weight_ * traj.vx;
        } else if (traj.vx < 0) {
            vel_score = 2 * velocity_cost_weight_ * traj.vx;
        }
        
        // 障碍物评分
        double obstacle_score = 0.0;
        if (traj.obstacle_dist < min_obstacle_dist_) {
            obstacle_score = -obstacle_cost_weight_ * 
                             (min_obstacle_dist_ - traj.obstacle_dist) / min_obstacle_dist_;
        } else {
            obstacle_score = clearance_weight_ * 
                             std::min(1.0, (traj.obstacle_dist - min_obstacle_dist_) / min_obstacle_dist_);
        }
        
        // 路径清除评分
        double clearance_score = clearance_weight_ * 
                                std::min(traj.obstacle_dist / (min_obstacle_dist_ * 2), 1.0);
        
        // 综合评分
        traj.score = goal_score + heading_score + orientation_score + 
                    vel_score + obstacle_score + clearance_score;

        if (traj.goal_dist < goal_pos_tolerance_) {
        traj.score += 50.0; // 终点额外奖励
}
        
        ROS_DEBUG("Scoring: tot=%.2f (g=%.2f, h=%.2f, o=%.2f, v=%.2f, obs=%.2f, clr=%.2f)", 
                 traj.score, goal_score, heading_score, orientation_score, 
                 vel_score, obstacle_score, clearance_score);
    }

    double getNearestObstacleDistance(const geometry_msgs::Pose& pose) {
        boost::mutex::scoped_lock lock(costmap_mutex_);
        if (!costmap_received_) return 0.0;
        
        double min_dist = std::numeric_limits<double>::max();
        double mx = (pose.position.x - costmap_origin_x_) / costmap_resolution_;
        double my = (pose.position.y - costmap_origin_y_) / costmap_resolution_;
        
        int search_radius = static_cast<int>((robot_radius_ + safety_margin_) / costmap_resolution_);
        search_radius = std::min(std::max(search_radius, 5), 30);
        
        for (int dx = -search_radius; dx <= search_radius; dx++) {
            for (int dy = -search_radius; dy <= search_radius; dy++) {
                int nx = static_cast<int>(mx) + dx;
                int ny = static_cast<int>(my) + dy;
                
                if (nx >= 0 && nx < costmap_.info.width && ny >= 0 && ny < costmap_.info.height) {
                    int index = ny * costmap_.info.width + nx;
                    if (costmap_.data[index] > 50) {
                        double dist = sqrt(dx*dx + dy*dy) * costmap_resolution_;
                        if (dist < min_dist) min_dist = dist;
                    }
                }
            }
        }
        return (min_dist == std::numeric_limits<double>::max()) ? 
               (search_radius * costmap_resolution_) : min_dist;
    }

    void publishTrajectory(const Trajectory& traj) {
        nav_msgs::Path path_msg;
        path_msg.header.frame_id = "map";
        path_msg.header.stamp = ros::Time::now();
        path_msg.poses = traj.path;
        trajectory_pub_.publish(path_msg);
    }

    ros::NodeHandle nh_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    ros::Subscriber costmap_sub_, path_sub_, odom_sub_; // 修改为path_sub_
    ros::Subscriber stop_sub_; // 新增停止订阅
    ros::Subscriber accelerate_sub_;
    ros::Publisher cmd_vel_pub_, trajectory_pub_, current_goal_pub_;
    
    boost::mutex costmap_mutex_, path_mutex_, odom_mutex_; // 添加path_mutex_
    boost::mutex stop_mutex_; // 新增停止互斥锁
    boost::mutex sim_time_mutex_;

    // 参数
    double max_vel_x_, min_vel_x_, max_vel_y_, min_vel_y_, max_rot_vel_, min_rot_vel_;
    double acc_lim_x_, acc_lim_y_, rot_acc_lim_;
    double sim_time_, step_size_, control_frequency_,normal_sim_time_;
    double goal_pos_tolerance_, goal_rot_tolerance_;
    double obstacle_cost_weight_, goal_cost_weight_, velocity_cost_weight_;
    double goal_orientation_weight_, heading_weight_, clearance_weight_;
    double robot_radius_, safety_margin_, min_obstacle_dist_;
    double lookahead_distance_; // 新增前视距离参数
    std::string base_frame_, odom_topic_;
    bool is_stopped_;

    nav_msgs::OccupancyGrid costmap_;
    nav_msgs::Path global_path_; // 存储全局路径
    geometry_msgs::PoseStamped current_goal_; // 当前跟踪的目标点
    bool path_received_, costmap_received_; // 修改为path_received_
    size_t path_index_; // 当前路径索引
    double current_vel_x_, current_vel_y_, current_rot_vel_;
    double costmap_resolution_, costmap_origin_x_, costmap_origin_y_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "dwaPlanner");
    DWAPlanner planner;
    planner.run();
    return 0;
}



