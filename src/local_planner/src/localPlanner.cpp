/**
 * @file localPlanner.cpp
 * @author BITLZX
 * @date 2025-09-23
 * @brief 实现基于A*算法的局部路径规划器。
 *
 * @details
 * 此节点的核心功能是在一个滚动的局部代价地图上，规划出一条从机器人当前位置到局部目标点的无碰撞路径。
 * 工作流程如下：
 * 1. **订阅全局目标**: 从`/move_base_simple/goal`等话题接收最终的目标点。
 * 2. **订阅局部代价地图**: 从`/local_costmap_demo`获取机器人周围的障碍物信息。
 * 3. **确定局部目标**: 如果全局目标在代价地图内，则直接使用；如果全局目标在代价地图外，则计算机器人与全局目标连线在地图边界上的交点作为局部目标。同时，会检查并确保局部目标点是安全的（非障碍物）。
 * 4. **A*路径规划**: 使用A*算法在代价地图上搜索从机器人当前位置到局部目标点的最短路径。
 * 5. **发布局部路径**: 将规划出的路径以`nav_msgs/Path`的形式发布到`/local_plan`话题，供DWA等下游路径跟随节点使用。
 */

#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <cmath>
#include <vector>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <boost/thread/mutex.hpp>
#include <std_msgs/Int32.h>
#include <magv_msgs/PositionCommand.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Transform.h>
#include <std_msgs/Bool.h>

/**
 * @struct AStarNode
 * @brief A*算法中用于搜索的节点结构体。
 */
struct AStarNode {
    int x, y;          ///< 节点在栅格地图中的x, y坐标
    double g, h, f;    ///< 代价: g=起点到此节点的实际代价, h=此节点到终点的启发式代价, f=g+h
    AStarNode* parent; ///< 指向父节点的指针，用于路径回溯

    /**
     * @brief AStarNode的构造函数。
     */
    AStarNode(int x, int y, double g, double h, AStarNode* parent = nullptr)
        : x(x), y(y), g(g), h(h), f(g + h), parent(parent) {}

    /**
     * @brief 优先队列的比较运算符重载。
     * @details 用于构建一个f值最小的最小堆。
     */
    bool operator>(const AStarNode& other) const {
        return f > other.f;
    }
};

/**
 * @struct NodeHasher
 * @brief 用于在`std::unordered_map`中对栅格坐标对(std::pair<int, int>)进行哈希计算。
 * @details 这使得我们可以高效地在`all_nodes` (功能类似close_list)中查找节点。
 */
struct NodeHasher {
    std::size_t operator()(const std::pair<int, int>& p) const {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

/**
 * @class AStarLocalPlanner
 * @brief 一个使用A*算法进行局部路径规划的ROS节点类。
 */
class AStarLocalPlanner {
public:
    /**
     * @brief AStarLocalPlanner的构造函数。
     * @details 初始化ROS句柄，加载参数，并设置订阅者、发布者和定时器。
     */
    AStarLocalPlanner() : tf_listener_(tf_buffer_) {
        ros::NodeHandle private_nh("~");
        
        // --- 从参数服务器加载参数 ---
        private_nh.param<std::string>("global_goal_topic", global_goal_topic_, "/move_base_simple/goal");
        private_nh.param<std::string>("costmap_topic", costmap_topic_, "/local_costmap_demo");
        private_nh.param<std::string>("local_plan_topic", local_plan_topic_, "/local_plan");
        private_nh.param<std::string>("robot_base_frame", robot_base_frame_, "car/base_link");
        private_nh.param<std::string>("global_frame", global_frame_, "map");
        private_nh.param<double>("planning_frequency", planning_frequency_, 5.0);
        private_nh.param<double>("goal_tolerance", goal_tolerance_, 1.5);

        // --- 初始化订阅者和发布者 ---
        goal_sub_ = nh_.subscribe(global_goal_topic_, 1, &AStarLocalPlanner::goalCallback, this);
        costmap_sub_ = nh_.subscribe(costmap_topic_, 1, &AStarLocalPlanner::costmapCallback, this);
        stop_sub_ = nh_.subscribe("/robot_control/stop", 1, &AStarLocalPlanner::stopCallback, this);
        path_pub_ = nh_.advertise<nav_msgs::Path>(local_plan_topic_, 1);
        local_goal_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/a_star_local_goal", 1);
        status_pub = nh_.advertise<std_msgs::Int32>("/status", 1);
        pos_cmd_pub_ = nh_.advertise<magv_msgs::PositionCommand>("/magv/planning/pos_cmd", 1, true);

        // --- 初始化状态变量 ---
        costmap_received_ = false;
        goal_received_ = false;
        is_stopped_ = false;
        local_path_past.header.frame_id = global_frame_;

        // --- 创建用于主规划循环的定时器 ---
        planning_timer_ = nh_.createTimer(ros::Duration(1.0 / planning_frequency_), &AStarLocalPlanner::planningStep, this);

        ROS_INFO("A* Local Planner initialized.");
    }

private:
    // --- ROS回调函数 ---

    /** @brief 接收全局目标点的回调函数。*/
    void goalCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
        boost::mutex::scoped_lock lock(goal_mutex_);
        if (msg->header.frame_id != global_frame_) {
            ROS_ERROR("Received goal in frame '%s', but planner expects '%s'. Ignoring goal.",
                      msg->header.frame_id.c_str(), global_frame_.c_str());
            return;
        }
        global_goal_ = *msg;
        goal_received_ = true;
        ROS_INFO("New global goal received at (%.2f, %.2f)", 
                 global_goal_.pose.position.x, global_goal_.pose.position.y);
    }

    /** @brief 接收局部代价地图的回调函数。*/
    void costmapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
        boost::mutex::scoped_lock lock(costmap_mutex_);
        costmap_ = *msg;
        costmap_received_ = true;
    }

    /** @brief 接收外部停止指令的回调函数。*/
    void stopCallback(const std_msgs::Bool::ConstPtr& msg) {
        boost::mutex::scoped_lock lock(stop_mutex_);
        is_stopped_ = msg->data;
        if (is_stopped_) {
            ROS_WARN("Received STOP command. Pausing planning.");
        } else {
            ROS_INFO("Received GO command. Resuming planning.");
        }
    }

    // --- 主规划逻辑 ---

    /**
     * @brief 定时器触发的主规划函数，是节点的核心循环。
     */
    void planningStep(const ros::TimerEvent&) {
        if (!costmap_received_ || !goal_received_) {
            ROS_WARN_THROTTLE(2.0, "Waiting for costmap and/or goal to start planning.");
            return;
        }
        
        // (停止逻辑已注释，当前节点不处理停止命令)

        // 1. 获取机器人当前位姿
        geometry_msgs::PoseStamped current_pose;
        if (!getCurrentPose(current_pose)) {
            ROS_WARN("Could not get current robot pose. Skipping planning step.");
            return;
        }
        
        // 2. 检查是否已到达全局目标
        if (isGoalReached(current_pose, global_goal_)) {
            ROS_INFO_THROTTLE(2.0, "Global goal reached!");
            publishEmptyPath(); // 发布空路径以停止DWA
            
            // 发布最终的位置控制指令
            magv_msgs::PositionCommand cmd;
            cmd.position.x = global_goal_.pose.position.x;
            cmd.position.y = global_goal_.pose.position.y;
            tf2::Quaternion q(current_pose.pose.orientation.x, current_pose.pose.orientation.y, current_pose.pose.orientation.z, current_pose.pose.orientation.w);
            tf2::Matrix3x3 m(q);
            double roll, pitch, yaw;
            m.getRPY(roll, pitch, yaw);
            cmd.yaw = yaw;
            pos_cmd_pub_.publish(cmd);
            return;
        }

        // 3. 计算用于A*搜索的局部目标点
        geometry_msgs::PoseStamped local_goal;
        if (!calculateLocalGoal(current_pose, local_goal)) {
            ROS_WARN("Could not determine a valid local goal. Skipping planning step.");
            publishEmptyPath();
            return;
        }
        local_goal_pub_.publish(local_goal);

        // 4. 使用A*算法规划路径
        nav_msgs::Path local_path;
        if (planAStar(current_pose, local_goal, local_path)) {
            if (!local_path.poses.empty()) {
                path_pub_.publish(local_path);
                local_path_past = local_path; // 存储本次成功的路径
            } else {
                ROS_WARN("A* planning resulted in an empty path. Publishing previous path.");
                path_pub_.publish(local_path_past); // 发布上一条成功的路径
            }
        } else {
            ROS_WARN("A* failed to find a path. Publishing previous path.");
            path_pub_.publish(local_path_past); // 规划失败时也发布上一条成功的路径
        }
    }
    
    // --- A* 算法核心实现 ---
    
    /**
     * @brief A*路径搜索核心实现。
     * @param start_pose 起点位姿 (世界坐标系)。
     * @param goal_pose 终点位姿 (世界坐标系)。
     * @param path [out] 用于存储规划结果的路径。
     * @return bool 如果成功找到路径则返回true。
     */
    bool planAStar(const geometry_msgs::PoseStamped& start_pose, const geometry_msgs::PoseStamped& goal_pose, nav_msgs::Path& path) {
        boost::mutex::scoped_lock lock(costmap_mutex_);

        // 将世界坐标转换为栅格坐标
        int start_x, start_y, goal_x, goal_y;
        if (!worldToGrid(start_pose.pose.position.x, start_pose.pose.position.y, start_x, start_y) ||
            !worldToGrid(goal_pose.pose.position.x, goal_pose.pose.position.y, goal_x, goal_y)) {
            ROS_ERROR("Start or goal is outside the costmap boundaries.");
            return false;
        }
        
        if (!isSafe(start_x, start_y) || !isSafe(goal_x, goal_y)) {
            ROS_ERROR("Start or goal position is in an obstacle!");
            return false;
        }

        // A* 数据结构
        std::priority_queue<AStarNode, std::vector<AStarNode>, std::greater<AStarNode>> open_set;
        std::unordered_map<std::pair<int, int>, AStarNode*, NodeHasher> all_nodes;
        
        auto heuristic = [&](int x, int y) { return std::hypot(x - goal_x, y - goal_y); };
        
        AStarNode* start_node = new AStarNode(start_x, start_y, 0.0, heuristic(start_x, start_y));
        open_set.push(*start_node);
        all_nodes[{start_x, start_y}] = start_node;

        AStarNode* goal_node_ptr = nullptr;
        
        // A* 主循环
        while(!open_set.empty()) {
            AStarNode* current_node = all_nodes.at({open_set.top().x, open_set.top().y});
            open_set.pop();

            if (current_node->x == goal_x && current_node->y == goal_y) {
                goal_node_ptr = current_node;
                break;
            }

            // 探索8个邻居
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    if (dx == 0 && dy == 0) continue;
                    int nx = current_node->x + dx;
                    int ny = current_node->y + dy;

                    if (nx < 0 || nx >= costmap_.info.width || ny < 0 || ny >= costmap_.info.height || !isSafe(nx, ny)) continue;

                    double move_cost = std::hypot(dx, dy);
                    double new_g = current_node->g + move_cost;

                    auto it = all_nodes.find({nx, ny});
                    if (it == all_nodes.end() || new_g < it->second->g) {
                        AStarNode* neighbor_node = (it == all_nodes.end()) ? 
                            new AStarNode(nx, ny, new_g, heuristic(nx, ny), current_node) : it->second;
                        if (it == all_nodes.end()) all_nodes[{nx, ny}] = neighbor_node;
                        else {
                            neighbor_node->g = new_g;
                            neighbor_node->f = new_g + neighbor_node->h;
                            neighbor_node->parent = current_node;
                        }
                        open_set.push(*neighbor_node);
                    }
                }
            }
        }
        
        bool path_found = (goal_node_ptr != nullptr);
        if (path_found) {
            reconstructPath(goal_node_ptr, path);
        }

        // 清理动态分配的内存
        for (auto const& pair : all_nodes) {
            delete pair.second;
        }

        return path_found;
    }

    // --- 辅助函数 ---

    /** @brief 使用TF获取机器人当前在全局坐标系下的位姿。*/
    bool getCurrentPose(geometry_msgs::PoseStamped& pose_stamped) {
        try {
            geometry_msgs::TransformStamped transform = tf_buffer_.lookupTransform(
                global_frame_, robot_base_frame_, ros::Time(0), ros::Duration(0.2));
            tf2::convert(transform, pose_stamped);
            pose_stamped.header.stamp = ros::Time::now();
            return true;
        } catch (tf2::TransformException& ex) {
            ROS_WARN_THROTTLE(1.0, "TF lookup failed: %s", ex.what());
            return false;
        }
    }

    /** @brief 计算局部目标点。如果全局目标在地图内，则直接使用；否则计算与地图边界的交点。*/
    bool calculateLocalGoal(const geometry_msgs::PoseStamped& start_pose, geometry_msgs::PoseStamped& local_goal) {
        boost::mutex::scoped_lock costmap_lock(costmap_mutex_);
        boost::mutex::scoped_lock goal_lock(goal_mutex_);

        const double map_min_x = costmap_.info.origin.position.x;
        const double map_min_y = costmap_.info.origin.position.y;
        const double map_max_x = map_min_x + costmap_.info.width * costmap_.info.resolution;
        const double map_max_y = map_min_y + costmap_.info.height * costmap_.info.resolution;

        if (global_goal_.pose.position.x >= map_min_x && global_goal_.pose.position.x <= map_max_x &&
            global_goal_.pose.position.y >= map_min_y && global_goal_.pose.position.y <= map_max_y) {
            local_goal = global_goal_;
        } else {
            // 计算机器人位置与全局目标点连线和地图边界的交点
            double start_x = start_pose.pose.position.x, start_y = start_pose.pose.position.y;
            double goal_x = global_goal_.pose.position.x, goal_y = global_goal_.pose.position.y;
            double dx = goal_x - start_x, dy = goal_y - start_y;
            double t = std::numeric_limits<double>::max();

            if (std::abs(dx) > 1e-9) {
                double tx1 = (map_min_x - start_x) / dx, tx2 = (map_max_x - start_x) / dx;
                if (tx1 > 0 && tx1 < 1.0) t = std::min(t, tx1);
                if (tx2 > 0 && tx2 < 1.0) t = std::min(t, tx2);
            }
            if (std::abs(dy) > 1e-9) {
                double ty1 = (map_min_y - start_y) / dy, ty2 = (map_max_y - start_y) / dy;
                if (ty1 > 0 && ty1 < 1.0) t = std::min(t, ty1);
                if (ty2 > 0 && ty2 < 1.0) t = std::min(t, ty2);
            }
            
            if (t == std::numeric_limits<double>::max()){
                 local_goal = global_goal_; // 没有交点，直接钳制
            } else {
                double safety_margin = -costmap_.info.resolution * 2; // 向内移动2个栅格
                double length = std::hypot(dx, dy);
                local_goal.pose.position.x = start_x + t * dx + safety_margin * dx / length;
                local_goal.pose.position.y = start_y + t * dy + safety_margin * dy / length;
            }
        }
        
        // 确保目标点在地图边界内并检查安全性
        int goal_gx, goal_gy;
        local_goal.pose.position.x = std::max(map_min_x, std::min(map_max_x - costmap_.info.resolution, local_goal.pose.position.x));
        local_goal.pose.position.y = std::max(map_min_y, std::min(map_max_y - costmap_.info.resolution, local_goal.pose.position.y));
        worldToGrid(local_goal.pose.position.x, local_goal.pose.position.y, goal_gx, goal_gy);

        if (!isSafe(goal_gx, goal_gy)) {
            ROS_WARN("Local goal is unsafe. Searching for nearest safe point.");
            return findBestReachableGoal(goal_gx, goal_gy, goal_gx, goal_gy) && 
                   (gridToWorld(goal_gx, goal_gy, local_goal.pose.position.x, local_goal.pose.position.y), true);
        }
        
        local_goal.header.frame_id = global_frame_;
        local_goal.header.stamp = ros::Time::now();
        local_goal.pose.orientation = global_goal_.pose.orientation;
        return true;
    }

    /** @brief 在不安全的目标点周围螺旋搜索一个最近的安全替代点。*/
    bool findBestReachableGoal(int goal_x, int goal_y, int& best_goal_x, int& best_goal_y) {
        for (int r = 1; r < 20; ++r) {
            double min_dist_sq = std::numeric_limits<double>::max();
            bool found_safe_point = false;
            for (int dx = -r; dx <= r; ++dx) {
                for (int dy = -r; dy <= r; ++dy) {
                    if (std::abs(dx) != r && std::abs(dy) != r) continue;
                    int nx = goal_x + dx, ny = goal_y + dy;
                    if (nx < 0 || nx >= costmap_.info.width || ny < 0 || ny >= costmap_.info.height) continue;
                    if (isSafe(nx, ny)) {
                        found_safe_point = true;
                        double dist_sq = dx * dx + dy * dy;
                        if (dist_sq < min_dist_sq) {
                            min_dist_sq = dist_sq;
                            best_goal_x = nx;
                            best_goal_y = ny;
                        }
                    }
                }
            }
            if (found_safe_point) {
                ROS_WARN("Found a substitute goal at (%d, %d).", best_goal_x, best_goal_y);
                return true;
            }
        }
        ROS_ERROR("Could not find any reachable substitute goal near (%d, %d).", goal_x, goal_y);
        return false;
    }

    /** @brief 从偏航角创建四元数。*/
    geometry_msgs::Quaternion createQuaternionFromYaw(double yaw) {
        tf2::Quaternion q;
        q.setRPY(0, 0, yaw);
        return tf2::toMsg(q);
    }

    /** @brief 从终点节点回溯，重建路径，并为路径点计算朝向。*/
    void reconstructPath(AStarNode* goal_node, nav_msgs::Path& path) {
        std::vector<geometry_msgs::PoseStamped> temp_path;
        AStarNode* current = goal_node;
        while (current) {
            geometry_msgs::PoseStamped pose;
            gridToWorld(current->x, current->y, pose.pose.position.x, pose.pose.position.y);
            temp_path.push_back(pose);
            current = current->parent;
        }
        std::reverse(temp_path.begin(), temp_path.end());

        // 为每个路径点计算偏航角
        for (size_t i = 0; i < temp_path.size(); ++i) {
            double yaw = 0.0;
            if (i < temp_path.size() - 1) {
                double dx = temp_path[i+1].pose.position.x - temp_path[i].pose.position.x;
                double dy = temp_path[i+1].pose.position.y - temp_path[i].pose.position.y;
                yaw = atan2(dy, dx);
            } else if (temp_path.size() > 1) {
                double dx = temp_path[i].pose.position.x - temp_path[i-1].pose.position.x;
                double dy = temp_path[i].pose.position.y - temp_path[i-1].pose.position.y;
                yaw = atan2(dy, dx);
            }
            temp_path[i].pose.orientation = createQuaternionFromYaw(yaw);
            temp_path[i].header.frame_id = global_frame_;
            path.poses.push_back(temp_path[i]);
        }
        path.header.stamp = ros::Time::now();
        path.header.frame_id = global_frame_;
    }

    /** @brief 检查是否到达目标点。*/
    bool isGoalReached(const geometry_msgs::PoseStamped& current, const geometry_msgs::PoseStamped& goal) {
        return std::hypot(current.pose.position.x - goal.pose.position.x, current.pose.position.y - goal.pose.position.y) < goal_tolerance_;
    }

    /** @brief 发布一条空路径以停止机器人。*/
    void publishEmptyPath() {
        nav_msgs::Path empty_path;
        empty_path.header.stamp = ros::Time::now();
        empty_path.header.frame_id = global_frame_;
        path_pub_.publish(empty_path);
    }
    
    // --- 坐标转换函数 ---
    /** @brief 将世界坐标转换为栅格坐标。*/
    bool worldToGrid(double wx, double wy, int& gx, int& gy) {
        gx = static_cast<int>((wx - costmap_.info.origin.position.x) / costmap_.info.resolution);
        gy = static_cast<int>((wy - costmap_.info.origin.position.y) / costmap_.info.resolution);
        return (gx >= 0 && gx < costmap_.info.width && gy >= 0 && gy < costmap_.info.height);
    }
    /** @brief 将栅格坐标转换为世界坐标。*/
    void gridToWorld(int gx, int gy, double& wx, double& wy) {
        wx = costmap_.info.origin.position.x + (gx + 0.5) * costmap_.info.resolution;
        wy = costmap_.info.origin.position.y + (gy + 0.5) * costmap_.info.resolution;
    }
    /** @brief 检查给定栅格是否安全(可通行)。*/
    bool isSafe(int gx, int gy) {
        return costmap_.data[gy * costmap_.info.width + gx] <= 0;
    }

private:
    // --- ROS通信对象 ---
    ros::NodeHandle nh_;                            ///< ROS节点句柄
    ros::Subscriber goal_sub_;                      ///< 全局目标订阅者
    ros::Subscriber costmap_sub_;                   ///< 代价地图订阅者
    ros::Subscriber stop_sub_;                      ///< 停止命令订阅者
    ros::Publisher status_pub;                      ///< 状态发布者
    ros::Publisher path_pub_;                       ///< 局部路径发布者
    ros::Publisher pos_cmd_pub_;                    ///< 最终位置指令发布者
    ros::Publisher local_goal_pub_;                 ///< 局部目标可视化发布者
    ros::Timer planning_timer_;                     ///< 主规划循环定时器
    
    // --- TF相关 ---
    tf2_ros::Buffer tf_buffer_;                     ///< TF缓冲区
    tf2_ros::TransformListener tf_listener_;        ///< TF监听器

    // --- 参数 ---
    std::string global_goal_topic_, costmap_topic_, local_plan_topic_; ///< 话题名称
    std::string robot_base_frame_, global_frame_;   ///< TF坐标系名称
    double planning_frequency_;                     ///< 规划频率 (Hz)
    double goal_tolerance_;                         ///< 到达目标的容忍距离 (米)
    bool is_stopped_;                               ///< 停止标志位

    // --- 状态数据 ---
    nav_msgs::OccupancyGrid costmap_;               ///< 存储的代价地图
    geometry_msgs::PoseStamped global_goal_;        ///< 存储的全局目标
    bool costmap_received_;                         ///< 标志位：是否已收到代价地图
    bool goal_received_;                            ///< 标志位：是否已收到全局目标
    nav_msgs::Path local_path_past;                 ///< 用于存储上一条成功的局部路径

    // --- 互斥锁 ---
    boost::mutex costmap_mutex_;                    ///< 代价地图数据锁
    boost::mutex goal_mutex_;                       ///< 全局目标数据锁
    boost::mutex stop_mutex_;                       ///< 停止标志位锁
};

/** @brief 主函数 */
int main(int argc, char** argv) {
    ros::init(argc, argv, "localPlanner");
    AStarLocalPlanner planner;
    ros::spin();
    return 0;
}