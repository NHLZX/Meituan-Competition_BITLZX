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
#include <tf2/LinearMath/Quaternion.h>   // 支持 tf2::Quaternion
#include <tf2/LinearMath/Matrix3x3.h>    // 支持 tf2::Matrix3x3
#include <tf2/LinearMath/Transform.h>    // 支持 tf2::Transform 
#include <std_msgs/Bool.h>

/**
 * @struct AStarNode
 * @brief A*算法中用于搜索的节点结构体.
 */
struct AStarNode {
    int x, y;          // 节点在栅格地图中的坐标
    double g, h, f;    // 代价: g=从起点到此节点的代价, h=启发式代价, f=g+h
    AStarNode* parent; // 指向父节点的指针，用于路径回溯

    AStarNode(int x, int y, double g, double h, AStarNode* parent = nullptr)
        : x(x), y(y), g(g), h(h), f(g + h), parent(parent) {}

    // 优先队列的比较函数 (构建最小堆)
    bool operator>(const AStarNode& other) const {
        return f > other.f;
    }
};

/**
 * @struct NodeHasher
 * @brief 用于在unordered_map中对栅格坐标对进行哈希计算.
 */
struct NodeHasher {
    std::size_t operator()(const std::pair<int, int>& p) const {
        // 将x和y坐标组合成一个唯一的哈希值
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};


/**
 * @class AStarLocalPlanner
 * @brief 一个使用A*算法进行局部路径规划的ROS节点.
 */
class AStarLocalPlanner {
public:
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
        path_pub_ = nh_.advertise<nav_msgs::Path>(local_plan_topic_, 1);
        local_goal_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/a_star_local_goal", 1); // 用于Rviz可视化局部目标点
        status_pub = nh_.advertise<std_msgs::Int32>("/status", 1); // 发布状态信息
        pos_cmd_pub_ = nh_.advertise<magv_msgs::PositionCommand>("/magv/planning/pos_cmd", 1,true); // 位置控制命令
        stop_sub_ = nh_.subscribe("/robot_control/stop", 1, &AStarLocalPlanner::stopCallback, this);

        // --- 初始化状态变量 ---
        costmap_received_ = false;
        goal_received_ = false;

        // --- 创建用于主规划循环的定时器 ---
        planning_timer_ = nh_.createTimer(ros::Duration(1.0 / planning_frequency_), &AStarLocalPlanner::planningStep, this);

        ROS_INFO("A* Local Planner initialized.");
        ROS_INFO("--> Subscribing to goal on: %s", global_goal_topic_.c_str());
        ROS_INFO("--> Subscribing to costmap on: %s", costmap_topic_.c_str());
        ROS_INFO("--> Publishing local plan for DWA on: %s", local_plan_topic_.c_str());
        is_stopped_ = false; // --- NEW ---: 默认不停止
        nav_msgs::Path empty_path;
        empty_path.header.stamp = ros::Time::now();
        empty_path.header.frame_id = global_frame_;
        local_path_past = empty_path; // 初始化过去的局部路径为空
    }

private:
    // --- ROS回调函数 ---

    /**
     * @brief 接收全局目标点的回调函数.
     */
    void goalCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
        boost::mutex::scoped_lock lock(goal_mutex_);
        if (msg->header.frame_id != global_frame_) {
            ROS_ERROR("Received goal in frame '%s', but planner expects goals in frame '%s'. Ignoring goal.",
                      msg->header.frame_id.c_str(), global_frame_.c_str());
            return;
        }
        global_goal_ = *msg;
        goal_received_ = true;
        ROS_INFO("New global goal received at (%.2f, %.2f)", 
                 global_goal_.pose.position.x, global_goal_.pose.position.y);
    }

    /**
     * @brief 接收局部代价地图的回调函数.
     */
    void costmapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
        boost::mutex::scoped_lock lock(costmap_mutex_);
        costmap_ = *msg;
        costmap_received_ = true;
    }

    void stopCallback(const std_msgs::Bool::ConstPtr& msg) {
        boost::mutex::scoped_lock lock(stop_mutex_); // 使用锁保护共享变量
        is_stopped_ = msg->data;
        if (is_stopped_) {
            ROS_WARN("Received STOP command. Pausing planning and publishing empty path.");
        } else {
            ROS_INFO("Received GO command. Resuming planning.");
        }
    }

    // --- 主规划逻辑 ---

    /**
     * @brief 定时器触发的主规划函数.
     */
    void planningStep(const ros::TimerEvent&) {
        if (!costmap_received_ || !goal_received_) {
            ROS_WARN_THROTTLE(2.0, "Waiting for costmap and/or goal to start planning.");
            return;
        }

        bool should_stop;
        {
            boost::mutex::scoped_lock lock(stop_mutex_);
            should_stop = is_stopped_;
        }

        // if (should_stop) {
        //     publishEmptyPath(); // 发布空路径以停止机器人
        //     ROS_WARN_THROTTLE(1.0, "Planning is paused due to stop command.");
        //     return; // 立即退出本次规划循环
        // }

        // 1. 获取机器人当前位姿
        geometry_msgs::PoseStamped current_pose;
        if (!getCurrentPose(current_pose)) {
            ROS_WARN("Could not get current robot pose. Skipping planning step.");
            return;
        }
        
        // 检查是否已到达全局目标
        if (isGoalReached(current_pose, global_goal_)) {
            ROS_INFO_THROTTLE(2.0, "Global goal reached!");
            publishEmptyPath(); // 发布空路径以停止DWA
            magv_msgs::PositionCommand cmd;
            // **关键修复**: 填充完整的指令
            cmd.position.x = global_goal_.pose.position.x;
            cmd.position.y = global_goal_.pose.position.y;
            // 假设 PositionCommand 消息中有 velocity 和 yaw_rate
            // 如果消息定义不同，请修改这里

            tf2::Quaternion q(
                current_pose.pose.orientation.x,
                current_pose.pose.orientation.y,
                current_pose.pose.orientation.z,
                current_pose.pose.orientation.w);

            tf2::Matrix3x3 m(q);
            double roll, pitch,yaw;
            m.getRPY(roll, pitch, yaw);
            cmd.yaw = yaw;
            pos_cmd_pub_.publish(cmd);
            // std_msgs::Int32 status_msg;
            // status_msg.data = 0;
            // status_pub.publish(status_msg);
            //goal_received_ = false; // 等待新目标
            return;
        }

        // 2. 计算用于A*搜索的局部目标点
        geometry_msgs::PoseStamped local_goal;
        if (!calculateLocalGoal(current_pose, local_goal)) {
            ROS_WARN("Could not determine a valid local goal. Skipping planning step.");
            publishEmptyPath();
            return;
        }
        local_goal_pub_.publish(local_goal); // 发布局部目标点用于调试

        // 3. 使用A*算法规划路径
        nav_msgs::Path local_path;
        if (planAStar(current_pose, local_goal, local_path)) {
            // 4. 发布路径供DWA跟踪
            if (!local_path.poses.empty()) {
                path_pub_.publish(local_path);
                local_path_past = local_path; // 存储过去的局部路径
            } else {
                ROS_WARN("A* planning resulted in an empty path.");
                path_pub_.publish(local_path_past);
                //publishEmptyPath();
            }
        } else {
            ROS_WARN("A* failed to find a path to the local goal.");
            path_pub_.publish(local_path_past);
            //publishEmptyPath(); // 规划失败时发布空路径
        }
    }
    
    // --- A* 算法核心实现 ---
    
    /**
     * @brief A*路径搜索实现.
     * @param start_pose 起点位姿.
     * @param goal_pose 终点位姿.
     * @param path 输出的路径.
     * @return 如果找到路径则返回true.
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
        
        // 检查起点和终点是否在障碍物内
        if (!isSafe(start_x, start_y)) {
            ROS_ERROR("Start position is in an obstacle!");
            return false;
        }
        if (!isSafe(goal_x, goal_y)) {
            ROS_ERROR("Goal position is in an obstacle!");
            return false;
        }

        // if (!worldToGrid(goal_pose.pose.position.x, goal_pose.pose.position.y, goal_x, goal_y) || !isSafe(goal_x, goal_y)) {
        // ROS_WARN("Original local goal at (%.2f, %.2f) is unsafe or outside map. Searching for a substitute.", 
        //          goal_pose.pose.position.x, goal_pose.pose.position.y);
        
        // A* 算法所需的数据结构
        std::priority_queue<AStarNode, std::vector<AStarNode>, std::greater<AStarNode>> open_set;
        std::unordered_map<std::pair<int, int>, AStarNode*, NodeHasher> all_nodes;
        
        // 启发式函数 (欧几里得距离)
        auto heuristic = [&](int x, int y) {
            return std::hypot(x - goal_x, y - goal_y);
        };
        
        // 将起点加入开放列表
        AStarNode* start_node = new AStarNode(start_x, start_y, 0.0, heuristic(start_x, start_y));
        open_set.push(*start_node);
        all_nodes[{start_x, start_y}] = start_node;

        AStarNode* goal_node_ptr = nullptr;
        
        // A* 主循环
        while(!open_set.empty()) {
            AStarNode current_node_val = open_set.top();
            open_set.pop();
            AStarNode* current_node = all_nodes.at({current_node_val.x, current_node_val.y});

            // 到达目标
            if (current_node->x == goal_x && current_node->y == goal_y) {
                goal_node_ptr = current_node;
                break;
            }

            // 探索8个邻居节点
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    if (dx == 0 && dy == 0) continue;

                    int nx = current_node->x + dx;
                    int ny = current_node->y + dy;

                    // 检查边界和障碍物
                    if (nx < 0 || nx >= costmap_.info.width || ny < 0 || ny >= costmap_.info.height || !isSafe(nx, ny)) {
                        continue;
                    }

                    double move_cost = std::hypot(dx, dy); // 直线移动代价为1，对角线为sqrt(2)
                    double new_g = current_node->g + move_cost;

                    auto it = all_nodes.find({nx, ny});
                    if (it == all_nodes.end() || new_g < it->second->g) {
                        AStarNode* neighbor_node;
                        if (it == all_nodes.end()) {
                            // 发现新节点
                            neighbor_node = new AStarNode(nx, ny, new_g, heuristic(nx, ny), current_node);
                            all_nodes[{nx, ny}] = neighbor_node;
                        } else {
                            // 更新已有节点
                            neighbor_node = it->second;
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
            reconstructPath(goal_node_ptr, path); // 回溯生成路径
        }

        // 清理动态分配的内存
        // for (auto const& [key, val] : all_nodes) {
        //     delete val;
        // }
        for (auto const& it : all_nodes) { // C++11/14 Compatible Syntax
        delete it.second; // 'it.second' is the value (the pointer)
}

        return path_found;
    }

    // --- 辅助函数 ---

    /**
     * @brief 使用TF获取机器人当前在全局坐标系下的位姿.
     */
    bool getCurrentPose(geometry_msgs::PoseStamped& pose_stamped) {
        try {
            geometry_msgs::TransformStamped transform = tf_buffer_.lookupTransform(
                global_frame_, robot_base_frame_, ros::Time(0), ros::Duration(0.2));
            
            pose_stamped.header.frame_id = global_frame_;
            pose_stamped.header.stamp = ros::Time::now();
            pose_stamped.pose.position.x = transform.transform.translation.x;
            pose_stamped.pose.position.y = transform.transform.translation.y;
            pose_stamped.pose.position.z = transform.transform.translation.z;
            pose_stamped.pose.orientation = transform.transform.rotation;
            return true;
        } catch (tf2::TransformException& ex) {
            ROS_WARN_THROTTLE(1.0, "TF lookup from '%s' to '%s' failed: %s", 
                global_frame_.c_str(), robot_base_frame_.c_str(), ex.what());
            return false;
        }
    }

    /**
     * @brief 计算局部目标点. 如果全局目标在地图内，则直接使用；否则计算与地图边界的交点.
     */
    bool calculateLocalGoal(const geometry_msgs::PoseStamped& start_pose, geometry_msgs::PoseStamped& local_goal) {
        boost::mutex::scoped_lock costmap_lock(costmap_mutex_);
        boost::mutex::scoped_lock goal_lock(goal_mutex_);

        const double map_min_x = costmap_.info.origin.position.x;
        const double map_min_y = costmap_.info.origin.position.y;
        const double map_max_x = map_min_x + costmap_.info.width * costmap_.info.resolution;
        const double map_max_y = map_min_y + costmap_.info.height * costmap_.info.resolution;

        // Case 1: 全局目标点在局部代价地图范围内
        if (global_goal_.pose.position.x >= map_min_x && global_goal_.pose.position.x <= map_max_x &&
            global_goal_.pose.position.y >= map_min_y && global_goal_.pose.position.y <= map_max_y) {
            local_goal = global_goal_;
            return true;
        }

        // Case 2: 全局目标点在地图外, 计算机器人位置与全局目标点连线和地图边界的交点
        double start_x = start_pose.pose.position.x;
        double start_y = start_pose.pose.position.y;
        double goal_x = global_goal_.pose.position.x;
        double goal_y = global_goal_.pose.position.y;

        double dx = goal_x - start_x;
        double dy = goal_y - start_y;

        double t = std::numeric_limits<double>::max();

        // 使用参数方程 line(t) = start + t * (goal - start)
        // 找到与四条边界相交的最小正t值
        if (std::abs(dx) > 1e-9) {
            double tx1 = (map_min_x - start_x) / dx;
            double tx2 = (map_max_x - start_x) / dx;
            if (tx1 > 0 && tx1 < 1.0) t = std::min(t, tx1);
            if (tx2 > 0 && tx2 < 1.0) t = std::min(t, tx2);
        }
        if (std::abs(dy) > 1e-9) {
            double ty1 = (map_min_y - start_y) / dy;
            double ty2 = (map_max_y - start_y) / dy;
            if (ty1 > 0 && ty1 < 1.0) t = std::min(t, ty1);
            if (ty2 > 0 && ty2 < 1.0) t = std::min(t, ty2);
        }
        
        if (t == std::numeric_limits<double>::max()){
            // 如果没有找到交点 (可能机器人已经在地图外), 将目标点钳制在地图边界
             local_goal = global_goal_;
             local_goal.pose.position.x = std::max(map_min_x, std::min(map_max_x, local_goal.pose.position.x));
             local_goal.pose.position.y = std::max(map_min_y, std::min(map_max_y, local_goal.pose.position.y));
        } else {
            // 计算交点坐标, 并向外移动一小段距离以确保目标点在地图
            double safety_margin = -costmap_.info.resolution * 2; // 向内移动2个栅格
            double length = std::hypot(dx, dy);
            local_goal.pose.position.x = start_x + t * dx + safety_margin * dx / length;
            local_goal.pose.position.y = start_y + t * dy + safety_margin * dy / length;
        }

        int goal_gx, goal_gy;
        if (!worldToGrid(local_goal.pose.position.x, local_goal.pose.position.y, goal_gx, goal_gy)) {
            ROS_WARN("Computed local goal is outside costmap. Clamping to boundary.");
            goal_gx = std::max(0, std::min(static_cast<int>(costmap_.info.width) - 1, goal_gx));
            goal_gy = std::max(0, std::min(static_cast<int>(costmap_.info.height) - 1, goal_gy));
            gridToWorld(goal_gx, goal_gy, local_goal.pose.position.x, local_goal.pose.position.y);
        }

    // 检查目标点是否在障碍物上
        if (!isSafe(goal_gx, goal_gy)) {
            ROS_WARN("Local goal at (%d, %d) is unsafe. Searching for nearest safe point.", goal_gx, goal_gy);
        
            // 逐步向起点方向移动，直到找到安全点
            const int max_steps = 100;  // 最大搜索步数
            double step_x = (start_pose.pose.position.x - local_goal.pose.position.x) / max_steps;
            double step_y = (start_pose.pose.position.y - local_goal.pose.position.y) / max_steps;
        
            for (int step = 1; step <= max_steps; ++step) {
                double test_x = local_goal.pose.position.x + step_x * step;
                double test_y = local_goal.pose.position.y + step_y * step;
            
                if (worldToGrid(test_x, test_y, goal_gx, goal_gy) && isSafe(goal_gx, goal_gy)) {
                    local_goal.pose.position.x = test_x;
                    local_goal.pose.position.y = test_y;
                    ROS_INFO("Found safe local goal at (%.2f, %.2f) after %d steps", 
                            test_x, test_y, step);
                    return true;
                }
            }
        ROS_ERROR("Failed to find safe local goal near target after %d steps", max_steps);
            return false;
        }
        
            local_goal.header.frame_id = global_frame_;
            local_goal.header.stamp = ros::Time::now();
            local_goal.pose.orientation = global_goal_.pose.orientation; // 保持全局目标的姿
        return true;
    }


    bool findBestReachableGoal(int goal_x, int goal_y, int& best_goal_x, int& best_goal_y) {
    // 从半径1开始向外螺旋搜索
    for (int r = 1; r < 20; ++r) { // 限制最大搜索半径为20个栅格
        double min_dist_sq = std::numeric_limits<double>::max();
        bool found_safe_point = false;

        for (int dx = -r; dx <= r; ++dx) {
            for (int dy = -r; dy <= r; ++dy) {
                // 只搜索当前半径的外环
                if (std::abs(dx) != r && std::abs(dy) != r) {
                    continue;
                }

                int nx = goal_x + dx;
                int ny = goal_y + dy;

                // 检查边界
                if (nx < 0 || nx >= costmap_.info.width || ny < 0 || ny >= costmap_.info.height) {
                    continue;
                }

                // 如果找到一个安全点
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
        // 如果在当前半径的外环上找到了安全点，就返回最近的那个
        if (found_safe_point) {
            ROS_WARN("Original goal is unsafe. Found a substitute goal at (%d, %d).", best_goal_x, best_goal_y);
            return true;
        }
    }

    ROS_ERROR("Could not find any reachable substitute goal near (%d, %d).", goal_x, goal_y);
    return false;
}

    geometry_msgs::Quaternion createQuaternionFromYaw(double yaw) {
    tf2::Quaternion q;
    q.setRPY(0, 0, yaw);
    return tf2::toMsg(q);
}

    /**
     * @brief 从终点节点回溯，重建路径.
     */
    void reconstructPath(AStarNode* goal_node, nav_msgs::Path& path) {
    
    std::vector<geometry_msgs::PoseStamped> temp_path;
    
    // 收集原始路径点（仅位置）
    AStarNode* current = goal_node;
    while (current) {
        geometry_msgs::PoseStamped pose;
        gridToWorld(current->x, current->y, pose.pose.position.x, pose.pose.position.y);
        pose.pose.orientation.w = 1.0; // 临时默认值
        temp_path.push_back(pose);
        current = current->parent;
    }
    std::reverse(temp_path.begin(), temp_path.end());

    // 计算每个点的偏航角
    for (size_t i = 0; i < temp_path.size(); ++i) {
        double yaw = 0.0;
        
        if (i < temp_path.size() - 1) {
            // 计算当前点到下一个点的方向向量
            double dx = temp_path[i+1].pose.position.x - temp_path[i].pose.position.x;
            double dy = temp_path[i+1].pose.position.y - temp_path[i].pose.position.y;
            yaw = atan2(dy, dx);  // 获得弧度制偏航角
        } else {
            // 终点：使用前一点的方向或全局目标朝向
            if (temp_path.size() > 1) {
                double dx = temp_path[i].pose.position.x - temp_path[i-1].pose.position.x;
                double dy = temp_path[i].pose.position.y - temp_path[i-1].pose.position.y;
                yaw = atan2(dy, dx);
            }
            // 可选：终点的姿态使用global_goal_.pose.orientation
        }
        
        // 更新姿态角
        temp_path[i].pose.orientation = createQuaternionFromYaw(yaw);
        temp_path[i].header.frame_id = global_frame_;
        path.poses.push_back(temp_path[i]);
    }
    
    // 设置路径头信息
    path.header.stamp = ros::Time::now();
    path.header.frame_id = global_frame_;
}
    //     std::vector<geometry_msgs::PoseStamped> temp_path;
    //     path.poses.clear();
        

    //     AStarNode* current = goal_node;
    //     while (current != nullptr) {
    //         double wx, wy;
    //         gridToWorld(current->x, current->y, wx, wy);

    //         geometry_msgs::PoseStamped pose;
    //         pose.header = path.header;
    //         pose.pose.position.x = wx;
    //         pose.pose.position.y = wy;
    //         pose.pose.orientation.w = 1.0;
    //         path.poses.push_back(pose);

    //         current = current->parent;
    //     }
    //     std::reverse(path.poses.begin(), path.poses.end()); // 反转路径为从起点到终点
    //     for (size_t i = 0; i < temp_path.size(); ++i) {
    //     double yaw = 0.0;
        
    //     if (i < temp_path.size() - 1) {
    //         // 计算当前点到下一个点的方向向量
    //         double dx = temp_path[i+1].pose.position.x - temp_path[i].pose.position.x;
    //         double dy = temp_path[i+1].pose.position.y - temp_path[i].pose.position.y;
    //         yaw = atan2(dy, dx);  // 获得弧度制偏航角
    //     } else {
    //         // 终点：使用前一点的方向或全局目标朝向
    //         if (temp_path.size() > 1) {
    //             double dx = temp_path[i].pose.position.x - temp_path[i-1].pose.position.x;
    //             double dy = temp_path[i].pose.position.y - temp_path[i-1].pose.position.y;
    //             yaw = atan2(dy, dx);
    //         }
    //         // 可选：终点的姿态使用global_goal_.pose.orientation
    //     }
        
    //     // 更新姿态角
    //     temp_path[i].pose.orientation = createQuaternionFromYaw(yaw);
    //     temp_path[i].header.frame_id = global_frame_;
    //     path.poses.push_back(temp_path[i]);
    // }
    //     path.header.stamp = ros::Time::now();
    //     path.header.frame_id = global_frame_;
    
    //}

    /**
     * @brief 检查是否到达目标点.
     */
    bool isGoalReached(const geometry_msgs::PoseStamped& current, const geometry_msgs::PoseStamped& goal) {
        double dx = current.pose.position.x - goal.pose.position.x;
        double dy = current.pose.position.y - goal.pose.position.y;
        return std::hypot(dx, dy) < goal_tolerance_;
    }

    /**
     * @brief 发布一条空路径.
     */
    void publishEmptyPath() {
        nav_msgs::Path empty_path;
        empty_path.header.stamp = ros::Time::now();
        empty_path.header.frame_id = global_frame_;
        path_pub_.publish(empty_path);
    }
    
    // --- 坐标转换函数 ---
    bool worldToGrid(double wx, double wy, int& gx, int& gy) {
        if (wx < costmap_.info.origin.position.x || wy < costmap_.info.origin.position.y) return false;
        
        gx = static_cast<int>((wx - costmap_.info.origin.position.x) / costmap_.info.resolution);
        gy = static_cast<int>((wy - costmap_.info.origin.position.y) / costmap_.info.resolution);

        return (gx >= 0 && gx < costmap_.info.width && gy >= 0 && gy < costmap_.info.height);
    }

    void gridToWorld(int gx, int gy, double& wx, double& wy) {
        wx = costmap_.info.origin.position.x + (gx + 0.5) * costmap_.info.resolution;
        wy = costmap_.info.origin.position.y + (gy + 0.5) * costmap_.info.resolution;
    }
    
    /**
     * @brief 检查给定栅格是否安全(可通行). -1也视为安全.
     */
    bool isSafe(int gx, int gy) {
        int index = gy * costmap_.info.width + gx;
        // 代价地图值: -1 (未知), 0 (空闲), 1-100 (障碍).
        // 任何大于0的值都被视为障碍物.
        return costmap_.data[index] <= 0;
    }

private:
    // ROS相关句柄、发布者、订阅者
    ros::NodeHandle nh_;
    ros::Subscriber goal_sub_;
    ros::Subscriber costmap_sub_;
    ros::Publisher status_pub;
    ros::Publisher path_pub_;
    ros::Publisher pos_cmd_pub_;
    ros::Publisher local_goal_pub_;
    ros::Timer planning_timer_;
    ros::Subscriber stop_sub_; // --- NEW ---
    
    // TF相关
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    // 参数
    std::string global_goal_topic_, costmap_topic_, local_plan_topic_;
    std::string robot_base_frame_, global_frame_;
    double planning_frequency_;
    double goal_tolerance_;
    bool is_stopped_;// 是否停止

    // 状态数据
    nav_msgs::OccupancyGrid costmap_;
    geometry_msgs::PoseStamped global_goal_;
    bool costmap_received_;
    bool goal_received_;
    nav_msgs::Path local_path_past; // 用于存储过去的局部路径

    // 用于线程安全的互斥锁
    boost::mutex costmap_mutex_;
    boost::mutex goal_mutex_;
    boost::mutex stop_mutex_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "localPlanner");
    AStarLocalPlanner planner;
    ros::spin();
    return 0;
}



