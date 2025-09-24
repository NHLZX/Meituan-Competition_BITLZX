/**
 * @file local_costmap_publisher.cpp
 * @author BITLZX
 * @date 2025-09-23
 * @brief 该节点实现了一个局部代价地图生成器和TF坐标变换发布器。
 *
 * @details
 * 此节点的主要功能有两个：
 * 1. **生成局部代价地图**: 订阅一个过滤后的点云话题 (`/filtered_cloud`)(map系下)，
 * 将点云数据转换为一个二维栅格地图。该地图以机器人为中心（“滚动窗口”），
 * 并对障碍物进行膨胀处理，以考虑机器人的体积和安全距离。最终的代价地图
 * 以 `nav_msgs/OccupancyGrid` 格式发布到 `/local_costmap_demo` 话题。
 * 2. **发布TF坐标变换**: 订阅里程计信息 (`/magv/odometry/gt`) 来发布
 * `map` -> `odom` 和 `odom` -> `car/base_link` 的动态变换。同时，在启动时
 * 发布 `car/base_link` -> `car/laser_link` 和 `car/base_link` -> `car/camera_link`
 * 的静态变换。
 */

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/OccupancyGrid.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/Int8.h>
#include <nav_msgs/Odometry.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>

/**
 * @class LocalCostmapPublisher
 * @brief 管理局部代价地图生成和TF发布的类。
 */
class LocalCostmapPublisher {
public:
    /**
     * @brief LocalCostmapPublisher的构造函数。
     * @details 初始化代价地图参数、膨胀参数、ROS订阅者和发布者，并发布静态TF变换。
     * @param nh ROS节点句柄。
     */
    LocalCostmapPublisher(ros::NodeHandle& nh)
    {
        // --- 固定参数设置 ---
        resolution_ = 0.1;      // 地图分辨率 (米/栅格)
        width_ = 120;           // 地图宽度 (栅格数)
        height_ = 120;          // 地图高度 (栅格数)
        origin_x_ = 0;          // 初始原点x坐标
        origin_y_ = 0;          // 初始原点y坐标
        
        // --- 膨胀参数设置 ---
        // 基于机器人半径0.6m，增加0.03m的安全裕度
        inflation_radius_ = 0.63; 
        // 将膨胀半径从米转换为栅格数
        inflation_cells_ = static_cast<int>(std::ceil(inflation_radius_ / resolution_));
        
        // --- 订阅者和发布者初始化 ---
        sub_ = nh.subscribe("/filtered_cloud", 1, &LocalCostmapPublisher::cloudCallback, this);
        sub_pos = nh.subscribe<nav_msgs::Odometry>("/magv/odometry/gt", 5, &LocalCostmapPublisher::posCallback, this);
        pub_ = nh.advertise<nav_msgs::OccupancyGrid>("/local_costmap_demo", 1);
        pub_2 = nh.advertise<std_msgs::Int8>("/my_int_topic2", 10); // 示例/调试话题

        // --- 发布静态TF变换 ---
        // 静态TF只需要发布一次，因此在构造函数中调用
        publishLidarTransform();
        publishCameraTransform();

        ROS_INFO("LocalCostmapPublisher initialized");
        ROS_INFO("  Resolution: %.2f m/cell", resolution_);
        ROS_INFO("  Grid size: %d x %d cells", width_, height_);
        ROS_INFO("  Inflation radius: %.2f m (%d cells)", inflation_radius_, inflation_cells_);
    }

    /**
     * @brief 里程计信息回调函数。
     * @details 此函数根据机器人的当前位置动态更新代价地图的原点，实现"滚动窗口"效果。
     * 同时，它还负责发布`map`->`odom`和`odom`->`base_link`的TF变换。
     * @param odom 从里程计话题接收到的消息。
     */
    void posCallback(const nav_msgs::Odometry::ConstPtr &odom)
    {
        time_stamp_ = odom->header.stamp.toSec();
        double current_x_ = odom->pose.pose.position.x;
        double current_y_ = odom->pose.pose.position.y;

        // 计算新的地图原点，使得机器人始终位于局部地图的中心
        origin_x_ = current_x_ - (width_ * resolution_) / 2.0;
        origin_y_ = current_y_ - (height_ * resolution_) / 2.0;

        // --- 发布 map -> odom 的变换 ---
        // 在这个简化的系统中，map和odom被认为是重合的，这是一个常见的设置。
        geometry_msgs::TransformStamped map_to_odom;
        map_to_odom.header.stamp = odom->header.stamp;
        map_to_odom.header.frame_id = "map";
        map_to_odom.child_frame_id = "odom";
        map_to_odom.transform.translation.x = 0.0;
        map_to_odom.transform.translation.y = 0.0;
        map_to_odom.transform.translation.z = 0.0;
        map_to_odom.transform.rotation.x = 0.0;
        map_to_odom.transform.rotation.y = 0.0;
        map_to_odom.transform.rotation.z = 0.0;
        map_to_odom.transform.rotation.w = 1.0;
        tf_broadcaster_.sendTransform(map_to_odom);
        
        // --- 发布 odom -> base_link 的变换 ---
        // 这个变换是动态的，直接使用里程计提供的位置和姿态信息。
        geometry_msgs::TransformStamped odom_to_base;
        odom_to_base.header.stamp = odom->header.stamp;
        odom_to_base.header.frame_id = "odom";
        odom_to_base.child_frame_id = "car/base_link";
        odom_to_base.transform.translation.x = odom->pose.pose.position.x;
        odom_to_base.transform.translation.y = odom->pose.pose.position.y;
        odom_to_base.transform.translation.z = odom->pose.pose.position.z;
        odom_to_base.transform.rotation = odom->pose.pose.orientation;
        tf_broadcaster_.sendTransform(odom_to_base);
    }
    
    /**
     * @brief 对栅格地图中的障碍物进行膨胀处理。
     * @details 此函数在原始障碍物周围创建一个代价递减的区域。这个区域的代价值从
     * 靠近障碍物的99逐渐降低到膨胀半径边缘的0。这有助于路径规划器
     * 生成更平滑、更安全的路径，避免过于贴近障碍物。
     * @param grid 对栅格地图数据的引用，函数将直接在此数据上进行修改。
     */
    void inflateObstacles(std::vector<int8_t>& grid) {
        // 创建一个临时的代价地图，用于存储膨胀层的代价值，避免迭代时互相干扰。
        std::vector<int8_t> temp_grid(grid.size(), -1); // 初始化为未知
        
        // 第一遍：遍历地图，找到所有原始障碍物，并计算它们的膨胀区域。
        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                if (grid[y * width_ + x] == 100) { // 如果是障碍物
                    // 计算此障碍物影响的膨胀区域边界（一个正方形邻域）
                    int min_x = std::max(0, x - inflation_cells_);
                    int max_x = std::min(width_ - 1, x + inflation_cells_);
                    int min_y = std::max(0, y - inflation_cells_);
                    int max_y = std::min(height_ - 1, y + inflation_cells_);
                    
                    // 在此膨胀区域内，为每个栅格计算代价值
                    for (int ny = min_y; ny <= max_y; ++ny) {
                        for (int nx = min_x; nx <= max_x; ++nx) {
                            // 计算当前栅格(nx, ny)到原始障碍物(x, y)的欧氏距离
                            double dist = resolution_ * std::hypot(nx - x, ny - y);
                            
                            // 如果距离在膨胀半径内
                            if (dist <= inflation_radius_) {
                                // 计算代价值：距离越近，代价值越高（线性递减）
                                double cost_factor = 1.0 - (dist / inflation_radius_);
                                int8_t cost = static_cast<int8_t>(99 * cost_factor);
                                
                                // 更新临时地图。如果一个点受多个障碍物影响，只保留最高的代价值。
                                if (cost > temp_grid[ny * width_ + nx]) {
                                    temp_grid[ny * width_ + nx] = cost;
                                }
                            }
                        }
                    }
                    // 确保原始障碍物本身在临时地图中也被标记为最高代价
                    temp_grid[y * width_ + x] = 100;
                }
            }
        }
        
        // 第二遍：将计算出的膨胀层合并回原始地图。
        for (size_t i = 0; i < grid.size(); ++i) {
            if (temp_grid[i] > 0) {
                // 用膨胀代价值覆盖原有的空闲或未知区域
                grid[i] = temp_grid[i];
            } else if (grid[i] == -1) {
                // 如果没有被膨胀，且原先是未知区域，则保持未知
                grid[i] = -1;
            }
            // 如果原先是空闲(0)，且没有被膨胀，则保持空闲(0)
        }
    }

    /**
     * @brief 发布从 `base_link` 到 `camera_link` 的静态TF变换。
     * @details 静态变换描述了机器人上固定不变的传感器或其他部件的相对位置关系。
     * 此函数在节点启动时被调用一次。
     */
    void publishCameraTransform() {
        geometry_msgs::TransformStamped camera_transform;
        
        camera_transform.header.stamp = ros::Time::now();
        camera_transform.header.frame_id = "car/base_link";     // 父坐标系
        camera_transform.child_frame_id = "car/camera_link";    // 子坐标系
        
        // 设置相机相对于base_link的位置偏移
        camera_transform.transform.translation.x = 0.5;
        camera_transform.transform.translation.y = -0.04;
        camera_transform.transform.translation.z = 0.57;
        
        // 设置相机相对于base_link的姿态（旋转）
        tf2::Quaternion q;
        q.setRPY(0.000, 0.314, 0.000); // Roll, Pitch, Yaw in radians
        camera_transform.transform.rotation = tf2::toMsg(q);
        
        static_broadcaster_.sendTransform(camera_transform);
        
        ROS_INFO("Published static TF: car/base_link -> car/camera_link");
    }

    /**
     * @brief 发布从 `base_link` 到 `laser_link` 的静态TF变换。
     * @details 描述了激光雷达相对于机器人基座的固定位置和姿态。
     * 此函数在节点启动时被调用一次。
     */
    void publishLidarTransform() {
        geometry_msgs::TransformStamped lidar_transform;
        
        lidar_transform.header.stamp = ros::Time::now();
        lidar_transform.header.frame_id = "car/base_link";    // 父坐标系
        lidar_transform.child_frame_id = "car/laser_link";    // 子坐标系
        
        // 设置雷达相对于base_link的位置偏移
        lidar_transform.transform.translation.x = -0.011;
        lidar_transform.transform.translation.y = 0.023;
        lidar_transform.transform.translation.z = 0.480;
        
        // 设置雷达相对于base_link的姿态（旋转180度roll）
        tf2::Quaternion q;
        q.setRPY(M_PI, 0.0, 0.0); // 180 degrees roll
        lidar_transform.transform.rotation = tf2::toMsg(q);
        
        static_broadcaster_.sendTransform(lidar_transform);
        
        ROS_INFO("Published static TF: car/base_link -> car/laser_link");
    }

    /**
     * @brief 点云消息的回调函数，用于生成代价地图。
     * @details 这是节点的核心处理函数。它将输入的点云数据转换为二维栅格地图，
     * 然后调用膨胀函数，最后发布生成的代价地图。
     * @param cloud_msg 从点云话题接收到的消息。
     */
    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
    {
        // 1. 初始化栅格地图，所有栅格默认为"未知"(-1)
        std::vector<int8_t> grid(width_ * height_, -1);

        // 2. 遍历点云中的每一个点
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*cloud_msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*cloud_msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*cloud_msg, "z");
        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
            float x = *iter_x;
            float y = *iter_y;
            float z = *iter_z;

            // 高度滤波：只考虑在特定高度范围内的点作为障碍物
            if (z < 0.1 || z > 2.0) continue;

            // 将点的世界坐标转换为栅格地图坐标
            int ix = static_cast<int>(std::round((x - origin_x_) / resolution_));
            int iy = static_cast<int>(std::round((y - origin_y_) / resolution_));

            // 如果坐标在地图范围内，则标记为"占用"(100)
            if (ix >= 0 && ix < width_ && iy >= 0 && iy < height_) {
                grid[iy * width_ + ix] = 100;
            }
        }

        // 3. 对已标记的障碍物进行膨胀处理
        inflateObstacles(grid);

        // 4. 构造并填充OccupancyGrid消息
        nav_msgs::OccupancyGrid occ;
        occ.header.stamp = cloud_msg->header.stamp;
        occ.header.frame_id = "map"; // 地图的坐标系为map
        occ.info.resolution = resolution_;
        occ.info.width = width_;
        occ.info.height = height_;
        occ.info.origin.position.x = origin_x_;
        occ.info.origin.position.y = origin_y_;
        occ.info.origin.position.z = 0;
        occ.info.origin.orientation.w = 1.0;
        occ.data = grid;

        // 5. 发布最终的代价地图
        pub_.publish(occ);
        
        // 发布调试信息
        std_msgs::Int8 msg;
        msg.data = time_stamp_;
        pub_2.publish(msg);
        
        ROS_DEBUG("Published costmap with origin: (%.2f, %.2f)", origin_x_, origin_y_);
    }

private:
    // --- ROS通信对象 ---
    ros::Subscriber sub_;                                   ///< 点云订阅者
    ros::Subscriber sub_pos;                                ///< 里程计订阅者
    ros::Publisher pub_;                                    ///< 代价地图发布者
    ros::Publisher pub_2;                                   ///< 调试话题发布者
    tf2_ros::TransformBroadcaster tf_broadcaster_;          ///< 动态TF发布者
    tf2_ros::StaticTransformBroadcaster static_broadcaster_;///< 静态TF发布者
    
    // --- 栅格地图参数 ---
    double resolution_;                                     ///< 地图分辨率 (米/栅格)
    int width_;                                             ///< 地图宽度 (栅格数)
    int height_;                                            ///< 地图高度 (栅格数)
    double origin_x_;                                       ///< 地图左下角原点的x坐标 (map系)
    double origin_y_;                                       ///< 地图左下角原点的y坐标 (map系)
    
    // --- 膨胀参数 ---
    double inflation_radius_;                               ///< 膨胀半径 (米)
    int inflation_cells_;                                   ///< 膨胀半径对应的栅格数

    // --- 状态变量 ---
    double time_stamp_;                                     ///< 时间戳（用于调试）
};

/**
 * @brief 主函数
 * @param argc 参数数量
 * @param argv 参数列表
 * @return int 退出码
 */
int main(int argc, char** argv)
{
    ros::init(argc, argv, "localCostMap");
    ros::NodeHandle nh;
    
    LocalCostmapPublisher lcp(nh);
    ros::spin();
    return 0;
}