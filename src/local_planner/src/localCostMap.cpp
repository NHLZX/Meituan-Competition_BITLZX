//该节点剩余工作：发布坐标转换（base_link->carme_link）
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
#include <tf2_ros/static_transform_broadcaster.h> // 添加静态变换广播器头文件
#include <tf2/LinearMath/Quaternion.h>            // 添加四元数头文件


class LocalCostmapPublisher {
public:

    LocalCostmapPublisher(ros::NodeHandle& nh)
    {
        // 固定参数设置
        resolution_ = 0.1;      // 米/栅格 (分辨率)
        width_ = 120;            // 栅格数 (宽度
        height_ = 120;           // 栅格数 (高度)
        origin_x_ = 0;           // 初始原点x
        origin_y_ = 0;           // 初始原点y
        
        // 膨胀参数设置（基于小车半径0.6m）
        inflation_radius_ = 0.63; // 膨胀半径（0.6m车身半径 + 0.1m安全裕度）
        inflation_cells_ = static_cast<int>(std::ceil(inflation_radius_ / resolution_));

        
        // 订阅者和发布者初始化
        sub_ = nh.subscribe("/filtered_cloud", 1, &LocalCostmapPublisher::cloudCallback, this);
        sub_pos = nh.subscribe<nav_msgs::Odometry>("/magv/odometry/gt", 5, &LocalCostmapPublisher::posCallback, this);
        pub_ = nh.advertise<nav_msgs::OccupancyGrid>("/local_costmap_demo", 1);
        pub_2 = nh.advertise<std_msgs::Int8>("/my_int_topic2", 10);

        publishLidarTransform();
        publishCameraTransform(); 

        ROS_INFO("LocalCostmapPublisher initialized");
        ROS_INFO("  Resolution: %.2f m/cell", resolution_);
        ROS_INFO("  Grid size: %d x %d cells", width_, height_);
        ROS_INFO("  Inflation radius: %.2f m (%d cells)", inflation_radius_, inflation_cells_);
    }

    void posCallback(const nav_msgs::Odometry::ConstPtr &odom)
    {
	    time_stamp_ = odom->header.stamp.toSec();
        double current_x_ = odom->pose.pose.position.x;
        double current_y_ = odom->pose.pose.position.y;
        // 计算新原点使小车位于地图中心
        origin_x_ = current_x_ - (width_ * resolution_) / 2.0;
        origin_y_ = current_y_ - (height_ * resolution_) / 2.0;

        //map->odom
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
        
        // 创建并发布odom->base_link变换
        geometry_msgs::TransformStamped odom_to_base;
        odom_to_base.header.stamp = odom->header.stamp;
        odom_to_base.header.frame_id = "odom";
        odom_to_base.child_frame_id = "car/base_link";
        odom_to_base.transform.translation.x = odom->pose.pose.position.x;
        odom_to_base.transform.translation.y = odom->pose.pose.position.y;
        odom_to_base.transform.translation.z = odom->pose.pose.position.z;
        odom_to_base.transform.rotation.x = odom->pose.pose.orientation.x;
        odom_to_base.transform.rotation.y = odom->pose.pose.orientation.y;
        odom_to_base.transform.rotation.z = odom->pose.pose.orientation.z;
        odom_to_base.transform.rotation.w = odom->pose.pose.orientation.w;
        tf_broadcaster_.sendTransform(odom_to_base);
        
    }
    
    // 膨胀障碍物函数
    void inflateObstacles(std::vector<int8_t>& grid) {
        // 创建临时网格用于膨胀计算
        std::vector<int8_t> temp_grid(grid.size(), -1);
        
        // 第一遍：标记原始障碍物
        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                if (grid[y * width_ + x] == 100) {
                    // 计算膨胀区域边界
                    int min_x = std::max(0, x - inflation_cells_);
                    int max_x = std::min(width_ - 1, x + inflation_cells_);
                    int min_y = std::max(0, y - inflation_cells_);
                    int max_y = std::min(height_ - 1, y + inflation_cells_);
                    
                    // 在膨胀区域内标记
                    for (int ny = min_y; ny <= max_y; ++ny) {
                        for (int nx = min_x; nx <= max_x; ++nx) {
                            // 计算实际距离（米）
                            double dist = resolution_ * std::hypot(nx - x, ny - y);
                            
                            // 如果在膨胀半径内且不是原始障碍物
                            if (dist <= inflation_radius_ && temp_grid[ny * width_ + nx] != 100) {
                                // 计算膨胀区域的代价值（距离越近值越高）
                                double cost_factor = 1.0 - (dist / inflation_radius_);
                                int8_t cost = static_cast<int8_t>(99 * cost_factor);
                                
                                // 保留最高值（避免被低值覆盖）
                                if (cost > temp_grid[ny * width_ + nx]) {
                                    temp_grid[ny * width_ + nx] = cost;
                                }
                            }
                        }
                    }
                    
                    // 确保原始障碍物保留（最高值100）
                    temp_grid[y * width_ + x] = 100;
                }
            }
        }
        
        // 第二遍：合并原始网格和膨胀网格
        for (size_t i = 0; i < grid.size(); ++i) {
            if (temp_grid[i] > 0) {
                // 膨胀区域覆盖未知区域和空闲区域
                grid[i] = temp_grid[i];
            } else if (grid[i] == -1) {
                // 保留未知区域
                grid[i] = -1;
            }
        }
    }

    // 发布相机TF变换
    void publishCameraTransform() {
        geometry_msgs::TransformStamped camera_transform;
        
        // 设置变换头信息
        camera_transform.header.stamp = ros::Time::now();
        camera_transform.header.frame_id = "car/base_link";     // 父坐标系
        camera_transform.child_frame_id = "car/camera_link";    // 子坐标系
        
        // 设置相机位置偏移 [0.5, -0.04, 0.57] 米
        camera_transform.transform.translation.x = 0.5;
        camera_transform.transform.translation.y = -0.04;
        camera_transform.transform.translation.z = 0.57;
        
        // 设置相机姿态 [0.000, 0.314, 0.000] 弧度 (roll, pitch, yaw)
        tf2::Quaternion q;
        q.setRPY(0.000, 0.314, 0.000);  // 直接使用给定的弧度值
        camera_transform.transform.rotation.x = q.x();
        camera_transform.transform.rotation.y = q.y();
        camera_transform.transform.rotation.z = q.z();
        camera_transform.transform.rotation.w = q.w();
        
        // 发布静态变换
        static_broadcaster_.sendTransform(camera_transform);
        
        ROS_INFO("Published static TF: car/base_link -> car/camera_link");
        ROS_INFO("  Translation: %.3f, %.3f, %.3f", 
                 camera_transform.transform.translation.x,
                 camera_transform.transform.translation.y,
                 camera_transform.transform.translation.z);
        ROS_INFO("  Rotation: %.3f, %.3f, %.3f, %.3f (RPY: %.3f, %.3f, %.3f rad)",
                 camera_transform.transform.rotation.x,
                 camera_transform.transform.rotation.y,
                 camera_transform.transform.rotation.z,
                 camera_transform.transform.rotation.w,
                 0.000, 0.314, 0.000);
    }

    // 发布雷达静态TF变换
    void publishLidarTransform() {
        geometry_msgs::TransformStamped lidar_transform;
        
        // 设置变换头信息
        lidar_transform.header.stamp = ros::Time::now();
        lidar_transform.header.frame_id = "car/base_link";    // 父坐标系
        lidar_transform.child_frame_id = "car/laser_link";    // 子坐标系
        
        // 设置雷达位置偏移 [ -0.011, 0.023, 0.480 ] 米
        lidar_transform.transform.translation.x = -0.011;
        lidar_transform.transform.translation.y = 0.023;
        lidar_transform.transform.translation.z = 0.480;
        
        // 设置雷达姿态 [ 180.000, 0.000, 0.000 ] 度
        // 将欧拉角转换为四元数
        tf2::Quaternion q;
        q.setRPY(
            M_PI,   // 滚转角 180度（π弧度）
            0.0,    // 俯仰角 0度
            0.0     // 偏航角 0度
        );
        lidar_transform.transform.rotation.x = q.x();
        lidar_transform.transform.rotation.y = q.y();
        lidar_transform.transform.rotation.z = q.z();
        lidar_transform.transform.rotation.w = q.w();
        
        // 发布静态变换
        static_broadcaster_.sendTransform(lidar_transform);
        
        ROS_INFO("Published static TF: car/base_link -> car/laser_link");
        ROS_INFO("  Translation: %.3f, %.3f, %.3f", 
                 lidar_transform.transform.translation.x,
                 lidar_transform.transform.translation.y,
                 lidar_transform.transform.translation.z);
        ROS_INFO("  Rotation: %.3f, %.3f, %.3f, %.3f",
                 lidar_transform.transform.rotation.x,
                 lidar_transform.transform.rotation.y,
                 lidar_transform.transform.rotation.z,
                 lidar_transform.transform.rotation.w);
    }

    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
    {
        // 初始化为未知(-1)
        std::vector<int8_t> grid(width_ * height_, -1);

        // 遍历点云
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*cloud_msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*cloud_msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*cloud_msg, "z");
        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
            float x = *iter_x;
            float y = *iter_y;
            float z = *iter_z;

            // 只考虑地面附近障碍（0.1m - 2.0m高度）
            if (z < 0.1 || z > 2.0) continue;

            int ix = static_cast<int>(std::round((x - origin_x_) / resolution_));
            int iy = static_cast<int>(std::round((y - origin_y_) / resolution_));
            if (ix >= 0 && ix < width_ && iy >= 0 && iy < height_) {
                grid[iy * width_ + ix] = 100; // 占用
            }
        }

        // 添加障碍物膨胀（考虑小车半径0.6m）
        inflateObstacles(grid);

        // 构造OccupancyGrid消息
        nav_msgs::OccupancyGrid occ;
        occ.header.stamp = cloud_msg->header.stamp;
        occ.header.frame_id = "map";
        occ.info.resolution = resolution_;
        occ.info.width = width_;
        occ.info.height = height_;
        occ.info.origin.position.x = origin_x_;
        occ.info.origin.position.y = origin_y_;
        occ.info.origin.position.z = 0;
        occ.info.origin.orientation.w = 1.0;
        
        occ.data = grid;

        pub_.publish(occ);
        std_msgs::Int8 msg;
        msg.data = time_stamp_;
        pub_2.publish(msg);
        
        ROS_DEBUG("Published costmap with origin: (%.2f, %.2f)", origin_x_, origin_y_);
    }

private:
    ros::Subscriber sub_;
    ros::Subscriber sub_pos;
    ros::Publisher pub_2;
    ros::Publisher pub_;
    
    // 栅格地图参数
    double resolution_;
    int width_;
    int height_;
    double origin_x_;
    double origin_y_;
    
    double time_stamp_;
    // 膨胀参数
    double inflation_radius_;  // 膨胀半径（米）
    int inflation_cells_;      // 膨胀栅格数

    tf2_ros::TransformBroadcaster tf_broadcaster_;  
    tf2_ros::StaticTransformBroadcaster static_broadcaster_; 
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "localCostMap");
    ros::NodeHandle nh;
    
    LocalCostmapPublisher lcp(nh);
    ros::spin();
    return 0;
}



