//该节点单纯过滤点云，并发布一些坐标系转换，已经完善
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2/exceptions.h>  // 添加这个头文件解决异常问题

class PointCloudFilter {
public:
    PointCloudFilter() : 
        min_height_(0.1), 
        min_distance_(1.0),
        tf_listener_(tf_buffer_)
    {
        // 初始化节点
        ros::NodeHandle nh;
        ros::NodeHandle private_nh("~");
        
        // 获取参数（如果没有设置则使用默认值）
        private_nh.param("min_height", min_height_, 0.1);
        private_nh.param("min_distance", min_distance_, 1.0);
        
        // 订阅原始点云
        sub_ = nh.subscribe<sensor_msgs::PointCloud2>(
            "/magv/scan/3d", 1, &PointCloudFilter::cloudCallback, this);
        
        // 发布过滤后的点云（在map坐标系）
        pub_ = nh.advertise<sensor_msgs::PointCloud2>("/filtered_cloud", 1);
        
        ROS_INFO("PointCloud Filter initialized with parameters:");
        ROS_INFO_STREAM("\tmin_height: " << min_height_);
        ROS_INFO_STREAM("\tmin_distance: " << min_distance_);
        
        // 初始化坐标系名称
        source_frame_ = "car/laser_link";
        target_frame_ = "map";
    }

    void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
        // 转换ROS消息到PCL点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*cloud_msg, *cloud);
        
        // 移除NaN点
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
        
        if (cloud->empty()) {
            ROS_DEBUG("No valid points in point cloud");
            return;
        }
        
        // 创建过滤后的点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        
        // 应用过滤条件
        for (const auto& point : *cloud) {
            // 高度过滤 (z < min_height)
            if (point.z >= min_height_) continue;
            
            // 距离过滤 (sqrt(x²+y²) > min_distance)
            float distance = sqrt(point.x*point.x + point.y*point.y);
            if (distance <= min_distance_) continue;
            
            filtered_cloud->push_back(point);
        }
        
        // 处理空点云情况
        if (filtered_cloud->empty()) {
            ROS_DEBUG("All points filtered out");
            return;
        }
        
        // 转换为ROS消息（保持在原始坐标系）
        sensor_msgs::PointCloud2 filtered_cloud_msg;
        pcl::toROSMsg(*filtered_cloud, filtered_cloud_msg);
        filtered_cloud_msg.header = cloud_msg->header;
        filtered_cloud_msg.header.frame_id = source_frame_; // 确保是car/laser_link
        
        try {
            // 获取从car/laser_link到map的变换
            geometry_msgs::TransformStamped transform = tf_buffer_.lookupTransform(
                target_frame_, source_frame_, cloud_msg->header.stamp, ros::Duration(0.1));
            
            // 将点云转换到map坐标系
            sensor_msgs::PointCloud2 transformed_cloud;
            tf2::doTransform(filtered_cloud_msg, transformed_cloud, transform);
            transformed_cloud.header.frame_id = target_frame_;
            
            // 发布转换后的点云
            pub_.publish(transformed_cloud);
            
            // 节流日志输出
            ROS_DEBUG_THROTTLE(5.0, "Transformed %lu points to map frame", filtered_cloud->size());
        }
        catch (const tf2::TransformException &ex) {  // 使用const引用
            ROS_WARN_STREAM("Failed to transform point cloud: " << ex.what());
        }
    }

private:
    ros::Subscriber sub_;
    ros::Publisher pub_;
    double min_height_;     // 高度阈值 (米)
    double min_distance_;   // 距离阈值 (米)
    
    // TF2相关对象
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::string source_frame_;  // 源坐标系 (car/laser_link)
    std::string target_frame_;  // 目标坐标系 (map)
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "pointcloud_filter_node");
    PointCloudFilter filter;
    ros::spin();
    return 0;
}
// #include <ros/ros.h>
// #include <sensor_msgs/PointCloud2.h>
// #include <pcl_conversions/pcl_conversions.h>
// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>
// #include <pcl/filters/filter.h>
// #include <pcl/common/common.h>
// #include <tf2_ros/transform_listener.h>
// #include <tf2_ros/buffer.h>
// #include <tf2_sensor_msgs/tf2_sensor_msgs.h>
// #include <tf2/exceptions.h>  // 添加这个头文件解决异常问题

// class PointCloudFilter {
// public:
//     PointCloudFilter() : min_height_(0.1), min_distance_(1.0) {
//         // 初始化节点
//         ros::NodeHandle nh;
//         ros::NodeHandle private_nh("~");
        
//         // 获取参数（如果没有设置则使用默认值）
//         private_nh.param("min_height", min_height_, 0.1);
//         private_nh.param("min_distance", min_distance_, 1.0);
        
//         // 订阅原始点云
//         sub_ = nh.subscribe<sensor_msgs::PointCloud2>(
//             "/magv/scan/3d", 1, &PointCloudFilter::cloudCallback, this);
        
//         // 发布过滤后的点云
//         pub_ = nh.advertise<sensor_msgs::PointCloud2>("/filtered_cloud", 1);
        
//         ROS_INFO("PointCloud Filter initialized with parameters:");
//         ROS_INFO_STREAM("\tmin_height: " << min_height_);
//         ROS_INFO_STREAM("\tmin_distance: " << min_distance_);
//         // 初始化坐标系名称
//         source_frame_ = "car/laser_link";
//         target_frame_ = "map";
//     }

//     void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
//         // 转换ROS消息到PCL点云
//         pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
//         pcl::fromROSMsg(*cloud_msg, *cloud);
        
//         // 移除NaN点
//         std::vector<int> indices;
//         pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
        
//         if (cloud->empty()) {
//             ROS_DEBUG("No valid points in point cloud");
//             //publishEmptyCloud(cloud_msg->header);
//             return;
//         }
        
//         // 创建过滤后的点云
//         pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        
//         // 应用过滤条件
//         for (const auto& point : *cloud) {
//             // 高度过滤 (z < min_height)
//             if (point.z >= min_height_) continue;
            
//             // 距离过滤 (sqrt(x²+y²+z²) > min_distance)
//             float distance = sqrt(point.x*point.x + point.y*point.y + point.z*point.z);
//             if (distance <= min_distance_) continue;
            
//             filtered_cloud->push_back(point);
//         }
        
//         // // 处理空点云情况
//         // if (filtered_cloud->empty()) {
//         //     ROS_DEBUG("All points filtered out");
//         //     publishEmptyCloud(cloud_msg->header);
//         //     return;
//         // }

//          if (filtered_cloud->empty()) {
//             ROS_DEBUG("All points filtered out");
//             return;
//         }

//         sensor_msgs::PointCloud2 filtered_cloud_msg;
//         pcl::toROSMsg(*filtered_cloud, filtered_cloud_msg);
//         filtered_cloud_msg.header = cloud_msg->header;
//         filtered_cloud_msg.header.frame_id = source_frame_; // 确保是car/laser_link
        
//         try {
//             // 获取从car/laser_link到map的变换
//             geometry_msgs::TransformStamped transform = tf_buffer_.lookupTransform(
//                 target_frame_, source_frame_, cloud_msg->header.stamp, ros::Duration(0.1));
            
//             // 将点云转换到map坐标系
//             sensor_msgs::PointCloud2 transformed_cloud;
//             tf2::doTransform(filtered_cloud_msg, transformed_cloud, transform);
//             transformed_cloud.header.frame_id = target_frame_;
            
//             // 发布转换后的点云
//             pub_.publish(transformed_cloud);
//             // 节流日志输出
//             ROS_DEBUG_THROTTLE(5.0, "Transformed %lu points to map frame", filtered_cloud->size());
//         }
//         catch (tf2::TransformException &ex) {
//             ROS_WARN_STREAM("Failed to transform point cloud: " << ex.what());
//         }
//         // // 转换为ROS消息
//         // sensor_msgs::PointCloud2 output;
//         // pcl::toROSMsg(*filtered_cloud, output);
//         // // 设置消息头（更改坐标系）
//         // output.header.stamp = cloud_msg->header.stamp;
//         // output.header.frame_id = "car/laser_link";
//         // 发布
//         // pub_.publish(output);
//         // 节流日志输出
//         ROS_DEBUG_THROTTLE(5.0, "Filtered %lu -> %lu points", 
//                           cloud->size(), filtered_cloud->size());
//     }

// private:
//     // void publishEmptyCloud(const std_msgs::Header& header) {
//     //     pcl::PointCloud<pcl::PointXYZ> empty_cloud;
//     //     sensor_msgs::PointCloud2 output;
//     //     pcl::toROSMsg(empty_cloud, output);
//     //     output.header.stamp = header.stamp;
//     //     output.header.frame_id = "car/laser_link";
//     //     pub_.publish(output);
//     // }
//     ros::Subscriber sub_;
//     ros::Publisher pub_;
//     double min_height_;     // 高度阈值 (米)
//     double min_distance_;   // 距离阈值 (米)
//     std::string source_frame_;  // 源坐标系 (car/laser_link)
//     std::string target_frame_;  // 目标坐标系 (map)
//     // TF2相关对象
//     tf2_ros::Buffer tf_buffer_;
//     tf2_ros::TransformListener tf_listener_;
// };

// int main(int argc, char** argv) {
//     ros::init(argc, argv, "pointcloud_filter_node");
//     PointCloudFilter filter;
//     ros::spin();
//     return 0;
// }
