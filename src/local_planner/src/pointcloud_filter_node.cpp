/**
 * @file pointcloud_filter_node.cpp
 * @author BITLZX
 * @date 2025-09-23
 * @brief 该节点用于过滤原始3D激光雷达点云，并将其转换到map坐标系。
 *
 * @details
 * 节点的主要工作流程：
 * 1. **订阅原始点云**: 监听来自激光雷达驱动的`/magv/scan/3d`话题。
 * 2. **初步过滤**: 对点云进行预处理，包括移除NaN点、根据高度阈值过滤掉地面以上的点、根据距离阈值过滤掉离雷达太近的点。
 * 3. **坐标系转换**: 使用TF2将过滤后的点云从其原始坐标系(`car/laser_link`)转换到全局的`map`坐标系。
 * 4. **发布过滤后的点云**: 将最终处理完成、位于`map`坐标系下的点云发布到`/filtered_cloud`话题，供代价地图生成等下游节点使用。
 */

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2/exceptions.h>  

/**
 * @class PointCloudFilter
 * @brief 实现点云过滤和坐标变换的核心逻辑类。
 */
class PointCloudFilter {
public:
    /**
     * @brief PointCloudFilter的构造函数。
     * @details 初始化ROS句柄，加载过滤参数，并设置订阅者和发布者。
     */
    PointCloudFilter() : 
        min_height_(0.1), 
        min_distance_(1.0),
        tf_listener_(tf_buffer_)
    {
        ros::NodeHandle nh;
        ros::NodeHandle private_nh("~");
        
        // 从参数服务器加载过滤参数
        private_nh.param("min_height", min_height_, 0.1);
        private_nh.param("min_distance", min_distance_, 1.0);
        
        // 初始化订阅者和发布者
        sub_ = nh.subscribe<sensor_msgs::PointCloud2>("/magv/scan/3d", 1, &PointCloudFilter::cloudCallback, this);
        pub_ = nh.advertise<sensor_msgs::PointCloud2>("/filtered_cloud", 1);
        
        ROS_INFO("PointCloud Filter initialized.");
        
        // 初始化坐标系名称
        source_frame_ = "car/laser_link";
        target_frame_ = "map";
    }

    /**
     * @brief 点云消息的回调函数，执行所有过滤和转换操作。
     * @param cloud_msg 接收到的原始点云消息。
     */
    void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
        // 1. 转换ROS消息到PCL点云格式
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*cloud_msg, *cloud);
        
        // 2. 移除无效点 (NaN)
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
        if (cloud->empty()) return;
        
        // 3. 应用自定义过滤条件
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& point : *cloud) {
            // 高度过滤 (只保留低于min_height的点)
            if (point.z >= min_height_) continue;
            // 距离过滤 (只保留远于min_distance的点)
            if (sqrt(point.x*point.x + point.y*point.y) <= min_distance_) continue;
            
            filtered_cloud->push_back(point);
        }
        
        if (filtered_cloud->empty()) return;
        
        // 4. 将过滤后的PCL点云转回ROS消息
        sensor_msgs::PointCloud2 filtered_cloud_msg;
        pcl::toROSMsg(*filtered_cloud, filtered_cloud_msg);
        filtered_cloud_msg.header = cloud_msg->header;
        filtered_cloud_msg.header.frame_id = source_frame_;
        
        // 5. 使用TF2将点云转换到目标坐标系(map)
        try {
            geometry_msgs::TransformStamped transform = tf_buffer_.lookupTransform(
                target_frame_, source_frame_, cloud_msg->header.stamp, ros::Duration(0.1));
            
            sensor_msgs::PointCloud2 transformed_cloud;
            tf2::doTransform(filtered_cloud_msg, transformed_cloud, transform);
            
            // 6. 发布最终的点云
            pub_.publish(transformed_cloud);
        }
        catch (const tf2::TransformException &ex) {
            ROS_WARN("Failed to transform point cloud: %s", ex.what());
        }
    }

private:
    ros::Subscriber sub_;       ///< 原始点云订阅者
    ros::Publisher pub_;        ///< 过滤后点云发布者
    double min_height_;         ///< 高度过滤阈值 (米)
    double min_distance_;       ///< 距离过滤阈值 (米)
    
    // TF2相关对象
    tf2_ros::Buffer tf_buffer_; ///< TF缓冲区
    tf2_ros::TransformListener tf_listener_; ///< TF监听器
    std::string source_frame_;  ///< 源坐标系 (通常是传感器坐标系)
    std::string target_frame_;  ///< 目标坐标系 (通常是map或odom)
};

/** @brief 主函数 */
int main(int argc, char** argv) {
    ros::init(argc, argv, "pointcloud_filter_node");
    PointCloudFilter filter;
    ros::spin();
    return 0;
}