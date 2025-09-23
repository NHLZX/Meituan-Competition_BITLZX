cmake_minimum_required(VERSION 3.0.2)
project(yolo_detect)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  message_generation
  tf2_ros
  tf2_geometry_msgs
  cv_bridge
  message_filters
  visualization_msgs
  std_msgs
  tf
)



add_message_files(
  DIRECTORY msg
    FILES
    Detection.msg
    DetectionArray.msg
    FullDetection.msg
    FullDetectionArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)


catkin_package(
  CATKIN_DEPENDS 
    cv_bridge
    geometry_msgs
    message_runtime  
    message_filters
    rospy
    sensor_msgs
    std_msgs
    tf
    tf2_ros
    visualization_msgs
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS scripts/yolo_detector.py
     DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
   )

install(DIRECTORY models/
     DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models
   )
install(DIRECTORY launch/
     DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
