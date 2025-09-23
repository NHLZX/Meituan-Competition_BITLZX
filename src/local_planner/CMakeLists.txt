cmake_minimum_required(VERSION 2.8.3)
project(local_planner)

set(CMAKE_BUILD_TYPE Release)
set(BUILD_STATIC_LIBS ON)
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  std_srvs
  sensor_msgs
  pcl_ros
  nav_msgs
  tf2_ros
  tf2
  magv_msgs
  geometry_msgs
  pcl_conversions
  pcl_ros
  tf2_sensor_msgs
  message_generation
  keyword_extractor
  yolo_detect        
)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

###################################
## catkin specific configuration ##
###################################

catkin_package(
  CATKIN_DEPENDS
  roscpp
  std_msgs
  std_srvs
  sensor_msgs
  pcl_ros
  nav_msgs
  tf2_ros
  tf2
  geometry_msgs
  pcl_conversions
  pcl_ros
  tf2_sensor_msgs
  keyword_extractor
  magv_msgs
  yolo_detect
  image_transport
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  "${PROJECT_SOURCE_DIR}/include"
  /usr/local/include # Location when using 'make system_install'
  /usr/include       # More usual location (e.g. when installing using a package)
)

## Specify additional locations for library files
link_directories(
  /usr/local/lib # Location when using 'make system_install'
  /usr/lib       # More usual location (e.g. when installing using a package)
)

## Declare executables
add_executable(localPlanner src/localPlanner.cpp)
add_executable(pathFollower src/pathFollower.cpp)
add_executable(localCostMap src/localCostMap.cpp)
add_executable(pointcloud_filter_node src/pointcloud_filter_node.cpp)
add_executable(dwaPlanner src/dwaPlanner.cpp)
add_executable(goalAdv src/goalAdv.cpp)
add_executable(state_machine src/state_machine.cpp)


## Specify libraries to link a library or executable target against
target_link_libraries(localPlanner ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(pathFollower ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(localCostMap ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(pointcloud_filter_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(dwaPlanner ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(goalAdv ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(state_machine ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})

add_dependencies(state_machine
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
  keyword_extractor_generate_messages_cpp # 添加消息生成依赖
  yolo_detect_generate_messages_cpp       # 添加消息生成依赖
)

install(TARGETS localPlanner pathFollower localCostMap pointcloud_filter_node dwaPlanner goalAdv state_machine
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
install(DIRECTORY paths/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/paths
)

