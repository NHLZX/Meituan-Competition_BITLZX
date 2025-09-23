cmake_minimum_required(VERSION 3.0.2)
project(magv_msgs)

find_package(
  catkin REQUIRED COMPONENTS 
    message_generation 
    std_msgs 
    geometry_msgs
    nav_msgs
    actionlib_msgs
)

add_message_files(
  DIRECTORY msg
    FILES
      PositionCommand.msg
)

generate_messages(
  DEPENDENCIES 
    std_msgs 
    geometry_msgs 
    nav_msgs
)

catkin_package(
  CATKIN_DEPENDS 
    message_runtime 
    std_msgs 
    geometry_msgs
    nav_msgs
)
