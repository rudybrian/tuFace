cmake_minimum_required(VERSION 2.8)
project( tuFace )
find_package( OpenCV REQUIRED )
add_executable( tuFace tuFace.cpp )
target_link_libraries( tuFace ${OpenCV_LIBS} boost_thread-mt boost_system-mt )

