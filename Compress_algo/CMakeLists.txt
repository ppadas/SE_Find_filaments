cmake_minimum_required(VERSION 3.1)

project(compress)

find_package(OpenCV REQUIRED)

add_executable(compress main.cpp)

target_link_libraries(compress PRIVATE ${OpenCV_LIBS})