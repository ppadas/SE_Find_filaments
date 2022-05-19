cmake_minimum_required(VERSION 3.1)

project(naive_ridge_detection)

find_package(OpenCV REQUIRED)

add_executable(ridges main.cpp)

target_link_libraries(ridges PRIVATE ${OpenCV_LIBS})