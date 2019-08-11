cmake_minimum_required(VERSION 3.14)
project(ar_core)

set(CMAKE_CXX_STANDARD 14)

set(OpenCV_DIR "D:\\opencv-4.1.1\\build\\install")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(.)
include_directories(src)

add_executable(ar_core
        src/Utils/CvUtils.hpp
        src/Recognition/Recognition.hpp
        src/Tracking/TrackerEntity.hpp
        src/Tracking/Tracking.hpp
        src/main.cpp
        src/Recognition/Recognition.cpp
        src/Tracking/TrackerEntity.cpp
        src/Tracking/Tracking.cpp)

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_videoio opencv_video)

target_link_libraries(ar_core ${OpenCV_LIBS})
