cmake_minimum_required(VERSION 3.4.1)
project(ar_core)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

file(GLOB_RECURSE SOURCE_FILES
        "src/*.hpp"
        "src/*.cpp"
        )

set(OpenCV_DIR "D:\\opencv-4.1.1\\build\\install")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(.)
include_directories(src)

add_executable(ar_core ${SOURCE_FILES})

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_videoio opencv_video opencv_xfeatures2d)

target_link_libraries(ar_core ${OpenCV_LIBS})
