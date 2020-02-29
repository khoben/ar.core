# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.6)


SET(OpenCV_DIR D:/Android/OpenCV42/sdk/native/jni)
find_package(OpenCV REQUIRED)
message(STATUS "opencv found: ${OpenCV_LIBS}")

SET(LIB_SRC_DIR ../..)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OpenCV_DIR}/include/
        ${LIB_SRC_DIR}/src/)

file(GLOB_RECURSE PROJECT_CXX_FILES
        ${LIB_SRC_DIR}/src/*.cpp
        ${LIB_SRC_DIR}/src/*.hpp
        )

list(FILTER PROJECT_CXX_FILES EXCLUDE REGEX ".+main\\.cpp")

add_library(ar_core-lib
        SHARED
        ${PROJECT_CXX_FILES}
        src/main/cpp/android.hpp
        src/main/cpp/android.cpp
        src/main/cpp/native-lib.cpp
        )

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib
        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        ar_core-lib
        # Links the target library to the log library
        # included in the NDK.
        ${log-lib}
        ${OpenCV_LIBS}
        )