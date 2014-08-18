
cmake_minimum_required(VERSION 2.6)

project(brisk)

find_package(OpenCV REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

add_definitions(-msse2 -mssse3)

# agast
add_subdirectory(thirdparty/agast)
include_directories(thirdparty/agast/include)

# brisk
include_directories(include)
link_directories(${OpenCV_LIB_DIR})
add_library(brisk src/brisk.cpp include/brisk/brisk.h)
target_link_libraries(brisk agast_static ${OpenCV_LIBS})


add_executable(demo src/demo.cpp)
target_link_libraries(demo brisk agast_static ${SPEC_OPENCV_LIBS})
