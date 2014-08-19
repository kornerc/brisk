
cmake_minimum_required(VERSION 2.6)

project(brisk)

find_package(OpenCV REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

add_definitions(-msse2 -mssse3)

option(BUILD_PYTHON_BINDING "Build the Python binding" ON)

# agast
add_subdirectory(thirdparty/agast)
include_directories(thirdparty/agast/include)

# brisk
include_directories(include)
link_directories(${OpenCV_LIB_DIR})
add_library(brisk src/brisk.cpp include/brisk/brisk.h)
target_link_libraries(brisk agast_static ${OpenCV_LIBS})

# brisk-demo
add_executable(demo src/demo.cpp)
target_link_libraries(demo brisk agast_static ${OPENCV_LIBS})

add_custom_command(TARGET demo PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory
                   ${CMAKE_SOURCE_DIR}/images ${PROJECT_BINARY_DIR}/res/images)

# opencv-ndarray-conversion
if(BUILD_PYTHON_BINDING)
    find_package(PythonLibs REQUIRED)
    if(PYTHONLIBS_FOUND)
        include_directories(${PYTHON_INCLUDE_DIRS})
    else()
        message(FATAL_ERROR "Unable to find PythonLibs.")
    endif()

    find_package(Boost)
    if(Boost_FOUND)
        include_directories(${Boost_INCLUDE_DIRS})
        set(Boost_USE_STATIC_LIBS OFF)
        set(Boost_USE_MULTITHREADED ON)
        set(Boost_USE_STATIC_RUNTIME OFF)
        find_package(Boost COMPONENTS python)
    else()
        message(FATAL_ERROR "Unable to find Boost.")
    endif()

    add_subdirectory(thirdparty/opencv-ndarray-conversion)

    include_directories(thirdparty/opencv-ndarray-conversion)

    add_library(pybrisk SHARED src/pybrisk.cpp)
    target_link_libraries(pybrisk brisk agast_static conversion ${OpenCV_LIBS} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
    set_target_properties(pybrisk PROPERTIES PREFIX "")

    add_custom_command(TARGET pybrisk PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_SOURCE_DIR}/src/brisk.py $<TARGET_FILE_DIR:pybrisk>)
endif()

