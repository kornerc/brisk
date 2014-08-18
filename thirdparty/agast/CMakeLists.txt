cmake_minimum_required(VERSION 2.4.6)

file(GLOB AGAST_SOURCE_FILES  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc")

file(GLOB AGAST_HEADER_FILES  "${CMAKE_CURRENT_SOURCE_DIR}/include/agast/*.h")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include/agast)

#build the library dynamic and static versions
IF(BRISK_BUILD_SHARED)
    add_library(agast SHARED ${AGAST_SOURCE_FILES} ${AGAST_HEADER_FILES})
ELSE(BRISK_BUILD_SHARED)
    add_library(agast_static STATIC ${AGAST_SOURCE_FILES} ${AGAST_HEADER_FILES})
    target_link_libraries(agast_static ${SPEC_OPENCV_LIBS} )
ENDIF(BRISK_BUILD_SHARED)
