cmake_minimum_required(VERSION 2.4.6)
PROJECT( briskProject )
set(CMAKE_BUILD_TYPE Release)
IF(WIN32 OR APPLE)
	set(BRISK_BUILD_SHARED FALSE)
ELSE()
	set(BRISK_BUILD_SHARED TRUE)
	set(OPENCV_RPATH FALSE)
ENDIF()
    
if(COMMAND cmake_policy)
     cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# Set default compile flags for GCC
add_definitions( -D_REENTRANT -DMX_COMPAT_32 -DMATLAB_MEX_FILE -Wall  )
if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions( -fPIC -msse2 -mssse3 -march=nocona -Wextra -Winit-self -Woverloaded-virtual -Wnon-virtual-dtor -Wsign-promo -Wno-long-long )
else(CMAKE_COMPILER_IS_GNUCXX)
	add_definitions( -fno-strict-aliasing )
endif(CMAKE_COMPILER_IS_GNUCXX)

# Set-up build directory
set(CMAKE_BINARY_DIR ${PROJECT_SOURCE_DIR}/build)

#search OpenCV
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
FIND_PACKAGE(OpenCV REQUIRED)
if(OpenCV2_FOUND)
#MESSAGE(STATUS "Found OpenCV libraries at ${OpenCV2_LIBRARIES} and include files at ${OpenCV2_INCLUDE_DIRS}")
endif(OpenCV2_FOUND)
include_directories(${OpenCV2_INCLUDE_DIRS})

IF(WIN32 OR NOT OPENCV_RPATH)
    set(SPEC_OPENCV_LIBS ${OpenCV2_LIBRARIES})
ELSE()
    set(SPEC_OPENCV_LIBS libopencv_features2d.so libopencv_imgproc.so libopencv_calib3d.so libopencv_highgui.so libopencv_core.so libopencv_flann.so)
ENDIF()

IF (APPLE)
    #where the demo will be placed
    set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/apple/bin)
    #where the libs will be placed
    set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/apple/lib)
    IF(OPENCV_RPATH)
        link_directories(${PROJECT_SOURCE_DIR}/apple/lib)
        #get the rpath right
        SET (CMAKE_EXE_LINKER_FLAGS
            "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,'../lib/'" )
        SET (CMAKE_SHARED_LINKER_FLAGS
            "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,'../lib'" )
    ELSE()
        link_directories(${PROJECT_SOURCE_DIR} ${OpenCV2_ROOT_DIR}/lib)
    ENDIF()
ENDIF ( )
 
IF (WIN32)
    #where the demo will be placed
    set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/win32/bin)
    #where the libs will be placed
    set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/win32/lib)
    link_directories(${PROJECT_SOURCE_DIR}/win32/lib)
ENDIF ( )
 
IF (UNIX AND NOT APPLE)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8) 
        MESSAGE(STATUS "Building for unix into ${PROJECT_SOURCE_DIR}/unix64/bin")
        #where the demo will be placed
        set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/unix64/bin)
        #where the libs will be placed
        set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/unix64/lib)
        IF(OPENCV_RPATH)
        	link_directories(${PROJECT_SOURCE_DIR}/unix64/lib)
        ELSE()
        	link_directories(${PROJECT_SOURCE_DIR} ${OpenCV2_ROOT_DIR}/lib)
        ENDIF()
    ELSE ( )
        MESSAGE(STATUS "Building for unix into ${PROJECT_SOURCE_DIR}/unix32/bin")
        #where the demo will be placed
        set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/unix32/bin)
        #where the libs will be placed
        set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/unix32/lib)
        IF(OPENCV_RPATH)
        	link_directories(${PROJECT_SOURCE_DIR}/unix32/lib)
        ELSE()
        	link_directories(${PROJECT_SOURCE_DIR} ${OpenCV2_ROOT_DIR}/lib)
        ENDIF()
    ENDIF ( )

    #get the rpath right if the libraries to be distributed are built
    IF(OPENCV_RPATH)
        SET (CMAKE_EXE_LINKER_FLAGS
            "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,'../lib/'" )
        SET (CMAKE_SHARED_LINKER_FLAGS
            "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,'../lib'" )
    ENDIF()
ENDIF ( )




#agast
add_subdirectory (${PROJECT_SOURCE_DIR}/thirdparty/agast) 
include_directories (${PROJECT_SOURCE_DIR}/thirdparty/agast/include) 

file(GLOB BRISK_SOURCE_FILES  "${PROJECT_SOURCE_DIR}/src/brisk.cpp")

file(GLOB BRISK_HEADER_FILES  "${PROJECT_SOURCE_DIR}/include/brisk/brisk.h")

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include/)

#build the brisk library dynamic and static versions
IF(BRISK_BUILD_SHARED)
    add_library(brisk SHARED ${BRISK_SOURCE_FILES} ${BRISK_HEADER_FILES})
ELSE(BRISK_BUILD_SHARED)
    add_library(brisk_static STATIC ${BRISK_SOURCE_FILES} ${BRISK_HEADER_FILES})
    TARGET_LINK_LIBRARIES(brisk_static agast_static ${SPEC_OPENCV_LIBS} )
ENDIF(BRISK_BUILD_SHARED)

#build the demo
ADD_EXECUTABLE( demo src/demo.cpp )

#link brisk
IF(BRISK_BUILD_SHARED)
    TARGET_LINK_LIBRARIES( demo ${SPEC_OPENCV_LIBS} brisk agast )
ELSE(BRISK_BUILD_SHARED)
    TARGET_LINK_LIBRARIES( demo brisk_static ${SPEC_OPENCV_LIBS} )
ENDIF(BRISK_BUILD_SHARED)
