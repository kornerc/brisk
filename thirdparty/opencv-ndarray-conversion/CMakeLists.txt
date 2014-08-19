
add_library(conversion conversion.cpp conversion.h)
target_link_libraries(conversion ${OpenCV_LIBS} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

