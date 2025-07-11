cmake_minimum_required(VERSION 3.10)
project(YOLOv5_CPP)

set(CMAKE_CXX_STANDARD 17)

# Path to your OpenCV
set(OpenCV_DIR "N:/opencv_4.5/opencv/build/x64/vc15/lib")
find_package(OpenCV REQUIRED)

add_executable(yolo_app main.cpp)
target_link_libraries(yolo_app ${OpenCV_LIBS})
include_directories(${OpenCV_INCLUDE_DIRS})

# Copy dependencies (DLL + Haar cascade) correctly
add_custom_command(TARGET yolo_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "N:/opencv_4.5/opencv/build/x64/vc15/bin/opencv_world455.dll"
        "$<TARGET_FILE_DIR:yolo_app>"
)

add_custom_command(TARGET yolo_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/haarcascade_frontalface_default.xml"
        "$<TARGET_FILE_DIR:yolo_app>"
)
