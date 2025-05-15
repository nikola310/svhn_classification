cmake_minimum_required(VERSION 3.31)

# add Eigen library
set(Eigen3_INCLUDE_DIR "/usr/include/eigen3/")
set(FLARE_INCLUDE_DIR "~/src/FLARE")
include_directories(${Eigen3_INCLUDE_DIR})
include_directories(${FLARE_INCLUDE_DIR})
project(eigen-test)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

add_definitions(-DUSING_OPENCV)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fopenmp")

# Set cmake standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(svhn main.cpp)
target_link_libraries(svhn ${OpenCV_LIBS})
