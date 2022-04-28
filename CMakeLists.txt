cmake_minimum_required(VERSION 3.16)
project(matrixdefo)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Libigl
option(LIBIGL_GLFW "Build target igl::glfw" ON)
include(libigl)

add_executable(${PROJECT_NAME} matrixdefo.cpp)
target_link_libraries(${PROJECT_NAME} igl::glfw)
