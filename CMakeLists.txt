cmake_minimum_required(VERSION 3.10)

#specifies the project name
project(cw1_ICP)

#What is the cmake module path?
#listfiles are files that have cmake code in them and usually have the
#file extension .cmake
#CMAKE_MODULE_PATH defines where to look for more of these .cmake files
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
message(CMAKE_MODULE_PATH=${CMAKE_MODULE_PATH})

#libigl
option(LIBIGL_WITH_OPENGL  "Use OpenGL" ON)
option(LIBIGL_WITH_OPENGL_GLFW  "Use GLFW" ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use GLFW_IMGUI" ON)

find_package(LIBIGL REQUIRED)

#Add in my project files
file(GLOB SRCFILES *.cpp)
file(GLOB INCLUDEFILES *.h)

#Set binary name (the executable file name)
set(EXE_NAME ${PROJECT_NAME}_bin)
set(CMAKE_CXX_STANDARD 11)

#add binary
add_executable(${EXE_NAME} ${SRCFILES} ${INCLUDEFILES})

message(SRCFILES=${SRCFILES})
message(INCLUDEFILES=${INCLUDEFILES})
message(SRCFILES=${SRCFILES})
message(INCLUDEFILES=${INCLUDEFILES})

target_link_libraries(${EXE_NAME}
        igl::core
        igl::opengl
        igl::opengl_glfw
        igl::opengl_glfw_imgui
        )




