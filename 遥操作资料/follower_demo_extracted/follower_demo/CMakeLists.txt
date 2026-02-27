cmake_minimum_required(VERSION 3.12)

project(demo)


# use c++ 17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)


# find aris
if(UNIX)
	add_definitions(-DUNIX)
        find_package(aris REQUIRED PATHS /usr/aris)
endif(UNIX)
IF(WIN32)
        find_package(aris REQUIRED PATHS C:/aris)
endif(WIN32)
include_directories(${aris_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR})

file(GLOB HEADERS 

    ./*.h ./*.hpp
)
file(GLOB SOURCES   ./*.cpp  ${HEADERS})

add_executable(${PROJECT_NAME} main.cpp ${SOURCES})

if(UNIX)
    target_link_libraries(${PROJECT_NAME} ${aris_LIBRARIES})
endif(UNIX)
IF(WIN32)
    target_link_libraries(${PROJECT_NAME} ${aris_LIBRARIES})
endif(WIN32)








