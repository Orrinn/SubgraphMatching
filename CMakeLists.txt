cmake_minimum_required(VERSION 3.5)
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
project(proj)

set(CMAKE_CXX_STANDARD 20)
include(CheckCXXCompilerFlag) 

find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(GTest)

find_package(spdlog)

check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_MAVX2)

if(COMPILER_SUPPORTS_MAVX2)
    message(STATUS "Compiler supports -mavx2")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
    set(INTERSECTION_AVX2 1)
else()
    message(WARNING "Compiler does NOT support -mavx2")
    set(INTERSECTION_AVX2 0)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -std=c++20 -O0 -g -Wall -Wno-sign-compare") # for debug -fsanitize=address
    # set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -std=c++20 -O0 -g -Wall -fsanitize=thread -Wno-sign-compare") # for debug and mem
    message(WARNING "Compiling in Debug mode, compile flags: ${CMAKE_CXX_FLAGS_DEBUG}")
    set(DEBUG 1)
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -std=c++20 -O3 -g -Wall -Wno-sign-compare") # for release
    # set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -std=c++20 -O3 -g -Wall -Wno-sign-compare -fdump-tree-original -ftime-report") # for release and template debug
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -O3 -g -Wall -fsanitize=thread ") # for release and mem
    message(WARNING "Compiling in Release mode, compile flags: ${CMAKE_CXX_FLAGS_RELEASE}")
    set(DEBUG 0)
endif()


if(spdlog_FOUND)
    message(STATUS "spdlog found")
    set(SPDLOG_LIBRARIES spdlog::spdlog)
    set(SPDLOG 1)
else()
    message(STATUS "spdlog not found")
    set(SPDLOG_LIBRARIES)
    set(SPDLOG 0)
endif()


SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/libs)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)


include_directories(inc)
include_directories(${PROJECT_BINARY_DIR}/config)
include_directories(${Boost_INCLUDE_DIRS})

if(GTest_FOUND)
    message(STATUS "Google Test found, adding tests...")
    
    include_directories(${GTEST_INCLUDE_DIRS})
   add_subdirectory(test)
    
else()
    message(WARNING "Google Test not found, tests will not be built.")
endif()


configure_file(
  ${PROJECT_SOURCE_DIR}/inc/config.h.in
  ${PROJECT_BINARY_DIR}/config/config.h
)

add_subdirectory(src)

