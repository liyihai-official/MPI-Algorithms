cmake_minimum_required(VERSION 3.15)
project(BruckAlgorithm LANGUAGES CXX)

set(PROJECT_AUTHOR "Yihai Li")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, defaulting to Release")
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(MPI REQUIRED)

# 1. INTERFACE Library)
add_library(common_settings INTERFACE)
target_compile_features(common_settings INTERFACE cxx_std_20)


# Public include
target_include_directories(common_settings INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Warning error
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(common_settings INTERFACE
    -Wall
    -Wextra
    -Wpedantic
    -Werror 
  )
endif()

# Execs
add_executable(bruck_algorithm main.cc)
add_executable(conways-game-2D conway2D.cc src/mpi/io_Cartesian.cc)
add_executable(conways-game-3D conway3D.cc src/mpi/io_Cartesian.cc)

# linkings
set(EXECUTABLES 
  bruck_algorithm 
  conways-game-2D 
  conways-game-3D
)

set(DIM_X 124 CACHE STRING "X dimension for Conway's Game")
set(DIM_Y 124 CACHE STRING "Y dimension for Conway's Game")
set(DIM_Z 124 CACHE STRING "Z dimension for Conway's Game")

set(DATA_TYPE "uint8_t" CACHE STRING "Datatype for elements in Conway's Game")


target_compile_definitions(common_settings INTERFACE
  DIM_X=${DIM_X}
  DIM_Y=${DIM_Y}
  DIM_Z=${DIM_Z}
  ELEMENT_TYPE_MACRO=${DATA_TYPE}
)

foreach(EXEC IN LISTS EXECUTABLES)
  target_link_libraries(${EXEC} PRIVATE common_settings MPI::MPI_CXX)
endforeach()