cmake_minimum_required(VERSION 3.10)
project(BruckAlgorithm)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, defaulting to Release")
  set(CMAKE_BUILD_TYPE Release)
endif()

# Package dependencies
find_package(MPI REQUIRED)

# Add EXEC
add_executable(
  bruck_algorithm 
  main.cc
)

add_executable(
  conways-game
  conway2D.cc
  src/mpi/io_Cartesian.cc
)

# -L Link lib
target_link_libraries(bruck_algorithm PRIVATE MPI::MPI_CXX)
target_link_libraries(conways-game PRIVATE MPI::MPI_CXX)

# -I include directories
target_include_directories(
  bruck_algorithm PRIVATE
  ${MPI_INCLUDE_PATH}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# -I include directories
target_include_directories(
  conways-game PRIVATE
  ${MPI_INCLUDE_PATH}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)