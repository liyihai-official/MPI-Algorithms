cmake_minimum_required(VERSION 3.15)
project(BruckAlgorithm LANGUAGES CXX)

set(PROJECT_AUTHOR "Yihai Li")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, defaulting to Release")
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(MPI REQUIRED)
option(CONWAY_ENABLE_OPENMP "Enable hybrid MPI + OpenMP/SIMD Conway kernels" ON)
if(CONWAY_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

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

# Compile Optimization Flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Execs
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bruck_Alltoall.cc")
  add_executable(bruck_Alltoall bruck_Alltoall.cc)
else()
  message(STATUS "bruck_Alltoall.cc is absent; skipping its executable")
endif()
add_executable(conways-game-2D conway2D.cc src/mpi/io_Cartesian.cc)
add_executable(conways-game-3D conway3D.cc src/mpi/io_Cartesian.cc)
add_executable(conways-game-2D-legacy conway2D_legacy.cc src/mpi/io_Cartesian.cc)
add_executable(conways-game-3D-legacy conway3D_legacy.cc src/mpi/io_Cartesian.cc)
add_executable(alltoallv_example_01 alltoallv_example.cc src/mpi/alltoall_examples.cc)
add_executable(alltoallv_example_02 alltoallv_example_02.cc src/mpi/alltoall_examples.cc)

# linkings
set(EXECUTABLES 
  conways-game-2D 
  conways-game-3D
  conways-game-2D-legacy
  conways-game-3D-legacy
  alltoallv_example_01
  alltoallv_example_02
)
if(TARGET bruck_Alltoall)
  list(APPEND EXECUTABLES bruck_Alltoall)
endif()

# Compile definition macros
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

add_library(conway_settings INTERFACE)
target_link_libraries(conway_settings INTERFACE common_settings MPI::MPI_CXX)
if(CONWAY_ENABLE_OPENMP)
  target_link_libraries(conway_settings INTERFACE OpenMP::OpenMP_CXX)
endif()
foreach(EXEC conways-game-2D conways-game-3D)
  target_link_libraries(${EXEC} PRIVATE conway_settings)
endforeach()

include(CTest)
if(BUILD_TESTING)
  add_executable(conway_correctness tests/conway_correctness.cc)
  target_link_libraries(conway_correctness PRIVATE conway_settings)
  foreach(RANKS 1 2 4 8)
    foreach(THREADS 1 2)
      math(EXPR TEST_PROCESSORS "${RANKS} * ${THREADS}")
      add_test(NAME conway_${RANKS}r_${THREADS}t
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${RANKS}
          ${MPIEXEC_PREFLAGS} $<TARGET_FILE:conway_correctness> ${MPIEXEC_POSTFLAGS})
      set_tests_properties(conway_${RANKS}r_${THREADS}t PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=${THREADS};OMP_DYNAMIC=FALSE"
        PROCESSORS ${TEST_PROCESSORS} TIMEOUT 60)
    endforeach()
  endforeach()
endif()
