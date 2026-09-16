/// @brief Exchange variable numbers of complete rows from local 2D matrices.
/// Build: mpicxx -std=c++20 -O2 -I/path/to/include alltoallv_2d_example.cpp -o alltoallv_2d_example
/// Run:   mpirun -np 4 ./alltoallv_2d_example
#include <mpi.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include "mpi/alltoalll_examples.hpp"
#include "mpi/print_in_order.hpp"
#include "mpi/type.hpp"
#include "multiarray.hpp"

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  using value_type = int;  // double
  int result{
    mpi_algorithm::Alltoallv_2d_example<value_type>(MPI_COMM_WORLD)};
  MPI_Finalize();
  return result;
}
