/// @brief Four-process variable-count exchange using the project's containers.
/// Build: mpicxx -std=c++20 -O2 -I/path/to/include alltoallv_example.cpp -o alltoallv_example
/// Run:   mpirun -np 4 ./alltoallv_example [--cartesian]
#include <mpi.h>

#include <iostream>
#include <stdexcept>
#include <string_view>
#include <type_traits>

#include "mpi/alltoalll_examples.hpp"
#include "mpi/print_in_order.hpp"
#include "mpi/topology_Cartesian.hpp"
#include "mpi/type.hpp"
#include "multiarray.hpp"

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int result{0};
  using value_type = int;  // double also works

  if (argc > 1 && std::string_view(argv[1]) == "--cartesian")
  {
    multi_array::multi_array_shape<2>
      global_shape(8, 8);

    mpi_topology::Cartesian<value_type, 2>
      topology(global_shape, MPI_COMM_WORLD);

    result = mpi_algorithm::Alltoallv_example<value_type>(
      topology.comm_cart);
  }  // Destroy topology (MPI_Comm_free) before MPI_Finalize.
  else
  {
    result = mpi_algorithm::Alltoallv_example<value_type>(MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return result;
}
