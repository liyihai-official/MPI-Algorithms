
#pragma once
#include <mpi.h>

#include <iostream>
#include <stdexcept>
#include <string_view>
#include <type_traits>

#include "mpi/print_in_order.hpp"
#include "mpi/topology_Cartesian.hpp"
#include "mpi/type.hpp"
#include "multiarray.hpp"

namespace mpi_algorithm
{

/// @brief Each rank sends rank + 2 * dest + 1 elements to dest.
/// @note This deliberately small example requires four processes.
template <typename T>
int Alltoallv_example(MPI_Comm);

/// @brief All ranks send 1, 2, 3 and 4 rows to ranks 0, 1, 2 and 3.
/// @note Each rank owns a separate 10-by-3 matrix. Counts and displacements
///       are numbers of T elements, not numbers of rows. Indices start at zero.
template <typename T>
int Alltoallv_2d_example(MPI_Comm);

}  // namespace mpi_algorithm
