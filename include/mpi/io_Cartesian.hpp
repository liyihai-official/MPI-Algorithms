///
/// @file io_Cartesian.hpp
/// @brief This file contains the implementation of the
///        MPI_IO under Cartesian structure in namespace
///        mpi_io.
///
/// @author Yihai Li
/// @date Sept. 4 2026
///

/// includes
#pragma once
#include <mpi.h>

#include <array>

#include "mpi/topology_Cartesian.hpp"
#include "mpi/type.hpp"

namespace mpi_io
{
template <typename T, size_t NumD>
void write_array_Cartesian_io_binary(
  const mpi_array::array_cartesian<T, NumD>&,
  const std::string&);
}  // namespace mpi_io