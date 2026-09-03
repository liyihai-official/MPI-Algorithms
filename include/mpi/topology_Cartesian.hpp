#ifndef TOPOLOGY_CARTESIAN_HPP_YIHAI
#define TOPOLOGY_CARTESIAN_HPP_YIHAI

#pragma once
#include <mpi.h>

#include <array>

#include "multiarray.hpp"

namespace mpi_topology
{
template <typename T, size_t NumD>
struct Cartesian
{
  typedef multi_array::multi_array_shape<NumD> array_shape;
  typedef std::array<int, NumD> array_idx;
  typedef std::array<MPI_Datatype, NumD> array_halos;

 public:
  array_shape global_shape, local_shape;
  array_idx starts, ends, dims = {0}, periods = {0};
  array_idx nbr_src, nbr_dest, coordinates;
  array_halos halos;
  int rank;

 public:
  MPI_Comm comm_cart;

 public:
  Cartesian() = default;
  Cartesian(const Cartesian&) = delete;
  Cartesian(Cartesian&&) = delete;

  Cartesian& operator=(const Cartesian&) = delete;
  Cartesian& operator=(Cartesian&&) = delete;
  ~Cartesian() = default;

};  // end of struct Cartesian
}  // end of namespace mpi_topology

#endif  // end if TOPOLOGY_CARTESIAN_HPP_YIHAI