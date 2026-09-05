///
/// @file topology_Cartesian.hpp
/// @brief This file is header file of Cartesian structure
///   in mpi_topology
///
/// @author Yihai Li
/// @date Sept. 2 2026
/// @version 1.0
/// @note This file is part of the Bruck Algorithm demonstration
///       and Conway's Life Game project.
///
///
#ifndef TOPOLOGY_CARTESIAN_HPP_YIHAI
#define TOPOLOGY_CARTESIAN_HPP_YIHAI

#pragma once
#include <mpi.h>

#include <array>

#include "mpi/type.hpp"
#include "multiarray.hpp"

/*
 *
 *
 * namespace of mpi topology, which contains types of MPI_Topology
 * structures and provides various features for distributed communication.
 *
 */

namespace mpi_topology
{

///
/// @brief This strucure provides a MPI Topology Cartesian
///         strucure, combining with resource management.
///
/// @tparam T Value type of array
/// @tparam NumD Number of dimensions.
///
template <typename T, size_t NumD>
struct Cartesian
{
  typedef multi_array::multi_array_shape<NumD> array_shape;
  typedef std::array<int, NumD> array_idx;
  typedef std::array<MPI_Datatype, NumD> array_halos;

 public:
  array_shape global_shape, local_shape;
  array_idx starts{}, ends{}, dims{}, periods{};
  array_idx nbr_src{}, nbr_dest{}, coordinates{};
  array_halos halos{};
  int rank{-1}, size{-1};

 public:
  MPI_Comm comm_cart{MPI_COMM_NULL};

 public:
  Cartesian() noexcept = default;

  // MPI Can not be copy (shallow)
  Cartesian(const Cartesian&) = delete;
  Cartesian& operator=(const Cartesian&) = delete;

  // MPI Can be move safely
  Cartesian(Cartesian&&) noexcept;
  Cartesian& operator=(Cartesian&&) noexcept;

  // Customized for MPI resource management
  ~Cartesian();

  // Constructor
  explicit Cartesian(const array_shape&, MPI_Comm);

};  // end of struct Cartesian
}  // end of namespace mpi_topology

/*
 *
 *
 *
 *
 *
 */

#include <unistd.h>

#include <iomanip>
/// @brief
namespace mpi_array
{

template <typename T, size_t NumD>
class array_cartesian
{
  // Private member variables
 public:
  mpi_topology::Cartesian<T, NumD> topology{};
  multi_array::array<T, NumD> current_data{}, next_data{};

  // Cons & Decons
 public:
  array_cartesian(
    const multi_array::multi_array_shape<NumD>& global_shape,
    MPI_Comm comm) : topology(global_shape,
                              comm),
                     current_data(topology.local_shape),
                     next_data(topology.local_shape)
  {
  }

  array_cartesian(const array_cartesian&) = delete;
  array_cartesian(array_cartesian&&) = delete;

  array_cartesian& operator=(const array_cartesian&) = delete;
  array_cartesian& operator=(array_cartesian&&) = delete;

  ~array_cartesian();

  // friend function
 public:
  template <class _T, size_t _NumD>
  friend std::ostream& operator<<(std::ostream&, const array_cartesian<_T, _NumD>&);

  // Communication Methods
 public:
  void commit_halo_mpi_datatypes();
  void exchange_halos_noneblocking();
  void exchange_halos_blocking();

 private:
};  // end of class array_cartesian
}  // end of namespace mpi_array

#include "topology_Cartesian.inl"
#include "topology_Cartesian_exhange.inl"

#endif  // end if TOPOLOGY_CARTESIAN_HPP_YIHAI