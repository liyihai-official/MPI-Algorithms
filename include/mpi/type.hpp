///
/// @file type.hpp
/// @brief get mpi datatypes based on given C datatype.
///
///

#ifndef TYPE_HPP
#define TYPE_HPP

#pragma once
#include <mpi.h>

namespace mpi_type
{

template <typename T>
struct mpi_type_traits;

template <>
struct mpi_type_traits<uint8_t>
{
  static MPI_Datatype get() { return MPI_UINT8_T; }
};

template <>
struct mpi_type_traits<uint16_t>
{
  static MPI_Datatype get() { return MPI_UINT16_T; }
};

template <>
struct mpi_type_traits<int>
{
  static MPI_Datatype get() { return MPI_INT; }
};

template <>
struct mpi_type_traits<long>
{
  static MPI_Datatype get() { return MPI_LONG; }
};

template <>
struct mpi_type_traits<float>
{
  static MPI_Datatype get() { return MPI_FLOAT; }
};

template <>
struct mpi_type_traits<double>
{
  static MPI_Datatype get() { return MPI_DOUBLE; }
};

}  // end of namespace mpi_type

#endif  // endif