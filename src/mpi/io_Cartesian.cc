///
/// @file io_Cartesian.cc
//

#include "mpi/io_Cartesian.hpp"

namespace mpi_io
{
///
/// @brief Write Cartesian distributed array to file in binary
///         using MPI-IO @c MPI_File_write_all .
/// @tparam T element datatype
/// @tparam NumD number of dimensions of array.
/// @param array distributed array.
/// @param filename output filename.
///
template <typename T, size_t NumD>
void write_array_Cartesian_io_binary(
  const mpi_array::array_cartesian<T, NumD>& array,
  const std::string& filename)
{
  MPI_File fh{};

  MPI_File_open(  // open file stores in MPI file header.
    array.topology.comm_cart,
    filename.c_str(),
    MPI_MODE_CREATE | MPI_MODE_WRONLY,
    MPI_INFO_NULL,
    &fh);

  // Prepare for MPI_File_write_all usage.
  MPI_Datatype mpi_old_type{mpi_type::mpi_type_traits<T>::get()};

  std::array<int, NumD>  // for datatype used in open file view.
    gsizes{},
    lsizes{},
    file_starts{};

  std::array<int, NumD>  // for datatype used in writing array.
    mem_gsizes{},
    mem_lsizes{},
    mem_starts{};

  for (size_t i = 0; i < NumD; ++i)
  {
    // File view usage
    gsizes[i] = static_cast<int>(array.topology.global_shape.dims[i]);
    lsizes[i] = static_cast<int>(array.topology.local_shape.dims[i]) - 2;

    // Comput the file starts' location - shift 1
    file_starts[i] = static_cast<int>(array.topology.starts[i]) - 1;

    // Writing array usage
    mem_gsizes[i] = static_cast<int>(array.topology.local_shape.dims[i]);
    mem_lsizes[i] = lsizes[i];
    mem_starts[i] = 1;
  }

  // datatype used in MPI file view and write.
  MPI_Datatype filetype{}, memtype{};

  // Set MPI File View.
  MPI_Type_create_subarray(
    NumD,
    gsizes.data(),
    lsizes.data(),
    file_starts.data(),
    MPI_ORDER_C,
    mpi_old_type,
    &filetype);
  MPI_Type_commit(&filetype);

  MPI_File_set_view(
    fh,
    0,
    mpi_old_type,
    filetype,
    "native",
    MPI_INFO_NULL);

  // Commit datatype of writing array.
  MPI_Type_create_subarray(
    NumD,
    mem_gsizes.data(),
    mem_lsizes.data(),
    mem_starts.data(),
    MPI_ORDER_C,
    mpi_old_type,
    &memtype);

  MPI_Type_commit(&memtype);

  // MPI write all
  MPI_File_write_all(
    fh,
    array.current_data.data(),
    1,
    memtype,
    MPI_STATUS_IGNORE);

  // free resources
  MPI_Type_free(&memtype);
  MPI_Type_free(&filetype);

  MPI_File_close(&fh);
}
}  // namespace mpi_io

/// Explicit template instantiation corresponding to mpi_type_traits

template void mpi_io::write_array_Cartesian_io_binary<uint8_t, 2ul>(
  const mpi_array::array_cartesian<uint8_t, 2ul>&,
  const std::string&);

template void mpi_io::write_array_Cartesian_io_binary<uint16_t, 2ul>(
  const mpi_array::array_cartesian<uint16_t, 2ul>&,
  const std::string&);

template void mpi_io::write_array_Cartesian_io_binary<int, 2ul>(
  const mpi_array::array_cartesian<int, 2ul>&,
  const std::string&);

template void mpi_io::write_array_Cartesian_io_binary<float, 2ul>(
  const mpi_array::array_cartesian<float, 2ul>&,
  const std::string&);

template void mpi_io::write_array_Cartesian_io_binary<double, 2ul>(
  const mpi_array::array_cartesian<double, 2ul>&,
  const std::string&);

template void mpi_io::write_array_Cartesian_io_binary<long, 2ul>(
  const mpi_array::array_cartesian<long, 2ul>&,
  const std::string&);