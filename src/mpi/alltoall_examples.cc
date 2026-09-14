#include "mpi/alltoalll_examples.hpp"

namespace mpi_algorithm
{
/// @brief Each rank sends rank + 2 * dest + 1 elements to dest.
/// @note This deliberately small example requires four processes.
template <typename T>
int Alltoallv_example(MPI_Comm comm)
{
  int rank{-1}, num_procs{0};
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &num_procs);

  if (num_procs != 4)
  {
    if (rank == 0)
      std::cerr << "Run this example with -np 4.\n";
    return 1;
  }

  multi_array::array<int, 1>
    sendcounts(num_procs),
    recvcounts(num_procs);

  multi_array::array<int, 1>
    sdispls(num_procs),
    rdispls(num_procs);

  for (int dest = 0; dest < num_procs; ++dest)
    sendcounts[dest] = rank + 2 * dest + 1;

  // Exchange one count per peer; counts always use the int MPI datatype.
  MPI_Datatype
    count_type{mpi_type::mpi_type_traits<int>::get()};
  MPI_Alltoall(
    sendcounts.data(),
    1,
    count_type,
    recvcounts.data(),
    1,
    count_type,
    comm);

  int total_send{0}, total_recv{0};
  for (int peer = 0; peer < num_procs; ++peer)
  {
    sdispls[peer] = total_send;
    rdispls[peer] = total_recv;

    total_send += sendcounts[peer];
    total_recv += recvcounts[peer];
  }

  // Direct construction avoids the uploaded array's assignment operators.
  multi_array::array<T, 1>
    send_arr(total_send),
    recv_arr(total_recv);

  for (int dest = 0; dest < num_procs; ++dest)
    for (int j = 0; j < sendcounts[dest]; ++j)
      send_arr[sdispls[dest] + j] = static_cast<T>(rank * 1000 + dest * 100 + j);

  mpi_io::print_multiarray_in_order(
    send_arr,
    comm);
  mpi_io::print_alltoallv_in_order(
    send_arr,
    sendcounts,
    sdispls,
    1,
    comm);

  // Displacements are in base_type extents, which here are T elements.
  MPI_Datatype base_type{mpi_type::mpi_type_traits<T>::get()};

  MPI_Alltoallv(
    send_arr.data(),
    sendcounts.data(),
    sdispls.data(),
    base_type,
    recv_arr.data(),
    recvcounts.data(),
    rdispls.data(),
    base_type,
    comm);

  int local_ok{1}, all_ok{0};
  for (int src = 0; src < num_procs; ++src)
  {
    if (recvcounts[src] != src + 2 * rank + 1)
      local_ok = 0;

    for (int j = 0; j < recvcounts[src]; ++j)
      if (recv_arr[rdispls[src] + j] != static_cast<T>(src * 1000 + rank * 100 + j))
        local_ok = 0;
  }
  MPI_Allreduce(
    &local_ok,
    &all_ok,
    1,
    count_type,
    MPI_MIN,
    comm);

  // Show one rank's complete layout, avoiding interleaved process output.
  mpi_io::print_alltoallv_in_order(
    recv_arr,
    recvcounts,
    rdispls,
    0,
    comm);
  mpi_io::print_multiarray_in_order(
    recv_arr,
    comm);

  return all_ok ? 0 : 1;
}

/// @brief All ranks send 1, 2, 3 and 4 rows to ranks 0, 1, 2 and 3.
/// @note Each rank owns a separate 10-by-3 matrix. Counts and displacements
///       are numbers of T elements, not numbers of rows. Indices start at zero.
template <typename T>
int Alltoallv_2d_example(MPI_Comm comm)
{
  int rank{-1}, num_procs{0};
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &num_procs);
  if (num_procs != 4)
  {
    if (rank == 0) std::cerr << "Run this example with -np 4.\n";
    return 1;
  }

  constexpr int ncols{3};
  constexpr int send_nrows{10};  // 1 + 2 + 3 + 4
  multi_array::array<T, 2>
    send_arr(send_nrows, ncols);

  // Values identify the original source rank, local row and column.
  for (int row = 0; row < send_nrows; ++row)
    for (int col = 0; col < ncols; ++col)
      send_arr(row, col) = static_cast<T>(1000 * rank + 10 * row + col);

  multi_array::array<int, 1>
    sendcounts(num_procs),
    recvcounts(num_procs);
  multi_array::array<int, 1>
    sdispls(num_procs),
    rdispls(num_procs);

  for (int dest = 0; dest < num_procs; ++dest)
  {
    int rows_to_dest{dest + 1};
    sendcounts[dest] = rows_to_dest * ncols;
  }

  MPI_Datatype count_type{mpi_type::mpi_type_traits<int>::get()};
  MPI_Alltoall(
    sendcounts.data(),
    1,
    count_type,
    recvcounts.data(),
    1,
    count_type,
    comm);

  int total_send{0}, total_recv{0};
  for (int peer = 0; peer < num_procs; ++peer)
  {
    sdispls[peer] = total_send;
    rdispls[peer] = total_recv;
    total_send += sendcounts[peer];
    total_recv += recvcounts[peer];
  }

  // The common column count makes every received block a whole set of rows.
  int recv_nrows{total_recv / ncols};
  multi_array::array<T, 2> recv_arr(recv_nrows, ncols);

  auto print_matrix = [&](
                        const multi_array::array<T, 2>& matrix,
                        const multi_array::array<int, 1>& counts,
                        const multi_array::array<int, 1>& displs,
                        bool sending)
  {
    std::ostringstream local;
    local << "shape: " << matrix.shape().dims[0] << " x " << ncols << '\n';
    local << (sending ? "sendcounts: " : "recvcounts: ") << counts;
    local << (sending ? "sdispls:    " : "rdispls:    ") << displs;
    for (int peer = 0; peer < num_procs; ++peer)
    {
      int first_row{displs[peer] / ncols};
      int rows{counts[peer] / ncols};
      local << (sending ? "  to rank " : "  from rank ") << peer
            << " | local rows " << first_row << ".." << first_row + rows - 1
            << " | " << rows << " rows, " << counts[peer] << " elements\n";
    }
    // Explicit separators keep floating-point entries distinct even when
    // their text is wider than the requested field width.
    local << "matrix (rows numbered from 0):\n"
          << std::defaultfloat;

    for (size_t row = 0; row < matrix.shape().dims[0]; ++row)
    {
      for (int col = 0; col < ncols; ++col)
        local << std::setw(10)
              << +matrix(row, col) << ' ';
      local << '\n';
    }
    mpi_io::print_in_order(
      local.str(),
      comm,
      sending ? "BEFORE: send matrices" : "AFTER: receive matrices");
  };

  print_matrix(send_arr, sendcounts, sdispls, true);

  MPI_Datatype base_type{mpi_type::mpi_type_traits<T>::get()};
  MPI_Alltoallv(
    send_arr.data(),
    sendcounts.data(),
    sdispls.data(),
    base_type,
    recv_arr.data(),
    recvcounts.data(),
    rdispls.data(),
    base_type,
    comm);

  print_matrix(recv_arr, recvcounts, rdispls, false);

  // Check every received cell against its original source row and column.
  int local_ok{1}, all_ok{0};
  int source_first_row{rank * (rank + 1) / 2};
  if (total_send != static_cast<int>(send_arr.size())) local_ok = 0;
  for (int src = 0; src < num_procs; ++src)
  {
    if (recvcounts[src] != (rank + 1) * ncols) local_ok = 0;
    int recv_first_row{rdispls[src] / ncols};
    for (int j = 0; j < rank + 1; ++j)
      for (int col = 0; col < ncols; ++col)
      {
        T expected{static_cast<T>(1000 * src + 10 * (source_first_row + j) + col)};
        if (recv_arr(recv_first_row + j, col) != expected) local_ok = 0;
      }
  }
  MPI_Allreduce(&local_ok, &all_ok, 1, count_type, MPI_MIN, comm);
  if (rank == 0)
    std::cout << "Validation: " << (all_ok ? "PASS" : "FAIL") << std::endl;
  return all_ok ? 0 : 1;
}

}  // namespace mpi_algorithm

/// Explicit template instantiation corresponding to mpi_type_traits

/// 1D instantiation
template int mpi_algorithm::Alltoallv_example<uint8_t>(MPI_Comm);
template int mpi_algorithm::Alltoallv_example<uint16_t>(MPI_Comm);
template int mpi_algorithm::Alltoallv_example<int>(MPI_Comm);
template int mpi_algorithm::Alltoallv_example<float>(MPI_Comm);
template int mpi_algorithm::Alltoallv_example<double>(MPI_Comm);
template int mpi_algorithm::Alltoallv_example<long>(MPI_Comm);

/// 2D instantiation
template int mpi_algorithm::Alltoallv_2d_example<uint8_t>(MPI_Comm);
template int mpi_algorithm::Alltoallv_2d_example<uint16_t>(MPI_Comm);
template int mpi_algorithm::Alltoallv_2d_example<int>(MPI_Comm);
template int mpi_algorithm::Alltoallv_2d_example<float>(MPI_Comm);
template int mpi_algorithm::Alltoallv_2d_example<double>(MPI_Comm);
template int mpi_algorithm::Alltoallv_2d_example<long>(MPI_Comm);