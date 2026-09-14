#ifndef PRINT_IN_ORDER_HPP_YIHAI
#define PRINT_IN_ORDER_HPP_YIHAI

#include <mpi.h>
#include <unistd.h>

#include <climits>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "multiarray.hpp"

namespace mpi_io
{

/// @brief Collect local text and print it on rank 0 in communicator rank order.
/// @note Every process in the intracommunicator must call this function in the
///       same order. The total text per call must fit in an MPI int count.
inline void print_in_order(
  const std::string& local_text,
  MPI_Comm comm,
  const std::string& title = "")
{
  int rank{-1}, num_procs{0};
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &num_procs);

  if (local_text.size() > static_cast<size_t>(INT_MAX))
  {
    std::cerr << "print_in_order: local text exceeds INT_MAX bytes.\n";
    MPI_Abort(comm, 1);
    return;
  }

  int local_size{static_cast<int>(local_text.size())};
  std::vector<int> sizes(rank == 0 ? num_procs : 0);
  std::vector<int> displs(rank == 0 ? num_procs : 0);

  MPI_Gather(
    &local_size,
    1,
    MPI_INT,
    sizes.data(),
    1,
    MPI_INT,
    0,
    comm);

  int total_size{0};
  if (rank == 0)
  {
    for (int p = 0; p < num_procs; ++p)
    {
      if (sizes[p] > INT_MAX - total_size)
      {
        std::cerr << "print_in_order: total text exceeds INT_MAX bytes.\n";
        MPI_Abort(comm, 1);
        return;
      }
      displs[p] = total_size;
      total_size += sizes[p];
    }
  }

  // Keep a valid root buffer even when every process supplies empty text.
  std::vector<char> text(rank == 0 ? (total_size > 0 ? total_size : 1) : 0);
  MPI_Gatherv(
    local_text.data(), local_size, MPI_CHAR, text.data(), sizes.data(), displs.data(), MPI_CHAR, 0, comm);

  if (rank == 0)
  {
    if (!title.empty()) std::cout << "\n========== " << title << " ==========\n";
    for (int p = 0; p < num_procs; ++p)
    {
      std::cout << "\n[rank " << p << "]\n";
      std::cout.write(text.data() + displs[p], sizes[p]);
      if (sizes[p] == 0 || text[displs[p] + sizes[p] - 1] != '\n')
        std::cout << '\n';
    }
    std::cout.flush();
  }
  MPI_Barrier(comm);
}

template <typename T, size_t NumD>
void print_multiarray_in_order(
  const multi_array::array<T, NumD>& arr,
  MPI_Comm comm)
{
  int num_procs{0}, rank{-1};
  MPI_Comm_size(comm, &num_procs);
  MPI_Comm_rank(comm, &rank);

  MPI_Barrier(comm);
  usleep(1000);
  MPI_Barrier(comm);

  for (int i = 0; i < num_procs; ++i)
  {
    if (i == rank)
    {
      std::cout << "\nProc : " << rank << " of "
                << num_procs << " is printing. \n";

      std::cout << arr;
    }
    fflush(stdout);
    usleep(1000);
    MPI_Barrier(comm);
  }
}

/// @brief Print each packed Alltoallv block with peer, count and flat indices.
/// @param sending True before exchange (to rank); false after (from rank).
/// @note counts and displs must have communicator-size entries and describe
///       valid regions in arr. All processes must call with the same sending flag.
template <typename T, size_t NumD>
void print_alltoallv_in_order(
  const multi_array::array<T, NumD>& arr,
  const multi_array::array<int, 1>& counts,
  const multi_array::array<int, 1>& displs,
  bool sending,
  MPI_Comm comm)
{
  int num_procs{0};
  MPI_Comm_size(comm, &num_procs);
  std::ostringstream local;
  const char* buffer{sending ? "send_arr" : "recv_arr"};

  local << (sending ? "sendcounts: " : "recvcounts: ");
  for (int p = 0; p < num_procs; ++p) local << counts[p] << ' ';
  local << '\n'
        << (sending ? "sdispls:    " : "rdispls:    ");
  for (int p = 0; p < num_procs; ++p) local << displs[p] << ' ';
  local << '\n';

  for (int peer = 0; peer < num_procs; ++peer)
  {
    local << (sending ? "  to rank " : "  from rank ") << peer
          << " | count=" << counts[peer] << " | offset=" << displs[peer] << '\n';
    if (counts[peer] == 0)
    {
      local << "    (empty)\n";
      continue;
    }
    size_t first{static_cast<size_t>(displs[peer])};
    local << "    " << buffer << '[' << first << ".."
          << first + static_cast<size_t>(counts[peer]) - 1 << "]: ";
    for (int j = 0; j < counts[peer]; ++j)
    {
      size_t idx{first + static_cast<size_t>(j)};
      local << +arr[idx] << ' ';
    }
    local << '\n';
  }

  print_in_order(local.str(), comm, sending ? "BEFORE MPI_Alltoallv" : "AFTER MPI_Alltoallv");
}

}  // namespace mpi_io

#endif
