///
/// @file bruck.cc
/// @brief The implementation of the Bruck algorithm demonstration.
///

#include <mpi.h>

#include <cstring>

#include "multiarray.hpp"
#include "randin.hpp"

namespace mpi_algorithm
{
template <typename T, size_t NumD>
void Bruck_Alltoall_noneblocking(
  const multi_array::array<T, NumD>& send_arr,
  multi_array::array<T, NumD>& recv_arr,
  MPI_Comm comm)
{
  int rank, num_procs;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &num_procs);

  size_t total_elements = send_arr.size();
  size_t elements_per_proc = total_elements / num_procs;

  const T* send_data = send_arr.data();
  T* recv_data = recv_arr.data();

  // Local data rotation
  for (size_t j = 0; j < num_procs; ++j)
  {
    int dest_rank = (j - rank + num_procs) % num_procs;
    std::memcpy(
      &recv_data[dest_rank * elements_per_proc],
      &send_data[j * elements_per_proc],
      elements_per_proc * sizeof(T));
  }

  // None blocking communication
  int num_steps = std::ceil(std::log2(num_procs));
  for (int step = 0; step < num_steps; ++step)
  {
    int distance = 1 << step;  // bitwise left shift to calculate 2^step
    int sent_to = (rank + distance) % num_procs;
    int recv_from = (rank - distance + num_procs) % num_procs;
  }
}
}  // end namespace mpi_algorithm