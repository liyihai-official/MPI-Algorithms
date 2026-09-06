///
/// @file bruck.hpp
/// @brief The implementation of the Bruck algorithm demonstration.
///
///

/// includes
#include <mpi.h>

#include <cstring>

#include "mpi/type.hpp"
#include "multiarray.hpp"
#include "randin.hpp"

/// @brief Namespace for MPI communication algorithms.
namespace mpi_algorithm
{

///
/// @brief Bruck's All-to-All communication algorithm using non-blocking MPI calls.
/// @tparam T Value type of the array elements.
/// @tparam NumD Number of dimensions of the array.
/// @param send_arr Send array containing data to be sent to other processes.
/// @param recv_arr Recv array to store received data from other processes.
/// @param comm MPI Communicator for the communication.
///
template <typename T, size_t NumD>
void Bruck_Alltoall_noneblocking(
  const multi_array::array<T, NumD>& send_arr,
  multi_array::array<T, NumD>& recv_arr,
  MPI_Comm comm)
{
  int rank, num_procs;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &num_procs);
  MPI_Datatype base_type = mpi_type::mpi_type_traits<T>::get();

  size_t total_elements{send_arr.size()};
  size_t elements_per_proc{total_elements / num_procs};

  const T* send_data{send_arr.data()};
  T* recv_data{recv_arr.data()};

  // Local data rotation
  for (int j = 0; j < num_procs; ++j)
  {
    int dest_rank{static_cast<int>((j - rank + num_procs) % num_procs)};
    std::memcpy(
      &recv_data[dest_rank * elements_per_proc],
      &send_data[j * elements_per_proc],
      elements_per_proc * sizeof(T));
  }

  // None blocking communication
  std::vector<T> recv_temp(total_elements);
  int num_steps{static_cast<int>(std::ceil(std::log2(num_procs)))};

  for (int step = 0; step < num_steps; ++step)
  {
    int distance{1 << step};  // bitwise left shift to calculate 2^step
    int sent_to{(rank + distance) % num_procs};
    int recv_from{(rank - distance + num_procs) % num_procs};

    // calculate the displacements
    std::vector<int> displacement;
    for (int j = 0; j < num_procs; ++j)
    {
      if ((j >> step) & 1)
      {
        displacement.push_back(j * elements_per_proc);
      }
    }

    int num_blocks{static_cast<int>(displacement.size())};
    if (num_blocks == 0) continue;  // no need for sending.

    // MPI_Datatype
    MPI_Datatype step_type;
    MPI_Type_create_indexed_block(
      num_blocks,
      elements_per_proc,
      displacement.data(),
      base_type,
      &step_type);

    MPI_Type_commit(&step_type);

    // Non-blocking send and receive
    MPI_Request reqs[2];

    MPI_Isend(
      recv_data,
      1,
      step_type,
      sent_to,
      0,
      comm,
      &reqs[0]);

    MPI_Irecv(
      recv_temp.data(),
      num_blocks * elements_per_proc,
      base_type,
      recv_from,
      0,
      comm,
      &reqs[1]);

    MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

    int position{0};
    int insize_bytes{static_cast<int>(num_blocks * elements_per_proc * sizeof(T))};
    MPI_Unpack(
      recv_temp.data(),
      insize_bytes,
      &position,
      recv_data,
      1,
      step_type,
      comm);
    MPI_Type_free(&step_type);
  }

  // Final data rotation
  std::vector<T> final_temp(total_elements);
  for (int j = 0; j < num_procs; ++j)
  {
    int orig_block{(num_procs + rank - j) % num_procs};
    std::memcpy(
      &final_temp[orig_block * elements_per_proc],
      &recv_data[j * elements_per_proc],
      elements_per_proc * sizeof(T));
  }
  std::memcpy(
    recv_data,
    final_temp.data(),
    total_elements * sizeof(T));
}

}  // end namespace mpi_algorithm