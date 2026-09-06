///
/// @file bruck_Alltoall.cc
/// @brief The main file of bruck algorithm demonstration.
///
/// @author Yihai Li
/// @date Sept. 2 2026
///
///

/// includes
#include <mpi.h>
#include <unistd.h>

#include <iostream>

#include "mpi/bruck.hpp"

/// using datatypes
using value_type = float;
using size_type = size_t;

/// Random Seed
constexpr int SEED{2027};

/// Problem Size
#if !defined(DIM_X) || !defined(DIM_Y) || !defined(DIM_Z)
#define DIM_X 1024
#define DIM_Y 1024
#define DIM_Z 128
#endif

int main(int argc, char** argv)
{
  int rank{-1}, num_proc{-1};

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

  if (num_proc % 2 != 0)
  {
    std::cout << "This application is meant to be run with 2 processes."
              << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  int elements_per_proc{static_cast<int>((DIM_X / num_proc) * DIM_Y)};
  multi_array::array<value_type, 2>
    send_mat(DIM_X, DIM_Y),
    recv_mat_MPI(DIM_X, DIM_Y),
    recv_mat_Bruck(DIM_X, DIM_Y);

  MPI_Datatype MPI_value_type{mpi_type::mpi_type_traits<value_type>::get()};

  array_randomizer<value_type> rng(
    rank + SEED,
    rank * 10,
    rank * 10.0 + 9.0);

  rng.fill(send_mat.begin(), send_mat.end());

  /// Preparing for benchmark tests
  MPI_Barrier(MPI_COMM_WORLD);

  // Iteration
  const int num_iters{100}, warmup_iters{10};

  // =============== Step 0. Warmup ========================
  for (int i = 0; i < warmup_iters; ++i)
  {
    mpi_algorithm::Bruck_Alltoall_noneblocking(
      send_mat,
      recv_mat_Bruck,
      MPI_COMM_WORLD);
  }

  // ============ Step 1. Bruck ============================
  MPI_Barrier(MPI_COMM_WORLD);
  double start_bruck{MPI_Wtime()};

  for (int i = 0; i < num_iters; ++i)
  {
    mpi_algorithm::Bruck_Alltoall_noneblocking(
      send_mat,
      recv_mat_Bruck,
      MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double end_bruck{MPI_Wtime()};
  double avg_time_bruck{(end_bruck - start_bruck) / num_iters};

  // ============ Step 2. MPI_Alltoall =====================
  for (int i = 0; i < warmup_iters; ++i)
  {
    MPI_Alltoall(
      send_mat.data(),
      elements_per_proc,
      MPI_value_type,
      recv_mat_MPI.data(),
      elements_per_proc,
      MPI_value_type,
      MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double start_mpi{MPI_Wtime()};

  for (int i = 0; i < num_iters; ++i)
  {
    MPI_Alltoall(
      send_mat.data(),
      elements_per_proc,
      MPI_value_type,
      recv_mat_MPI.data(),
      elements_per_proc,
      MPI_value_type,
      MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double end_mpi{MPI_Wtime()};
  double avg_time_mpi{(end_mpi - start_mpi) / num_iters};

  // ============ Step 3. Benchmark Results ================
  if (0 == rank)
  {
    std::cout << "Bruck avg. time consumption: "
              << avg_time_bruck
              << " s\n";
    std::cout << "MPI_Alltoall avg. time consumption: "
              << avg_time_mpi
              << " s";
    std::cout << std::endl;
  }

  //
  MPI_Barrier(MPI_COMM_WORLD);

  /**
   * @note Print the sum difference if needed.
   */

  // auto diff{multi_array::sum(recv_mat_MPI - recv_mat_Bruck)};
  // for (int i = 0; i < num_proc; ++i)
  // {
  //   if (rank == i)
  //   {
  //     std::cout << "========== After: proc " << rank << " diff ==========\n";
  //     std::cout << diff << std::endl;
  //   }
  //   MPI_Barrier(MPI_COMM_WORLD);
  // }
  MPI_Finalize();
  return EXIT_SUCCESS;
}