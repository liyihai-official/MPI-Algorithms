#pragma once

#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <charconv>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace conway
{
struct run_options
{
  int generations = 0;
  bool write_output = true;
  bool help = false;
  std::filesystem::path output_directory = "output";
};

inline bool initialize_mpi(int& argc, char**& argv)
{
  int provided = MPI_THREAD_SINGLE;
#ifdef _OPENMP
  constexpr int required = MPI_THREAD_FUNNELED;
#else
  constexpr int required = MPI_THREAD_SINGLE;
#endif
  if (MPI_Init_thread(&argc, &argv, required, &provided) != MPI_SUCCESS)
    return false;
  if (provided < required)
  {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
      std::cerr << "MPI does not provide the required thread support.\n";
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    return false;
  }
  return true;
}

inline run_options parse_options(int argc,
                                 char** argv,
                                 int default_generations)
{
  run_options result;
  result.generations = default_generations;
  for (int i = 1; i < argc; ++i)
  {
    const std::string_view argument(argv[i]);
    if (argument == "--no-output")
      result.write_output = false;
    else if (argument == "--help")
      result.help = true;
    else if (argument == "--generations" && i + 1 < argc)
    {
      const std::string_view value(argv[++i]);
      const auto [end, error] = std::from_chars(
        value.data(), value.data() + value.size(), result.generations);
      if (error != std::errc{} || end != value.data() + value.size() ||
          result.generations < 0)
        throw std::invalid_argument(
          "--generations requires a nonnegative integer.");
    }
    else if (argument == "--output-dir" && i + 1 < argc)
      result.output_directory = argv[++i];
    else
      throw std::invalid_argument("Unknown or incomplete option: " +
                                  std::string(argument));
  }
  return result;
}

inline void print_help()
{
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0)
    std::cout
      << "Options: --generations N --no-output --output-dir PATH --help\n"
      << "Set OMP_NUM_THREADS, OMP_PLACES, and OMP_PROC_BIND for hybrid "
         "runs.\n";
}

inline void prepare_output(const run_options& options)
{
  if (!options.write_output)
    return;
  int rank = 0, success = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0)
  {
    std::error_code error;
    std::filesystem::create_directories(options.output_directory, error);
    success = !error;
  }
  MPI_Bcast(&success, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (!success)
    throw std::runtime_error("Cannot create output directory.");
}

inline void report_times(MPI_Comm comm,
                         double evolving,
                         double total,
                         int generations)
{
  const double local[2] = {evolving, total};
  double maximum[2] = {};
  MPI_Reduce(local, maximum, 2, MPI_DOUBLE, MPI_MAX, 0, comm);
  int rank = 0, ranks = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &ranks);
  if (rank == 0)
  {
    std::cout << "Total evolving time: " << maximum[0] << "\n"
              << "Total simulation time (including I/O): " << maximum[1] << "\n"
              << "Generations: " << generations << "\n"
              << "MPI ranks: " << ranks << "\n";
#ifdef _OPENMP
    std::cout << "OpenMP max threads per rank: " << omp_get_max_threads()
              << "\n";
#else
    std::cout << "OpenMP disabled (MPI-only build)\n";
#endif
  }
}
}  // namespace conway
