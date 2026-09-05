///
/// @file conway2d.cc
/// @brief MPI Parallel Conway's Game of Life using custom
///         Cartesian topology.
/// @author Yihai Li
/// @date Sept. 4 2026
///
///

/// includes
#include <mpi.h>

#include <cstdint>
#include <iostream>

#include "mpi/io_Cartesian.hpp"
#include "mpi/topology_Cartesian.hpp"
#include "mpi/type.hpp"
#include "multiarray.hpp"
#include "randin.hpp"  // Integrated randomizer class

constexpr int SEED{2027}, NUM_GENERATION{200};
constexpr size_t DIMENSIONS{2}, DIM_X{300}, DIM_Y{300};
typedef uint8_t ELEMENT_TYPE;

void evolve(mpi_array::array_cartesian<ELEMENT_TYPE, DIMENSIONS>&);

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  multi_array::multi_array_shape<DIMENSIONS> global_shape(DIM_Y, DIM_X);

  {
    mpi_array::array_cartesian<ELEMENT_TYPE, DIMENSIONS>
      grid(global_shape, MPI_COMM_WORLD);  // shape of entire board.

    array_randomizer<ELEMENT_TYPE>
      rng(2027, 0, 1);  // random number generator

    if (grid.topology.comm_cart != MPI_COMM_NULL)
    {
      grid.commit_halo_mpi_datatypes();

      // initialize entire board.
      array_randomizer<ELEMENT_TYPE> rng(SEED, 0, 1);

      for (size_t gi = 0; gi < global_shape.dims[0]; ++gi)
      {
        for (size_t gj = 0; gj < global_shape.dims[1]; ++gj)
        {
          ELEMENT_TYPE val{rng()};

          size_t start_i = grid.topology.starts[0] - 1;
          size_t end_i = grid.topology.ends[0] - 1;
          size_t start_j = grid.topology.starts[1] - 1;
          size_t end_j = grid.topology.ends[1] - 1;

          if (gi >= start_i && gi <= end_i && gj >= start_j && gj <= end_j)
          {  // exclude halos
            size_t local_i = gi - start_i + 1;
            size_t local_j = gj - start_j + 1;
            grid.current_data(local_i, local_j) = val;
          }
        }
      }

      // Run the simulation
      double begin_time{MPI_Wtime()};
      for (int generation = 0; generation < NUM_GENERATION; ++generation)
      {
        evolve(grid);

        /// Store data to "output" folder if needed
        std::string out_filename{
          ("output/output_step_" + std::to_string(generation) + ".bin")};

        mpi_io::write_array_Cartesian_io_binary(
          grid,
          out_filename);
      }
      double total_time{MPI_Wtime() - begin_time};

      if (0 == grid.topology.rank)
        std::cout << "Total evolving time: " << total_time << "\n"
                  << "Generations: " << NUM_GENERATION
                  << std::endl;
    }
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}

///
/// @brief Evolve function based on Conway's Life Rule
/// @param grid Game input grid.
///
void evolve(mpi_array::array_cartesian<ELEMENT_TYPE, DIMENSIONS>& grid)
{
  auto &curr{grid.current_data}, &next{grid.next_data};
  const auto& shape{grid.topology.local_shape};

  // Step 0: exchange halos
  grid.exchange_halos_noneblocking();

  // Step 1: Apply Conway's Life rules in 2D world. (C ROW MAJOR)
  for (size_t i = 1; i < shape.dims[0] - 1; ++i)
  {
    for (size_t j = 1; j < shape.dims[1] - 1; ++j)
    {
      // Count the 8 surrounding neighbors
      ELEMENT_TYPE alive_neighbors =
        curr(i - 1, j - 1) + curr(i - 1, j) + curr(i - 1, j + 1) +
        curr(i, j - 1) + curr(i, j + 1) +
        curr(i + 1, j - 1) + curr(i + 1, j) + curr(i + 1, j + 1);

      // Rule 1 & 2: Any live cell with 2 or 3 live neighbors survives.
      if (curr(i, j) == 1)
      {
        next(i, j) = (alive_neighbors == 2 || alive_neighbors == 3) ? 1 : 0;
      }
      // Rule 3 & 4: Any dead cell with exactly 3 live neighbors becomes a live cell.
      else
      {
        next(i, j) = (alive_neighbors == 3) ? 1 : 0;
      }
    }
  }

  // Step 2: Swap
  curr.swap(next);
}