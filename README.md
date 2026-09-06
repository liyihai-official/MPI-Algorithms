# MPI-Algorithms: Bruck All-to-all & Conway's Game of Life

**[English](#english-version)** | **[中文](#chinese-version)**

---

<a id="english-version"></a>

## Documentation

### Project Overview

This project is a high-performance computing (HPC) and parallel algorithms
library built with modern C++20 and OpenMPI. The core repository provides a
robust implementation of the non-blocking Bruck All-to-all communication
algorithm in`bruck.hpp`,
alongside comprehensive benchmarking against the standard `MPI_Alltoall`.
Furthermore, it features a custom Cartesian topology framework used to simulate
both 2D (`conway2D.cc`) and 3D (`conway3D.cc`) parallel Conway's Game of Life.

### Core Features

- **Bruck All-to-all Algorithm**:
  - Implements a non-blocking All-to-all collective
    communication algorithm `mpi_algorithm::Bruck_Alltoall_noneblocking`
    using bitwise shift calculations, `MPI_Isend`, and `MPI_Irecv`.

- **Parallel Game of Life**:
  - **2D Evolution**: `conway2D.cc` Utilizes standard Game of Life rules
    (Survival: 2-3 neighbors, Reproduction: exactly 3 neighbors) over a
    distributed 2D grid.
  - **3D Evolution**: `conway3D.cc` Extends to 3D space applying the Carter Bays
    4555 rule (Survival: 4-5 neighbors, Reproduction: exactly 5 neighbors).

- **High-Performance Topology & Halo Exchange**:
  Features an advanced `mpi_array::array_cartesian` container.
  It manages local/global array shapes `multi_array::multi_array_shape` and
  supports automated MPI datatype creation for blocking (`MPI_Sendrecv`)
  and non-blocking halo exchanges.
- **Parallel Binary I/O**:
  Implements highly efficient distributed file writing using `MPI_File_write_all`
  via MPI-IO and custom subarrays, enabling fast dumps of Cartesian data
  grids to disk.
- **Modern C++ Utilities**:
  Includes a lightweight N-dimensional array container (`multi_array`) with
  multi-index access, compile-time C++ to MPI datatype deduction mapping,
  and a C++20 span-compatible random number generator.

### Prerequisites

- A C++ compiler fully supporting C++20 standards (GCC, Clang, etc.).
- An MPI library implementation (e.g., OpenMPI, MPICH).
- CMake (Minimum version 3.15).
- _(Optional)_ Doxygen and Graphviz for API documentation generation.

### Build Instructions

The project uses CMake as its build system. Compilation definitions such as
grid dimensions and data types can be overridden via CMake cache variables:

```bash
mkdir build && cd build
# Configure the project with custom dimensions and data types
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DDIM_X=1024 -DDIM_Y=1024 -DDIM_Z=128 \
         -DDATA_TYPE="uint8_t"

# Build the executables
cmake --build . -j 4

```

_Note: Valid `DATA_TYPE` options include `uint8_t`, `uint16_t`, `int`, `float`, `double`, and `long` as explicitly instantiated in the MPI-IO module._

### Running the Executables

Once compiled, three executables are generated:

1. **`bruck_Alltoall`**: Runs the benchmark comparing the custom Bruck
   algorithm against the standard `MPI_Alltoall`.

```bash
# Note: This specific application is meant to be run with 2 processes
mpiexec -np 2 ./bruck_Alltoall

```

2. **`conways-game-2D`**:
   Simulates the 2D Game of Life for 200 generations.

```bash
mpiexec -np 4 ./conways-game-2D

```

<img src="figure/conways_game.gif" width="400" height="400" />

3. **`conways-game-3D`**:
   Simulates the 3D Game of Life for 100 generations.

```bash
mpiexec -np 8 ./conways-game-3D

```

<img src="figure/conways_3d.gif" width="400" height="400" />

### Automated Benchmarks

The repository includes Bash scripts to automate compilation, execution,
and performance data extraction (saved to CSV files) across varying
problem sizes and process counts:

- `benchmark_bruck2D.sh` & `benchmark_bruck3D.sh`:
  the Bruck collective communication time.
- `benchmark_conways2D.sh`:
  Benchmarks the total execution time of the 2D Game of Life evolution.

### Getting Documentation

Execute Doxygen in the root directory to generate HTML and LaTeX documentation:

```bash
doxygen Doxyfile

```

---

<a id="chinese-version"></a>

## 描述

```

```
