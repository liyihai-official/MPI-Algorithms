#include <cuda_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "mpi/io_Cartesian.hpp"
#include "mpi/topology_Cartesian.hpp"

int world_rank = 0;
using Cell = uint8_t;
using Grid = std::vector<Cell>;
void check(cudaError_t e)
{
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}

// MPI 库已经填充一圈 halo；CUDA 仅更新内部细胞。
__global__ void evolve(const Cell* current, Cell* next, int width, int height)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int y = blockIdx.y * blockDim.y + threadIdx.y + 1;

  if (x >= width - 1 || y >= height - 1)
    return;

  int neighbors = 0;
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx)
      if (dx || dy)
        neighbors += current[(y + dy) * width + x + dx];
  int i = y * width + x;

  next[i] = neighbors == 3 || (current[i] && neighbors == 2);
}

class Simulation
{
  using Distributed = mpi_array::array_cartesian<Cell, 2>;

  // 构造库对象之前检查分解，避免空子域的无效 halo 数据类型。
  static multi_array::multi_array_shape<2> shape(int w, int h)
  {
    int size, dims[2] = {0, 0};
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Dims_create(size, 2, dims);
    if (dims[0] > h || dims[1] > w)
      throw std::runtime_error("Too many MPI ranks for this grid");
    return multi_array::multi_array_shape<2>(h, w);
  }

  Distributed grid;
  Cell *a = nullptr, *b = nullptr;
  int width, height, local_w, local_h;

 public:
  Simulation(
    const Grid& initial,
    int w,
    int h)
    : grid(shape(w, h), MPI_COMM_WORLD),
      width(w),
      height(h),
      local_w(static_cast<int>(grid.topology.local_shape.dims[1])),
      local_h(static_cast<int>(grid.topology.local_shape.dims[0]))
  {
    grid.commit_halo_mpi_datatypes();
    for (int y = 1; y < local_h - 1; ++y)
      for (int x = 1; x < local_w - 1; ++x)
        grid.current_data(y, x) = initial[(grid.topology.starts[0] + y - 2) * w +
                                          grid.topology.starts[1] + x - 2];
    try
    {
      check(cudaMalloc(&a, grid.current_data.size()));
      check(cudaMalloc(&b, grid.current_data.size()));
      check(cudaMemset(b, 0, grid.current_data.size()));
    }
    catch (...)
    {
      cudaFree(a);
      cudaFree(b);
      throw;
    }
  }
  Simulation(const Simulation&) = delete;
  Simulation& operator=(const Simulation&) = delete;
  ~Simulation()
  {
    cudaFree(a);
    cudaFree(b);
  }

  void step()
  {
    // 按维度交换，第二维会传递第一维的 halo，从而覆盖对角邻居。
    grid.exchange_halos_noneblocking();
    check(
      cudaMemcpy(a,
                 grid.current_data.data(),
                 grid.current_data.size(),
                 cudaMemcpyHostToDevice));

    dim3 threads(16, 16);
    dim3 blocks((local_w - 2 + 15) / 16, (local_h - 2 + 15) / 16);
    evolve<<<blocks, threads>>>(a, b, local_w, local_h);

    check(
      cudaGetLastError());
    check(
      cudaMemcpy(
        grid.next_data.data(),
        b,
        grid.next_data.size(),
        cudaMemcpyDeviceToHost));

    grid.current_data.swap(grid.next_data);
  }

  Grid read() const
  {
    // 用于显示和校验的全局组装；无动画模式仅在最后调用。
    Grid result(static_cast<size_t>(width) * height, 0);
    for (int y = 1; y < local_h - 1; ++y)
      for (int x = 1; x < local_w - 1; ++x)
        result[(grid.topology.starts[0] + y - 2) * width +
               grid.topology.starts[1] + x - 2] = grid.current_data(y, x);
    MPI_Allreduce(
      MPI_IN_PLACE,
      result.data(),
      static_cast<int>(result.size()),
      MPI_UNSIGNED_CHAR,
      MPI_MAX,
      grid.topology.comm_cart);

    return result;
  }

  void write(const std::string& path) const
  {
    // 上游写入器不截断已有文件；先集体设置文件长度。
    MPI_File file;
    int error = MPI_File_open(
      grid.topology.comm_cart,
      path.c_str(),
      MPI_MODE_CREATE | MPI_MODE_WRONLY,
      MPI_INFO_NULL,
      &file);

    if (error != MPI_SUCCESS)
      throw std::runtime_error("Cannot open output: " + path);
    int resized = MPI_File_set_size(file, static_cast<MPI_Offset>(width) * height);
    MPI_File_close(&file);

    if (resized != MPI_SUCCESS)
      throw std::runtime_error("Cannot resize output: " + path);

    mpi_io::write_array_Cartesian_io_binary(grid, path);
  }
};

Grid cpu_step(const Grid& input, int w, int h)
{
  Grid output(input.size());
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
    {
      int count = 0;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
        {
          if (!dx && !dy) continue;
          int nx = x + dx, ny = y + dy;
          if (nx < 0) nx += w;
          if (nx >= w) nx -= w;
          if (ny < 0) ny += h;
          if (ny >= h) ny -= h;
          count += input[ny * w + nx];
        }
      output[y * w + x] = count == 3 || (input[y * w + x] && count == 2);
    }
  return output;
}

Grid initial_grid(int w, int h, const std::string& pattern, unsigned seed)
{
  Grid grid(static_cast<size_t>(w) * h, 0);
  if (pattern == "random")
  {
    std::mt19937 rng(seed);
    for (auto& cell : grid)
      cell = rng() % 100 < 25;
  }
  else if (pattern == "glider")
  {
    int x = w / 2 - 1, y = h / 2 - 1;
    grid[y * w + x + 1] = 1;
    grid[(y + 1) * w + x + 2] = 1;
    for (int dx = 0; dx < 3; ++dx)
      grid[(y + 2) * w + x + dx] = 1;
  }
  else
    throw std::runtime_error("Pattern must be random or glider");
  return grid;
}

void self_test()
{
  auto compare = [](Grid grid, int w, int h, int steps)
  {
    Simulation gpu(grid, w, h);
    for (int i = 0; i < steps; ++i)
    {
      grid = cpu_step(grid, w, h);
      gpu.step();
      if (gpu.read() != grid)
        throw std::runtime_error("CPU/GPU mismatch at step " + std::to_string(i + 1));
    }
  };
  Grid block(64, 0);
  block[27] = block[28] = block[35] = block[36] = 1;
  Simulation still(block, 8, 8);
  still.step();

  if (still.read() != block)
    throw std::runtime_error("Block should remain stable");

  Grid blinker(64, 0), vertical(64, 0);

  blinker[26] = blinker[27] = blinker[28] = 1;

  vertical[19] = vertical[27] = vertical[35] = 1;

  Simulation oscillator(blinker, 8, 8);

  oscillator.step();

  if (oscillator.read() != vertical)
    throw std::runtime_error("Blinker first phase failed");

  oscillator.step();

  if (oscillator.read() != blinker)
    throw std::runtime_error("Blinker period failed");

  Grid glider = initial_grid(9, 7, "glider", 1), translated(63, 0);

  for (int y = 0; y < 7; ++y)
    for (int x = 0; x < 9; ++x)
      translated[((y + 1) % 7) * 9 + (x + 1) % 9] = glider[y * 9 + x];

  Simulation moving(glider, 9, 7);

  for (int i = 0; i < 4; ++i) moving.step();
  if (moving.read() != translated)
    throw std::runtime_error("Glider translation failed");

  compare(glider, 9, 7, 100);  // 跨边界移动
  compare(Grid(35, 0), 7, 5, 3);
  compare(Grid(35, 1), 7, 5, 3);
  compare(initial_grid(37, 23, "random", 42), 37, 23, 100);
  compare(initial_grid(5, 5, "random", 7), 5, 5, 100);

  if (world_rank == 0)
    std::cout << "PASS: block, blinker, glider, wraparound, empty/full and random CPU/GPU checks\n";
}

int number(const char* value, int low, int high)
{
  std::string s(value);
  size_t end = 0;
  long long n = std::stoll(s, &end);
  if (end != s.size() || n < low || n > high) throw std::runtime_error("Argument out of range: " + s);
  return static_cast<int>(n);
}

int run(int argc, char** argv)
{
  try
  {
    int w = 60,
        h = 25,
        steps = 200,
        delay = 80,
        seed = 42;

    bool headless = false,
         test = false;

    std::string
      pattern = "glider",
      output;

    for (int i = 1; i < argc; ++i)
    {
      std::string arg(argv[i]);
      if (arg == "--help")
      {
        if (world_rank == 0)
          std::cout << "MPI + CUDA Conway's Game of Life (toroidal boundaries, B3/S23)\n"
                    << "--width N      5..8192 (default 60)\n--height N     5..8192 (default 25)\n"
                    << "--steps N      0..1000000 (default 200)\n--delay-ms N   0..10000 (default 80)\n"
                    << "--pattern P    glider or random (default glider)\n--seed N       0..2147483647 (default 42)\n"
                    << "--headless     No animation or delay; print final statistics\n--self-test    Run correctness checks\n--output FILE  Save final uint8 row-major grid using upstream MPI-IO\n";
        return 0;
      }
      if (arg == "--headless")
      {
        headless = true;
        continue;
      }
      if (arg == "--self-test")
      {
        test = true;
        continue;
      }
      if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
      const char* value = argv[++i];
      if (arg == "--width")
        w = number(value, 5, 8192);
      else if (arg == "--height")
        h = number(value, 5, 8192);
      else if (arg == "--steps")
        steps = number(value, 0, 1000000);
      else if (arg == "--delay-ms")
        delay = number(value, 0, 10000);
      else if (arg == "--seed")
        seed = number(value, 0, 2147483647);
      else if (arg == "--pattern")
        pattern = value;
      else if (arg == "--output")
        output = value;
      else
        throw std::runtime_error("Unknown option: " + arg);
    }
    if (test)
    {
      self_test();
      return 0;
    }

    Grid grid = initial_grid(w, h, pattern, static_cast<unsigned>(seed));

    Simulation sim(grid, w, h);

    auto render = [&](int generation)
    {
      if (world_rank != 0) return;
      std::cout << "\033[HGeneration "
                << generation << " / " << steps << "   \n";

      for (int y = 0; y < h; ++y)
      {
        for (int x = 0; x < w; ++x)
          std::cout << (grid[y * w + x] ? '#' : '.');
        std::cout << '\n';
      }
      std::cout << std::flush;
    };
    if (!headless)
    {
      if (world_rank == 0) std::cout << "\033[2J";
      render(0);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::steady_clock::now();
    for (int generation = 1; generation <= steps; ++generation)
    {
      sim.step();
      if (!headless)
      {
        grid = sim.read();
        render(generation);
        std::this_thread::sleep_for(std::chrono::milliseconds(delay));
      }
    }
    check(cudaDeviceSynchronize());
    double ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - start)
                  .count();
    double max_ms = 0;
    MPI_Reduce(
      &ms,
      &max_ms,
      1,
      MPI_DOUBLE,
      MPI_MAX,
      0,
      MPI_COMM_WORLD);

    grid = sim.read();
    if (!output.empty())
      sim.write(output);

    if (world_rank == 0)
      std::cout << "Grid: " << w << 'x' << h << ", generations: " << steps
                << ", alive: " << std::count(grid.begin(), grid.end(), Cell{1})
                << ", elapsed: " << max_ms << " ms"
                << (headless ? " (simulation wall time)\n" : " (includes display and delay)\n");
  }
  catch (const std::exception& e)
  {
    std::cerr << "Rank " << world_rank << " error: " << e.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  return 0;
}

// 同一节点内按 local rank 选择可见 GPU；单卡也支持多个进程共享。
int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  try
  {
    bool help = false;
    for (int i = 1; i < argc; ++i)
      if (std::string(argv[i]) == "--help") help = true;
    if (!help)
    {
      int devices = 0, local_rank = 0;

      check(cudaGetDeviceCount(&devices));
      if (!devices)
        throw std::runtime_error("No CUDA devices available");

      MPI_Comm local;
      MPI_Comm_split_type(
        MPI_COMM_WORLD,
        MPI_COMM_TYPE_SHARED,
        world_rank,
        MPI_INFO_NULL,
        &local);

      MPI_Comm_rank(local, &local_rank);
      MPI_Comm_free(&local);
      check(cudaSetDevice(local_rank % devices));
    }
    int result = run(argc, argv);
    MPI_Finalize();
    return result;
  }
  catch (const std::exception& e)
  {
    std::cerr << "Rank " << world_rank
              << " error: " << e.what()
              << '\n';

    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
}
