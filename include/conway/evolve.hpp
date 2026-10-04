#pragma once

#include "mpi/topology_Cartesian.hpp"

namespace conway
{
// X is contiguous. Distinct output buffers and branch-free rules allow the
// compiler to vectorize without selecting architecture-specific intrinsics.
template <typename T>
inline void update_row_2d(const T* curr,
                          T* next,
                          size_t row,
                          size_t stride,
                          size_t begin,
                          size_t end)
{
  const T* above = curr + row - stride;
  const T* center = curr + row;
  const T* below = curr + row + stride;
  T* output = next + row;

#ifdef _OPENMP
#pragma omp simd
#endif
  for (size_t x = begin; x < end; ++x)
  {
    const auto count = above[x - 1] + above[x] + above[x + 1] + center[x - 1] +
                       center[x + 1] + below[x - 1] + below[x] + below[x + 1];

    output[x] =
      static_cast<T>((count == 3) | ((center[x] == 1) & (count == 2)));
  }
}

template <typename T>
inline void update_row_3d(const T* curr,
                          T* next,
                          size_t row,
                          size_t plane,
                          size_t stride,
                          size_t begin,
                          size_t end)
{
  const T* a = curr + row - plane - stride;
  const T* b = curr + row - plane;
  const T* c = curr + row - plane + stride;
  const T* d = curr + row - stride;
  const T* e = curr + row;
  const T* f = curr + row + stride;
  const T* g = curr + row + plane - stride;
  const T* h = curr + row + plane;
  const T* i = curr + row + plane + stride;
  T* output = next + row;

#ifdef _OPENMP
#pragma omp simd
#endif
  for (size_t x = begin; x < end; ++x)
  {
    const auto count = a[x - 1] + a[x] + a[x + 1] + b[x - 1] + b[x] + b[x + 1] +
                       c[x - 1] + c[x] + c[x + 1] + d[x - 1] + d[x] + d[x + 1] +
                       e[x - 1] + e[x + 1] + f[x - 1] + f[x] + f[x + 1] +
                       g[x - 1] + g[x] + g[x + 1] + h[x - 1] + h[x] + h[x + 1] +
                       i[x - 1] + i[x] + i[x + 1];

    output[x] = static_cast<T>((count == 5) | ((e[x] == 1) & (count == 4)));
  }
}

template <typename T>
void evolve(mpi_array::array_cartesian<T, 2>& grid)
{
  const auto& shape = grid.topology.local_shape;
  const size_t ny = shape.dims[0] - 2, nx = shape.dims[1] - 2;
  const size_t stride = shape.strides[0];
  const T* curr = grid.current_data.data();
  T* next = grid.next_data.data();

  grid.begin_halo_exchange();

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    // Strict interior never reads receive buffers. nowait allows the MPI
    // thread to finish communication while workers complete their chunks.
#ifdef _OPENMP
#pragma omp for schedule(static) nowait
#endif
    for (size_t y = 2; y < ny; ++y)
      update_row_2d(curr, next, y * stride, stride, 2, nx);

#ifdef _OPENMP
#pragma omp master
#endif
    {
      grid.finish_halo_exchange();
    }

#ifdef _OPENMP
#pragma omp barrier
#pragma omp for schedule(static)
#endif
    for (size_t y = 1; y <= ny; ++y)
    {
      if (y == 1 || y == ny)
        update_row_2d(curr, next, y * stride, stride, 1, nx + 1);
      else
      {
        update_row_2d(curr, next, y * stride, stride, 1, 2);
        if (nx > 1)
          update_row_2d(curr, next, y * stride, stride, nx, nx + 1);
      }
    }
  }

  grid.current_data.swap(grid.next_data);
}

template <typename T>
void evolve(mpi_array::array_cartesian<T, 3>& grid)
{
  const auto& shape = grid.topology.local_shape;
  const size_t nz = shape.dims[0] - 2, ny = shape.dims[1] - 2;
  const size_t nx = shape.dims[2] - 2;
  const size_t plane = shape.strides[0], stride = shape.strides[1];
  const T* curr = grid.current_data.data();
  T* next = grid.next_data.data();

  grid.begin_halo_exchange();

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifdef _OPENMP
#pragma omp for collapse(2) schedule(static) nowait
#endif
    for (size_t z = 2; z < nz; ++z)
      for (size_t y = 2; y < ny; ++y)
        update_row_3d(curr,
                      next,
                      z * plane + y * stride,
                      plane,
                      stride,
                      2,
                      nx);

#ifdef _OPENMP
#pragma omp master
#endif
    {
      grid.finish_halo_exchange();
    }
#ifdef _OPENMP
#pragma omp barrier
#pragma omp for collapse(2) schedule(static)
#endif

    for (size_t z = 1; z <= nz; ++z)
      for (size_t y = 1; y <= ny; ++y)
      {
        const auto row = z * plane + y * stride;

        if (z == 1 || z == nz || y == 1 || y == ny)
          update_row_3d(curr,
                        next,
                        row,
                        plane,
                        stride,
                        1,
                        nx + 1);
        else
        {
          update_row_3d(curr,
                        next,
                        row,
                        plane,
                        stride,
                        1,
                        2);
          if (nx > 1)
            update_row_3d(curr,
                          next,
                          row,
                          plane,
                          stride,
                          nx,
                          nx + 1);
        }
      }
  }
  grid.current_data.swap(grid.next_data);
}
}  // namespace conway
