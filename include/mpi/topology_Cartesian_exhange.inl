///
/// @file topology_Cartesian_exchange.inl
///
///

namespace mpi_array
{

template <typename T, size_t NumD>
inline void array_cartesian<T, NumD>::commit_halo_mpi_datatypes()
{
  // Prepare for MPI_Type_create_subarray usage.
  std::array<int, NumD> array_size{};
  std::array<int, NumD> array_subsize{};
  std::array<int, NumD> array_starts{};

  for (size_t i = 0; i < NumD; ++i)
  {
    array_size[i] = topology.local_shape.dims[i];
    array_starts[i] = 0;
  }

  for (size_t i = 0; i < NumD; ++i)
  {
    for (size_t j = 0; j < NumD; ++j)
      array_subsize[j] = topology.local_shape.dims[j];

    array_subsize[i] = 1;

    MPI_Type_create_subarray(
      NumD,
      array_size.data(),
      array_subsize.data(),
      array_starts.data(),
      MPI_ORDER_C,
      mpi_type::mpi_type_traits<T>::get(),
      &topology.halos[i]);

    MPI_Type_commit(&topology.halos[i]);
  }
}

template <typename T, size_t NumD>
void array_cartesian<T, NumD>::exchange_halos_noneblocking()
{
  for (size_t d = 0; d < NumD; ++d)
  {
    int
      left_rank{topology.nbr_src[d]},
      right_rank{topology.nbr_dest[d]};

    size_t
      stride{topology.local_shape.strides[d]},
      dim_size{topology.local_shape.dims[d]};

    size_t
      send_left_idx{1 * stride},
      recv_right_idx{(dim_size - 1) * stride};
    size_t
      send_right_idx{(dim_size - 2) * stride},
      recv_left_idx{0 * stride};

    MPI_Request requests[4]{};
    int req_count{0};

    if (right_rank != MPI_PROC_NULL)
      MPI_Irecv(
        &current_data[recv_right_idx],
        1,
        topology.halos[d],
        right_rank,
        0,
        topology.comm_cart,
        &requests[req_count++]);

    if (left_rank != MPI_PROC_NULL)
      MPI_Irecv(
        &current_data[recv_left_idx],
        1,
        topology.halos[d],
        left_rank,
        1,
        topology.comm_cart,
        &requests[req_count++]);

    if (left_rank != MPI_PROC_NULL)
      MPI_Isend(
        &current_data[send_left_idx],
        1,
        topology.halos[d],
        left_rank,
        0,
        topology.comm_cart,
        &requests[req_count++]);

    if (right_rank != MPI_PROC_NULL)
      MPI_Isend(
        &current_data[send_right_idx],
        1,
        topology.halos[d],
        right_rank,
        1,
        topology.comm_cart,
        &requests[req_count++]);

    if (req_count > 0)
      MPI_Waitall(
        req_count,
        requests,
        MPI_STATUS_IGNORE);
  }
}

template <typename T, size_t NumD>
void array_cartesian<T, NumD>::exchange_halos_blocking()
{
  for (size_t d = 0; d < NumD; ++d)
  {
    int
      left_rank{topology.nbr_src[d]},
      right_rank{topology.nbr_dest[d]};

    size_t
      stride{topology.local_shape.strides[d]},
      dim_size{topology.local_shape.dims[d]};

    size_t
      send_left_idx{1 * stride},
      recv_right_idx{(dim_size - 1) * stride};
    size_t
      send_right_idx{(dim_size - 2) * stride},
      recv_left_idx{0 * stride};

    MPI_Sendrecv(
      &current_data[send_left_idx],
      1,
      topology.halos[d],
      left_rank,
      0,
      &current_data[recv_right_idx],
      1,
      topology.halos[d],
      right_rank,
      0,
      topology.comm_cart,
      MPI_STATUS_IGNORE);

    MPI_Sendrecv(
      &current_data[send_right_idx],
      1,
      topology.halos[d],
      right_rank,
      0,
      &current_data[recv_left_idx],
      1,
      topology.halos[d],
      left_rank,
      0,
      topology.comm_cart,
      MPI_STATUS_IGNORE);
  }
}

}  // namespace mpi_array