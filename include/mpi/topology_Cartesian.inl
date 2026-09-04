///
/// @file topology_Cartesian.inl
/// @brief This file contains the implementation of the
///        Cartesian structure implementation in mpi_topology
///        namespace.
///
/// @author Yihai Li
/// @date Sept. 3 2026
///

namespace mpi_topology
{

///
/// @brief Move constructor of MPI Cartesian structure
/// @tparam T Value type
/// @tparam NumD Number of dimensions.
/// @param other Other Cartesian structure
template <typename T, size_t NumD>
Cartesian<T, NumD>::Cartesian(Cartesian&& other) noexcept
  : global_shape(std::move(other.global_shape)),
    local_shape(std::move(other.local_shape)),
    starts(std::move(other.starts)),
    ends(std::move(other.ends)),
    dims(std::move(other.dims)),
    periods(std::move(other.periods)),
    nbr_src(std::move(other.nbr_src)),
    nbr_dest(std::move(other.nbr_dest)),
    coordinates(std::move(other.coordinates)),
    halos(std::move(other.halos)),
    rank(std::move(other.rank)),
    comm_cart(other.comm_cart)
{
  other.comm_cart = MPI_COMM_NULL;
  other.rank = -1;
}

///
/// @brief Move assignment operator of MPI Cartesian structure
/// @tparam T Value type
/// @tparam NumD Number of dimensions.
/// @param other Other Cartesian structure
/// @return Reference of Cartesian structure
template <typename T, size_t NumD>
Cartesian<T, NumD>&
Cartesian<T, NumD>::operator=(Cartesian<T, NumD>&& other) noexcept
{
  if (this != &other)
  {
    // Avoiding communicator conflicts
    if (comm_cart != MPI_COMM_NULL && comm_cart != MPI_COMM_WORLD)
    {
      MPI_Comm_free(&comm_cart);
    }

    global_shape(std::move(other.global_shape));
    local_shape(std::move(other.local_shape));
    starts(std::move(other.starts));
    ends(std::move(other.ends));
    dims(std::move(other.dims));
    periods(std::move(other.periods));
    nbr_src(std::move(other.nbr_src));
    nbr_dest(std::move(other.nbr_dest));
    coordinates(std::move(other.coordinates));
    halos(std::move(other.halos));
    rank(std::move(other.rank));
    comm_cart(other.comm_cart);

    other.comm_cart = MPI_COMM_NULL;
    other.rank = -1;
  }
}

///
/// @brief Deconstructor of MPI Cartesian structure manage
///       MPI resources.
/// @tparam T Value type
/// @tparam NumD Number of dimensions.
template <typename T, size_t NumD>
Cartesian<T, NumD>::~Cartesian()
{
  if (comm_cart != MPI_COMM_NULL && comm_cart != MPI_COMM_WORLD)
  {
    MPI_Comm_free(&comm_cart);
  }
  /// TODO:
  /// 注意：如果 halos 中的 MPI_Datatype 是被 Commit 过的，这里也需要循环调用 MPI_Type_free
}

///
/// @brief explicit Constructor from a global multiarray_shape
/// @tparam T Value type
/// @tparam NumD Number of dimension
/// @param glob_shape multiarray_shape structure, shows the
///                  shape of big array been divided.
/// @param comm The correspond MPI Communicator.
template <typename T, size_t NumD>
Cartesian<T, NumD>::Cartesian(
  const array_shape& glob_shape,
  MPI_Comm comm)
  : global_shape(glob_shape),
    local_shape(glob_shape)
{
  int size, orig_rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  // Initializing
  dims.fill(0);
  periods.fill(1);  // Turn On Periods of Boundaries
  halos.fill(MPI_DATATYPE_NULL);
  nbr_src.fill(0);
  nbr_dest.fill(0);

  // MPI_Topology
  MPI_Dims_create(size, NumD, dims.data());

  MPI_Cart_create(
    comm,
    NumD,
    dims.data(),
    periods.data(),
    0,
    &comm_cart);

  if (comm_cart != MPI_COMM_NULL)
  {
    MPI_Comm_rank(comm_cart, &rank);
    MPI_Comm_size(comm_cart, &size);

    MPI_Cart_coords(
      comm_cart,
      rank,
      NumD,
      coordinates.data());

    /// @brief A helper Function, provides
    ///         the decomposition routine.
    auto Decomp = [](
                    const int n,
                    const int prob_size,
                    const int rank,
                    int& s,
                    int& e)
    {
      int n_loc{n / prob_size}, remain{n % prob_size};

      s = rank * n_loc + 1;
      s += ((rank < remain) ? rank : remain);

      if (rank < remain) ++n_loc;
      e = s + n_loc - 1;

      if (e > n || rank == prob_size - 1) e = n;
      return 0;
    };

    // Get neighbors and local shapes
    for (size_t i = 0; i < NumD; ++i)
    {
      Decomp(
        global_shape.dims[i],
        dims[i],
        coordinates[i],
        starts[i],
        ends[i]);

      local_shape.dims[i] = ends[i] - starts[i] + 1 + 2;  // Add halos
      MPI_Cart_shift(
        comm_cart,
        i,
        1,
        &nbr_src[i],
        &nbr_dest[i]);
    }

    local_shape.compute_strides();
  }
}

}  // namespace mpi_topology
