///
/// @file multiarray.inl
/// @brief This file contains the implementation of the
///        multi-dimensional array class and its associated
///        shape class.
///
/// @author Yihai Li
/// @date Sept. 2 2026
/// @note This file provides the implementation of the
/// multi-dimensional array class and its associated shape class.
///

/// @brief Namespace for multi-dimensional array utilities.
namespace multi_array
{

////////////////////////////////////////////////////////////////////////////////
/*
 *              Member functions of struct multi_array_shape
 */
////////////////////////////////////////////////////////////////////////////////

/// @brief Constructs a multi-dimensional array shape with all
///         dimensions initialized to 0.
/// @tparam NumD The number of dimensions.
template <size_t NumD>
inline multi_array_shape<NumD>::multi_array_shape() noexcept
{
  dims.fill(0);
  strides.fill(0);
}

/// @brief Copy constructor. Constructs a multi-dimensional
///        array shape by copying another shape.
///
/// @tparam NumD The number of dimensions.
/// @param other The shape to copy.
template <size_t NumD>
inline multi_array_shape<NumD>::multi_array_shape(
  const multi_array_shape& other) noexcept
  : dims(other.dims),
    strides(other.strides)
{
}

/// @brief Move constructor. Constructs a multi-dimensional
/// @tparam NumD Number of dimensions.
/// @param other The shape to move from.
template <size_t NumD>
inline multi_array_shape<NumD>::multi_array_shape(
  multi_array_shape&& other) noexcept
  : dims(std::move(other.dims)),
    strides(std::move(other.strides))
{
}

/// @brief Copy assignment operator.
/// @tparam NumD Number of dimensions.
/// @param other The shape to copy for.
/// @return A reference to the assigned shape.
template <size_t NumD>
inline multi_array_shape<NumD>&
multi_array_shape<NumD>::operator=(
  const multi_array_shape& other) noexcept
{
  if (this != &other)
  {
    dims = other.dims;
    strides = other.strides;
  }
  return *this;
}

/// @brief Move assignment operator.
/// @tparam NumD Number of dimensions.
/// @param other The shape to move from.
/// @return A reference to the assigned shape.
template <size_t NumD>
inline multi_array_shape<NumD>&
multi_array_shape<NumD>::operator=(
  multi_array_shape&& other) noexcept
{
  if (this != &other)
  {
    dims = std::move(other.dims);
    strides = std::move(other.strides);
  }
  return *this;
}

/// @brief Constructs a multi-dimensional array shape with
///         the specified dimensions.
/// @tparam NumD Number of dimensions.
/// @param ...exts The extents for each dimension.
template <size_t NumD>
template <typename... Exts>
inline multi_array_shape<NumD>::multi_array_shape(Exts... exts)
{
  static_assert(
    sizeof...(exts) == NumD,
    "Arguments count must match NumD");

  dims = {static_cast<size_t>(exts)...};
  compute_strides();
}

/// @brief Computes the strides for the multi-dimensional array shape.
/// @tparam NumD Number of dimensions.
template <size_t NumD>
inline void multi_array_shape<NumD>::compute_strides()
{
  if constexpr (NumD == 0) return;

  strides[NumD - 1] = 1;
  for (size_t i = NumD - 1; i > 0; --i)
    strides[i - 1] = strides[i] * dims[i];
}

/// @brief Constructor for the multi-dimensional array class.
/// @tparam T The type of the elements in the array.
/// @tparam NumD Number of dimensions.
template <typename T, size_t NumD>
array<T, NumD>::array() noexcept
  : shape_(),
    data_(nullptr),
    total_size_(0)
{
}

////////////////////////////////////////////////////////////////////////////////
/*
 *                     Member functions of class array
 */
////////////////////////////////////////////////////////////////////////////////

/// @brief Constructor for the multi-dimensional array class
///         with specified dimensions.
/// @tparam T The type of the elements in the array.
/// @tparam NumD Number of dimensions.
/// @param ...exts The extents for each dimension.
template <typename T, size_t NumD>
template <std::integral... Exts>
array<T, NumD>::array(Exts... exts)
  : shape_(exts...)
{
  total_size();
  data_ = std::make_unique<T[]>(total_size_);
}

/// @brief Copy constructor for the multi-dimensional array class.
/// @tparam T The type of the elements in the array.
/// @tparam NumD Number of dimensions.
/// @param other The array to copy from.
template <typename T, size_t NumD>
array<T, NumD>::array(const array& other) noexcept
  : shape_(other.shape_),
    total_size_(other.total_size_)
{
  if (total_size_ > 0)
  {
    data_ = std::make_unique<T[]>(total_size_);
    std::copy(
      other.data_.get(),
      other.data_.get() + total_size_,
      data_.get());
  }
}

/// @brief Move constructor for the multi-dimensional array class.
/// @tparam T The type of the elements in the array.
/// @tparam NumD Number of dimensions.
/// @param other The array to move from.
template <typename T, size_t NumD>
array<T, NumD>::array(array&& other) noexcept
  : shape_(std::move(other.shape_)),
    data_(std::move(other.data_)),
    total_size_(other.total_size_)
{
  other.total_size_ = 0;
}

/// @brief Move assignment operator for the multi-dimensional
///         array class.
/// @tparam T The type of the elements in the array.
/// @tparam NumD Number of dimensions.
/// @param other The array to move from.
/// @return A reference to the assigned array.
template <typename T, size_t NumD>
array<T, NumD>&
array<T, NumD>::operator=(array&& other) noexcept
{
  if (this != &other)
  {
    shape_ = std::move(other.shape_);
    data_ = std::move(other.data_);

    other.data_ = nullptr;
    other.shape_ = multi_array_shape<NumD>();
    other.total_size_ = 0;
  }
  return *this;
}

/// @brief Copy assignment operator for the multi-dimensional
///         array class.
/// @tparam T The type of the elements in the array.
/// @tparam NumD Number of dimensions.
/// @param other The array to copy from.
/// @return A reference to the assigned array.
template <typename T, size_t NumD>
array<T, NumD>&
array<T, NumD>::operator=(const array& other) noexcept
{
  if (this != &other)
  {
    shape_ = other.shape_;
    data_ = other.data_;
    total_size_ = other.total_size_;
  }
  return *this;
}

/// @brief Access an element in the multi-dimensional array
///         using multi-dimensional indices.
/// @tparam ...Exts The types of the indices.
/// @param ...exts The indices for each dimension.
/// @return A reference to the accessed element.
template <typename T, size_t NumD>
template <std::integral... Exts>
inline typename array<T, NumD>::reference
array<T, NumD>::operator()(Exts... exts)
{
  static_assert(
    sizeof...(exts) == NumD,
    "Arguments count must match NumD.");

  std::array<size_t, NumD> indices = {static_cast<size_t>(exts)...};

  size_t flat_idx = 0;
  for (size_t i = 0; i < NumD; ++i)
  {
    assert(indices[i] < shape_.dims[i] && "Index out of bounds!");
    flat_idx += indices[i] * shape_.strides[i];
  }

  return data_[flat_idx];
}

// @brief Access a constant element in the multi-dimensional
///         array using multi-dimensional indices.
/// @tparam ...Exts The types of the indices.
/// @param ...exts The indices for each dimension.
/// @return A constreference to the accessed element.
template <typename T, size_t NumD>
template <std::integral... Exts>
inline typename array<T, NumD>::const_reference
array<T, NumD>::operator()(Exts... exts) const
{
  static_assert(
    sizeof...(exts) == NumD,
    "Arguments count must match NumD.");

  std::array<size_t, NumD> indices = {static_cast<size_t>(exts)...};

  size_t flat_idx = 0;
  for (size_t i = 0; i < NumD; ++i)
  {
    assert(indices[i] < shape_.dims[i] && "Index out of bounds!");
    flat_idx += indices[i] * shape_.strides[i];
  }

  return data_[flat_idx];
}

/// @brief Access an element in the multi-dimensional array
///         using a flat index.
/// @tparam T The type of the elements in the array.
/// @tparam NumD Number of dimensions.
/// @param flat_idx The flat index of the element to access.
/// @return A reference to the accessed element.
template <typename T, size_t NumD>
inline array<T, NumD>::reference
array<T, NumD>::operator[](size_t flat_idx)
{
  assert(flat_idx < total_size_ && "Flat index out of range!");
  return data_[flat_idx];
}

/// @brief Access a constant element in the multi-dimensional array
///         using a flat index.
/// @tparam T The type of the elements in the array.
/// @tparam NumD Number of dimensions.
/// @param flat_idx The flat index of the element to access.
/// @return A const reference to the accessed element.
template <typename T, size_t NumD>
inline array<T, NumD>::const_reference
array<T, NumD>::operator[](size_t flat_idx) const
{
  assert(flat_idx < total_size_ && "Flat index out of range!");
  return data_[flat_idx];
}

/// @brief Calculate the total size of the array.
/// @tparam T The type of the elements in the array.
/// @tparam NumD Number of dimensions.
template <typename T, size_t NumD>
inline void array<T, NumD>::total_size() noexcept
{
  total_size_ = 1;
  for (size_t i = 0; i < NumD; ++i)
    total_size_ *= shape_.dims[i];
}

template <typename T, size_t NumD>
inline void array<T, NumD>::swap(array<T, NumD>& other) noexcept
{
  std::swap(data_, other.data_);
  std::swap(shape_, other.shape_);
  std::swap(total_size_, other.total_size_);
}

}  // end of namespace multi_array
