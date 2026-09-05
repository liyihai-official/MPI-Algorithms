///
/// @file multiarray.hpp
/// @brief This file provides a multi-dimensional array and its shape templates.
///
/// @author Yihai Li
/// @date Sept. 2 2026
/// @version 1.0
/// @note This file is part of the bruck algorithm demonstration project.
///
#ifndef MULTIARRAY_HPP_YIHAI
#define MULTIARRAY_HPP_YIHAI

#pragma once
/// includes
#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <iomanip>
#include <memory>

/// @brief Namespace for multi-dimensional array utilities.
namespace multi_array
{

/// @brief Represents the shape of a multi-dimensional array.
/// @tparam NumD The number of dimensions.
template <size_t NumD>
struct multi_array_shape
{
  // Type Aliases
  typedef const size_t const_size;
  typedef size_t& reference;
  typedef const size_t& const_reference;

  // Member Variables
  std::array<size_t, NumD> dims, strides;

  // Cons & Decons
  multi_array_shape() noexcept;
  multi_array_shape(const multi_array_shape&) noexcept;
  multi_array_shape(multi_array_shape&&) noexcept;
  multi_array_shape& operator=(const multi_array_shape&) noexcept;
  multi_array_shape& operator=(multi_array_shape&&) noexcept;

  // Operators
  bool operator==(const multi_array_shape&) noexcept;
  bool operator!=(const multi_array_shape&) noexcept;

  /// @brief Constructs a multi-dimensional array shape.
  /// @tparam ...Exts The types of the shape dimensions.
  /// @param ... Shape dimensions.
  template <typename... Exts>
  explicit multi_array_shape(Exts...);

  /// @brief Computes the strides for the multi-dimensional array shape.
  void compute_strides();
};

/// @brief Represents a multi-dimensional array.
/// @tparam T Type of the elements in the array.
/// @tparam NumD Number of dimensions of the array.
template <typename T, size_t NumD>
class array
{
  /// @brief Type aliases for the array class.
 public:
  using value_type = T;
  using reference = T&;
  using const_reference = const T&;
  using pointer = T*;
  using const_pointer = const T*;
  using iterator = T*;
  using const_iterator = const T*;

  /// Member Variables
 private:
  multi_array_shape<NumD> shape_;
  std::unique_ptr<T[]> data_;
  size_t total_size_;

  // cons & decon
 public:
  array() noexcept;
  template <std::integral... Exts>
  explicit array(Exts...);
  explicit array(const multi_array_shape<NumD>&);

  array(const array&) noexcept;
  array(array&&) noexcept;
  array& operator=(array&&) noexcept;
  array& operator=(const array&) noexcept;
  ~array() noexcept = default;

  // Interface for MPI
 public:
  pointer data() noexcept { return data_.get(); }
  const_pointer data() const noexcept { return data_.get(); }
  size_t size() const noexcept { return total_size_; }
  const multi_array_shape<NumD>& shape() const noexcept { return shape_; }

  // Iterator
  iterator begin() noexcept { return data_.get(); }
  const_iterator begin() const noexcept { return data_.get(); }
  const_iterator cbegin() const noexcept { return data_.get(); }

  iterator end() noexcept { return data_.get() + total_size_; }
  const_iterator end() const noexcept { return data_.get() + total_size_; }
  const_iterator cend() const noexcept { return data_.get() + total_size_; }

  // Member Operation
  void swap(array&) noexcept;

  // Element Access
 public:
  template <std::integral... Exts>
  inline reference operator()(Exts...);

  template <std::integral... Exts>
  inline const_reference operator()(Exts...) const;

  inline reference operator[](size_t);
  inline const_reference operator[](size_t) const;

 private:
  inline void total_size() noexcept;
};  // end of array

/// @brief Overloads the << operator for printing the array.
/// @tparam T Type of the elements in the array.
/// @tparam NumD Number of dimensions of the array.
/// @param os Output stream.
/// @param arr  The multi-dimensional array to be printed.
/// @return Reference to the output stream.
template <typename T, size_t NumD>
std::ostream& operator<<(
  std::ostream& os,
  const array<T, NumD>& arr)
{
  auto print_recursive = [&](
                           auto&& self,
                           size_t current_dim,
                           size_t offset) -> void
  {
    if (current_dim == NumD - 1)
    {
      // os << "|";
      for (size_t i = 0; i < arr.shape().dims[current_dim]; ++i)
      {
        os << std::fixed
           << std::setprecision(0)
           << std::setw(6)
           << arr[offset + i];
      }
      // os << " |\n";
      os << "\n";
    }
    else
    {
      for (size_t i = 0; i < arr.shape().dims[current_dim]; ++i)
      {
        self(
          self,
          current_dim + 1,
          offset + i * arr.shape().strides[current_dim]);
      }
      os << "\n";
    }
  };
  print_recursive(print_recursive, 0, 0);

  return os;
}
}  // end of namespace multi_array

#include "multiarray.inl"

#endif