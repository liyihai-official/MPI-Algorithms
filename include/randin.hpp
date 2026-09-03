///
/// @file randin.cc
/// @brief This file provides class of random number generator
///        based on given seed. And number of filling methods
///        for multi-dimensional array and C style array.
///
/// @author Yihai Li
/// @date Sept. 2 2026
///
///

// includes
#include <algorithm>
#include <concepts>
#include <random>
#include <span>
#include <type_traits>
#include <vector>

#include "multiarray.hpp"

///
/// @brief Random number generator (rng) class for filling arrays
/// @tparam T Value type of rng and the requirement of arithmetic type.
/// @note This class uses std::mt19937_64 as the random number generator.
///
template <typename T>
  requires std::is_arithmetic_v<T>
class array_randomizer
{
 private:
  using Distribution_Type = std::conditional_t<
    std::is_integral_v<T>,
    std::uniform_int_distribution<T>,
    std::uniform_real_distribution<T>>;

 private:
  std::mt19937_64 gen;
  Distribution_Type dist;

 public:
  /// @brief Constructor with seed for reproducibility.
  /// @param min_val The min value for the distribution.
  /// @param max_val The max value for the distribution.
  array_randomizer(
    T min_val = T{0},
    T max_val = T{1.0})
    : gen(std::random_device{}()),
      dist(min_val, max_val) {}

  /// @brief Constructor with seed for reproducibility.
  /// @param seed The seed for the random number generator.
  /// @param min_val The min value for the distribution.
  /// @param max_val The max value for the distribution.
  array_randomizer(
    std::mt19937_64::result_type seed,
    T min_val = T{0},
    T max_val = T{1.0})
    : gen(seed),
      dist(min_val, max_val) {}

  /// @brief Operator operator() to generate a random number.
  /// @return A random number of type T.
  T operator()() { return dist(gen); }

  /// @brief C style array randomizer.
  /// @tparam T The type of the elements in the array.
  /// @param matrix The pointer to the array to be filled
  ///               with random numbers.
  /// @param total_elem The total number of elements in the array.
  void fill(T* matrix, size_t total_elem)
  {
    std::generate_n(
      matrix,
      total_elem,
      [this]()
      { return dist(gen); });
  }

  ///
  /// @brief Fills a multi-dimensional array with random numbers.
  /// @tparam T The type of the elements in the array.
  /// @param matrix The multi-dimensional array to be filled
  ///                with random numbers.
  /// @note This function uses the std::ranges::generate algorithm.
  /// @note std::span is used as C++20 requirements.
  ///
  void fill(std::span<T> matrix)
  {
    std::ranges::generate(
      matrix,
      [this]()
      { return dist(gen); });
  }

  /// @brief Fills a multi-dimensional array with random numbers.
  /// @param begin beginning of iterator
  /// @param end end of iterator
  template <std::forward_iterator Iter>
  void fill(Iter begin, Iter end)
  {
    std::generate(
      begin,
      end,
      [this]()
      { return dist(gen); });
  }

  /// @brief Fills a multi-dimensional array with random numbers.
  /// @tparam Range Overload for output range.
  /// @param range The output range to be filled with random numbers.
  template <std::ranges::output_range<T> Range>
    requires(!std::convertible_to<Range, std::span<T>>)
  void fill(Range& range)
  {
    std::ranges::generate(
      range,
      [this]()
      { return dist(gen); });
  }

};  // end of class array_randomizer
