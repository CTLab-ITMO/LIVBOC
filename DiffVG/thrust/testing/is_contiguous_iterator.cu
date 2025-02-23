#include <unittest/unittest.h>
#include <thrust/detail/static_assert.h>
#include <iterator>
#include <vector>
#if THRUST_CPP_DIALECT >= 2011
  #include <array>
  #include <unordered_map>
  #include <unordered_set>
#endif
#include <string>
#if THRUST_CPP_DIALECT >= 2017
  #include <string_view>
#endif
#include <deque>
#include <list>
#include <map>
#include <set>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/type_traits/is_contiguous_iterator.h>

THRUST_STATIC_ASSERT((thrust::is_contiguous_iterator<
  std::string::iterator
>::value));

THRUST_STATIC_ASSERT((thrust::is_contiguous_iterator<
  std::wstring::iterator
>::value));

#if THRUST_CPP_DIALECT >= 2017
THRUST_STATIC_ASSERT((thrust::is_contiguous_iterator<
  std::string_view::iterator
>::value));

THRUST_STATIC_ASSERT((thrust::is_contiguous_iterator<
  std::wstring_view::iterator
>::value));
#endif

THRUST_STATIC_ASSERT((!thrust::is_contiguous_iterator<
  std::vector<bool>::iterator
>::value));

template <typename T>
__host__
void test_is_contiguous_iterator()
{
  THRUST_STATIC_ASSERT((thrust::is_contiguous_iterator<
    T*
  >::value));

  THRUST_STATIC_ASSERT((thrust::is_contiguous_iterator<
    T const*
  >::value));

  THRUST_STATIC_ASSERT((thrust::is_contiguous_iterator<
    thrust::device_ptr<T>
  >::value));

  THRUST_STATIC_ASSERT((thrust::is_contiguous_iterator<
    typename std::vector<T>::iterator
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_contiguous_iterator<
    typename std::vector<T>::reverse_iterator
  >::value));

  #if THRUST_CPP_DIALECT >= 2011
  THRUST_STATIC_ASSERT((thrust::is_contiguous_iterator<
    typename std::array<T, 1>::iterator
  >::value));
  #endif

  THRUST_STATIC_ASSERT((!thrust::is_contiguous_iterator<
    typename std::list<T>::iterator
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_contiguous_iterator<
    typename std::deque<T>::iterator
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_contiguous_iterator<
    typename std::set<T>::iterator
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_contiguous_iterator<
    typename std::multiset<T>::iterator
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_contiguous_iterator<
    typename std::map<T, T>::iterator
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_contiguous_iterator<
    typename std::multimap<T, T>::iterator
  >::value));

  #if THRUST_CPP_DIALECT >= 2011
  THRUST_STATIC_ASSERT((!thrust::is_contiguous_iterator<
    typename std::unordered_set<T>::iterator
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_contiguous_iterator<
    typename std::unordered_multiset<T>::iterator
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_contiguous_iterator<
    typename std::unordered_map<T, T>::iterator
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_contiguous_iterator<
    typename std::unordered_multimap<T, T>::iterator
  >::value));
  #endif

  THRUST_STATIC_ASSERT((!thrust::is_contiguous_iterator<
    std::istream_iterator<T>
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_contiguous_iterator<
    std::ostream_iterator<T>
  >::value));
}
DECLARE_GENERIC_UNITTEST(test_is_contiguous_iterator);

template <typename Vector>
__host__
void test_is_contiguous_iterator_vectors()
{
  THRUST_STATIC_ASSERT((thrust::is_contiguous_iterator<
    typename Vector::iterator
  >::value));
}
DECLARE_VECTOR_UNITTEST(test_is_contiguous_iterator_vectors);

