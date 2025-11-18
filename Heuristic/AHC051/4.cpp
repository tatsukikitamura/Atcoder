// -*- compile-command: "make" -*-
#pragma GCC optimize "-O3,omit-frame-pointer,inline,unroll-all-loops,fast-math"
 // #pragma GCC target "tune=native"  // Commented out for ARM architecture
 #include <iostream>
 #include <vector>
 #include <string>
 #include <queue>
 #include <deque>
 #include <set>
 #include <map>
 #include <tuple>      // tie を使うため
 #include <algorithm>
 #include <chrono>   
 #include <cmath>     
 #include <random>    
 #include <numeric>    // iota を使うため (もしC++11より前なら不要)
#include <sys/time.h>
// #include <immintrin.h>  // x86 only, not available on ARM
// #include <x86intrin.h>  // x86 only, not available on ARM
#include <ext/pb_ds/assoc_container.hpp>
/*
    pdqsort.h - Pattern-defeatin g quicksort.
    Copyright (c) 2015 Orson Peters
    This software is provided 'as-is', without any express or implied warranty. In no event will the
    authors be held liable for any damages arising from the use of this software.
    Permission is granted to anyone to use this software for any purpose, including commercial
    applications, and to alter it and redistribute it freely, subject to the following restrictions:
    1. The origin of this software must not be misrepresented; you must not claim that you wrote the
       original software. If you use this software in a product, an acknowledgment in the product
       documentation would be appreciated but is not required.
    2. Altered source versions must be plainly marked as such, and must not be misrepresented as
       being the original software.
    3. This notice may not be removed or altered from any source distribution.
*/
namespace pdqsort_detail {
    enum {
        // Partitions below this size are sorted using insertion sort.
        insertion_sort_threshold = 24,
        // Partitions above this size use Tukey's ninther to select the pivot.
        ninther_threshold = 128,
        // When we detect an already sorted partition, attempt an insertion sort that allows this
        // amount of element moves before giving up.
        partial_insertion_sort_limit = 8,
        // Must be multiple of 8 due to loop unrolling, and < 256 to fit in unsigned char.
        block_size = 64,
        // Cacheline size, assumes power of two.
        cacheline_size = 64
    };
    template<class T> struct is_default_compare : std::false_type { };
    template<class T> struct is_default_compare<std::less<T>> : std::true_type { };
    template<class T> struct is_default_compare<std::greater<T>> : std::true_type { };
    // Returns floor(log2(n)), assumes n > 0.
    template<class T>
    inline int log2(T n) {
        int log = 0;
        while (n >>= 1) ++log;
        return log;
    }
    // Sorts [begin, end) using insertion sort with the given comparison function.
    template<class Iter, class Compare>
    inline void insertion_sort(Iter begin, Iter end, Compare comp) {
        typedef typename std::iterator_traits<Iter>::value_type T;
        if (begin == end) return;
        for (Iter cur = begin + 1; cur != end; ++cur) {
            Iter sift = cur;
            Iter sift_1 = cur - 1;
            // Compare first so we can avoid 2 moves for an element already positioned correctly.
            if (comp(*sift, *sift_1)) {
                T tmp = std::move(*sift);
                do { *sift-- = std::move(*sift_1); }
                while (sift != begin && comp(tmp, *--sift_1));
                *sift = std::move(tmp);
            }
        }
    }
    // Sorts [begin, end) using insertion sort with the given comparison function. Assumes
    // *(begin - 1) is an element smaller than or equal to any element in [begin, end).
    template<class Iter, class Compare>
    inline void unguarded_insertion_sort(Iter begin, Iter end, Compare comp) {
        typedef typename std::iterator_traits<Iter>::value_type T;
        if (begin == end) return;
        for (Iter cur = begin + 1; cur != end; ++cur) {
            Iter sift = cur;
            Iter sift_1 = cur - 1;
            // Compare first so we can avoid 2 moves for an element already positioned correctly.
            if (comp(*sift, *sift_1)) {
                T tmp = std::move(*sift);
                do { *sift-- = std::move(*sift_1); }
                while (comp(tmp, *--sift_1));
                *sift = std::move(tmp);
            }
        }
    }
    // Attempts to use insertion sort on [begin, end). Will return false if more than
    // partial_insertion_sort_limit elements were moved, and abort sorting. Otherwise it will
    // successfully sort and return true.
    template<class Iter, class Compare>
    inline bool partial_insertion_sort(Iter begin, Iter end, Compare comp) {
        typedef typename std::iterator_traits<Iter>::value_type T;
        if (begin == end) return true;
        int limit = 0;
        for (Iter cur = begin + 1; cur != end; ++cur) {
            if (limit > partial_insertion_sort_limit) return false;
            Iter sift = cur;
            Iter sift_1 = cur - 1;
            // Compare first so we can avoid 2 moves for an element already positioned correctly.
            if (comp(*sift, *sift_1)) {
                T tmp = std::move(*sift);
                do { *sift-- = std::move(*sift_1); }
                while (sift != begin && comp(tmp, *--sift_1));
                *sift = std::move(tmp);
                limit += cur - sift;
            }
        }
        return true;
    }
    template<class Iter, class Compare>
    inline void sort2(Iter a, Iter b, Compare comp) {
        if (comp(*b, *a)) std::iter_swap(a, b);
    }
    // Sorts the elements *a, *b and *c using comparison function comp.
    template<class Iter, class Compare>
    inline void sort3(Iter a, Iter b, Iter c, Compare comp) {
        sort2(a, b, comp);
        sort2(b, c, comp);
        sort2(a, b, comp);
    }
    template<class T>
    inline T* align_cacheline(T* p) {
        std::size_t ip = reinterpret_cast<std::size_t>(p);
        ip = (ip + cacheline_size - 1) & -cacheline_size;
        return reinterpret_cast<T*>(ip);
    }
    template<class Iter>
    inline void swap_offsets(Iter first, Iter last,
                             unsigned char* offsets_l, unsigned char* offsets_r,
                             int num, bool use_swaps) {
        typedef typename std::iterator_traits<Iter>::value_type T;
        if (use_swaps) {
            // This case is needed for the descending distribution, where we need
            // to have proper swapping for pdqsort to remain O(n).
            for (int i = 0; i < num; ++i) {
                std::iter_swap(first + offsets_l[i], last - offsets_r[i]);
            }
        } else if (num > 0) {
            Iter l = first + offsets_l[0]; Iter r = last - offsets_r[0];
            T tmp(std::move(*l)); *l = std::move(*r);
            for (int i = 1; i < num; ++i) {
                l = first + offsets_l[i]; *r = std::move(*l);
                r = last - offsets_r[i]; *l = std::move(*r);
            }
            *r = std::move(tmp);
        }
    }
    // Partitions [begin, end) around pivot *begin using comparison function comp. Elements equal
    // to the pivot are put in the right-hand partition. Returns the position of the pivot after
    // partitioning and whether the passed sequence already was correctly partitioned. Assumes the
    // pivot is a median of at least 3 elements and that [begin, end) is at least
    // insertion_sort_threshold long. Uses branchless partitioning.
    template<class Iter, class Compare>
    inline std::pair<Iter, bool> partition_right_branchless(Iter begin, Iter end, Compare comp) {
        typedef typename std::iterator_traits<Iter>::value_type T;
        // Move pivot into local for speed.
        T pivot(std::move(*begin));
        Iter first = begin;
        Iter last = end;
        // Find the first element greater than or equal than the pivot (the median of 3 guarantees
        // this exists).
        while (comp(*++first, pivot));
        // Find the first element strictly smaller than the pivot. We have to guard this search if
        // there was no element before *first.
        if (first - 1 == begin) while (first < last && !comp(*--last, pivot));
        else while ( !comp(*--last, pivot));
        // If the first pair of elements that should be swapped to partition are the same element,
        // the passed in sequence already was correctly partitioned.
        bool already_partitioned = first >= last;
        if (!already_partitioned) {
            std::iter_swap(first, last);
            ++first;
        }
        // The following branchless partitioning is derived from "BlockQuicksort: How Branch
        // Mispredictions don’t affect Quicksort" by Stefan Edelkamp and Armin Weiss.
        unsigned char offsets_l_storage[block_size + cacheline_size];
        unsigned char offsets_r_storage[block_size + cacheline_size];
        unsigned char* offsets_l = align_cacheline(offsets_l_storage);
        unsigned char* offsets_r = align_cacheline(offsets_r_storage);
        int num_l, num_r, start_l, start_r;
        num_l = num_r = start_l = start_r = 0;
        while (last - first > 2 * block_size) {
            // Fill up offset blocks with elements that are on the wrong side.
            if (num_l == 0) {
                start_l = 0;
                Iter it = first;
                for (unsigned char i = 0; i < block_size;) {
                    offsets_l[num_l] = i++; num_l += !comp(*it, pivot); ++it;
                    offsets_l[num_l] = i++; num_l += !comp(*it, pivot); ++it;
                    offsets_l[num_l] = i++; num_l += !comp(*it, pivot); ++it;
                    offsets_l[num_l] = i++; num_l += !comp(*it, pivot); ++it;
                    offsets_l[num_l] = i++; num_l += !comp(*it, pivot); ++it;
                    offsets_l[num_l] = i++; num_l += !comp(*it, pivot); ++it;
                    offsets_l[num_l] = i++; num_l += !comp(*it, pivot); ++it;
                    offsets_l[num_l] = i++; num_l += !comp(*it, pivot); ++it;
                }
            }
            if (num_r == 0) {
                start_r = 0;
                Iter it = last;
                for (unsigned char i = 0; i < block_size;) {
                    offsets_r[num_r] = ++i; num_r += comp(*--it, pivot);
                    offsets_r[num_r] = ++i; num_r += comp(*--it, pivot);
                    offsets_r[num_r] = ++i; num_r += comp(*--it, pivot);
                    offsets_r[num_r] = ++i; num_r += comp(*--it, pivot);
                    offsets_r[num_r] = ++i; num_r += comp(*--it, pivot);
                    offsets_r[num_r] = ++i; num_r += comp(*--it, pivot);
                    offsets_r[num_r] = ++i; num_r += comp(*--it, pivot);
                    offsets_r[num_r] = ++i; num_r += comp(*--it, pivot);
                }
            }
            // Swap elements and update block sizes and first/last boundaries.
            int num = std::min(num_l, num_r);
            swap_offsets(first, last, offsets_l + start_l, offsets_r + start_r,
                         num, num_l == num_r);
            num_l -= num; num_r -= num;
            start_l += num; start_r += num;
            if (num_l == 0) first += block_size;
            if (num_r == 0) last -= block_size;
        }
        int l_size = 0, r_size = 0;
        int unknown_left = (last - first) - ((num_r || num_l) ? block_size : 0);
        if (num_r) {
            // Handle leftover block by assigning the unknown elements to the other block.
            l_size = unknown_left;
            r_size = block_size;
        } else if (num_l) {
            l_size = block_size;
            r_size = unknown_left;
        } else {
            // No leftover block, split the unknown elements in two blocks.
            l_size = unknown_left/2;
            r_size = unknown_left - l_size;
        }
        // Fill offset buffers if needed.
        if (unknown_left && !num_l) {
            start_l = 0;
            Iter it = first;
            for (unsigned char i = 0; i < l_size;) {
                offsets_l[num_l] = i++; num_l += !comp(*it, pivot); ++it;
            }
        }
        if (unknown_left && !num_r) {
            start_r = 0;
            Iter it = last;
            for (unsigned char i = 0; i < r_size;) {
                offsets_r[num_r] = ++i; num_r += comp(*--it, pivot);
            }
        }
        int num = std::min(num_l, num_r);
        swap_offsets(first, last, offsets_l + start_l, offsets_r + start_r, num, num_l == num_r);
        num_l -= num; num_r -= num;
        start_l += num; start_r += num;
        if (num_l == 0) first += l_size;
        if (num_r == 0) last -= r_size;
        // We have now fully identified [first, last)'s proper position. Swap the last elements.
        if (num_l) {
            offsets_l += start_l;
            while (num_l--) std::iter_swap(first + offsets_l[num_l], --last);
            first = last;
        }
        if (num_r) {
            offsets_r += start_r;
            while (num_r--) std::iter_swap(last - offsets_r[num_r], first), ++first;
            last = first;
        }
        // Put the pivot in the right place.
        Iter pivot_pos = first - 1;
        *begin = std::move(*pivot_pos);
        *pivot_pos = std::move(pivot);
        return std::make_pair(pivot_pos, already_partitioned);
    }
    // Partitions [begin, end) around pivot *begin using comparison function comp. Elements equal
    // to the pivot are put in the right-hand partition. Returns the position of the pivot after
    // partitioning and whether the passed sequence already was correctly partitioned. Assumes the
    // pivot is a median of at least 3 elements and that [begin, end) is at least
    // insertion_sort_threshold long.
    template<class Iter, class Compare>
    inline std::pair<Iter, bool> partition_right(Iter begin, Iter end, Compare comp) {
        typedef typename std::iterator_traits<Iter>::value_type T;
        // Move pivot into local for speed.
        T pivot(std::move(*begin));
        Iter first = begin;
        Iter last = end;
        // Find the first element greater than or equal than the pivot (the median of 3 guarantees
        // this exists).
        while (comp(*++first, pivot));
        // Find the first element strictly smaller than the pivot. We have to guard this search if
        // there was no element before *first.
        if (first - 1 == begin) while (first < last && !comp(*--last, pivot));
        else while ( !comp(*--last, pivot));
        // If the first pair of elements that should be swapped to partition are the same element,
        // the passed in sequence already was correctly partitioned.
        bool already_partitioned = first >= last;
        // Keep swapping pairs of elements that are on the wrong side of the pivot. Previously
        // swapped pairs guard the searches, which is why the first iteration is special-cased
        // above.
        while (first < last) {
            std::iter_swap(first, last);
            while (comp(*++first, pivot));
            while (!comp(*--last, pivot));
        }
        // Put the pivot in the right place.
        Iter pivot_pos = first - 1;
        *begin = std::move(*pivot_pos);
        *pivot_pos = std::move(pivot);
        return std::make_pair(pivot_pos, already_partitioned);
    }
    // Similar function to the one above, except elements equal to the pivot are put to the left of
    // the pivot and it doesn't check or return if the passed sequence already was partitioned.
    // Since this is rarely used (the many equal case), and in that case pdqsort already has O(n)
    // performance, no block quicksort is applied here for simplicity.
    template<class Iter, class Compare>
    inline Iter partition_left(Iter begin, Iter end, Compare comp) {
        typedef typename std::iterator_traits<Iter>::value_type T;
        T pivot(std::move(*begin));
        Iter first = begin;
        Iter last = end;
        while (comp(pivot, *--last));
        if (last + 1 == end) while (first < last && !comp(pivot, *++first));
        else while ( !comp(pivot, *++first));
        while (first < last) {
            std::iter_swap(first, last);
            while (comp(pivot, *--last));
            while (!comp(pivot, *++first));
        }
        Iter pivot_pos = last;
        *begin = std::move(*pivot_pos);
        *pivot_pos = std::move(pivot);
        return pivot_pos;
    }
    template<class Iter, class Compare, bool Branchless>
    inline void pdqsort_loop(Iter begin, Iter end, Compare comp, int bad_allowed, bool leftmost = true) {
        typedef typename std::iterator_traits<Iter>::difference_type diff_t;
        // Use a while loop for tail recursion elimination.
        while (true) {
            diff_t size = end - begin;
            // Insertion sort is faster for small arrays.
            if (size < insertion_sort_threshold) {
                if (leftmost) insertion_sort(begin, end, comp);
                else unguarded_insertion_sort(begin, end, comp);
                return;
            }
            // Choose pivot as median of 3 or pseudomedian of 9.
            diff_t s2 = size / 2;
            if (size > ninther_threshold) {
                sort3(begin, begin + s2, end - 1, comp);
                sort3(begin + 1, begin + (s2 - 1), end - 2, comp);
                sort3(begin + 2, begin + (s2 + 1), end - 3, comp);
                sort3(begin + (s2 - 1), begin + s2, begin + (s2 + 1), comp);
                std::iter_swap(begin, begin + s2);
            } else sort3(begin + s2, begin, end - 1, comp);
            // If *(begin - 1) is the end of the right partition of a previous partition operation
            // there is no element in [begin, end) that is smaller than *(begin - 1). Then if our
            // pivot compares equal to *(begin - 1) we change strategy, putting equal elements in
            // the left partition, greater elements in the right partition. We do not have to
            // recurse on the left partition, since it's sorted (all equal).
            if (!leftmost && !comp(*(begin - 1), *begin)) {
                begin = partition_left(begin, end, comp) + 1;
                continue;
            }
            // Partition and get results.
            std::pair<Iter, bool> part_result =
                Branchless ? partition_right_branchless(begin, end, comp)
                           : partition_right(begin, end, comp);
            Iter pivot_pos = part_result.first;
            bool already_partitioned = part_result.second;
            // Check for a highly unbalanced partition.
            diff_t l_size = pivot_pos - begin;
            diff_t r_size = end - (pivot_pos + 1);
            bool highly_unbalanced = l_size < size / 8 || r_size < size / 8;
            // If we got a highly unbalanced partition we shuffle elements to break many patterns.
            if (highly_unbalanced) {
                // If we had too many bad partitions, switch to heapsort to guarantee O(n log n).
                if (--bad_allowed == 0) {
                    std::make_heap(begin, end, comp);
                    std::sort_heap(begin, end, comp);
                    return;
                }
                if (l_size >= insertion_sort_threshold) {
                    std::iter_swap(begin, begin + l_size / 4);
                    std::iter_swap(pivot_pos - 1, pivot_pos - l_size / 4);
                    if (l_size > ninther_threshold) {
                        std::iter_swap(begin + 1, begin + (l_size / 4 + 1));
                        std::iter_swap(begin + 2, begin + (l_size / 4 + 2));
                        std::iter_swap(pivot_pos - 2, pivot_pos - (l_size / 4 + 1));
                        std::iter_swap(pivot_pos - 3, pivot_pos - (l_size / 4 + 2));
                    }
                }
                if (r_size >= insertion_sort_threshold) {
                    std::iter_swap(pivot_pos + 1, pivot_pos + (1 + r_size / 4));
                    std::iter_swap(end - 1, end - r_size / 4);
                    if (r_size > ninther_threshold) {
                        std::iter_swap(pivot_pos + 2, pivot_pos + (2 + r_size / 4));
                        std::iter_swap(pivot_pos + 3, pivot_pos + (3 + r_size / 4));
                        std::iter_swap(end - 2, end - (1 + r_size / 4));
                        std::iter_swap(end - 3, end - (2 + r_size / 4));
                    }
                }
            } else {
                // If we were decently balanced and we tried to sort an already partitioned
                // sequence try to use insertion sort.
                if (already_partitioned && partial_insertion_sort(begin, pivot_pos, comp)
                                        && partial_insertion_sort(pivot_pos + 1, end, comp)) return;
            }
            // Sort the left partition first using recursion and do tail recursion elimination for
            // the right-hand partition.
            pdqsort_loop<Iter, Compare, Branchless>(begin, pivot_pos, comp, bad_allowed, leftmost);
            begin = pivot_pos + 1;
            leftmost = false;
        }
    }
}
template<class Iter, class Compare>
inline void pdqsort(Iter begin, Iter end, Compare comp) {
    if (begin == end) return;
    pdqsort_detail::pdqsort_loop<Iter, Compare,
        pdqsort_detail::is_default_compare<typename std::decay<Compare>::type>::value &&
        std::is_arithmetic<typename std::iterator_traits<Iter>::value_type>::value>(
        begin, end, comp, pdqsort_detail::log2(end - begin));
}
template<class Iter>
inline void pdqsort(Iter begin, Iter end) {
    typedef typename std::iterator_traits<Iter>::value_type T;
    pdqsort(begin, end, std::less<T>());
}
template<class Iter, class Compare>
inline void pdqsort_branchless(Iter begin, Iter end, Compare comp) {
    if (begin == end) return;
    pdqsort_detail::pdqsort_loop<Iter, Compare, true>(
        begin, end, comp, pdqsort_detail::log2(end - begin));
}
template<class Iter>
inline void pdqsort_branchless(Iter begin, Iter end) {
    typedef typename std::iterator_traits<Iter>::value_type T;
    pdqsort_branchless(begin, end, std::less<T>());
}

using namespace std;
// Macros
using i8 = int8_t;
using u8 = uint8_t;
using i16 = int16_t;
using u16 = uint16_t;
using i32 = int32_t;
using u32 = uint32_t;
using i64 = int64_t;
using u64 = uint64_t;
using f32 = float;
using f64 = double;
template <class T> using min_queue = priority_queue<T, vector<T>, greater<T>>;
template <class T> using max_queue = priority_queue<T>;
struct uint64_hash {
  static inline uint64_t rotr(uint64_t x, unsigned k) {
    return (x >> k) | (x << (8U * sizeof(uint64_t) - k));
  }
  static inline uint64_t hash_int(uint64_t x) noexcept {
    auto h1 = x * (uint64_t)(0xA24BAED4963EE407);
    auto h2 = rotr(x, 32U) * (uint64_t)(0x9FB21C651E98DF25);
    auto h = rotr(h1 + h2, 32U);
    return h;
  }
  size_t operator()(uint64_t x) const {
    static const uint64_t FIXED_RANDOM = std::chrono::steady_clock::now().time_since_epoch().count();
    return hash_int(x + FIXED_RANDOM);
  }
};
template <typename K, typename V, typename Hash = uint64_hash>
using hash_map = __gnu_pbds::gp_hash_table<K, V, Hash>;
template <typename K, typename Hash = uint64_hash>
using hash_set = hash_map<K, __gnu_pbds::null_type, Hash>;
// Types
template<class T>
using min_queue = priority_queue<T, vector<T>, greater<T>>;
template<class T>
using max_queue = priority_queue<T>;
// Printing
template<class T>
void print_collection(ostream& out, T const& x);
template<class T, size_t... I>
void print_tuple(ostream& out, T const& a, index_sequence<I...>);
namespace std {
  template<class... A>
  ostream& operator<<(ostream& out, tuple<A...> const& x) {
    print_tuple(out, x, index_sequence_for<A...>{});
    return out;
  }
  template<class... A>
  ostream& operator<<(ostream& out, pair<A...> const& x) {
    print_tuple(out, x, index_sequence_for<A...>{});
    return out;
  }
  template<class A, size_t N>
  ostream& operator<<(ostream& out, array<A, N> const& x) { print_collection(out, x); return out; }
  template<class A>
  ostream& operator<<(ostream& out, vector<A> const& x) { print_collection(out, x); return out; }
  template<class A>
  ostream& operator<<(ostream& out, deque<A> const& x) { print_collection(out, x); return out; }
  template<class A>
  ostream& operator<<(ostream& out, multiset<A> const& x) { print_collection(out, x); return out; }
  template<class A, class B>
  ostream& operator<<(ostream& out, multimap<A, B> const& x) { print_collection(out, x); return out; }
  template<class A>
  ostream& operator<<(ostream& out, set<A> const& x) { print_collection(out, x); return out; }
  template<class A, class B>
  ostream& operator<<(ostream& out, map<A, B> const& x) { print_collection(out, x); return out; }
  template<class A, class B>
  ostream& operator<<(ostream& out, unordered_set<A> const& x) { print_collection(out, x); return out; }
}
template<class T, size_t... I>
void print_tuple(ostream& out, T const& a, index_sequence<I...>){
  using swallow = int[];
  out << '(';
  (void)swallow{0, (void(out << (I == 0? "" : ", ") << get<I>(a)), 0)...};
  out << ')';
}
template<class T>
void print_collection(ostream& out, T const& x) {
  int f = 0;
  out << '[';
  for(auto const& i: x) {
    out << (f++ ? "," : "");
    out << i;
  }
  out << "]";
}
// Random
struct RNG {
  uint64_t s[2];
  RNG(u64 seed) {
    reset(seed);
  }
  RNG() {
    reset(time(0));
  }
  using result_type = u32;
  constexpr u32 min(){ return numeric_limits<u32>::min(); }
  constexpr u32 max(){ return numeric_limits<u32>::max(); }
  u32 operator()() { return randomInt32(); }
  static __attribute__((always_inline)) inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
  }
  inline void reset(u64 seed) {
    struct splitmix64_state {
      u64 s;
      u64 splitmix64() {
        u64 result = (s += 0x9E3779B97f4A7C15);
        result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
        result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
        return result ^ (result >> 31);
      }
    };
    splitmix64_state sm { seed };
    s[0] = sm.splitmix64();
    s[1] = sm.splitmix64();
  }
  uint64_t next() {
    const uint64_t s0 = s[0];
    uint64_t s1 = s[1];
    const uint64_t result = rotl(s0 * 5, 7) * 9;
    s1 ^= s0;
    s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16); // a, b
    s[1] = rotl(s1, 37); // c
    return result;
  }
  inline u32 randomInt32() {
    return next();
  }
  inline u64 randomInt64() {
    return next();
  }
  inline u32 random32(u32 r) {
    return (((u64)randomInt32())*r)>>32;
  }
  inline u64 random64(u64 r) {
    return randomInt64()%r;
  }
  inline u32 randomRange32(u32 l, u32 r) {
    return l + random32(r-l+1);
  }
  inline u64 randomRange64(u64 l, u64 r) {
    return l + random64(r-l+1);
  }
  inline double randomDouble() {
    return (double)randomInt32() / 4294967296.0;
  }
  inline float randomFloat() {
    return (float)randomInt32() / 4294967296.0;
  }
  inline double randomRangeDouble(double l, double r) {
    return l + randomDouble() * (r-l);
  }
  template<class T>
  void shuffle(vector<T>& v) {
    i32 sz = v.size();
    for(i32 i = sz; i > 1; i--) {
      i32 p = random32(i);
      swap(v[i-1],v[p]);
    }
  }
  template<class T>
  void shuffle(T* fr, T* to) {
    i32 sz = distance(fr,to);
    for(int i = sz; i > 1; i--) {
      int p = random32(i);
      swap(fr[i-1],fr[p]);
    }
  }
  template<class T>
  inline int sample_index(vector<T> const& v) {
    return random32(v.size());
  }
  template<class T>
  inline T sample(vector<T> const& v) {
    return v[sample_index(v)];
  }
} rng;
// Timer
struct timer {
  chrono::high_resolution_clock::time_point t_begin;
  timer() {
    t_begin = chrono::high_resolution_clock::now();
  }
  void reset() {
    t_begin = chrono::high_resolution_clock::now();
  }
  float elapsed() const {
    return chrono::duration<float>(chrono::high_resolution_clock::now() - t_begin).count();
  }
};
// Util
template<class T>
T& smin(T& x, T const& y) { x = min(x,y); return x; }
template<class T>
T& smax(T& x, T const& y) { x = max(x,y); return x; }
template<typename T>
int sgn(T val) {
  if(val < 0) return -1;
  if(val > 0) return 1;
  return 0;
}
static inline
string int_to_string(int val, int digits = 0) {
  string s = to_string(val);
  reverse(begin(s), end(s));
  while((int)s.size() < digits) s.push_back('0');
  reverse(begin(s), end(s));
  return s;
}
// Debug
static inline void debug_impl_seq() {
  cerr << "}";
}
template <class T, class... V>
void debug_impl_seq(T const& t, V const&... v) {
  cerr << t;
  if(sizeof...(v)) { cerr << ", "; }
  debug_impl_seq(v...);
}
// Bits
__attribute__((always_inline)) inline
u64 bit(u64 x) { return 1ull<<x; }
__attribute__((always_inline)) inline
u64 popcount(u64 x) { return __builtin_popcountll(x); }
__attribute__((always_inline)) inline
void setbit(u64& a, u32 b, u64 value = 1) {
  a = (a&~bit(b)) | (value<<b);
}
__attribute__((always_inline)) inline
u64 getbit(u64 a, u32 b) {
  return (a>>b)&1;
}
__attribute__((always_inline)) inline
u64 lsb(u64 a) {
  return __builtin_ctzll(a);
}
__attribute__((always_inline)) inline
int msb(uint64_t bb) {
  return __builtin_clzll(bb) ^ 63;
}
f32 env_param(char const* name, f32 value) {
  if(char const* param = std::getenv(name)) {
    value = stof(param);
  }
  cerr << name << ": " << value << endl;
  return value;
}
/*
 * TODO list:
 *  when computing the binary tree, use different size budgets.
 *    budget(mask) = (popcount(mask) / (nwaste-1)) ^ alpha    * nlocations
 *    for alpha >= 1.0
 *  find optimal value of alpha for each N
 */
const f32 TL_DECOMP = 0.23;
const f32 TL0 = 1.84;
const f32 TL1 = 1.98;
const f32 sa_temp0 = 0.00015;
const f32 sa_temp1 = 0.00015;
const f32 TL_ANGLE_RATIO = 0.24;
const int SMASK_LIMIT = 8;
const f32 SPLITTING_3_SIZE_MULT = 0.93;
int BW = 9;
int SIZES_STEPS;
int SIZES_RANGE = 25;
f32 SIZES_LOWER_LIMIT, SIZES_UPPER_LIMIT;
void init_params(int n) {
  if(n <= 10) {
    SIZES_STEPS = lerp(2000, 1800, (n-5)/5.0);
    SIZES_LOWER_LIMIT = lerp(0.706, 0.680, (n-5)/5.0);
    SIZES_UPPER_LIMIT = lerp(2.023, 2.039, (n-5)/5.0);
  }else if(n <= 15) {
    BW = 8;
    SIZES_STEPS = lerp(1800, 1300, (n-10)/5.0);
    SIZES_LOWER_LIMIT = lerp(0.680, 0.620, (n-10)/5.0);
    SIZES_UPPER_LIMIT = lerp(2.039, 1.961, (n-10)/5.0);
  }else{
    BW = 6;
    SIZES_STEPS = lerp(1300, 600, (n-15)/5.0);
    SIZES_LOWER_LIMIT = lerp(0.620, 0.5, (n-15)/5.0);
    SIZES_UPPER_LIMIT = lerp(1.961, 1.62, (n-15)/5.0);
  }
}
const int MAXBW = 9;
timer TIMER;
const int MAXN = 20;
const int MAXM = 1000;
const int MAXK = 160;
const int MAXPOS = 1+MAXM+MAXN;
namespace geom_2d {
struct pt {
  i32 x,y;
  pt() { x = y = 0; }
  pt(int x_, int y_) { x = x_; y = y_; }
  void read() { cin>>x>>y; }
  pt operator+(pt const& o) const {
    return pt(x-o.x, y-o.y);
  }
  pt operator-(pt const& o) const {
    return pt(x-o.x, y-o.y);
  }
  auto operator<=>(const pt&) const = default;
  f32 atan2() const {
    return std::atan2<f32>(y, x);
  }
};
ostream& operator<<(ostream& os, pt const& p) {
  return os << make_tuple(p.x, p.y);
}
int dot(pt const& a, pt const& b) {
  return a.x*b.x+a.y*b.y;
}
int len2(pt const& a) {
  return dot(a,a);
}
int dist2(pt const& a, pt const& b) {
  return len2(a-b);
}
int det(pt const& a, pt const& b) {
  return a.x*b.y - a.y*b.x;
}
int signed_area(pt const &a, pt const& b, pt const& c) {
  return det(b-a, c-a);
}
int orientation(pt const& a, pt const& b, pt const& c) {
  return det(b-a, c-a);
}
bool intersects(pt const& p1, pt const& p2, pt const& q1, pt const& q2) {
  if(p1 == q1 || p1 == q2 || p2 == q1 || p2 == q2) {
    return false;
  }
  i64 o1 = orientation(p1, p2, q1);
  i64 o2 = orientation(p1, p2, q2);
  i64 o3 = orientation(q1, q2, p1);
  i64 o4 = orientation(q1, q2, p2);
  return (o1 * o2 <= 0) && (o3 * o4 <= 0);
}
bool intersects(array<pt, 2> const& p, array<pt, 2> const& q) {
  return intersects(p[0],p[1],q[0],q[1]);
}
void ch_add_point(vector<pt> &s, int sign, pt p) {
  while(s.size() >= 2 &&
        det((*(s.end()-2))-s.back(), p-s.back())*sign >= 0) {
    s.pop_back();
  }
  s.push_back(p);
}
vector<pt> convex_hull(vector<pt> A) {
  if(A.size() <= 2) return A;
  vector<pt> c[2];
  sort(A.begin(), A.end());
  for(i32 i = 0; i < (i32)(2); ++i) for(pt p : A) ch_add_point(c[i], 2 * i - 1, p);
  c[0].insert(c[0].end(), c[1].rbegin() + 1, c[1].rend() - 1);
  return c[0];
}
bool point_is_inside_convex(vector<pt> const& hull, pt p) {
  for(i32 i = 0; i < (i32)(hull.size()); ++i) {
    auto a = hull[i], b = hull[(i+1)%hull.size()];
    if(det(p-a, p-b) >= 0) return false;
  }
  return true;
}
}
using namespace geom_2d;
int nwaste;
int nlocations;
int nsorters;
int npos;
pt pos[MAXPOS];
f32 sorter_prob[2*MAXK][MAXN];
f32 log_sorter_prob[2*MAXK][MAXN];
f32 log_sorter_prob_inv[2*MAXK][MAXN];
int num_targets[1+MAXM+MAXK];
void read() {
  cin>>nwaste>>nlocations>>nsorters;
  cerr << "[DATA] N = " << nwaste << endl;
  cerr << "[DATA] M = " << nlocations << endl;
  cerr << "[DATA] MN = " << nlocations/nwaste << endl;
  cerr << "[DATA] K = " << nsorters << endl;
  pos[0] = {0, 5000};
  num_targets[0] = 1;
  for(i32 i = 0; i < (i32)(nwaste); ++i) { pos[1+i].read(); num_targets[1+i] = 0; }
  for(i32 i = 0; i < (i32)(nlocations); ++i) { pos[1+nwaste+i].read(); num_targets[1+nwaste+i] = 2; }
  npos = 1+nwaste+nlocations;
  for(i32 i = 0; i < (i32)(nsorters); ++i) for(i32 j = 0; j < (i32)(nwaste); ++j) {
    cin>>sorter_prob[2*i][j];
    sorter_prob[2*i+1][j] = 1-sorter_prob[2*i][j];
  }
  nsorters *= 2;
  for(i32 i = 0; i < (i32)(nsorters); ++i) for(i32 j = 0; j < (i32)(nwaste); ++j) {
    log_sorter_prob[i][j] = log(sorter_prob[i][j]);
    log_sorter_prob_inv[i][j] = log(1 - sorter_prob[i][j]);
  }
}
f32 angle[MAXPOS][MAXPOS];
int angle_sorted[MAXPOS][MAXPOS];
map<pt, int> inv_pos;
void init() {
  init_params(nwaste);
  for(i32 i = 0; i < (i32)(npos); ++i) inv_pos[pos[i]] = i;
  for(i32 i = 0; i < (i32)(npos); ++i) for(i32 j = 0; j < (i32)(npos); ++j) if(i != j) angle[i][j] = (pos[j]-pos[i]).atan2();
  for(i32 i = 0; i < (i32)(npos); ++i) {
    int sz = 0;
    for(i32 j = 0; j < (i32)(npos); ++j) if(j != i) angle_sorted[i][sz++] = j;
    pdqsort_branchless
      (angle_sorted[i], angle_sorted[i]+npos-1, [&](int a, int b){
        return angle[i][a] < angle[i][b];
      });
  }
}
struct solution {
  void reset() {
    for(i32 i = 0; i < (i32)(nwaste); ++i) proc_at[1+i] = i;
    for(i32 i = 0; i < (i32)(npos); ++i) sort_at[i] = 0;
    for(i32 i = 0; i < (i32)(npos); ++i) graph[i] = {-1,-1};
  }
  int proc_at[MAXPOS];
  int sort_at[MAXPOS];
  array<int, 2> graph[MAXPOS];
  void print() {
    for(i32 i = (1); i <= (i32)(nwaste); ++i) cout << proc_at[i] << ' ';
    cout << endl;
    cout << graph[0][0]-1 << endl;
    for(i32 i = (nwaste+1); i <= (i32)(npos-1); ++i) {
      auto [x,y] = graph[i];
      if(sort_at[i]&1) swap(x, y);
      if(x <= 0 || y <= 0) cout << -1 << endl;
      else cout << sort_at[i]/2 << ' ' << x-1 << " " << y-1 << endl;
    }
  }
  vector<int> get_top_order() const {
    vector<int> N(npos);
    for(i32 i = 0; i < (i32)(npos); ++i) for(i32 j = 0; j < (i32)(2); ++j) if(graph[i][j] != -1) N[graph[i][j]] += 1;
    vector<int> O;
    for(i32 i = 0; i < (i32)(npos); ++i) if(N[i] == 0) O.push_back(i);
    for(i32 io = 0; io < (i32)(O.size()); ++io) {
      int i = O[io];
      for(i32 j = 0; j < (i32)(2); ++j) if(graph[i][j] != -1) {
        N[graph[i][j]] -= 1;
        if(N[graph[i][j]] == 0) O.push_back(graph[i][j]);
      }
    }
    return O;
  }
  f32 simulate(vector<int> const& order) {
    static array<f32, MAXN> at[MAXPOS];
    for(i32 i = 0; i < (i32)(npos); ++i) for(i32 j = 0; j < (i32)(nwaste); ++j) at[i][j] = 0.0;
    for(i32 j = 0; j < (i32)(nwaste); ++j) at[0][j] = 1.0;
    for(int i : order) {
      if(i == 0) {
        for(i32 j = 0; j < (i32)(nwaste); ++j) at[graph[i][0]][j] += at[i][j];
      }else if(num_targets[i] == 2){
        for(i32 k = 0; k < (i32)(2); ++k) if(graph[i][k] != -1) for(i32 j = 0; j < (i32)(nwaste); ++j) {
            auto p = sorter_prob[sort_at[i]][j];
            at[graph[i][k]][j] += at[i][j] * (k ? 1-p : p);
          }
      }
    }
    f32 ret = 0.0;
    for(i32 i = 0; i < (i32)(nwaste); ++i) ret += at[1+i][proc_at[i+1]];
    return 1.0 - ret / nwaste;
  }
};
void fail_with_default() {
  solution S; S.reset();
  S.graph[0] = {1,1};
  S.print();
  exit(0);
}
struct vec {
  f32 data[MAXN];
  vec() { }
  __attribute__((always_inline)) inline
  vec(f32 v) { for(i32 x = 0; x < (i32)(nwaste); ++x) data[x] = v; }
  __attribute__((always_inline)) inline
  f32& operator[](int ix) { return data[ix]; }
  __attribute__((always_inline)) inline
  f32 const& operator[](int ix) const { return data[ix]; }
  __attribute__((always_inline)) inline
  vec& operator+=(vec const& o) {
    for(i32 x = 0; x < (i32)(nwaste); ++x) data[x] += o[x];
    return *this;
  }
  __attribute__((always_inline)) inline
  vec operator+(vec const& o) const {
    vec r = *this;
    r += o;
    return r;
  }
  __attribute__((always_inline)) inline
  vec& operator*=(vec const& o) {
    for(i32 x = 0; x < (i32)(nwaste); ++x) data[x] *= o[x];
    return *this;
  }
  __attribute__((always_inline)) inline
  vec operator*(vec const& o) const {
    vec r = *this;
    r *= o;
    return r;
  }
  __attribute__((always_inline)) inline
  f32 dot(vec const& o) const {
    f32 r = 0;
    for(i32 x = 0; x < (i32)(nwaste); ++x) r += data[x] * o[x];
    return r;
  }
  __attribute__((always_inline)) inline
  f32 sum() const {
    f32 s = 0;
    for(i32 x = 0; x < (i32)(nwaste); ++x) s += data[x];
    return s;
  }
  // FORCE_INLINE
  // vec(vec const& o) {
  //   FOR(x, nwaste) data[x] = o[x];
  // }
  // FORCE_INLINE
  // vec& operator=(vec const& o) {
  //   FOR(x, nwaste) data[x] = o[x];
  //   return *this;
  // }
};
ostream& operator<<(ostream& os, vec const& v) {
  vector<f32> w; for(i32 x = 0; x < (i32)(nwaste); ++x) w.push_back(v[x]);
  return os << w;
}
vec find_candidate_splits(int mask1, int mask2, vector<int>* sorters = 0) {
  array<int,2> key = {mask1,mask2};
  static map<array<i32,2>, vector<int>> cache_sorters;
  static map<array<i32,2>, vec> cache_result;
  if(cache_result.count(key)) {
    if(sorters) *sorters = cache_sorters[key];
    return cache_result[key];
  }
  static f32 candidate_q[2*MAXK];
  static int candidates[2*MAXK];
  static int indices1[MAXN]; int n1 = 0;
  static int indices2[MAXN]; int n2 = 0;
  while(mask1) { indices1[n1++] = lsb(mask1); mask1 ^= bit(lsb(mask1)); }
  while(mask2) { indices2[n2++] = lsb(mask2); mask2 ^= bit(lsb(mask2)); }
  for(i32 isorter = 0; isorter < (i32)(nsorters); ++isorter) {
    candidate_q[isorter] = 0;
    for(i32 x = 0; x < (i32)(n1); ++x) { candidate_q[isorter] += log_sorter_prob[isorter][indices1[x]]; }
    for(i32 x = 0; x < (i32)(n1); ++x) { candidate_q[isorter] += log_sorter_prob_inv[isorter][indices2[x]]; }
    // FOR(x, nwaste) candidate_q[isorter] += c[x] * sorter_prob[isorter][x];
    candidates[isorter] = isorter;
  }
  pdqsort_branchless(candidates, candidates + nsorters, [&](int i, int j) __attribute__((always_inline)) -> bool {
    return candidate_q[i] > candidate_q[j];
  });
  const int ncandidates = 4;
  static f32 w[ncandidates];
  for(i32 i = 0; i < (i32)(ncandidates); ++i) w[i] = rng.randomDouble();
  { f32 sum_w = 0.0;
    for(i32 i = 0; i < (i32)(ncandidates); ++i) sum_w += w[i];
    for(i32 i = 0; i < (i32)(ncandidates); ++i) w[i] /= sum_w;
  }
  f32 lr = 0.65;
  f32 last_obj = 1e9;
  for(i32 iter = 0; iter < (i32)(250); ++iter) {
    lr *= 0.99;
    f32 obj = 0.0;
    static f32 grad[2*MAXK];
    for(i32 i = 0; i < (i32)(ncandidates); ++i) grad[i] = 0.0;
    for(i32 x = 0; x < (i32)(n1); ++x) {
      f32 sum_wp = 0.0;
      for(i32 i = 0; i < (i32)(ncandidates); ++i) sum_wp += w[i] * log_sorter_prob[candidates[i]][indices1[x]];
      f32 t = exp(sum_wp);
      obj += t;
      for(i32 j = 0; j < (i32)(ncandidates); ++j) {
        grad[j] += t * (log_sorter_prob[candidates[j]][indices1[x]] - sum_wp);
      }
    }
    for(i32 x = 0; x < (i32)(n2); ++x) {
      f32 sum_wp = 0.0;
      for(i32 i = 0; i < (i32)(ncandidates); ++i) sum_wp += w[i] * log_sorter_prob[candidates[i]][indices2[x]];
      f32 t = -exp(sum_wp);
      obj += t;
      for(i32 j = 0; j < (i32)(ncandidates); ++j) {
        grad[j] += t * (log_sorter_prob[candidates[j]][indices2[x]] - sum_wp);
      }
    }
    if(abs(last_obj - obj) < 1e-5) break;
    last_obj = obj;
    for(i32 j = 0; j < (i32)(ncandidates); ++j) {
      w[j] += lr * grad[j];
      w[j] = max(w[j], 0.0f);
    }
    { f32 sum_w = 0.0;
      for(i32 i = 0; i < (i32)(ncandidates); ++i) sum_w += w[i];
      for(i32 i = 0; i < (i32)(ncandidates); ++i) w[i] /= sum_w;
    }
  }
  vec out;
  for(i32 x = 0; x < (i32)(nwaste); ++x) {
    f32 sum_wp = 0.0;
    for(i32 i = 0; i < (i32)(ncandidates); ++i) sum_wp += w[i] * log_sorter_prob[candidates[i]][x];
    out[x] = exp(sum_wp);
  }
  { auto& v = cache_sorters[key];
    for(i32 i = 0; i < (i32)(ncandidates); ++i) if(w[i] > 1e-5) v.push_back(candidates[i]);
  }
  cache_result[key] = out;
  if(sorters) *sorters = cache_sorters[key];
  return cache_result[key];
}
vector<array<int, 3>> build_graph(f32 p0, f32 p1, int LEN) {
  vector<array<f32, 2>> points;
  vector<array<int, 3>> graph;
  vector<int> indices;
  int next_index = 2;
  points = { {0,1}, {1,0} };
  indices = { 0, 1 };
  auto combine = [&](array<f32, 2> q1, array<f32, 2> q2) __attribute__((always_inline)) -> array<f32, 2> {
    auto [xi,yi] = q1;
    auto [xj,yj] = q2;
    auto x = p0*xj+(1-p0)*xi;
    auto y = p1*yj+(1-p1)*yi;
    return {x,y};
  };
  f32 best_exit = 0.0;
  for(i32 step = 0; step < (i32)(LEN); ++step) {
    int npoints = points.size();
    int bi = -1, bj = -1;
    f32 best_score = 0.0;
    // TODO: optimize from O(n^3) to O(n^2)
    for(i32 j = 0; j < (i32)(npoints); ++j) for(i32 i = (j-2); i <= (i32)(j-1); ++i) if(i >= 0) {
      f32 score = 0;
      f32 lx = 0;
      for(i32 k = (1); k <= (i32)(i); ++k) {
        auto [x,y] = points[k];
        score += y*(x-lx);
        lx = x;
      }
      f32 ly = 0;
      for(i32 k = (npoints-2); k >= (i32)(j); --k) {
        auto [x,y] = points[k];
        score += x*(y-ly);
        ly = y;
      }
      score -= lx * ly;
      auto [x,y] = combine(points[i], points[j]);
      score += (x-lx)*(y-ly);
      if(score > best_score) {
        best_score = score;
        bi = i;
        bj = j;
      }
    }
    vector<array<f32, 2>> new_points;
    for(i32 k = (0); k <= (i32)(bi); ++k) new_points.push_back(points[k]);
    auto p = combine(points[bi], points[bj]);
    new_points.push_back(p);
    for(i32 k = (bj); k <= (i32)(npoints-1); ++k) new_points.push_back(points[k]);
    points = new_points;
    best_exit = max(best_exit, p[0]+p[1]);
    graph.push_back({ indices[bi], indices[bj], step });
    vector<i32> new_indices;
    for(i32 k = (0); k <= (i32)(bi); ++k) new_indices.push_back(indices[k]);
    new_indices.push_back(next_index++);
    for(i32 k = (bj); k <= (i32)(npoints-1); ++k) new_indices.push_back(indices[k]);
    indices = new_indices;
  }
  // debug(best_exit);
  return graph;
}
template<class F>
void instantiate_graph
(vector<array<i32, 3>> const& g, vec const& p, vec const& v1, vec const& v2,
 F&& f)
{
  static vector<vec> V;
  V.resize(g.size()+2);
  V[0] = v1;
  V[1] = v2;
  for(i32 i = 0; i < (i32)(g.size()); ++i) {
    auto [a, b, l] = g[i];
    auto const& va = V[a];
    auto const& vb = V[b];
    vec& o = V[2+i];
    for(i32 x = 0; x < (i32)(nwaste); ++x) o[x] = p[x]*va[x] + (1-p[x])*vb[x];
    f32 sum = 0;
    for(i32 x = 0; x < (i32)(nwaste); ++x) sum += o[x];
    f(l, sum, o);
  }
}
struct tree_node {
  int children[2];
  int masks[2];
  int mask;
  vec target;
  int target_size;
  int total_size;
  int children_size[2];
  vec best_cand;
};
struct tree {
  vector<tree_node> nodes;
};
vector<array<i32, 3>> build_dag(vector<vector<array<i32, 3>>> const& graphs) {
  vector<array<i32, 3>> dag;
  map<array<i32, 2>, i32> M;
  for(auto const& g : graphs) {
    vector<int> v;
    v.push_back(0);
    v.push_back(1);
    for(auto [a,b,l] : g) {
      array<i32,2> key = {v[a],v[b]};
      if(!M.count(key)) {
        M[key] = dag.size()+2;
        dag.push_back({v[a],v[b],l});
      }
      v.push_back(M[key]);
    }
  }
  return dag;
}
vector<array<int, 3>> make_dag(int max_size) {
  vector<vector<array<i32, 3>>> graphs;
  f32 p0_fr = 0.6, p0_to = 0.85;
  f32 p1_fr = 0.15, p1_to = 0.35;
  int num_p0 = 5;
  int num_p1 = 5;
  for(i32 ip0 = 0; ip0 < (i32)(num_p0); ++ip0) for(i32 ip1 = 0; ip1 < (i32)(num_p1); ++ip1) {
    f32 p0 = p0_fr + (p0_to-p0_fr) * ip0 / (num_p0-1);
    f32 p1 = p1_fr + (p1_to-p1_fr) * ip1 / (num_p1-1);
    graphs.push_back(build_graph(p0, p1, max_size));
  }
  return build_dag(graphs);
}
tree solve_binary_tree() {
  auto dag = make_dag(nlocations / (nwaste-1));
  vector<vec> C(1<<nwaste);
  vector<vec> V(1<<nwaste);
  vector<array<i32, 2>> F(1<<nwaste, {-1,-1});
  for(i32 i = 0; i < (i32)(nwaste); ++i) {
    vec o; for(i32 x = 0; x < (i32)(nwaste); ++x) o[x] = 0;
    o[i] = 1;
    V[bit(i)] = o;
  }
  int count = 0;
  vector<u8> seen(bit(nwaste));
  auto go = [&](auto go, int mask) -> void {
    if(popcount(mask) == 1) return;
    if(seen[mask]) return;
    if(TIMER.elapsed() > TL0) fail_with_default();
    seen[mask] = 1;
    count += 1;
    if(count % 100 == 0) do{}while(0);
    hash_set<u32> masks1;
    for(i32 isorter = 0; isorter < (i32)(nsorters); ++isorter) if(isorter & 1) {
      int mask1 = 0, mask2 = 0;
      for(i32 x = 0; x < (i32)(nwaste); ++x) {
        if(sorter_prob[isorter][x] > 0.5) {
          mask1 |= bit(x);
        }else{
          mask2 |= bit(x);
        }
      }
      mask1 &= mask;
      mask2 &= mask;
      if(mask1 == 0) {
        int bx = lsb(mask);
        for(i32 x = 0; x < (i32)(nwaste); ++x) {
          if((mask&bit(x)) && sorter_prob[isorter][x] > sorter_prob[isorter][bx]) {
            bx = x;
          }
        }
        mask1 ^= bit(bx);
        mask2 ^= bit(bx);
      }
      if(mask2 == 0) {
        int bx = lsb(mask);
        for(i32 x = 0; x < (i32)(nwaste); ++x) {
          if((mask&bit(x)) && sorter_prob[isorter][x] < sorter_prob[isorter][bx]) {
            bx = x;
          }
        }
        mask1 ^= bit(bx);
        mask2 ^= bit(bx);
      }
      masks1.insert(mask1);
    }
    if(popcount(mask) <= 5){
      int mask1 = (mask-1)&mask;
      while(mask1) {
        masks1.insert(mask1);
        mask1 = (mask1-1)&mask;
      }
    }
    vector<tuple<f32, int> > smasks1;
    for(auto mask1 : masks1) {
      int mask2 = mask^mask1;
      auto cand = find_candidate_splits(mask1,mask2);
      f32 q = 0.0;
      for(i32 i = 0; i < (i32)(nwaste); ++i) {
        if(mask1&bit(i)) q += log(cand[i]);
        if(mask2&bit(i)) q += log(1-cand[i]);
      }
      smasks1.push_back({exp(q/popcount(mask)),mask1});
    }
    sort(begin(smasks1), end(smasks1));
    reverse(begin(smasks1), end(smasks1));
    // TODO: vary 8 depending on popcount(mask)
    if(smasks1.size() > SMASK_LIMIT) smasks1.resize(SMASK_LIMIT);
    for(auto [q,mask1] : smasks1) {
      go(go,mask1);
      go(go,mask^mask1);
    }
    f32 best_sum = -1.0;
    for(i32 x = 0; x < (i32)(nwaste); ++x) V[mask][x] = 0.0;
    for(auto [q,mask1] : smasks1) {
      int mask2 = mask ^ mask1;
      auto cand = find_candidate_splits(mask1, mask2);
      instantiate_graph
        (dag, cand, V[mask1], V[mask2],
         [&](i32, f32 sum, vec o) {
           if(sum > best_sum) {
             best_sum = sum;
             V[mask] = o;
             F[mask] = {mask1, mask2};
             C[mask] = cand;
           }
         });
    }
  };
  go(go, bit(nwaste)-1);
  tree out;
  auto recon = [&](auto recon, int mask) -> int {
    if(popcount(mask) > 1) {
      auto [m1,m2] = F[mask];
      auto a = recon(recon, m1);
      auto b = recon(recon, m2);
      int ix = out.nodes.size();
      tree_node node;
      node.mask = mask;
      node.masks[0] = m1;
      node.masks[1] = m2;
      node.children[0] = a;
      node.children[1] = b;
      node.target = V[mask];
      node.best_cand = C[mask];
      for(i32 x = 0; x < (i32)(nwaste); ++x) {
        if(m1&bit(x)) node.target[x] /= V[m1][x];
        if(m2&bit(x)) node.target[x] /= V[m2][x];
      }
      out.nodes.push_back(node);
      return 2*ix;
    }else{
      return 2*lsb(mask)+1;
    }
  };
  recon(recon, bit(nwaste)-1);
  vec v = V[bit(nwaste)-1];
  do{}while(0);
  return out;
}
vector<array<i32, 3>> dag;
f32 evaluate_sizes(tree& t, vector<int> const& L) {
  static f32 S[MAXN];
  static vec V[MAXN];
  auto eval = [&](auto eval, int i) -> void {
    auto const& node = t.nodes[i];
    if(!(node.children[0]&1)) eval(eval, node.children[0]/2);
    if(!(node.children[1]&1)) eval(eval, node.children[1]/2);
    S[i] = 0.0;
    V[i] = vec(0.0);
    vec v1, v2;
    if(node.children[0]&1) for(i32 x = 0; x < (i32)(nwaste); ++x) v1[x] = x == node.children[0]/2;
    else v1 = V[node.children[0]/2];
    if(node.children[1]&1) for(i32 x = 0; x < (i32)(nwaste); ++x) v2[x] = x == node.children[1]/2;
    else v2 = V[node.children[1]/2];
    instantiate_graph
      (dag, node.best_cand, v1, v2,
       [&](i32 l, f32 sum, vec o) {
         if(l <= L[i] && sum > S[i]) {
           S[i] = sum;
           V[i] = o;
         }
       });
  };
  eval(eval, nwaste-2);
  for(i32 i = 0; i < (i32)(nwaste-1); ++i) {
    t.nodes[i].target = V[i];
  }
  return 1.0 - S[nwaste-2] / nwaste;
}
void optimize_sizes(tree& t) {
  dag = make_dag(min<int>(nlocations, SIZES_UPPER_LIMIT * nlocations/(nwaste-1)));
  vector<int> L(nwaste-1, nlocations/(nwaste-1));
  vector<vec> B(nwaste, vec(0.0));
  for(i32 i = 0; i < (i32)(nwaste); ++i) for(i32 j = 0; j < (i32)(nwaste); ++j) B[i][j] = (i==j?1.0:0.0);
  vector<vec> V(nwaste-1);
  vector<f32> S(nwaste-1);
  f32 score = evaluate_sizes(t, L);
  do{}while(0);
  vector<int> best_L = L;
  f32 best = score;
  for(i32 iter = 0; iter < (i32)(SIZES_STEPS); ++iter) {
    int i = rng.random32(nwaste-1);
    int j = rng.random32(nwaste-1);
    if(i == j) continue;
    int d = 1+rng.random32(SIZES_RANGE);
    if(d >= L[i]) continue;
    if(L[i] - d < SIZES_LOWER_LIMIT * nlocations / (nwaste-1)) continue;
    L[i] -= d; L[j] += d;
    f32 new_score = evaluate_sizes(t, L);
    if(new_score <= score) {
      score = new_score;
    }else{
      L[i] += d; L[j] -= d;
    }
    if(score < best) {
      best = score;
      best_L = L;
      do{}while(0);
    }
  }
  L = best_L;
  evaluate_sizes(t, L);
  for(i32 i = 0; i < (i32)(nwaste-1); ++i) {
    t.nodes[i].target_size = L[i];
    t.nodes[i].total_size = L[i];
    for(i32 j = 0; j < (i32)(2); ++j) if(!(t.nodes[i].children[j]&1)) {
      t.nodes[i].children_size[j] = t.nodes[t.nodes[i].children[j]/2].total_size;
      t.nodes[i].total_size += t.nodes[i].children_size[j];
    }else{
      t.nodes[i].children_size[j] = 0;
    }
  }
}
const int NODE_MAX_POINTS = 200;
const int NODE_MAX_SEGMENTS = NODE_MAX_POINTS * (NODE_MAX_POINTS+1) / 2;
const int NODE_MAX_INTERSECTIONS = (NODE_MAX_SEGMENTS+255)/256*4;
struct decomposition_preinfo {
  int root;
  int exit[2];
};
struct decomposition_info {
  int root;
  int exit[2];
  // int exit_to[2];
  int npoints;
  int points[NODE_MAX_POINTS];
  int nsegments;
  int segment[NODE_MAX_POINTS][NODE_MAX_POINTS];
  array<int, 2> segments[NODE_MAX_SEGMENTS];
  int intersections_size;
  alignas(64)
  u64 intersections[NODE_MAX_SEGMENTS][NODE_MAX_INTERSECTIONS];
  void init(vector<int> const& points_) {
    npoints = points_.size();
    if(npoints > NODE_MAX_POINTS) {
      do{}while(0);
      fail_with_default();
    }
    do { if(!(npoints <= NODE_MAX_POINTS)) { throw runtime_error("main.cpp" ":" "854" " Assertion failed: " "npoints <= NODE_MAX_POINTS"); } } while(0);
    for(i32 i = 0; i < (i32)(npoints); ++i) points[i] = points_[i];
    for(i32 i = 0; i < (i32)(npoints); ++i) if(points[i] == root) { root = i; break; }
    for(i32 j = 0; j < (i32)(2); ++j) if(exit[j] != -1) {
      bool found = 0;
      for(i32 i = 0; i < (i32)(npoints); ++i) if(points[i] == exit[j]) { exit[j] = i; found = 1; break; }
      do { if(!(found)) { throw runtime_error("main.cpp" ":" "861" " Assertion failed: " "found"); } } while(0);
    }
    nsegments = 0;
    for(i32 i = 0; i < (i32)(npoints); ++i) for(i32 j = 0; j < (i32)(i); ++j) {
      segment[i][j] = segment[j][i] = nsegments;
      segments[nsegments] = {i,j};
      nsegments += 1;
    }
    intersections_size = (nsegments+255) / 256 * 4;
    for(i32 i = 0; i < (i32)(nsegments); ++i) for(i32 j = 0; j < (i32)(intersections_size); ++j) intersections[i][j] = 0;
    for(i32 i = 0; i < (i32)(nsegments); ++i) for(i32 j = 0; j < (i32)(i); ++j) {
      auto [a,b] = segments[i];
      auto [u,v] = segments[j];
      if(intersects(pos[points[a]], pos[points[b]], pos[points[u]], pos[points[v]])) {
        intersections[i][j/64] |= bit(j%64);
        intersections[j][i/64] |= bit(i%64);
      }
    }
  }
};
const int CHECK_EVERY = 20;
tuple<vector<int>, vector<int>, int>
find_splitting_2
(vector<int> const& points,
 int size1, int size2,
 int root)
{
  int n = points.size();
  int best_value = 0;
  int best_count = 0;
  int bi=-1,bx=-1,by=-1;
  int check_every = max(1, n / CHECK_EVERY); // TODO
  static bitset<MAXPOS> in_points;
  in_points.reset();
  for(i32 i = 0; i < (i32)(n); ++i) in_points[points[i]] = 1;
  static tuple<f32,int> others[2*MAXPOS];
  for(i32 i = 0; i < (i32)(n); ++i) if(rng.random32(check_every) == 0 &&
               (points[i] != root) && num_targets[points[i]] == 2) {
    int nothers = 0;
    for(i32 j0 = 0; j0 < (i32)(npos-1); ++j0) if(int j = angle_sorted[points[i]][j0]; in_points[j]) {
      others[nothers++] = {angle[points[i]][j], j};
    }
    for(i32 x = 0; x < (i32)(nothers); ++x) {
      auto [alpha, j] = others[x];
      others[nothers+x] = {alpha + 2*M_PI, j};
    }
    int r1 = 0, g1 = 0, s1 = 0, r2 = 0, g2 = 0, s2 = 0;
    for(i32 x = 0; x < (i32)(nothers); ++x) {
      auto [alpha, j] = others[x];
      if(j == root) r2 += 1;
      else if(num_targets[j] == 2) s2 += 1;
      else g2 += 1;
    }
    do { if(!(s1+1+s2+1 >= size1+size2)) { throw runtime_error("main.cpp" ":" "924" " Assertion failed: " "s1+1+s2+1 >= size1+size2"); } } while(0);
    int y = 0;
    f32 alpha = -M_PI;
    auto advance_y = [&](){
      while(get<0>(others[y]) < alpha + M_PI) {
        int j = get<1>(others[y]);
        if(j == root) { r2 -= 1; r1 += 1; }
        else if(num_targets[j] == 2) { s2 -= 1; s1 += 1; }
        else { g2 -= 1; g1 += 1; }
        y += 1;
      }
    };
    advance_y();
    for(i32 x = 0; x < (i32)(nothers); ++x) {
      alpha = get<0>(others[x]);
      int j = get<1>(others[x]);
      advance_y();
      do { if(!(x < y)) { throw runtime_error("main.cpp" ":" "943" " Assertion failed: " "x < y"); } } while(0);
      if(g1 == 1 && r1 == 1 && s2+1 >= size2) {
        int value = s1+1;
        if(value > best_value) {
          best_value = value;
          best_count = 0;
        }
        if(value == best_value) {
          best_count += 1;
          if(rng.random32(best_count) == 0) {
            bi = i; bx = x; by = y;
          }
        }
      }
      if(j == root) { r2 += 1; r1 -= 1; }
      else if(num_targets[j] == 2) { s2 += 1; s1 -= 1; }
      else { g2 += 1; g1 -= 1; }
      advance_y();
    }
  }
  if(best_count == 0) {
    // fail
    return { {}, {}, -1 };
  }
  int nothers = 0;
  for(i32 j0 = 0; j0 < (i32)(npos-1); ++j0) if(int j = angle_sorted[points[bi]][j0]; in_points[j]) {
    others[nothers++] = {angle[points[bi]][j], j};
  }
  for(i32 x = 0; x < (i32)(nothers); ++x) {
    auto [alpha, j] = others[x];
    others[nothers+x] = {alpha + 2*M_PI, j};
  }
  vector<int> left_points, right_points;
  left_points.push_back(points[bi]);
  for(i32 i = (bx); i <= (i32)(by-1); ++i) {
    left_points.push_back(get<1>(others[i]));
  }
  right_points.push_back(points[bi]); // TODO: take into account before
  for(i32 i = (by); i <= (i32)(bx + nothers-1); ++i) {
    right_points.push_back(get<1>(others[i]));
  }
  return { left_points, right_points, points[bi] };
}
tuple<vector<int>, vector<int>, vector<int>, int, int>
find_splitting_3
(vector<int> const& points,
 int size1, int size2, int size3,
 int goals2, int goals3,
 int root,
 f32 limit_size1
 ) {
  int n = points.size();
  int best_count = 0;
  int bi=-1,bx=-1,by=-1,bz=-1;
  int check_every = max(1, n / CHECK_EVERY); // TODO
  static bitset<MAXPOS> in_points;
  in_points.reset();
  for(i32 i = 0; i < (i32)(n); ++i) in_points[points[i]] = 1;
  static tuple<f32,int> others[2*MAXPOS];
  while(1) {
    for(i32 i = 0; i < (i32)(n); ++i) if(rng.random32(check_every) == 0 &&
                 (points[i] != root) && num_targets[points[i]] == 2) {
      int nothers = 0;
      for(i32 j0 = 0; j0 < (i32)(npos-1); ++j0) if(int j = angle_sorted[points[i]][j0]; in_points[j]) {
        others[nothers++] = {angle[points[i]][j], j};
      }
      for(i32 x = 0; x < (i32)(nothers); ++x) {
        auto [alpha, j] = others[x];
        others[nothers+x] = {alpha + 2*M_PI, j};
      }
      int r1 = 0, g1 = 0, s1 = 0, r2 = 0, g2 = 0, s2 = 0, r3 = 0, g3 = 0, s3 = 0;
      for(i32 x = 0; x < (i32)(nothers); ++x) {
        auto [alpha, j] = others[x];
        if(j == root) r3 += 1;
        else if(num_targets[j] == 2) s3 += 1;
        else g3 += 1;
      }
      do { if(!(s1+1+s2+s3+1 >= size1+size2+size3)) { throw runtime_error("main.cpp" ":" "1034" " Assertion failed: " "s1+1+s2+s3+1 >= size1+size2+size3"); } } while(0);
      // [x, y-1], [y, z-1], [z, x+nothers-1]
      int x = 0, y = 0, z = 0;
      auto incr_z = [&](){
        auto [alpha, j] = others[z];
        if(j == root) { r3 -= 1; r2 += 1; }
        else if(num_targets[j] == 2) { s3 -= 1; s2 += 1; }
        else { g3 -= 1; g2 += 1; }
        z += 1;
      };
      auto incr_y = [&](){
        auto [alpha, j] = others[y];
        if(j == root) { r2 -= 1; r1 += 1; }
        else if(num_targets[j] == 2) { s2 -= 1; s1 += 1; }
        else { g2 -= 1; g1 += 1; }
        y += 1;
      };
      auto incr_x = [&](){
        auto [alpha, j] = others[x];
        if(j == root) { r1 -= 1; r3 += 1; }
        else if(num_targets[j] == 2) { s1 -= 1; s3 += 1; }
        else { g1 -= 1; g3 += 1; }
        x += 1;
      };
      auto advance = [&](){
        while(s1+1 < size1) {
          incr_y();
        }
        while(s2 < size2) {
          incr_z();
        }
      };
      for(i32 step = 0; step < (i32)(nothers); ++step) {
        advance();
        if(r1 == 1 && g1 == 0 && g2 == goals2 && g3 == goals3) {
          f32 axy = get<0>(others[y]) - get<0>(others[x]);
          f32 ayz = get<0>(others[z-1]) - get<0>(others[y]);
          f32 azx = get<0>(others[x+nothers-1]) - get<0>(others[z]);
          int vy = get<1>(others[y]);
          if(max({axy,ayz,azx}) < M_PI && (vy != root) && num_targets[vy] == 2) {
            best_count += 1;
            if(rng.random32(best_count) == 0) {
              bi = i; bx = x; by = y; bz = z;
            }
          }
        }
        incr_x();
      }
    }
    if(best_count > 0) break;
    int new_size1 = size1 * SPLITTING_3_SIZE_MULT;
    int d = size1 - new_size1;
    size1 = new_size1;
    if(rng.random32(2) == 0) {
      size2 += d;
    }else{
      size3 += d;
    }
    if(size1 < limit_size1) {
      // fail
      return { {}, {}, {}, -1, -1 };
    }
  }
  int nothers = 0;
  for(i32 j0 = 0; j0 < (i32)(npos-1); ++j0) if(int j = angle_sorted[points[bi]][j0]; in_points[j]) {
    others[nothers++] = {angle[points[bi]][j], j};
  }
  for(i32 x = 0; x < (i32)(nothers); ++x) {
    auto [alpha, j] = others[x];
    others[nothers+x] = {alpha + 2*M_PI, j};
  }
  vector<int> root_points, left_points, right_points;
  root_points.push_back(get<1>(others[by]));
  root_points.push_back(points[bi]);
  for(i32 i = (bx); i <= (i32)(by-1); ++i) {
    root_points.push_back(get<1>(others[i]));
  }
  for(i32 i = (by); i <= (i32)(bz-1); ++i) {
    left_points.push_back(get<1>(others[i]));
  }
  right_points.push_back(points[bi]);
  for(i32 i = (bz); i <= (i32)(bx+nothers-1); ++i) {
    right_points.push_back(get<1>(others[i]));
  }
  return {root_points, left_points, right_points, get<1>(others[by]), points[bi]};
}
vector<int> get_goals(vector<int> const& points) {
  vector<int> R;
  for(int i : points) if(num_targets[i] == 0) R.push_back(i);
  return R;
}
void find_splitting
(bool& failed,
 vector<vector<int>>& S,
 vector<decomposition_preinfo>& D,
 f32 limit,
 int block_size,
 tree const& t,
 int i,
 vector<int> const& points,
 int root)
{
  if(failed) return;
  auto const& node = t.nodes[i];
  if(popcount(node.mask) == 2) {
    auto goals = get_goals(points);
    S[i] = points;
    D[i].root = root;
    D[i].exit[0] = goals[0];
    D[i].exit[1] = goals[1];
    return;
  }
  int target_size = node.target_size;
  int limit_target_size = limit * target_size;
  { int need = node.total_size - node.target_size + 3;
    int avail = 0;
    for(int i : points) if(num_targets[i] == 2) avail += 1;
    do { if(!(avail >= need)) { throw runtime_error("main.cpp" ":" "1169" " Assertion failed: " "avail >= need"); } } while(0);
    target_size = avail - (node.total_size - node.target_size);
    // debug(i, target_size, need, avail);
  }
  if(target_size < limit_target_size) {
    failed = true;
    return;
  }
  if((node.children[0]&1) || (node.children[1]&1)) {
    int child_size = max(node.children_size[0], node.children_size[1]);
    auto [L, R, R_root] =
      find_splitting_2
      (points,
       target_size,
       child_size,
       root);
    if(L.empty()) { failed = 1; return; } // FAIL
    S[i] = L;
    D[i].root = root;
    if(node.children[0]&1) {
      D[i].exit[0] = get_goals(L)[0];
      D[i].exit[1] = R_root;
      find_splitting(failed,S,D,limit,block_size, t, node.children[1]/2, R, R_root);
    }else{
      D[i].exit[0] = R_root;
      D[i].exit[1] = get_goals(L)[0];
      find_splitting(failed,S,D,limit,block_size, t, node.children[0]/2, R, R_root);
    }
  }else{
    auto [A1,L1,R1,L1_root,R1_root] = find_splitting_3
      (points,
       target_size,
       node.children_size[0],
       node.children_size[1],
       popcount(node.masks[0]),
       popcount(node.masks[1]),
       root,
       limit_target_size
       );
    // check3(A1, L1, R1);
    if(!A1.empty()) {
      S[i] = A1;
      D[i].root = root;
      D[i].exit[0] = L1_root;
      D[i].exit[1] = R1_root;
      find_splitting(failed,S,D,limit,block_size, t, node.children[0]/2, L1, L1_root);
      find_splitting(failed,S,D,limit,block_size, t, node.children[1]/2, R1, R1_root);
      return;
    }
    auto [A2,L2,R2,L2_root,R2_root] = find_splitting_3
      (points,
       target_size,
       node.children_size[1],
       node.children_size[0],
       popcount(node.masks[1]),
       popcount(node.masks[0]),
       root,
       limit_target_size
       );
    if(!A2.empty()) {
      S[i] = A2;
      D[i].root = root;
      D[i].exit[0] = R2_root;
      D[i].exit[1] = L2_root;
      find_splitting(failed,S,D,limit,block_size, t, node.children[1]/2, L2, L2_root);
      find_splitting(failed,S,D,limit,block_size, t, node.children[0]/2, R2, R2_root);
      return;
    }
    failed = true;
  }
}
struct beam_state_elem {
  int u;
  array<f32, 2> proj;
  vec v;
};
struct history_list {
  int u, c1, c2;
  int prev;
};
const int POOL_SIZE = 1<<20;
beam_state_elem elem_pool[POOL_SIZE];
history_list history_pool[POOL_SIZE];
alignas(64)
u64 intersections_pool[POOL_SIZE];
int elem_pool_next;
int history_pool_next;
int intersections_pool_next;
void reset_pool() {
  elem_pool_next = 0;
  history_pool_next = 0;
  intersections_pool_next = 0;
}
int make_history(history_list const& l) {
  int index = history_pool_next++;
  history_pool[index] = l;
  return index;
}
__attribute__((always_inline)) inline
array<f32, 2> get_vec_proj
(int mask1, int mask2, vec const& w) {
  f32 w1 = 0, w2 = 0;
  while(mask1) { w1 += w[lsb(mask1)]; mask1 ^= bit(lsb(mask1)); }
  while(mask2) { w2 += w[lsb(mask2)]; mask2 ^= bit(lsb(mask2)); }
  return {w1,w2};
}
struct beam_candidate {
  f32 score;
  int from;
  int i,j;
};
struct beam_state {
  int nelems;
  beam_state_elem* elems;
  u64* intersections;
  int history;
  void pre(int intersections_size) {
    nelems = 0;
    elems = elem_pool + elem_pool_next;
    intersections = intersections_pool + intersections_pool_next;
    intersections_pool_next += intersections_size;
  }
  __attribute__((always_inline)) inline
  void add_segment(decomposition_info const& info, int a, int b) {
    int seg = info.segment[a][b];
    for(i32 x = 0; x < (i32)(info.intersections_size); ++x) intersections[x] |= info.intersections[seg][x];
  }
  __attribute__((always_inline)) inline
  void add_elem(beam_state_elem const& e) {
    elems[nelems] = e;
    elem_pool_next++;
    nelems++;
  }
  void reset
  (decomposition_info const& info,
   int mask1, int mask2,
   vec const& v1, vec const& v2,
   int e1, int e2) {
    pre(info.intersections_size);
    add_elem
      (beam_state_elem
       { .u = e2, .proj = get_vec_proj(mask1, mask2, v2), .v = v2 });
    add_elem
      (beam_state_elem
       { .u = e1, .proj = get_vec_proj(mask1, mask2, v1), .v = v1 });
    for(i32 x = 0; x < (i32)(info.intersections_size); ++x) intersections[x] = 0;
    history = -1;
  }
  template<class F>
  void children
  (decomposition_info const& info,
   int mask1, int mask2, vec const& p,
   int u, F&& f) const
  {
    static bool valid[NODE_MAX_POINTS];
    for(i32 i = 0; i < (i32)(nelems); ++i) {
      int seg = info.segment[elems[i].u][u];
      valid[i] = !(intersections[seg/64]&bit(seg%64));
    }
    bool valid_root = 1;
    { int seg = info.segment[info.root][u];
      valid_root = !(intersections[seg/64]&bit(seg%64));
    }
    static f32 scorei[MAXPOS], scorej[MAXPOS];
    { scorei[0] = 0;
      f32 lx = 0;
      for(i32 i = 0; i < (i32)(nelems); ++i) {
        auto [x,y] = elems[i].proj;
        scorei[i+1] = scorei[i] + (x-lx) * y;
        lx = x;
      }
    }
    { f32 ly = 0;
      scorej[nelems] = 0;
      for(i32 i = (nelems-1); i >= (i32)(0); --i) {
        auto [x,y] = elems[i].proj;
        scorej[i] = scorej[i+1] + (y-ly) * x;
        ly = y;
      }
    }
    for(i32 j = 0; j < (i32)(nelems); ++j) if(valid[j]) for(i32 i = 0; i < (i32)(j); ++i) if(valid[i]) {
      vec v;
      // FOR(x, nwaste) v[x] = elems[i].v[x] + p[x] * (elems[j].v[x] - elems[i].v[x]);
      for(i32 x = 0; x < (i32)(nwaste); ++x) v[x] = (1-p[x]) * elems[i].v[x] + p[x] * elems[j].v[x];
      auto v_proj = get_vec_proj(mask1, mask2, v);
      f32 score = scorei[i+1] + scorej[j];
      auto [xi,yi] = elems[i].proj;
      auto [xj,yj] = elems[j].proj;
      score -= xi*yj;
      score += (v_proj[0]-xi)*(v_proj[1]-yj);
      f(valid_root?v_proj[0]+v_proj[1]:0.0,
        history_list { u, elems[i].u, elems[j].u, history },
        score,
        i, j);
    }
  }
  template<bool RECON>
  void make_child
  (decomposition_info const& info,
   int mask1, int mask2, int u, vec const& p,
   beam_state& child, beam_candidate const& c) {
    int i = c.i, j = c.j;
    vec v;
    for(i32 x = 0; x < (i32)(nwaste); ++x) v[x] = (1-p[x]) * elems[i].v[x] + p[x] * elems[j].v[x];
    auto v_proj = get_vec_proj(mask1, mask2, v);
    child.pre(info.intersections_size);
    // FOR(x, info.intersections_size) child.intersections[x] = intersections[x];
    // child.add_segment(info, u, elems[i].u);
    // child.add_segment(info, u, elems[j].u);
    int segi = info.segment[u][elems[i].u];
    int segj = info.segment[u][elems[j].u];
    for(i32 x = 0; x < (i32)(info.intersections_size/4); ++x) {
      __m256i v1 = _mm256_load_si256((__m256i *)(intersections + 4*x));
      __m256i v2 = _mm256_load_si256((__m256i *)(info.intersections[segi] + 4*x));
      __m256i v3 = _mm256_load_si256((__m256i *)(info.intersections[segj] + 4*x));
      _mm256_store_si256((__m256i *)(child.intersections+4*x),
                         _mm256_or_si256(v1, _mm256_or_si256(v2, v3)));
      // child.intersections[x] =
      //   intersections[x] |
      //   info.intersections[segi][x] |
      //   info.intersections[segj][x];
    }
    elem_pool_next += (i+1) + 1 + (nelems-j);
    child.nelems += (i+1) + 1 + (nelems-j);
    for(i32 k = (0); k <= (i32)(i); ++k) child.elems[k] = elems[k];
    child.elems[i+1] = beam_state_elem { .u = u, .proj = v_proj, .v = v };
    for(i32 k = (j); k <= (i32)(nelems-1); ++k) child.elems[i+2+k-j] = elems[k];
    if(RECON) {
      child.history = make_history(history_list { u, elems[i].u, elems[j].u, history });
    }
  }
};
i64 nevaluate = 0;
template<bool RECON>
tuple<f32, int> evaluate
(vector<int> const& order,
 int mask1, int mask2,
 vec v1, vec v2, vec p,
 decomposition_info const& info)
{
  nevaluate += 1;
  reset_pool();
  const int BRANCH = NODE_MAX_POINTS * NODE_MAX_POINTS;
  static beam_state BEAM0[MAXBW], BEAM1[MAXBW];
  beam_state *BEAM_A = BEAM0, *NEW_BEAM_A = BEAM1;
  static beam_candidate BEAM_B[MAXBW * BRANCH];
  int na = 0, nb = 0;
  na += 1;
  BEAM_A[0].reset
    (info,
     mask1, mask2,
     v1, v2,
     info.exit[0], info.exit[1]);
  f32 best_sum = 0.0;
  int best_h = -1;
  for(int u : order) {
    nb = 0;
    for(i32 ia = 0; ia < (i32)(na); ++ia) {
      auto const& sa = BEAM_A[ia];
      auto const& add =
        [&](f32 sum, history_list const& h, f32 score, int i, int j)
        __attribute__((always_inline)) -> void
        {
          BEAM_B[nb++] = beam_candidate { score, ia, i, j };
          if(sum > best_sum) {
            best_sum = sum;
            if(RECON) {
              best_h = make_history(h);
            }
          }
        };
      sa.children
        (info, mask1, mask2, p, u, add);
    }
    if(nb == 0) {
    }else{
      pdqsort_branchless(BEAM_B, BEAM_B+nb, [&](auto const& s1, auto const& s2) __attribute__((always_inline)) -> bool {
        return s1.score > s2.score;
      });
      na = 0;
      f32 last = 0.0;
      for(i32 ib = 0; ib < (i32)(nb); ++ib) {
        auto const& sb = BEAM_B[ib];
        if(abs(last - sb.score) < 1e-7) continue;
        BEAM_A[sb.from].make_child<RECON>
          (info,mask1,mask2,u,p,
           NEW_BEAM_A[na], sb);
        na += 1;
        last = sb.score;
        if(na == BW) break;
      }
      swap(BEAM_A, NEW_BEAM_A);
    }
  }
  f32 best_score = 1.0 - 1.0 * best_sum / (popcount(mask1)+popcount(mask2));
  return { best_score, best_h };
}
vector<int> get_decomposition_sizes(tree const& t, vector<vector<int>> const& decomposition) {
  vector<int> L(nwaste-1);
  for(i32 i = 0; i < (i32)(nwaste-1); ++i) {
    for(int j : decomposition[i]) {
      if(num_targets[j] == 2) L[i] += 1;
    }
    L[i] = min(L[i], t.nodes[i].target_size);
  }
  return L;
}
void solve() {
  auto tree = solve_binary_tree();
  optimize_sizes(tree);
  vector<vector<int>> decomposition(nwaste-1);
  vector<decomposition_preinfo> preinfos(nwaste-1);
  vector<int> root_path;
  { vector<pt> hull;
    for(i32 i = (1); i <= (i32)(npos-1); ++i) hull.push_back(pos[i]);
    hull = convex_hull(hull);
    set<pt> shull(begin(hull), end(hull));
    vector<int> ihull;
    for(auto p : hull) ihull.push_back(inv_pos[p]);
    vector<vector<int>> G(npos);
    for(i32 i = 0; i < (i32)(ihull.size()); ++i) {
      int a = ihull[i], b = ihull[(i+1)%ihull.size()];
      if(num_targets[a] == 2 && num_targets[b] == 2) {
        G[a].push_back(b);
        G[b].push_back(a);
      }
    }
    for(int i : ihull) if(num_targets[i] == 2) {
      bool valid = true;
      for(i32 j = 0; j < (i32)(hull.size()); ++j) {
        pt a = hull[j], b = hull[(j+1)%hull.size()];
        if(intersects(a,b,pos[0],pos[i])) valid = false;
      }
      if(valid) G[0].push_back(i);
    }
    const int INF = 1e9;
    static int Q[MAXPOS]; int nq = 0;
    static int D[MAXPOS];
    static int F[MAXPOS];
    for(i32 i = 0; i < (i32)(npos); ++i) D[i] = INF;
    Q[nq++] = 0; D[0] = 0; F[0] = -1;
    vector<int> roots;
    for(i32 iq = 0; iq < (i32)(nq); ++iq) {
      int u = Q[iq];
      for(int v : G[u]) if(D[v] == INF) {
          D[v] = D[u]+1;
          Q[nq++] = v;
          F[v] = u;
          roots.push_back(v);
        }
    }
    do{}while(0);
    if(roots.empty()) {
      do{}while(0);
      fail_with_default();
    }
    vector<int> points;
    for(i32 i = (1); i <= (i32)(npos-1); ++i) points.push_back(i);
    f32 best_score = 1e9;
    vector<vector<int>> best_decomposition;
    vector<decomposition_preinfo> best_preinfos;
    vector<int> best_root_path;
    bool found = 0;
    int ntry = 0;
    (void)ntry;
    f32 t0 = TIMER.elapsed();
    while(!found || (TIMER.elapsed() - t0) < TL_DECOMP) {
      if(TIMER.elapsed() > TL0) {
        do{}while(0);
        fail_with_default();
      }
      f32 done = min(1.0f, (TIMER.elapsed() - t0) / TL_DECOMP);
      f32 limit = lerp(0.8, 0.2, done);
      bool failed = false;
      int root = rng.sample(roots);
      set<int> apoints(begin(points), end(points));
      for(int x = F[root]; x != 0; x = F[x]) {
        apoints.erase(x);
      }
      vector<int> the_root_path;
      for(int x = root; x != -1; x = F[x]) {
        the_root_path.push_back(x);
      }
      int block_size = 0;
      for(int x : apoints) if(num_targets[x] == 2) block_size += 1;
      block_size /= nwaste-1;
      find_splitting
        (failed,decomposition,preinfos,limit,block_size,
         tree, nwaste-2, vector<int>(begin(apoints), end(apoints)),
         root
         );
      if(!failed) {
        ntry ++;
        found = 1;
        auto L = get_decomposition_sizes(tree, decomposition);
        auto new_score = evaluate_sizes(tree, L);
        if(new_score < best_score) {
          best_score = new_score;
          best_decomposition = decomposition;
          best_preinfos = preinfos;
          best_root_path = the_root_path;
          do{}while(0);
        }
      }
    }
    do{}while(0);
    decomposition = best_decomposition;
    preinfos = best_preinfos;
    root_path = best_root_path;
    auto L = get_decomposition_sizes(tree, decomposition);
    do{}while(0);
  }
  static decomposition_info infos[MAXN];
  for(i32 i = 0; i < (i32)(nwaste-1); ++i) {
    infos[i].root = preinfos[i].root;
    for(i32 j = 0; j < (i32)(2); ++j) infos[i].exit[j] = preinfos[i].exit[j];
    infos[i].init(decomposition[i]);
  }
  solution S; S.reset();
  vector<vector<int>> candidates_at(npos);
  vector<vec> node_targets(nwaste-1);
  for(i32 iwaste = 0; iwaste < (i32)(nwaste-1); ++iwaste) {
    auto const& node = tree.nodes[iwaste];
    node_targets[iwaste] = node.target;
  }
  for(i32 iwaste = 0; iwaste < (i32)(nwaste-1); ++iwaste) {
    auto const& node = tree.nodes[iwaste];
    auto const& points = decomposition[iwaste];
    auto const& info = infos[iwaste];
    vec v1, v2;
    for(i32 x = 0; x < (i32)(nwaste); ++x) v1[x] = (node.masks[0]&bit(x))?1:0;
    for(i32 x = 0; x < (i32)(nwaste); ++x) v2[x] = (node.masks[1]&bit(x))?1:0;
    for(i32 j = 0; j < (i32)(nwaste-1); ++j) if(j != iwaste) {
      for(i32 x = 0; x < (i32)(nwaste); ++x) if(tree.nodes[j].mask&bit(x)) {
        if((node.masks[0]&bit(x))) v1[x] *= node_targets[j][x];
        if((node.masks[1]&bit(x))) v2[x] *= node_targets[j][x];
      }
    }
    auto target_score = 1.0 - 1.0 * (v1.dot(node.target) + v2.dot(node.target)) / popcount(node.mask);
    do{}while(0);
    vector<int> candidate_sorters;
    vec p = find_candidate_splits(node.masks[0], node.masks[1], &candidate_sorters);
    for(i32 j = 0; j < (i32)(2); ++j) {
      int c = tree.nodes[iwaste].children[j];
      if(c&1) {
        S.proc_at[info.points[info.exit[j]]] = c/2;
      }
    }
    vector<int> perm;
    for(i32 i = 0; i < (i32)(info.npoints); ++i) {
      if(i != info.root &&
         i != info.exit[0] && i != info.exit[1]) {
        perm.push_back(i);
      }
    }
    rng.shuffle(perm);
    f32 t0 = TIMER.elapsed();
    f32 tl = (TL0 - t0) / (nwaste-1-iwaste);
    tl = max(tl, 0.001f);
    { f32 best = 1e9;
      auto best_perm = perm;
      while(TIMER.elapsed()-t0 < tl*TL_ANGLE_RATIO) {
        f32 alpha = rng.randomDouble() * 2*M_PI;
        f32 x = cos(alpha), y = sin(alpha);
        sort(begin(perm), end(perm), [&](int i, int j){
          return
            pos[info.points[i]].x * x + pos[info.points[i]].y * y
            < pos[info.points[j]].x * x + pos[info.points[j]].y * y;
        });
        auto [score, h0] = evaluate<false>
          (perm,
           node.masks[0], node.masks[1],
           v1, v2, p, info);
        if(score < best) {
          best = min(best, score);
          // debug(score, best);
          best_perm = perm;
        }
      }
      perm = best_perm;
    }
    auto [score, h0] = evaluate<false>
      (perm,
       node.masks[0], node.masks[1],
       v1, v2, p, info);
    f32 best = score;
    auto best_perm = perm;
    int iter = 0;
    (void)iter;
    while(1) {
      iter++;
      f32 done = (TIMER.elapsed() - (t0+tl*TL_ANGLE_RATIO)) / (tl*(1.0-TL_ANGLE_RATIO));
      if(done > 1.0) break;
      f32 temp = sa_temp0*pow(sa_temp1/sa_temp0, done);
      auto old_perm = perm;
      int sty = rng.random32(2);
      if(sty == 0) {
        int u = rng.random32(perm.size());
        int v = rng.random32(perm.size());
        int x = perm[u];
        perm.erase(begin(perm)+u);
        perm.insert(begin(perm)+v, x);
      }else if(sty == 1){
        int u = rng.random32(perm.size());
        int v = rng.random32(perm.size());
        swap(perm[u], perm[v]);
      }else{
        int u = rng.random32(perm.size());
        int v = rng.random32(perm.size());
        if(u > v) swap(u,v);
        reverse(begin(perm)+u, begin(perm)+v+1);
      }
      auto [new_score, new_h] = evaluate<false>
        (perm,
         node.masks[0], node.masks[1],
         v1, v2, p, info);
      f32 delta = new_score-score;
      if(delta <= 0 || delta <= temp * rng.randomDouble()) {
        score = new_score;
      }else{
        perm = old_perm;
      }
      if(score < best) {
        best = score;
        best_perm = perm;
        do{}while(0);
      }
    }
    auto [final_score, h] = evaluate<true>
      (best_perm,
       node.masks[0], node.masks[1],
       v1, v2, p, info);
    do{}while(0);
    if(final_score > 2*target_score) {
      do{}while(0);
    }
    int lastu = history_pool[h].u;
    S.graph[info.points[info.root]] = {info.points[lastu],info.points[lastu]};
    vector<array<int, 3>> A;
    while(h != -1) {
      auto e = history_pool[h];
      A.push_back({e.u,e.c1,e.c2});
      candidates_at[info.points[e.u]] = candidate_sorters;
      S.sort_at[info.points[e.u]] = rng.sample(candidate_sorters);
      S.graph[info.points[e.u]] = {info.points[e.c2], info.points[e.c1]};
      h = e.prev;
    }
    vector<vec> O(npos);
    O[info.exit[0]] = vec(0.0);
    O[info.exit[1]] = vec(0.0);
    for(i32 x = 0; x < (i32)(nwaste); ++x) {
      for(i32 j = 0; j < (i32)(2); ++j) {
        if(node.masks[j]&bit(x)) O[info.exit[j]][x] = 1.0;
      }
    }
    reverse(begin(A), end(A));
    for(auto [u,c1,c2] : A) {
      for(i32 x = 0; x < (i32)(nwaste); ++x) O[u][x] = (1-p[x])*O[c1][x]+p[x]*O[c2][x];
    }
    node_targets[iwaste] = O[lastu];
  }
  for(i32 i = 0; i < (i32)(root_path.size()-1); ++i) {
    int a = root_path[i];
    int b = root_path[i+1];
    S.graph[b] = {a,a};
  }
  { auto order = S.get_top_order();
    f32 score = S.simulate(order);
    f32 best = score;
    do{}while(0);
    int iter = 0;
    while(1) {
      iter++;
      if(iter % 512 == 0) {
        if(TIMER.elapsed() > TL1) break;
      }
      int ty = rng.random32(2);
      if(ty == 0) {
        int i = nwaste+1+rng.random32(nlocations);
        if(candidates_at[i].empty()) {
          continue;
        }
        int j = rng.sample(candidates_at[i]);
        int old = S.sort_at[i];
        if(old == j) continue;
        S.sort_at[i] = j;
        f32 new_score = S.simulate(order);
        if(new_score <= score) {
          score = new_score;
          if(score < best) {
            best = score;
            do{}while(0);
          }
        }else{
          S.sort_at[i] = old;
        }
      }else{
        int i1 = nwaste+1+rng.random32(nlocations);
        int i2 = nwaste+1+rng.random32(nlocations);
        if(candidates_at[i1].empty()) continue;
        if(candidates_at[i2].empty()) continue;
        if(i1 == i2) continue;
        swap(S.sort_at[i1], S.sort_at[i2]);
        f32 new_score = S.simulate(order);
        if(new_score <= score) {
          score = new_score;
          if(score < best) {
            best = score;
            do{}while(0);
          }
        }else{
          swap(S.sort_at[i1], S.sort_at[i2]);
        }
      }
    }
    do{}while(0);
    cerr << score << endl;
  }
  S.print();
}
int main() {
  cerr << setprecision(5) << fixed;
  TIMER.reset();
  rng.reset(time(0));
  read();
  init();
  solve();
  cerr << "[DATA] nevaluate = " << nevaluate << endl;
  cerr << "[DATA] time = " << (i32)(TIMER.elapsed()*1000) << endl;
  return 0;
}
