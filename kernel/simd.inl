#ifndef G6K_SIMD_H
#error Do not include simd.inl without simd.h
#endif

#include <cstring> // Needed for memcpy. If (for some unknown reason) this is prohibitive you can instead
                   // use __builtin_memcpy.


// Clang and GCC don't have a shared syntax for built-in shuffling.
// This was fixed in GCC-12 and later, but GCC's shuffle expects a vector mask,
// whereas Clang needs a compile-time known list of indices.
#ifdef __clang__
#define SHUFFLE16(a, b, ...) __builtin_shufflevector(a, b, __VA_ARGS__)
// We just re-use SHUFFLE16 here, so __builtin_shufflevector is variadic.
#define SHUFFLE8(a, b, ...) SHUFFLE16(a, b, __VA_ARGS__)
#define SHUFFLE4(a, b, ...) SHUFFLE16(a, b, __VA_ARGS__)
#elif defined(__GNUG__)
#define SHUFFLE16(a, b, ...) __builtin_shuffle(a, b, Vec16s{__VA_ARGS__});
#define SHUFFLE8(a, b, ...) __builtin_shuffle(a, b, Vec8s{__VA_ARGS__});
#define SHUFFLE4(a, b, ...) __builtin_shuffle(a, b, Vec4q{__VA_ARGS__});
#endif

template<int pos>
inline Simd::SmallVecType Simd::m128_slli_epi64(const SmallVecType a) {
#ifdef HAVE_AVX2
  return _mm_slli_epi64(a, pos);
#else
  return a << pos;
#endif
}

inline Simd::VecType Simd::m256_loadu_si256(const int16_t *const a) {
#ifdef HAVE_AVX2
  return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a));
#else
  Vec16s out;
  memcpy(&out, a, sizeof(out));
  return out;
#endif
}

inline void Simd::m256_storeu_si256(int16_t *a, const VecType b) {
#ifdef HAVE_AVX2
  return _mm256_storeu_si256(reinterpret_cast<__m256i *>(a), b);
#else
  memcpy(a, &b, sizeof(VecType));
#endif
}

inline Simd::SmallVecType Simd::m128_set_epi64x(const int64_t e1,
                                                const int64_t e0) {
#ifdef HAVE_AVX2
  return _mm_set_epi64x(e1, e0);
#else
  // NOTE the swap: this is for endianness.
  // All else acts exactly the same, this is just the one weird bit of
  // inconsistency.
  int64_t arr[]{e0, e1};
  SmallVecType vec;
  memcpy(&vec, &arr, sizeof(arr));
  return vec;
#endif
}

inline Simd::SmallVecType Simd::m128_set_epi64x(const uint64_t e1,
                                                const uint64_t e0) {
#ifdef HAVE_AVX2
  return _mm_set_epi64x(e1, e0);
#else
  // NOTE the swap: this is for endianness.
  // All else acts exactly the same, this is just the one weird bit of
  // inconsistency.
  uint64_t arr[]{e0, e1};
  SmallVecType vec;
  memcpy(&vec, &arr, sizeof(arr));
  return vec;
#endif
}

inline Simd::VecType Simd::m256_hadd_epi16(const Simd::VecType a,
                                           const Simd::VecType b) {
#ifdef HAVE_AVX2
  return _mm256_hadd_epi16(a, b);
#else
  // The trick in this function is the following; we simulate horizontal
  // addition by adding `a` to a shifted version of `a` (i.e shifted to the
  // right by 1) and adding. This gives us a vector that almost has exactly what
  // we want: it has (a[0] + a[1], a[1] + a[2],....). Note that every `odd`
  // position has something useless in it with this approach, so we'll need to
  // do another shuffle at the end to recombine these results into something
  // useful.

#define SHIFT_RIGHT_1_MASK 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0
#define HADD_SHIFT_MASK 0,2,4,6,16,18,20,22,8,10,12,14,24,26,28,30

  const auto a1 = SHUFFLE16(a, a, SHIFT_RIGHT_1_MASK);
  const auto b1 = SHUFFLE16(b, b, SHIFT_RIGHT_1_MASK);
  

  // a2 = (a[0] + a[1], a[1] + a[2]  , a[2] + a[3], a[3] + a[4],
  //       a[4] + a[5], a[5] + a[6]  , a[6] + a[7], a[7] + a[8],
  //       a[8] + a[9], a[9] + a[10] , a[10] + a[11], a[11] + a[12],
  //       a[12] + a[13], a[13] + a[14], a[14] + a[15], a[15] + a[0]);

  const auto a2 = a + a1;
  const auto b2 = b + b1;

  // This is a multi-lane shuffle!
  // The mask works by shuffling mod the length of the vector.
  // This means that (for example) a value of `18` refers to b2[2], whereas `2`
  // refers to a2[2].
  return SHUFFLE16(a2, b2, HADD_SHIFT_MASK);
  #undef SHIFT_RIGHT_1_MASK
  #undef HADD_SHIFT_MASK
#endif
}

inline Simd::SmallVecType Simd::m128_add_epi64(const SmallVecType a,
                                               const SmallVecType b) {
#ifdef HAVE_AVX2
  return _mm_add_epi64(a, b);
#else
  Vec2q a_as_q, b_as_q;
  memcpy(&a_as_q, &a, sizeof(a));
  memcpy(&b_as_q, &b, sizeof(b));
  const auto c = a_as_q + b_as_q;
  SmallVecType result;
  static_assert(sizeof(result) == sizeof(c), "Error: sizeof(SmallVecType) != sizeof(Vec2q)");
  memcpy(&result, &c, sizeof(result));
  return result;
#endif
}

inline Simd::VecType Simd::m256_add_epi16(const VecType a, const VecType b) {
#ifdef HAVE_AVX2
  return _mm256_add_epi16(a, b);
#else
  return a + b;
#endif
}

inline Simd::VecType Simd::m256_sub_epi16(const VecType a, const VecType b) {
#ifdef HAVE_AVX2
  return _mm256_sub_epi16(a, b);
#else
  return a - b;
#endif
}

inline Simd::SmallVecType Simd::m128_xor_si128(const SmallVecType a,
                                               const SmallVecType b) {
#ifdef HAVE_AVX2
  return _mm_xor_si128(a, b);
#else
  return a ^ b;
#endif
}

template<int pos>
inline Simd::SmallVecType Simd::m128_srli_epi64(const SmallVecType a) {
#ifdef HAVE_AVX2
  return _mm_srli_si128(a, pos);
#else
  return a >> pos;
#endif
}

inline Simd::VecType Simd::m256_xor_si256(const VecType a, const VecType b) {
#ifdef HAVE_AVX2
  return _mm256_xor_si256(a, b);
#else
  return a ^ b;
#endif
}

inline Simd::VecType Simd::m256_and_si256(const VecType a, const VecType b) {
#ifdef HAVE_AVX2
  return _mm256_and_si256(a, b);
#else
  return a & b;
#endif
}

template <uint8_t mask>
inline Simd::VecType Simd::m256_permute4x64_epi64(const VecType a) {
#ifdef HAVE_AVX2
  return _mm256_permute4x64_epi64(a, mask);
#else
  // NOTE: we need to extract the bits of `mask` into a
  // vector so that we can shuffle. This involves us isolating each pair of bits
  // in mask and placing them into a VecType.

  // You could do this with a lookup table (it would only require a bit of
  // storage) but it's probably not worth it: this is just a general function.

  // NOTE: the memcpying here is a type-punning guard. Technically reinterpret casting
  // here would be UB (GCC has an extension for this, but Clang doesn't).
  // The compiler knows that this isn't actually an operation that it needs to do:
  // the backend optimises out the memcpys (see https://godbolt.org/z/zxKvnKdeP: the code
  // for this function compiles down to a vectorised permutation).
  Vec4q tmp_a;
  memcpy(&tmp_a, &a, sizeof(tmp_a));
  const auto res = SHUFFLE4(tmp_a, tmp_a, mask&3, (mask&12)>>2,
                      (mask&48) >> 4, (mask&192) >> 6);
  VecType out;
  memcpy(&out, &res, sizeof(out));
  return out;
#endif
}

inline Simd::VecType
Simd::m256_permute4x64_epi64_for_hadamard(const VecType a) {
#ifdef HAVE_AVX2
  return _mm256_permute4x64_epi64(a, 0b01001110);
#else
  return m256_permute4x64_epi64<0b01001110>(a);
#endif
}

inline int Simd::m256_testz_si256(const VecType a, const VecType b) {
#ifdef HAVE_AVX2
  return _mm256_testz_si256(a, b);
#else
  // This doesn't have a neat implementation.
  // Basically, GCC's == operator produces a vector as a result, which is really
  // useful in most cases (but not here).
  // To get around this, we just copy into a fixed-size array and compare against the all zero.
  // Again, the copy here is a type-punning guard.
  const auto res = a & b;
  constexpr static std::array<int16_t, 16> zero{0};
  std::array<int16_t, 16> res_as_16;
  memcpy(&res_as_16, &res, sizeof(res_as_16));
  return memcmp(&res_as_16, &zero, sizeof(zero)) == 0;
#endif
}

inline Simd::VecType Simd::m256_abs_epi16(const VecType a) {
#ifdef HAVE_AVX2
  return _mm256_abs_epi16(a);
#else
  // Produce a "negative" version
  const auto a_copy = a * -1;
  // Now we'll check what elements in a < 0
  const auto gt_0 = a > 0;
  // If the element is < 0 then we'll return it as is
  return gt_0 ? a : a_copy;
#endif
}

template <int pos> inline int64_t Simd::m256_extract_epi64(const VecType a) {
  static_assert(pos < 4, "Error: the requested index is too high.");
#ifdef HAVE_AVX2
  return _mm256_extract_epi64(a, pos);
#else
  int64_t arr[4];
  memcpy(&arr, &a, sizeof(a));
  return a[pos];
#endif
}

template <int pos>
inline int64_t Simd::m128_extract_epi64(const SmallVecType a) {
  static_assert(pos < 2, "Error: the requested index is too high.");
#ifdef HAVE_AVX2
  return _mm_extract_epi64(a, pos);
#else
  int64_t arr[2];
  static_assert(sizeof(arr) == sizeof(a), "Error: sizeof(arr) != sizeof(a)");
  memcpy(&arr, &a, sizeof(a));
  return a[pos];
#endif
}

inline Simd::VecType Simd::m256_sign_epi16_ternary(const VecType a,
                                                   const VecType mask) {
#ifdef HAVE_AVX2
  // Just use a regular sign operation here.
  return _mm256_sign_epi16(a, mask);
#else
  // Since `mask` is ternary we can just multiply here.
  return a * mask;
#endif
}

inline Simd::VecType Simd::m256_sign_epi16(const VecType a,
                                           const VecType mask) {
#ifdef HAVE_AVX2
  return _mm256_sign_epi16(a, mask);
#else
  // NOTE: if you can guarantee that mask is ternary then you can just do a
  // multiply here. For that see m256_sign_epi16 ternary.

  // For the sake of absolute compatibility though we need to be slightly
  // cleverer. First of all, we'll need to zero some things, so
  constexpr static Vec16s zeroes{0};

  // Now: GCC's intrinsics are weird. If you do pairwise comparision you'll get
  // a vector that contains 0 for `true` and `-1` for `false` (I guess this is
  // because -1 = 0xFFFF in this world).

  // Now we need to find all of those entries in `mask` that are < 0
  const auto lt_0 = mask < zeroes;
  // And all of those that are exactly 0
  const auto are_0 = mask == zeroes;

  // We'll need to be able to make a negative choice shortly, so we'll just
  // negate the elements
  const auto neg_a = (-1 * a);

  // Now we can recombine. It's pretty simple: we use a vector select to return
  // the right values at each step.
  Vec16s intermediate = lt_0 ? neg_a : a;

  // We'll save on the final comparison by noting that we already have the
  // positive outputs.
  Vec16s result = are_0 ? zeroes : intermediate;
  return result;
#endif
}

template<int pos>
inline Simd::VecType Simd::m256_slli_epi16(const VecType a) {
#ifdef HAVE_AVX2
  return _mm256_slli_epi16(a, pos);
#else
  return a << pos;
#endif
}

inline Simd::VecType Simd::m256_cmpgt_epi16(const VecType a, const VecType b) {
#ifdef HAVE_AVX2
  return _mm256_cmpgt_epi16(a, b);
#else
  return a > b;
#endif
}

inline Simd::VecType Simd::m256_broadcastsi128_si256(const SmallVecType in) {
#ifdef HAVE_AVX2
  return _mm256_broadcastsi128_si256(in);
#else
  // N.B we need to copy into an array here, since Clang doesn't let us do
  // element-wise access into vectors.
  std::array<int16_t, 16> out_as_arr;
  memcpy(&out_as_arr, &in, sizeof(in));
  memcpy(&out_as_arr[8], &in, sizeof(in));  
  Vec16s out;
  memcpy(&out, &out_as_arr, sizeof(out));
  return out;
#endif
}

inline Simd::SmallVecType Simd::m128_shuffle_epi8(const SmallVecType in,
                                                  const SmallVecType mask) {
#ifdef HAVE_AVX2
  return _mm_shuffle_epi8(in, mask);
#elif defined(__GNUG__) & !defined(__clang__)
  // We have separate code here for GCC and Clang. The reason why
  // is because Clang's shuffle doesn't support variable inputs,
  // and GCC's shuffle would be pessimised by just using the generic code.
  
  // The mm_shuffle_epi8 intrinsic is a bit weird.
  // First of all, we need to extract the lowest 4 bits of each word (since
  // there's only 16 options this is all we're allowed). We then shuffle
  // according to that.
  Vec16c shuffle_mask, in_as_16c;
  memcpy(&shuffle_mask, &mask, sizeof(shuffle_mask));
  shuffle_mask = shuffle_mask & 15;
  memcpy(&in_as_16c, &in, sizeof(in));

  // We'll use GCC@s shuffle here since we're inside the GNUG block.
  const auto intermediate = __builtin_shuffle(in_as_16c, shuffle_mask);

  // It turns out the mm_shuffle_epi8 intrinsic is a bit weird.
  // Essentially, if the top-most bit of `mask[i]` is set then `out[i] == 0`.
  Vec16uc mask_as_uc;
  memcpy(&mask_as_uc, &mask, sizeof(mask_as_uc));
  const auto gt_64 = mask_as_uc & 0x80;

  // And now if the element is > 64 we choose 0, otherwise we choose the
  // shuffled version
  const auto result = gt_64 ? 0 : intermediate;
  SmallVecType res;
  memcpy(&res, &result, sizeof(result));
  return res;
#else
  // We'll shuffle by hand.
  std::array<uint8_t, 16> in_arr, mask_arr, out_arr;
  static_assert(sizeof(in_arr) == sizeof(SmallVecType), "Error: wrong array size for copy");
  memcpy(&in_arr, &in, sizeof(in_arr));
  memcpy(&mask_arr, &mask, sizeof(mask_arr));
  for(unsigned i = 0; i < 16; i++) {
    out_arr[i] = in_arr[mask_arr[i] & 15];
  }
  
  // Now we've shuffled we'll go back to explicitly vectorised instructions.
  Vec16uc intermediate;
  memcpy(&intermediate, &out_arr, sizeof(intermediate));

  // We'll also now do the same masking as in the GNUG block.
  Vec16uc mask_as_uc;
  memcpy(&mask_as_uc, &mask, sizeof(mask));
  const auto gt_64 = mask_as_uc & 0x80;
  const auto result = gt_64 ? 0 : intermediate;
  SmallVecType res;
  memcpy(&res, &result, sizeof(result));
  return res;
#endif
}

inline Simd::VecType Simd::m256_shuffle_epi8(const VecType in,
                                             const VecType mask) {
#ifdef HAVE_AVX2
  return _mm256_shuffle_epi8(in, mask);
#else
  // WARNING: you cannot use the native shuffle here.
  // As tempting as it might seem, the reason why is that shuffle
  // let's you do cross-lane shuffles, whereas the _mm256_shuffle_epi8 intrinsic
  // does not. To fix this problem, we sub-divide: we deal with each 128-bit
  // segment separately and then re-combine at the end.
  std::array<int16_t, 16> mask_as_arr, in_as_arr;
  static_assert(sizeof(mask_as_arr) == sizeof(in), "Error: wrong vector size for copy");
  memcpy(&mask_as_arr, &mask, sizeof(mask_as_arr));
  memcpy(&in_as_arr, &in, sizeof(in_as_arr));

  SmallVecType first_mask, last_mask;
  Vec8s first, last;

  // NOTE: the compiler is likely to turn these into moves, since these
  // variables are most likely in registers.
  memcpy(&first, &in_as_arr, sizeof(Vec8s));
  memcpy(&last, &in_as_arr[8], sizeof(Vec8s));
  memcpy(&first_mask, &mask_as_arr, sizeof(Vec8s));
  memcpy(&last_mask, &mask_as_arr[8], sizeof(Vec8s));
    
  const auto res_1 = Simd::m128_shuffle_epi8(
      first, first_mask);
  const auto res_2 =
      Simd::m128_shuffle_epi8(last, last_mask);

  std::array<int16_t, 16> out_as_arr;
  memcpy(&out_as_arr, &res_1, sizeof(res_1));
  memcpy(&out_as_arr[8], &res_2, sizeof(res_2));
  VecType result;
  memcpy(&result, &out_as_arr, sizeof(result));
  return result;
#endif
}

inline void Simd::m256_hadamard16_epi16(const VecType x1, VecType &r1) {
  // Apply a permutation 0123 -> 1032 to x1 (this operates on 64-bit words).
  auto a1 = m256_permute4x64_epi64_for_hadamard(x1);

  // From here we go back to treating x1 as a 16x16 vector.
  // Negate the first 8 of the elements in the vector
  auto t1 = m256_sign_epi16(x1, reinterpret_cast<VecType>(sign_mask_8));

  // Add the permutation to the recently negated portion & apply the second sign
  // mask. (BTW the Wikipedia page for the Hadamard transform is really useful
  // for understanding what's going on here!)
  a1 = m256_add_epi16(a1, t1);
  auto b1 = m256_sign_epi16(a1, reinterpret_cast<VecType>(sign_mask_2));
  a1 = m256_hadd_epi16(a1, b1);
  b1 = m256_sign_epi16(a1, reinterpret_cast<VecType>(sign_mask_2));
  a1 = m256_hadd_epi16(a1, b1);
  b1 = m256_sign_epi16(a1, reinterpret_cast<VecType>(sign_mask_2));
  r1 = m256_hadd_epi16(a1, b1);
}

inline void Simd::m256_hadamard32_epi16(const VecType x1, const VecType x2,
                                        VecType &r1, VecType &r2) {
  auto a1 = m256_permute4x64_epi64_for_hadamard(x1);
  auto a2 = m256_permute4x64_epi64_for_hadamard(x2);

  auto t1 = m256_sign_epi16(x1, reinterpret_cast<VecType>(sign_mask_8));
  auto t2 = m256_sign_epi16(x2, reinterpret_cast<VecType>(sign_mask_8));

  a1 = m256_add_epi16(a1, t1);
  a2 = m256_add_epi16(a2, t2);

  auto b1 = m256_sign_epi16(a1, reinterpret_cast<VecType>(sign_mask_2));
  auto b2 = m256_sign_epi16(a2, reinterpret_cast<VecType>(sign_mask_2));

  // Now apply the 16-bit Hadamard transforms and repeat the process
  a1 = m256_hadd_epi16(a1, b1);
  a2 = m256_hadd_epi16(a2, b2);
  b1 = m256_sign_epi16(a1, reinterpret_cast<VecType>(sign_mask_2));
  b2 = m256_sign_epi16(a2, reinterpret_cast<VecType>(sign_mask_2));
  a1 = m256_hadd_epi16(a1, b1);
  a2 = m256_hadd_epi16(a2, b2);
  b1 = m256_sign_epi16(a1, reinterpret_cast<VecType>(sign_mask_2));
  b2 = m256_sign_epi16(a2, reinterpret_cast<VecType>(sign_mask_2));
  a1 = m256_hadd_epi16(a1, b1);
  a2 = m256_hadd_epi16(a2, b2);

  r1 = m256_add_epi16(a1, a2);
  r2 = m256_sub_epi16(a1, a2);
}

inline void Simd::m256_mix(VecType &v0, VecType &v1, const VecType &mask) {
  VecType diff;
  diff = m256_xor_si256(v0, v1);
  diff = m256_and_si256(diff, mask);
  v0 = m256_xor_si256(v0, diff);
  v1 = m256_xor_si256(v1, diff);
}



inline Simd::SmallVecType Simd::m128_random_state(SmallVecType prg_state,
                                                  SmallVecType key,
                                                  SmallVecType *extra_state) {
#ifdef HAVE_AVX2
  (void)extra_state;
  return _mm_aesenc_si128(prg_state, key);
#else
  // Silence the fact it isn't used.
  (void)key;
  (void)prg_state;

  // Rand should be fine here: it stops us falling into random cycles.


  //srand(m128_extract_epi64<1>(prg_state));


  *extra_state = m128_set_epi64x(static_cast<uint64_t>(rand()),
                                 static_cast<uint64_t>(rand()));

  return *extra_state;
#endif
}

/**
 * m256_permute_epi16. The goal of this function is to permute the input vector
 * v, according to the randomness from prg_state & the tailmask. \tparam[in]
 * regs: the number of registers to use. \param[in] v: a pointer to a sequence
 * of VecTypes. \param[in] prgstate: this is the state of the prg for the
 * current hashing round \param[in] tailmask: a mask for handling mixing when
 * the length of v is not a multiple of 16. \param[in] key: the rest of the
 * state of the random number generator.
 */
template <int regs_>
inline void Simd::m256_permute_epi16(VecType *const v, SmallVecType &prg_state,
                                     const VecType tailmask,
                                     const SmallVecType &key,
                                     SmallVecType *extra_state) {

  // NOTE: this is a specialisation for regs_ = 2. Essentially, we
  // might not have enough space for permutations if we have just two registers.
  // The compiler should optimise this check away.
  if (regs_ == 2) {
    // double pack the prg state in rnd (has impact of doubly repeating the prg
    // state in rnd) Though we will use different threshold on each part
    // decorrelating the permutation on each halves

    auto rnd = m256_broadcastsi128_si256(prg_state);
    VecType mask;

    // With only 2 registers, we may not have enough room to randomize via
    // m256_mix, so we also choose at random among a few precomputed permutation
    // to apply on the first register

    uint32_t x = m128_extract_epi64<0>(prg_state);
    uint32_t x1 = (x >> 16) & 0x03;
    uint32_t x2 = x & 0x03;

    // Apply the precomputed permutations to the input vector
    v[0] = m256_shuffle_epi8(v[0],
                             reinterpret_cast<VecType>(permutations_epi16[x1]));
    m256_mix(v[0], v[1], tailmask);
    v[0] = m256_permute4x64_epi64<0b10010011>(v[0]);
    v[0] = m256_shuffle_epi8(v[0],
                             reinterpret_cast<VecType>(permutations_epi16[x2]));

    mask = m256_cmpgt_epi16(rnd, reinterpret_cast<VecType>(mixmask_threshold));
    mask = m256_and_si256(mask, tailmask);
    m256_mix(v[0], v[1], mask);

    // Update the randomness
    prg_state = m128_random_state(prg_state, key, extra_state);
    return;
  }

  // regs_ > 2 version.

  // double pack the prg state in rnd (has impact of doubly repeating the prg
  // state in rnd) Though we will use different threshold on each part
  // decorrelating the permutation on each halves

  auto rnd = m256_broadcastsi128_si256(prg_state);
  VecType tmp;

  // We treat the even and the odd positions differently
  // This is for the goal of decorrelating the permutation on the
  // double packed prng state.
  for (int i = 0; i < (regs_ - 1) / 2; ++i) {
    // shuffle 8 bit parts in each 128 bit lane
    // Note - the exact semantics of what this function does are a bit
    // confusing. See the Intel intrinsics guide if you're curious
    v[2 * i] = m256_shuffle_epi8(
        v[2 * i], reinterpret_cast<VecType>(permutations_epi16[i % 3]));
    // For the odd positions we permute each 64-bit chunk according to the mask.
    v[2 * i + 1] = m256_permute4x64_epi64<0b10010011>(v[2 * i + 1]);
  }

  // Now we negate the first two vectors according to the negation masks
  v[0] =
      m256_sign_epi16(v[0], reinterpret_cast<VecType>(negation_masks_epi16[0]));
  v[1] =
      m256_sign_epi16(v[1], reinterpret_cast<VecType>(negation_masks_epi16[1]));

  // swap int16 entries of v[0] and v[1] where rnd > threshold
  tmp = m256_cmpgt_epi16(rnd, reinterpret_cast<VecType>(mixmask_threshold));
  m256_mix(v[0], v[1], tmp);
  // Shift the randomness around before extracting more (somewhat independent)
  // mixing bits
  rnd = m256_slli_epi16<1>(rnd);

  // Now do random swaps between v[0] and v[last-1]
  m256_mix(v[0], v[regs_ - 2], tmp);
  rnd = m256_slli_epi16<1>(rnd);

  // Now do swaps between v[1] and v[last], avoiding padding data
  m256_mix(v[1], v[regs_ - 1], tailmask);

  // More permuting
  for (int i = 2; i + 2 < regs_; i += 2) {
    rnd = m256_slli_epi16<1>(rnd);
    tmp = m256_cmpgt_epi16(rnd, reinterpret_cast<VecType>(mixmask_threshold));
    m256_mix(v[0], v[i], tmp);
    rnd = m256_slli_epi16<1>(rnd);
    tmp = m256_cmpgt_epi16(rnd, reinterpret_cast<VecType>(mixmask_threshold));
    m256_mix(v[1], v[i + 1], tmp);
  }

  // Update the randomness.
  prg_state = m128_random_state(prg_state, key, extra_state);
}
