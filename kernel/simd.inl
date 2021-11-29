#ifndef G6K_SIMD_H
#error Do not include simd.inl without simd.h
#endif

#include <cstring> // Needed for memcpy. If (for some unknown reason) this is prohibitive you can instead
                   // use __builtin_memcpy.

inline Simd::SmallVecType Simd::m128_slli_epi64(const SmallVecType a,
                                                const int mask) {
#ifdef HAVE_AVX2
  return _mm_slli_epi64(a, mask);
#else
  return a << mask;
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
  return (SmallVecType)(Vec2q{e0, e1});
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
  return (SmallVecType)(Vec2uq{e0, e1});
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

  static constexpr Vec16s hadd_shift_mask_epi16 = {
      0, 2, 4, 6, 16, 18, 20, 22, 8, 10, 12, 14, 24, 26, 28, 30};
  static constexpr Vec16s shift_right_1_epi16 = {1, 2,  3,  4,  5,  6,  7,  8,
                                                 9, 10, 11, 12, 13, 14, 15, 0};

  const auto a1 = __builtin_shuffle(a, shift_right_1_epi16);
  const auto b1 = __builtin_shuffle(b, shift_right_1_epi16);

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
  return __builtin_shuffle(a2, b2, hadd_shift_mask_epi16);
#endif
}

inline Simd::SmallVecType Simd::m128_add_epi64(const SmallVecType a,
                                               const SmallVecType b) {
#ifdef HAVE_AVX2
  return _mm_add_epi64(a, b);
#else
  return (SmallVecType)((Vec2uq)(a) + (Vec2uq)(b));
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

inline Simd::SmallVecType Simd::m128_srli_epi64(const SmallVecType a,
                                                const int pos) {
#ifdef HAVE_AVX2
  return _mm_srli_si128(a, pos);
#else
  return (SmallVecType)(((Vec2q)a) >> pos);
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

  constexpr Vec4q temp_mask{mask & 3, (mask & 12) >> 2, (mask & 48) >> 4,
                            (mask & 192) >> 6};
  return reinterpret_cast<Vec16s>(
      __builtin_shuffle(reinterpret_cast<Vec4q>(a), temp_mask));
#endif
}

inline Simd::VecType
Simd::m256_permute4x64_epi64_for_hadamard(const VecType a) {
#ifdef HAVE_AVX2
  return _mm256_permute4x64_epi64(a, 0b01001110);
#else
  // The shuffle vector should be built at compile-time.
  static constexpr int64_t arr[4] = {1, 0, 3, 2};
  // NOTE: for some unknown reason *this* needs to be backwards, even if the
  // version for the general shuffle doesn't need to be backwards.
  // My guess is that it's to do with the endianness of `a` but I've
  // got no idea beyond that.
  constexpr Vec4q mask{arr[3], arr[2], arr[1], arr[0]};
  return reinterpret_cast<Vec16s>(
      __builtin_shuffle(reinterpret_cast<Vec4q>(a), mask));
#endif
}

inline int Simd::m256_testz_si256(const VecType a, const VecType b) {
#ifdef HAVE_AVX2
  return _mm256_testz_si256(a, b);
#else
  // This doesn't have a neat implementation.
  // Basically, GCC's == operator produces a vector as a result, which is really
  // useful in most cases (but not here). To hack around this we need to cast
  // each part to a __int128_t and compare against zero.
  const auto res = a & b;

  Vec8s p1, p2;
  memcpy(&p1, &res, sizeof(p1));
  memcpy(&p2, &res[8], sizeof(p2));

  __int128_t lhs = reinterpret_cast<__int128_t>(p1);
  __int128_t rhs = reinterpret_cast<__int128_t>(p2);

  return (lhs == 0) & (rhs == 0);
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
  return ((Vec4q)a)[pos];
#endif
}

template <int pos>
inline int64_t Simd::m128_extract_epi64(const SmallVecType a) {
  static_assert(pos < 2, "Error: the requested index is too high.");
#ifdef HAVE_AVX2
  return _mm_extract_epi64(a, pos);
#else
  return ((Vec2q)a)[pos];
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

inline Simd::VecType Simd::m256_slli_epi16(const VecType a, const int count) {
#ifdef HAVE_AVX2
  return _mm256_slli_epi16(a, count);
#else
  return a << count;
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
  // The simple solution here is to copy all of the elements of `in` in order
  // into a new vector, but that's really slow and GCC produces _awful_ object
  // code. A better solution (although still slower than the ideal case) is to
  // use memcpy, since GCC seems to do better there: I have no idea why.
  Vec16s out;
  memcpy(&out[0], &in[0], sizeof(Vec8s));
  memcpy(&out[8], &in[0], sizeof(Vec8s));
  return out;
#endif
}

inline Simd::SmallVecType Simd::m128_shuffle_epi8(const SmallVecType in,
                                                  const SmallVecType mask) {
#ifdef HAVE_AVX2
  return _mm_shuffle_epi8(in, mask);
#else
  // The mm_shuffle_epi8 intrinsic is a bit weird.
  // First of all, we need to extract the lowest 4 bits of each word (since
  // there's only 16 options this is all we're allowed). We then shuffle
  // according to that.
  const auto shuffle_mask = reinterpret_cast<Vec16c>(mask) & 15;
  // So now we've gotten that match, we'll want to make the shuffle. Sounds
  // easy, right?
  const auto intermediate =
      __builtin_shuffle(reinterpret_cast<Vec16c>(in), shuffle_mask);

  // Aha! Gotcha.
  // It turns out the mm_shuffle_epi8 intrinsic is a bit weird.
  // Essentially, if the top-most bit of `mask[i]` is set then `out[i] == 0`.
  const auto gt_64 = reinterpret_cast<Vec16uc>(mask) & 0x80;

  // And now if the element is > 64 we choose 0, otherwise we choose the
  // shuffled version
  const auto result = gt_64 ? 0 : intermediate;
  return reinterpret_cast<SmallVecType>(result);
#endif
}

inline Simd::VecType Simd::m256_shuffle_epi8(const VecType in,
                                             const VecType mask) {
#ifdef HAVE_AVX2
  return _mm256_shuffle_epi8(in, mask);
#else
  // WARNING: you cannot use the native __builtin_shuffle here.
  // As tempting as it might seem, the reason why is that __builtin_shuffle
  // let's you do cross-lane shuffles, whereas the _mm256_shuffle_epi8 intrinsic
  // does not. To fix this problem, we sub-divide: we deal with each 128-bit
  // segment separately and then re-combine at the end.
  Vec16s result;
  Vec8s first, last;
  Vec16c first_mask, last_mask;

  // NOTE: the compiler is likely to turn these into moves, since these
  // variables are most likely in registers.
  memcpy(&first, &in, sizeof(Vec8s));
  memcpy(&last, &in[8], sizeof(Vec8s));
  memcpy(&first_mask, &mask, sizeof(Vec8s));
  memcpy(&last_mask, &mask[8], sizeof(Vec8s));

  // Delegate to the 128-bit version.
  auto res_1 = Simd::m128_shuffle_epi8(
      first, reinterpret_cast<SmallVecType>(first_mask));
  auto res_2 =
      Simd::m128_shuffle_epi8(last, reinterpret_cast<SmallVecType>(last_mask));

  // Same caveat as above.
  memcpy(&result, &res_1, sizeof(Vec8s));
  memcpy(&result[8], &res_2, sizeof(Vec8s));
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

  SmallVecType s1 = prg_state;
  const SmallVecType s0 = *extra_state;

  s1 = m128_xor_si128(s1, m128_slli_epi64(s1, 23));
  *extra_state = m128_xor_si128(
      m128_xor_si128(m128_xor_si128(s1, s0), m128_srli_epi64(s1, 5)),
      m128_srli_epi64(s0, 5));
  return m128_add_epi64(*extra_state, s0);
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
  rnd = m256_slli_epi16(rnd, 1);

  // Now do random swaps between v[0] and v[last-1]
  m256_mix(v[0], v[regs_ - 2], tmp);
  rnd = m256_slli_epi16(rnd, 1);

  // Now do swaps between v[1] and v[last], avoiding padding data
  m256_mix(v[1], v[regs_ - 1], tailmask);

  // More permuting
  for (int i = 2; i + 2 < regs_; i += 2) {
    rnd = m256_slli_epi16(rnd, 1);
    tmp = m256_cmpgt_epi16(rnd, reinterpret_cast<VecType>(mixmask_threshold));
    m256_mix(v[0], v[i], tmp);
    rnd = m256_slli_epi16(rnd, 1);
    tmp = m256_cmpgt_epi16(rnd, reinterpret_cast<VecType>(mixmask_threshold));
    m256_mix(v[1], v[i + 1], tmp);
  }

  // Update the randomness.
  prg_state = m128_random_state(prg_state, key, extra_state);
}
