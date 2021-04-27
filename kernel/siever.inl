/***\
*
*   Copyright (C) 2018-2021 Team G6K
*
*   This file is part of G6K. G6K is free software:
*   you can redistribute it and/or modify it under the terms of the
*   GNU General Public License as published by the Free Software Foundation,
*   either version 2 of the License, or (at your option) any later version.
*
*   G6K is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with G6K. If not, see <http://www.gnu.org/licenses/>.
*
****/


#ifndef G6K_SIEVER_INL
#define G6K_SIEVER_INL

#ifndef G6K_SIEVER_H
#error Do not include siever.inl directly
#endif

#include "parallel_algorithms.hpp"
namespace pa = parallel_algorithms;

// a += c*b
template <typename Container, typename Container2>
inline void Siever::addmul_vec(Container &a, Container2 const &b, const typename Container::value_type c, int num)
{
    auto ita = a.begin();
    auto itb = b.cbegin();
    auto const ite = ita + num;

    for (; ita != ite; ++ita, ++itb)
    {
        *ita += c * (*itb);
    }
}

template <typename Container, typename Container2>
inline void Siever::addmul_vec(Container &a, Container2 const &b, const typename Container::value_type c)
{
    auto ita = a.begin();
    auto itb = b.cbegin();
    auto const ite = ita + n;

    for (; ita != ite; ++ita, ++itb)
    {
        *ita += c * (*itb);
    }
}

template <typename Container, typename Container2>
inline void Siever::addsub_vec(Container &a, Container2 const &b, const typename Container::value_type c)
{
    auto ita = a.begin();
    auto itb = b.cbegin();
    auto const ite = ita + n;
    assert( c == 1 || c == -1 );

    if (c == 1)
    {
        for (; ita != ite; ++ita, ++itb)
            *ita += *itb;
    } else {
        for (; ita != ite; ++ita, ++itb)
            *ita -= *itb;
    }
}


inline size_t Siever::histo_index(double l) const
{
    int const i = std::ceil((l - 1.) * size_of_histo - .001);
    if (i > static_cast<int>(size_of_histo-1)) return size_of_histo-1; // the static_cast is just to silence a warning.
    if (i < 0) return 0;
    return i;
}

template<class Functor>
void Siever::apply_to_all_entries(Functor const &functor)
{
    int th_n = std::min<int>(this->params.threads, 1 + db.size()/MIN_ENTRY_PER_THREAD);
    threadpool.run([this,functor](int th_i, int th_n)
        {
	    for (auto i : pa::subrange(db.size(), th_i, th_n))
		functor(this->db[i]);
        }, th_n);
}

template<class Functor>
void Siever::apply_to_all_compressed_entries(Functor const &functor)
{
    int th_n = std::min<int>(this->params.threads, 1 + cdb.size()/MIN_ENTRY_PER_THREAD);
    threadpool.run([this,functor](int th_i, int th_n)
        {
	    for (auto i : pa::subrange(cdb.size(), th_i, th_n))
		functor(this->cdb[i]);
        }, th_n);
}

template<unsigned int THRESHOLD>
inline bool Siever::is_reducible_maybe(const uint64_t *left, const uint64_t *right)
{

    // No idea if this makes a difference, mirroring previous code
    unsigned wa = unsigned(0) - THRESHOLD;
    for (size_t k = 0; k < XPC_WORD_LEN; ++k)
    {
        // NOTE return type of __builtin_popcountl is int not unsigned int
        wa += __builtin_popcountl(left[k] ^ right[k]);
    }
    return (wa > (XPC_BIT_LEN - 2 * THRESHOLD));
}


template<unsigned int THRESHOLD>
inline bool Siever::is_reducible_maybe(const CompressedVector &left, const CompressedVector &right)
{
    return is_reducible_maybe<THRESHOLD>(&left.front(), &right.front());
}

/*
    Same as the above is_reducible_maybe but for the inner most loop of triple sieve
    We look for pairs that are *far apart* (i.e. PopCnt is bigger than expected rather than smaller)
*/
template<unsigned int THRESHOLD>
inline bool Siever::is_far_away(const uint64_t *left, const uint64_t *right)
{
    unsigned w = unsigned(0);
    for (size_t k = 0; k < XPC_WORD_LEN; ++k)
    {
        // NOTE return type of __builtin_popcountl is int not unsigned int
        w += __builtin_popcountl(left[k] ^ right[k]);
    }
    return (w > THRESHOLD);
}


template<unsigned int THRESHOLD>
inline bool Siever::is_far_away(const CompressedVector &left, const CompressedVector &right)
{
    return is_far_away<THRESHOLD>(&left.front(), &right.front());
}

/**
  ENABLE_BITOPS_FOR_ENUM enables bit-operations on enum classes without having to static_cast
  This macro is defined in compat.hpp.
*/

ENABLE_BITOPS_FOR_ENUM(Siever::Recompute)

// if you change one function template, you probably have to change all 2 (vanilla, babai)
template<Siever::Recompute what_to_recompute>
inline void Siever::recompute_data_for_entry(Entry &e)
{
    ATOMIC_CPUCOUNT(214);
    bool constexpr rec_yr = (what_to_recompute & Recompute::recompute_yr) != Recompute::none;
    bool constexpr rec_len = (what_to_recompute & Recompute::recompute_len) != Recompute::none;
    bool constexpr rec_c = (what_to_recompute & Recompute::recompute_c) != Recompute::none;
    bool constexpr rec_uid = (what_to_recompute & Recompute::recompute_uid) != Recompute::none;
    bool constexpr consider_lift = (what_to_recompute & Recompute::consider_otf_lift) != Recompute::none;
    bool constexpr rec_otf_helper = (what_to_recompute & Recompute::recompute_otf_helper) != Recompute::none;


    CPP17CONSTEXPRIF(rec_len) e.len = 0.;
    for (unsigned int i = 0; i < n; ++i)
    {
        if (rec_yr || rec_len)
        {
            FT const yri = std::inner_product(e.x.cbegin()+i, e.x.cbegin()+n, muT[i].cbegin()+i,  static_cast<FT>(0.)) * sqrt_rr[i];
            if (rec_yr) e.yr[i] = yri; // Note : conversion to lower precision
            if (rec_len) e.len+=yri * yri; // slightly inefficient if we only compute the lenght and not yr, but that does not happen anyway.
        }
    }

    // No benefit of merging those loops, I think.

    CPP17CONSTEXPRIF (rec_uid)
    {
        e.uid  = uid_hash_table.compute_uid(e.x);
    }

    CPP17CONSTEXPRIF (rec_c)
    {
        e.c = sim_hashes.compress(e.yr);
    }

    CPP17CONSTEXPRIF (rec_otf_helper)
    {
        for (int k = 0; k < OTF_LIFT_HELPER_DIM; ++k)
        {
            int const i = l - (k + 1);
            if (i < static_cast<signed int>(ll)) break;
            e.otf_helper[k] = std::inner_product(e.x.cbegin(), e.x.cbegin()+n, full_muT[i].cbegin()+l,  static_cast<FT>(0.));
        }
    }

    if (consider_lift && params.otf_lift && e.len < params.lift_radius)
    {
        lift_and_compare(e);
    }

    return;
}


template<Siever::Recompute what_to_recompute>
inline void Siever::recompute_data_for_entry_babai(Entry &e, int babai_index)
{
    ATOMIC_CPUCOUNT(215);
    bool constexpr rec_yr = (what_to_recompute & Recompute::recompute_yr) != Recompute::none;
    bool constexpr rec_len = (what_to_recompute & Recompute::recompute_len) != Recompute::none;
    bool constexpr rec_c = (what_to_recompute & Recompute::recompute_c) != Recompute::none;
    bool constexpr rec_uid = (what_to_recompute & Recompute::recompute_uid) != Recompute::none;
    bool constexpr consider_lift = (what_to_recompute & Recompute::consider_otf_lift) != Recompute::none;
    bool constexpr rec_otf_helper = (what_to_recompute & Recompute::recompute_otf_helper) != Recompute::none;

    CPP17CONSTEXPRIF(rec_len) e.len = 0.;

  // recompute y, yr, len for the other indices (if requested)
    for (int i = n-1; i >= babai_index; --i)
    {
        CPP17CONSTEXPRIF(rec_yr || rec_len)
        {
            FT const yri = std::inner_product(e.x.cbegin()+i, e.x.cbegin()+n, muT[i].cbegin()+i,  static_cast<FT>(0.)) * sqrt_rr[i];
            CPP17CONSTEXPRIF (rec_yr) e.yr[i] = yri;    // Note : conversion to lower precision
            CPP17CONSTEXPRIF (rec_len) e.len+=yri * yri; // slightly inefficient if we only compute the lenght and not yr, but that does not happen anyway.
        }
    }

    for (int i = babai_index -1 ; i >= 0; --i)
    {
        FT yi = std::inner_product(e.x.cbegin()+i+1, e.x.cbegin()+n, muT[i].cbegin()+i+1, static_cast<FT>(0.));
        int c = -std::floor(yi+0.5);
        e.x[i] = c;
        yi += c;
        yi *= sqrt_rr[i];
        e.yr[i] = yi; // ( original_value(yi) + c ) * sqrt_rr[i]. Note that assignment loses precision.
        CPP17CONSTEXPRIF (rec_len) e.len += yi * yi; // adds e.yr[i]^2 (but with higher precision)
    }

    // No benefit of merging those loops, I think.

    CPP17CONSTEXPRIF (rec_uid)
    {
        e.uid = uid_hash_table.compute_uid(e.x);
    }

    CPP17CONSTEXPRIF (rec_c)
    {
        e.c = sim_hashes.compress(e.yr);
    }

    CPP17CONSTEXPRIF (rec_otf_helper)
    {
        for (int k = 0; k < OTF_LIFT_HELPER_DIM; ++k)
        {
            int const i = l - (k + 1);
            
            if (i < static_cast<signed int>(ll)) break;
            e.otf_helper[k] = std::inner_product(e.x.cbegin(), e.x.cbegin()+n, full_muT[i].cbegin()+l,  static_cast<FT>(0.));
        }
    }

    if (consider_lift && params.otf_lift && e.len < params.lift_radius)
    {
        lift_and_compare(e);
    }

    return;
}

#endif
