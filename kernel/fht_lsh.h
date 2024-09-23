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


// Please note that this version of FastHadamardLSH
// was modified from  https://github.com/lducas/AVX2-BDGL-bucketer commit 630c2286a440fae1eddd9f90341ff2020f18b614.
// This version can be found at: https://github.com/joerowell/gcc-bucketer commit 868c3ef

#ifndef G6K_FHTLSH_H
#define G6K_FHTLSH_H

#include <stdio.h>
#include <assert.h>
#include <cstring>
#include <algorithm>
#include <array>
#include <vector>
#include <math.h>
#include <iostream>

// All of the SIMD stuff lives here
#include "simd.h"


class FastHadamardLSH
    {

    private:
    // full_seed contains the source of randomness. We extract from this in our permutations and update
    // it to keep it fresh.

    // aes_key contains the key we use for using the singular AES tour for updating our randomness.
      Simd::SmallVecType aes_key;
      // This contains some extra state for the random number generator in non-AVX2 builds.
      // Essentially the AVX2 version uses AES instructions to generate randomness: in non AVX2 we don't assume this and instead
      // use a generator based on xorshift128. This requires extra state. Note that the prg_state is also used for this, but we
      // overwrite that via the output of the call to the rng. 
      Simd::SmallVecType extra_state;

      inline void insert_in_maxs(int32_t * const maxs, const int16_t val, const int32_t index);
      inline void insert_in_maxs_epi16(int32_t * const maxs, const int i_high, const Simd::VecType vals);

      template<int regs_> 
      void hash_templated(const int16_t * const vv, int32_t * const res);

    public:
    Simd::SmallVecType full_seed;//TODO:move back to private

    // n is the adjusted hashing dimension that we use in this subcode.
        size_t n;

    // codesize is the size of this subcode
        size_t codesize;
    // multi_hash tells us how many subcode blocks we are hashing against.
        unsigned multi_hash;

        // regs is the number of registers we have available to use for hashing
        unsigned regs;
        int64_t seed;

    // Prints out an m256i vector as 16bit chunks
      void pprint(const Simd::VecType x)
        {
            int16_t* f = (int16_t*) &x;
            printf("%4i %4i %4i %4i %4i %4i %4i %4i %4i %4i %4i %4i %4i %4i %4i %4i\n",
                    f[  0], f[  1], f[  2], f[  3], f[  4], f[  5], f[  6], f[  7],
                    f[8+0], f[8+1], f[8+2], f[8+3], f[8+4], f[8+5], f[8+6], f[8+7]);
        }


        explicit FastHadamardLSH(const size_t _n,const  size_t _codesize, 
                                 const unsigned _multi_hash, const  int64_t _seed) : 
        n(_n), codesize(_codesize), multi_hash(_multi_hash), seed(_seed)
        { 
            // Generate some initial randomness - this is used for building the permutations later
	  aes_key = Simd::m128_set_epi64x(0xDEADBEAF * _seed + 0xC00010FF, 0x00BAB10C * _seed + 0xBAADA555);
	  full_seed = Simd::m128_set_epi64x(0xC0DED00D * _seed + 0xBAAAAAAD, 0x000FF1CE * _seed + 0xCAFED00D);
	  extra_state = aes_key;
	  
        // It only makes sense to hash to one or more buckets, so we do this check here
            if (multi_hash == 0) throw std::invalid_argument( "multi_hash should be >= 1" );

            if (n <= 16 || n > 128) throw std::invalid_argument( "lsh dimension invalid (must be 16 < n <= 128)");
            regs = (n+15)/16;
            if (regs < 2 || regs > 8) throw std::invalid_argument( "lsh dimension invalid (must be 16 < n <= 128)");
        };

    // This hashes the vector v against this subcode, producing an array of hash values (stored in hashes)
        void hash(const float * v, float * coeff, int32_t * hashes);

    };

class ProductLSH
    {

    private:
        // Permutation & sign are used to permute the input vector
            std::vector<size_t> permutation;
            std::vector<int> sign;

        // codesizes denotes how long each subcode is
            std::vector<size_t> codesizes;
            std::vector<int> ns, is;

            // This holds all of the different subcodes
        std::vector<FastHadamardLSH> lshs;
            
        // This function inserts the score into the correct position of the scores array
        // as well as the index into the right position of the indices array.
            inline bool insert_in_maxs(float * const scores, int32_t * const indices, const float score, const int32_t index);

        // These are used for the prg_state when building the permutation
      Simd::SmallVecType full_seed;
      Simd::SmallVecType aes_key;
      Simd::SmallVecType extra_state;

        // This function is a specialisation of the hash function - the only difference here is that we handle differently
        // depending on the number of blocks we're hashing against 
            template<int blocks_> 
        void hash_templated(const float * vv, int32_t * res);

        void pprint(const Simd::SmallVecType x)
        {
            int8_t* f = (int8_t*) &x;
            printf("%4i %4i %4i %4i %4i %4i %4i %4i \n",
                   f[  0], f[  1], f[  2], f[  3], f[  4], f[  5], f[  6], f[  7]);
        }

    public: 
        
        // n denotes the dimension of the lattice, blocks denotes the width of each hash block
        size_t n, blocks;
        // codesize denotes how long this code is 
        int64_t codesize;

        //multi_hash is how many buckets we're targeting
        //multi_hash_block is the size of each hash_block
        unsigned multi_hash;
        unsigned multi_hash_block;
        explicit ProductLSH(const size_t _n,const size_t _blocks, const  int64_t _codesize, 
                            const unsigned _multi_hash, const int64_t _seed) : 
        permutation(_n), sign(_n),
        codesizes(_blocks), ns(_blocks), is(_blocks), 
        n(_n), blocks(_blocks), codesize(_codesize), multi_hash(_multi_hash)
        {
        // Set up our permutation randomness    
            aes_key =   Simd::m128_set_epi64x(0xFACEFEED * _seed + 0xDEAD10CC, 0xFFBADD11 * _seed + 0xDEADBEEF);
            extra_state = aes_key;
            full_seed = Simd::m128_set_epi64x(0xD0D0FA11 * _seed + 0xD15EA5E5, 0xFEE1DEAD * _seed + 0xB105F00D);
            Simd::SmallVecType prg_state = full_seed;


            // Taken is a vector denoting if we've used this position in our permutation before
            std::vector<bool> taken(n,false);


            // Build the permutation that's applied to each vector upon hashing
            for (size_t i = 0; i < n;)
            {
            // produce a new prng state & take the first 64-bits as output
            // We then use this to correspond to an array position - repeating if we fail
            // to find an unused element

                prg_state = Simd::m128_random_state(prg_state, aes_key, &extra_state);
                size_t pos = Simd::m128_extract_epi64<0>(prg_state) % n;

                if (taken[pos] ){
                    continue;
                }
            // Note that we've used this element, and put it in the permutation array
                taken[pos] = true;
                permutation[i] = pos;
            // We also take this chance to permute the signs too - if the second 64-bit number is odd then
            // we will negate in future. Then, just continue producing the permutation
                sign[pos] = Simd::m128_extract_epi64<1>(prg_state) % 2 ? 1 : -1;


                ++i;
            }

	
        // rn is the number of remaining dimensions we have to divide up
        // Similarly with rblocks and rcodesize
            int rn = n;
            int rblocks = blocks;
            double rcodesize = codesize / (1 << (blocks - 1));
            multi_hash_block = multi_hash;
    
        // We take this opportunity to reserve some memory, so that we don't have to keep 
        // allocating more as we push back    
            lshs.reserve(blocks);
	    
            for (size_t i = 0; i < blocks; ++i)
            {
        // Divide up the number of dimensions 
                ns[i] = rn / rblocks;
                is[i] = n - rn;
                codesizes[i] = int64_t(pow(rcodesize, 1./rblocks)+0.0001);
        // Check that we haven't given more work than necessary to a single sub-code
                assert(multi_hash <= codesizes[i]);
        // Put the new subcode into the lshs array at position i.
                lshs.emplace_back(FastHadamardLSH(ns[i], codesizes[i], multi_hash, _seed + uint64_t(i) * 1641633149));
        // Subtract the number of dimensions we've allocated to lsh[i] & the code size we've dealt with 
                rn -= ns[i];
                rcodesize /= codesizes[i];
                rblocks --;
            }
        codesize = 1;
        for (size_t i = 0; i < blocks; ++i)
        {
            codesize *= codesizes[i];
            if (i>0) codesize *= 2;
        }
        assert(codesize <= _codesize);
        //std::cout << "codesize: " << codesize << " blocks: " << blocks << std::endl;
        //for (int i=0; i<blocks; i++){
        //    pprint(lshs[i].full_seed);
        //}
        };

    // Hash. Given a vector v as input, hash it against the subcodes and produce the results, stored in res. 
        void hash(const float* v, int32_t * res);
    };

#endif
