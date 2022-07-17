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

#include "g6k_config.h"
#include "fht_lsh.h"
#include<cmath> 

// Please note that this file originally came from:
// https://github.com/lducas/AVX2-BDGL-bucketer commit 630c2286a440fae1eddd9f90341ff2020f18b614
// This has since been modified: this version can be found at:
// https://github.com/joerowell/gcc-bucketer commit 868c3ef

// insert_in_maxs. Given a hash value val, insert it into maxs along with its index.
inline void FastHadamardLSH::insert_in_maxs(int32_t * const maxs, const int16_t val, const int32_t index)
{
    
    // Firstly if the value isn't bigger than our minimum then we exit.
    // Note that we offset by one - this is because we pack the value next to its index
    if (std::abs(val) < maxs[2*(multi_hash - 1)]) 
    {
        return;
    }

    // If we're only hashing against one bucket then pack the value and its index in maxs (maxs is of size 2) and return
    int i = multi_hash-1;
    if (!i) 
    {
        maxs[0] = std::abs(val); 
        maxs[1] = (val > 0) ? index : -index; 
        return; 
    }
    
    // As we deal with pairs, we need to decrement this here to make sure we start in the right position
    i--;

    // Starting at the end of xs, we shift all of the existing hash pairs along
    // This means that we end up in the right place for inserting this value and its index
    while((i >= 0) && (std::abs(val) > maxs[2*i]))
    {
        maxs[2*(i + 1)] = maxs[2*i];
        maxs[2*(i + 1) + 1] = maxs[2*i + 1];
        i--;
    }
    
    // Now we're in the right position, so insert
    maxs[2*(i+1)]  = std::abs(val);
    maxs[2*(i+1) + 1]  = (val > 0) ? index : -index;
}

// insert_in_maxs_epi16. Insert the 16-bit vals in vals into the maxs array. 
inline void FastHadamardLSH::insert_in_maxs_epi16(int32_t * const maxs, const int i_high, const Simd::VecType vals)
{	
    // Grab the smallest absolute  value and put it in the threshold vector
    int16_t T = maxs[2*(multi_hash - 1)];
    auto threshold = Simd::build_vec_type(T);
    
    // Convert each value to their abs value - then compare to the threshold
    auto tmp = Simd::m256_abs_epi16(vals);
    // N.b tmp is full of 1s and 0s at this point.
    tmp = Simd::m256_cmpgt_epi16(tmp, threshold);
    // If the bitwise AND is 1 then return (this shouldn't happen!)
    if (Simd::m256_testz_si256(tmp, tmp)) return;

    // Otherwise, we extract the 16-bit values and put them in an array
    int16_t vals_[16];
    Simd::m256_storeu_si256(vals_, vals);

    // Set up the iteration condition - we want a value of at max 16 (as it's a 256-bit vector), but
    // codesize-i_high encodes what the length should be.
    int imax = codesize - i_high;
    imax = imax > 16 ? 16 : imax;

    // Insert each element
    for (int i = 0; i < imax; ++i)
    {
        insert_in_maxs(maxs, vals_[i], i_high + i + 1);
    }
}

/**
 * hash_templated. This is the hash function for the 
 * FastHadamardLSH code that we're operating on. 
 * @param v - a pointer to the vector which is hashed
 * @param res - a pointer to receive the multi_hash many (coefficient, hashes) pairs
    (pairs are written next to each other in this array of size 2*multi_hash)
 */
template<int regs_>
void FastHadamardLSH::hash_templated(const int16_t * const vv, int32_t * const res)
{
 
    // Firstly we set up our matrix - we have regs_ many values to work on,
    // so we create a matrix with regs_ * 16 many 16-bit integer entries.
  Simd::VecType v[regs_] = {0};
    // We grab the appropriate tailmask & set up the randomness for the permutations
  auto tailmask = Simd::tailmasks[n % 16];
  auto prg_state = full_seed;

    // Copy over each run of 16-bit integers from vv
    for (int i = 0; i < regs_; ++i)
    {
      v[i] = Simd::m256_loadu_si256(&vv[16 * i]);
    }
    

    // If we have more than 2 registers to operate on then we apply the Hadamard transform on 32 entries at a given time
    // However, if we've got only 2 registers to operate on then we need to use different permutations 
    // and we only apply the Hadamard transform on 16 entries at a given time.
    if(regs_ > 2) {
    	//h0 and h1 contains the Hadamard transforms of v[0] and v[1] on each iteration respectively.
      Simd::VecType h0, h1;
    	// Permute the 16-bit integers in the matrix
      Simd::m256_permute_epi16<regs_>(v, prg_state, reinterpret_cast<Simd::VecType>(tailmask), aes_key, &extra_state);
    
    	// Now we apply the Hadamard transformations and permutations, inserting the scores
    	// in to the res array (note that these are double-packed)
    	for(uint64_t i_high = 0; i_high < codesize; i_high += 32) 
    	{
	  Simd::m256_hadamard32_epi16(v[0],v[1], h0, h1);
	  Simd::m256_permute_epi16<regs_>(v, prg_state, reinterpret_cast<Simd::VecType>(tailmask), aes_key, &extra_state);
          insert_in_maxs_epi16(res, i_high, h0);
          insert_in_maxs_epi16(res, i_high+16, h1);     
    	}
   } else {
	// h0 contains the result of applying the Hadamard on v[0]
        Simd::VecType h0;
	// Permute and apply the Hadamard transformations as above.
	Simd::m256_permute_epi16<regs_>(v, prg_state, reinterpret_cast<Simd::VecType>(tailmask), aes_key, &extra_state);

    	for(uint64_t i_high = 0; i_high < codesize; i_high += 16)
    	{
	  Simd::m256_hadamard16_epi16(v[0], h0);
	  Simd::m256_permute_epi16<regs_>(v, prg_state, reinterpret_cast<Simd::VecType>(tailmask), aes_key, &extra_state);
          insert_in_maxs_epi16(res, i_high, h0);
    	}
  }
}

/**
 * Hash. This hashes a given input vector v against the subcode.
 * Note that this function is separate from the different "hash_templated" functions - that's
 * because we apply some normalisation and safety checks first.
 * @param v - a pointer to the vector which is hashed
 * @param coeff - a pointer to receive the multi_hash many coefficient of the selected hash values
 * @param hashes - a pointer to receive the multi_hash many hash values
 */
void FastHadamardLSH::hash(const float * const v, float * const coeff, int32_t * const hashes)
{
    // vv. This contains the normalised values of v that are hashed against.
    int16_t vv[regs*16];

    // res. This contains the results of the hashing.
    // Note that we double pack this array:
    // 		The even positions contain the coefficients
    // 		The odd positions contain the hashes
    int32_t res[2*multi_hash];
    for (unsigned i = 0; i < multi_hash; ++i)
    {
        res[2*i] = 0;
        res[2*i + 1] = 1;
    }

    // Compute the normalisation factor

    double norm = 0.0;
    for (size_t i = 0; i < n; ++i) 
    {
        norm += std::fabs(v[i]); 
    }
   

    // This is the 0-vector case - in that case, we reply with the default response
	
    if (!norm)
    {
        // This is the 0-vector, answer the first possible hashes
        for (unsigned i = 0; i < multi_hash; ++i)
        {
            coeff[i] = 1.;
            hashes[i] = i+1 > codesize ? codesize : i+1;
        }
        return;
    }

    // This is 1023.0 for the sake of overflow - we're dealing with 16 bit signed
    // ints & we want to prevent the Hadamard sums over up to 32 of them to
    // overflowing these values.
    // Could be more agressive or clever to get more precision (like computing
    // the largest sum over 32 values by sorting them), but this may be enough already

    float renorm = 1023.0 / norm;
    for (size_t i = 0; i < n; ++i) 
    {
        vv[i] = v[i] * renorm; 
    }

    // Now depending on how many registers we have available, we 
    // choose the right templated hash function.
    switch(regs) {
        case 2: {FastHadamardLSH::hash_templated<2>(vv, res); break;}
        case 3: {FastHadamardLSH::hash_templated<3>(vv, res); break;}
        case 4: {FastHadamardLSH::hash_templated<4>(vv, res); break;}
        case 5: {FastHadamardLSH::hash_templated<5>(vv, res); break;}
        case 6: {FastHadamardLSH::hash_templated<6>(vv, res); break;}
        case 7: {FastHadamardLSH::hash_templated<7>(vv, res); break;}
        case 8: {FastHadamardLSH::hash_templated<8>(vv, res); break;}
	default:
        // Note that as we're dealing with 16 bit integers, then 
        // this error is equivalent to saying that 2 < regs <= 15
        throw std::invalid_argument("lsh dimension invalid (must be 16 < n <=128");
    }

    // With this finished, we have to renormalise the coefficients of the results array
    // We place the results of this into the output parameters
    // Note that we double pack the res array so that we can split out nicely
    renorm = 1./renorm;
    for (unsigned i = 0; i < multi_hash; ++i)
    {
        coeff[i] = res[2*i] * renorm;
        hashes[i] = res[2*i + 1];
    }
}

/*
 * insert_in_maxs. Given a score, insert that score into the right position in scores, whilst also 
 * inserting the index into the right position in the indices array. */
inline bool ProductLSH::insert_in_maxs(float * const scores, int32_t * const indices, const float score, const int32_t index)
{
	
    // multi_hash -1 to prevent off-by-ones
    int i = multi_hash-1;
    // If our score isn't bigger than the old smallest then fail
    if (score < scores[i]) return false;

    // If only hashing against 1 bucket then just copy over the values and exit
    if (!i) 
    {
        scores[0] = score; 
        indices[0] = index; 
        return true; 
    }

    // Realign ourselves to make sure we start in the right place
    i--;
    // Insert in the right place - iterate until we find the right insertion point
    while((i >= 0) && (score > scores[i]))
    {
        scores[i+1] = scores[i];
        indices[i+1] = indices[i];
        i--;
    }

    // Insert into scores and indices, returning 
    scores[i+1]  = score;
    indices[i+1]  = index;
    return true;
}


/*
 * hash_templated. This hashes the vector vv against the three blocks. 
 */
template<> void ProductLSH::hash_templated<3>(const float * const vv, int32_t * const res)
{

    //These arrays are temporaries that we use while inserting the hashes to res
    int32_t h0[multi_hash_block], h1[multi_hash_block], h2[multi_hash_block];
    float c0[multi_hash_block], c1[multi_hash_block], c2[multi_hash_block];

    // This memset is inserted to prevent some compilers from complaining about initialising
    // variable-length arrays. This is because variable-length arrays are a compiler extension and
    // not endorsed by the C++ standard.
    // Note that this call to memset actually appears to generate better code on more recent versions of gcc.
    // See https://godbolt.org/z/vGh9c69br for an example
    float c[multi_hash];
    memset(&c, 0, sizeof(float)*multi_hash);
    
    // Hash against the subcodes
    lshs[0].hash(&(vv[0]), c0, h0);
    lshs[1].hash(&(vv[is[1]]), c1, h1);
    lshs[2].hash(&(vv[is[2]]), c2, h2);

    for (unsigned i0 = 0; i0 < multi_hash_block; ++i0)
    {
	// Grab the current coefficient and adjust 
        float c0_ = c0[i0];
        int32_t h0_ = std::abs(h0[i0]) - 1;
        int sign0 = h0[i0] > 0 ? 1 : -1;
        unsigned i1 = 0;


	// Now we transform the inputs for the hash results of lshs[1]
        for (; i1 < multi_hash_block; ++i1)
        {
            int32_t h1_ = sign0 * h1[i1];
            h1_ = 2*(std::abs(h1_) - 1) + (h1_ > 0);
            h1_ = 2*codesizes[1]*h0_ + h1_;
            float c1_ = c0_ + c1[i1];

            unsigned i2 = 0;
            for (; i2 < multi_hash_block; ++i2)
            {
		// apply the same adjustments as above, but this time on h2[i2]
                int32_t h2_ = sign0 * h2[i2];
                h2_ = 2*(std::abs(h2_) - 1) + (h2_ > 0);
                h2_ = 2*codesizes[2]*h1_ + h2_;
                float c2_ = c1_ + c2[i2];
		// if the insertion fails then exist
                if (!insert_in_maxs(c, res, c2_, sign0 * (h2_+1))) break;
            }
	    // if the inner loop failed on i2 = 0 then break - we move on to the next coefficient
            if (!i2) break;
        }
	//if we broke on i1 = 0 from the loop then we're done - nothing is 
	//gained from list decoding further.
        if (!i1) break;
    }
}

/*
 * hash_templated. This function is a specialisation of hashing vv against 2 sub-code blocks.
 */
template<> void ProductLSH::hash_templated<2>(const float * const vv, int32_t * const res)
{
    int32_t h0[multi_hash_block], h1[multi_hash_block];
    float c0[multi_hash_block], c1[multi_hash_block];
    float c[multi_hash];
    memset(&c, 0, sizeof(float) * multi_hash);
    
    // Now hash against the two subcode blocks.
    lshs[0].hash(&(vv[0]), c0, h0);
    lshs[1].hash(&(vv[is[1]]), c1, h1);

    // Now we need to process the results of hashing.
    for (unsigned i0 = 0; i0 < multi_hash_block; ++i0)
    {
	// Grab the current coefficient 
	// and take the magnitude of the current hash
        float c0_ = c0[i0];
        int32_t h0_ = std::abs(h0[i0]) - 1;
        int sign0 = h0[i0] > 0 ? 1 : -1;
	
	// For each of the multi-hash-block many elements, 
	// we try and insert the hash into the maximal list.
        unsigned i1 = 0;
	for (; i1 < multi_hash_block; ++i1)
        {
	    // Transform the inputs - apply the sign of the first hash against the second
	    // Then do the same as above, but offset the coefficient by c1[i1].
            int32_t h1_ = sign0 * h1[i1];
            h1_ = 2*(std::abs(h1_) - 1) + (h1_ > 0);
            h1_ = 2*codesizes[1]*h0_ + h1_;
            float c1_ = c0_ + c1[i1];
            // Insert the value into maxs - breaking if it fails.
            if (!insert_in_maxs(c, res, c1_, sign0 * (h1_+1))) break;
        }

	// If we fail to insert on the first iteration then break from the outer loop - we're done
        if (!i1) break;
    }
}

/*
 * hash_templated. This is the product hash for vv against a single block.
 * The result of this hash is stored in res.
 */

template<> void ProductLSH::hash_templated<1>(const float * const vv, int32_t * const res)
{
    //  Hash against the single block, copy the results and return
    //  We memcpy over the results into the res array.
    int32_t h0[multi_hash_block];
    float c0[multi_hash_block];

    lshs[0].hash(&(vv[0]), c0, h0);    
    memcpy(res, h0, multi_hash_block * sizeof(int32_t));
}

/**
 * hash. Accepts a vector, v, and hashes it against all of the relevant subcodes.
 * @param v - a pointer to the vector which is hashed
 * @param res - a pointer to receive the multi_hash many hash values
 */

void ProductLSH::hash(const float * const v, int32_t * const res)
{
    // Firstly we apply the permutation - we apply the same permutation to every input vector.
    // The sign and permutation vectors are built in the constructor for this class.
    float vv[n];
    for (size_t i = 0; i < n; ++i) 
    {
        vv[i] = sign[i] * v[permutation[i]];
    }

    for (size_t i = 0; i < multi_hash; ++i)
    {
        res[i] = 1;
    }

    // With all of the permutations done, we apply the multi_block hash. 
    // Note that right now we've only got support for between 1-3 blocks.
    if      (blocks==1)  {hash_templated<1>(vv, res); }
    else if (blocks==2)  {hash_templated<2>(vv, res); }
    else if (blocks==3)  {hash_templated<3>(vv, res); }
    else {throw std::invalid_argument( "not implemented" );}
    for (size_t i = 0; i < multi_hash; ++i)
    {
        res[i] = abs(res[i]) - 1;
    }

}
