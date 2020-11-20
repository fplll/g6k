// https://github.com/lducas/AVX2-BDGL-bucketer commit 62fb8bc2d882a39b9d83257dd320d904b6cbc407

#include "fht_lsh.h"

/* m256d_hadamard8_ps.
 * This function computes a Hadamard transform on a vector of 8 single precision floats
 */
inline __m256 m256d_hadamard8_ps(const __m256& x1)
{
    __m256 tmp, res = x1;

    // Note that here we use a 64-bit permutation and operate on res
    // as if it's full of ints - no harm. We do this as an explicit reinterpret_cast.
    tmp = (__m256) _mm256_permute4x64_epi64(reinterpret_cast<__m256i>(res), 0b01001110);
    // Negate the first 4 floats of res, add to temp & put back in res
    res = _mm256_fmadd_ps(res, _mm256_set_ps(-1,-1,-1,-1,1,1,1,1), tmp);
    // Then negate the even components and add, repeat and then done
    tmp = _mm256_mul_ps(res, _mm256_set_ps(-1,1,-1,1,-1,1,-1,1));
    res = _mm256_hadd_ps(tmp, res);
    tmp = _mm256_mul_ps(res, _mm256_set_ps(-1,1,-1,1,-1,1,-1,1));
    res = _mm256_hadd_ps(tmp, res);
    return res;
}


/*
 * m256_hadamard16_epi16. This function applies the Hadamard transformation
 * over 16 entries of 16-bit integers stored in a single __m256i vector x1, 
 * storing result in r1.
 */

inline void FastHadamardLSH::m256_hadamard16_epi16(const __m256i &x1, __m256i &r1)
{
   /* Apply a permutation 0123 -> 1032 to 64 bit words. */ 
    __m256i a1 = _mm256_permute4x64_epi64(x1, 0b01001110);
    
    // From here we treat the input vector x1 as 16, 16-bit integers - which is what the entries are
    // Now negate the first 8 16-bit integers of x1
    __m256i t1 = _mm256_sign_epi16(x1, sign_mask_8);

    // Add the permutation to the recently negated portion & apply the second sign mask
    a1 = _mm256_add_epi16(a1, t1);
    __m256i b1 = _mm256_sign_epi16(a1, sign_mask_2);

    // With this, we can now build what we want by repeatedly applying the sign mask and adding
    a1 = _mm256_hadd_epi16(a1, b1);
    b1 = _mm256_sign_epi16(a1, sign_mask_2);
    a1 = _mm256_hadd_epi16(a1, b1);
    b1 = _mm256_sign_epi16(a1, sign_mask_2);
    r1 = _mm256_hadd_epi16(a1, b1);

}

/**
 * m256_hadamard32_epi16. This function applies the Hadamard transformation 
   over 2 __m256i vectors, x1 and x2, storing the results in two distinct vectors r1 & r2.

   Note: appart for the final mixing, this is equivalent to running the above twice,
   but the interleaving makes it faster, mitigating delays.
 */
inline void FastHadamardLSH::m256_hadamard32_epi16(const __m256i &x1,const __m256i &x2, __m256i &r1, __m256i &r2)
{
    // Permute 64-bit chunks of a1 and a2 and then negate the first 8 16-bit integers of each 
    // x1 and x2.
    __m256i a1 = _mm256_permute4x64_epi64(x1, 0b01001110);
    __m256i a2 = _mm256_permute4x64_epi64(x2, 0b01001110);
    __m256i t1 = _mm256_sign_epi16(x1, sign_mask_8);
    __m256i t2 = _mm256_sign_epi16(x2, sign_mask_8);

    // Add the results and negate the second 8 16-bit integers 
    a1 = _mm256_add_epi16(a1, t1);
    a2 = _mm256_add_epi16(a2, t2);
    __m256i b1 = _mm256_sign_epi16(a1, sign_mask_2);
    __m256i b2 = _mm256_sign_epi16(a2, sign_mask_2);

    // Now apply the 16-bit Hadamard transforms and repeat the process
    a1 = _mm256_hadd_epi16(a1, b1);
    a2 = _mm256_hadd_epi16(a2, b2);
    b1 = _mm256_sign_epi16(a1, sign_mask_2);
    b2 = _mm256_sign_epi16(a2, sign_mask_2);
    a1 = _mm256_hadd_epi16(a1, b1);
    a2 = _mm256_hadd_epi16(a2, b2);
    b1 = _mm256_sign_epi16(a1, sign_mask_2);
    b2 = _mm256_sign_epi16(a2, sign_mask_2);
    a1 = _mm256_hadd_epi16(a1, b1);
    a2 = _mm256_hadd_epi16(a2, b2);

    r1 = _mm256_add_epi16(a1, a2);
    r2 = _mm256_sub_epi16(a1, a2);
}


/*
 * m256_mix. Swaps V0[i] and V1[i] iff mask[i] = 1 for 0 <= i < 255.
 */
inline void FastHadamardLSH::m256_mix(__m256i &v0, __m256i &v1, const __m256i &mask)
{
    __m256i diff;
    diff = _mm256_xor_si256(v0, v1);
    diff = _mm256_and_si256(diff, mask);
    v0 = _mm256_xor_si256(v0, diff);
    v1 = _mm256_xor_si256(v1, diff);
}

/*
 * m256_permute_epi16. The goal of this function is to permute the input vector v, 
 * according to the randomness from prg_state & the tailmask.  Note that this function is a specialisation of the 
 * broader m256_permute_epi16.
 * @param v - this is a pointer to the input vector. 
 * @param prgstate - this is the state of the prg for the current hashing round
 * @param tailmask - a mask for handling mixing when the length of v is not a multiple of 16.

 This is specialized for inputs of 2 AVX vectors, ie for v of length 17 to 32.
 */
template<>
inline void FastHadamardLSH::m256_permute_epi16<2>(__m256i * const v, __m128i &prg_state, const __m256i &tailmask)
{
    // double pack the prg state in rnd (has impact of doubly repeating the prg state in rnd)
    // Though we will use different threshold on each part decorrelating the permutation
    // on each halves 

    __m256i rnd = _mm256_broadcastsi128_si256(prg_state);
    __m256i mask;

    // With only 2 registers, we may not have enough room to randomize via m256_mix, 
    // so we also choose at random among a few precomputed permutation to apply on
    // the first register

    uint32_t x = _mm_extract_epi64(prg_state, 0);
    uint32_t x1 = (x  >> 16) & 0x03;
    uint32_t x2 = x & 0x03;

    // Apply the precomputed permutations to the input vector
    v[0] = _mm256_shuffle_epi8(v[0], permutations_epi16[x1]);
    m256_mix(v[0], v[1], tailmask);
    v[0] = _mm256_permute4x64_epi64(v[0], 0b10010011);
    v[0] = _mm256_shuffle_epi8(v[0], permutations_epi16[x2]);

    mask = _mm256_cmpgt_epi16(rnd, mixmask_threshold);
    mask = _mm256_and_si256(mask, tailmask);
    m256_mix(v[0], v[1], mask);

    // update the very fast but non-cryptographic PRG (one tour of AES)
    prg_state = _mm_aesenc_si128(prg_state, aes_key);    
}



/* Same as above, but for inputs of arbitrary lengths.  */

template<int regs_> 
inline void FastHadamardLSH::m256_permute_epi16(__m256i * const v, __m128i &prg_state, const __m256i &tailmask)
{
    // double pack the prg state in rnd (has impact of doubly repeating the prg state in rnd)
    // Though we will use different threshold on each part decorrelating the permutation
    // on each halves 

    __m256i rnd = _mm256_broadcastsi128_si256(prg_state);
    __m256i tmp;

    // We treat the even and the odd positions differently
    // This is for the goal of decorrelating the permutation on the 
    // double packed prng state.
    for (int i = 0; i < (regs_-1)/2; ++i)
    {
        // shuffle 8 bit parts in each 128 bit lane
	// Note - the exact semantics of what this function does are a bit confusing.
	// See the Intel intrinsics guide if you're curious
        v[2*i  ] = _mm256_shuffle_epi8(v[2*i], permutations_epi16[i % 3]);
	// For the odd positions we permute each 64-bit chunk according to the mask.
        v[2*i+1] = _mm256_permute4x64_epi64(v[2*i+1], 0b10010011);
    }


    // Now we negate the first two vectors according to the negation masks
    v[0] = _mm256_sign_epi16(v[0], negation_masks_epi16[0]);
    v[1] = _mm256_sign_epi16(v[1], negation_masks_epi16[1]);


    // swap int16 entries of v[0] and v[1] where rnd > threshold
    tmp = _mm256_cmpgt_epi16(rnd, mixmask_threshold);
    m256_mix(v[0], v[1], tmp);
    // Shift the randomness around before extracting more (sonmewhat independent) mixing bits
    rnd = _mm256_slli_epi16(rnd, 1);

    // Now do random swaps between v[0] and v[last-1]
    m256_mix(v[0], v[regs_- 2], tmp);
    rnd = _mm256_slli_epi16(rnd, 1);

    // Now do swaps between v[1] and v[last], avoiding padding data
    m256_mix(v[1], v[regs_ - 1], tailmask);


    // More permuting
    for (int i = 2; i + 2 < regs_; i+=2)
    {
        rnd = _mm256_slli_epi16(rnd, 1);
        tmp = _mm256_cmpgt_epi16(rnd, mixmask_threshold);
        m256_mix(v[0], v[i], tmp);
        rnd = _mm256_slli_epi16(rnd, 1);
        tmp = _mm256_cmpgt_epi16(rnd, mixmask_threshold);
        m256_mix(v[1], v[i+1], tmp);
    }

    // update the very fast but non-cryptographic PRG (one tour of AES)
    prg_state = _mm_aesenc_si128(prg_state, aes_key);    
}

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
inline void FastHadamardLSH::insert_in_maxs_epi16(int32_t * const maxs, const int i_high, const __m256i &vals)
{	
    // Grab the smallest absolute  value and put it in the threshold vector
    int16_t T = maxs[2*(multi_hash - 1)];
    __m256i threshold = _mm256_set_epi16(T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T);
    // Convert each value to their abs value - then compare to the threshold
    __m256i tmp = _mm256_abs_epi16(vals);
    // N.b tmp is full of 1s and 0s at this point.
    tmp = _mm256_cmpgt_epi16(tmp, threshold);
    // If the bitwise AND is 1 then return (this shouldn't happen!)
    if (_mm256_testz_si256(tmp, tmp)) return;

    // Otherwise, we extract the 16-bit values and put them in an array
    int16_t vals_[16];

    // N.B we use a reinterpret cast to be explicit
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(vals_), vals);

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
 * @param res - a pointer to receive the multihash many (coefficient, hashes) pairs
    (pairs are written next to each other in this array of size 2*multihash)
 */
template<int regs_>
void FastHadamardLSH::hash_templated(const int16_t * const vv, int32_t * const res)
{
 
    // Firstly we set up our matrix - we have regs_ many values to work on,
    // so we create a matrix with regs_ * 16 many 16-bit integer entries.
    __m256i v[regs_] = {0};
    // We grab the appropriate tailmask & set up the randomness for the permutations
    __m256i tailmask = tailmasks[n % 16];
    __m128i prg_state = full_seed;

    // Copy over each run of 16-bit integers from vv
    for (int i = 0; i < regs_; ++i)
    {
        v[i] = _mm256_loadu_si256((__m256i*)&vv[16 * i]);
    }
    

    // Here we use some C++17 magic.
    // If we have more than 2 registers to operate on then we apply the Hadamard transform on 32 entries at a given time
    // However, if we've got only 2 registers to operate on then we need to use different permutations 
    // and we only apply the Hadamard transform on 16 entries at a given time.
    //
    // But we don't want to use a regular if! That's slow.
    // Thankfully, if constexpr is a thing - it will throw away the branch that we don't care about in our template - 
    // thus it lets us write simpler code, but get the benefits we want in the end. 
    if constexpr(regs_ > 2) {
    	//h0 and h1 contains the Hadamard transforms of v[0] and v[1] on each iteration respectively.
    	__m256i h0, h1;
    	// Permute the 16-bit integers in the matrix
    	m256_permute_epi16<regs_>(v, prg_state, tailmask);
    
    	// Now we apply the Hadamard transformations and permutations, inserting the scores
    	// in to the res array (note that these are double-packed)
    	for(uint64_t i_high = 0; i_high < codesize; i_high += 32) 
    	{
        	m256_hadamard32_epi16(v[0],v[1], h0, h1);
        	m256_permute_epi16<regs_>(v, prg_state, tailmask);
        	insert_in_maxs_epi16(res, i_high, h0);
        	insert_in_maxs_epi16(res, i_high+16, h1);     
    	}
   } else {
	// h0 contains the result of applying the Hadamard on v[0]
	__m256i h0;
	// Permute and apply the Hadamard transformations as above.
    	m256_permute_epi16<regs_>(v, prg_state, tailmask);

    	for(uint64_t i_high = 0; i_high < codesize; i_high += 16)
    	{
        	m256_hadamard16_epi16(v[0], h0);
        	m256_permute_epi16<regs_>(v, prg_state, tailmask);
        	insert_in_maxs_epi16(res, i_high, h0);
    	}
  }
}

/**
 * Hash. This hashes a given input vector v against the subcode.
 * Note that this function is separate from the different "hash_templated" functions - that's
 * because we apply some normalisation and safety checks first.
 * @param v - a pointer to the vector which is hashed
 * @param coeff - a pointer to receive the multihash many coefficient of the selected hash values
 * @param hashes - a pointer to receive the multihash many hash values
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
    float c[multi_hash] = {0};

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
    float c[multi_hash] = {0};

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
 * @param res - a pointer to receive the multihash many hash values
 */

void ProductLSH::hash(const float * const v, int32_t * const res)
{
    // Firstly we apply the permutation - we apply the same permutation to every input vector.
    // The sign and permutation vectors are built in the constructor for this class.
    float vv[n], tmp[n];
    for (size_t i = 0; i < n; ++i) 
    {
        vv[i] = sign[i] * v[permutation[i]];
    }


    // Now we perform the pre-Hadamard rounds. 
    // You can think of this as applying another set of permutations to the input.
    // This code consists of fast, floating point Hadamard code. 
    for (int iter = 0; iter < pre_hadamards; ++iter)
    {
	// For all but the last 8 floats, apply the fp Hadamard and store it in the tmp array 
        unsigned i = 0;
        for (; i+7 < n; i+=8)
        {
            __m256 x = _mm256_loadu_ps(&vv[i]);
            x = m256d_hadamard8_ps(x);
            _mm256_storeu_ps(&tmp[i], x);
        }

	
	// Normalise the last 8 floats
        for (; i < n; ++i)
        {
            tmp[i] = vv[i] * 2.82842712474619; // sqrt(8.);
        }

	
	// Now normalize all of Hadamard'd components
        i = 0;
        for (; i < n-8; ++i)
        {
            tmp[i] = tmp[i] * 2.82842712474619; // sqrt(8.);
        }


	// And finally apply the Hadamard transform to the last 8 floats
        __m256 x = _mm256_loadu_ps(&tmp[i]);
        x = m256d_hadamard8_ps(x);
        _mm256_storeu_ps(&tmp[i], x);




	// Now renormalize the components, applying the permutation as we go
        for (size_t i = 0; i < n; ++i) 
        {
            vv[i] = .125 * sign[i] * tmp[permutation[i]];
        }        
    }

    for (int i = 0; i < multi_hash; ++i)
    {
        res[i] = 0;
    }

    // With all of the permutations done, we apply the multi_block hash. 
    // Note that right now we've only got support for between 1-3 blocks.
    if      (blocks==1)  { return hash_templated<1>(vv, res); }
    else if (blocks==2)  { return hash_templated<2>(vv, res); }
    else if (blocks==3)  { return hash_templated<3>(vv, res); }

    throw std::invalid_argument( "not implemented" );
}
