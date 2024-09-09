#include "siever.h"
#include "slicer.h"


void RandomizedSlicer::randomize_target_small_task(Entry_t &t)
{
    size_t fullS = this->sieve.cdb.size();
    size_t max_trial = 2 + std::pow(fullS, .3);
    CompressedEntry* fast_cdb = &(this->sieve.cdb.front());
    size_t partial = 5 * std::pow(fullS, .6);

    //std::array<LFT,MAX_SIEVING_DIM> tmp;

    Entry_t tmp;

    size_t i = (rng_t() % partial);
    size_t j = (rng_t() % partial);

    for (unsigned int trial=0;;++trial) {
        ++j;
        unsigned int w1 = 0;
        //unsigned int w2 = 0;
        for (size_t k = 0; k < XPC_WORD_LEN; ++k) {
            w1 += __builtin_popcountl(fast_cdb[i].c[k] ^ fast_cdb[j].c[k]);
        }
        if (w1 < XPC_SLICER_SAMPLING_THRESHOLD || w1 > (XPC_BIT_LEN - XPC_SLICER_SAMPLING_THRESHOLD) || trial > max_trial) {
            if (i == j) continue;
            ZT sign = w1 < XPC_SLICER_SAMPLING_THRESHOLD / 2 ? -1 : 1;

            std::array<LFT,MAX_SIEVING_DIM> new_yr = db_t[cdb_t[i].i].yr;
            for( size_t ii = 0; ii < n; ii++ ) {
                new_yr[ii] += sign*db_t[cdb_t[j].i].yr[ii];
            }
            //std::cout << "w1: " << w1 << std::endl;
            //addsub_vec(new_yr, db[cdb[j].i].x, static_cast<ZT>(sign));

            //TODO: change to XPC (if makes sense)
            LFT const inner = std::inner_product(t.yr.begin(), t.yr.begin() + n, new_yr.begin(), static_cast<LFT>(0.));
            sign = inner < 0 ? 1 : -1;
            this->sieve.addsub_vec(t.yr, new_yr, static_cast<ZT>(sign));

            this->sieve.recompute_data_for_entry_t<Siever::Recompute::recompute_all>(t);
            break;
        }
    }

}


//void RandomizedSlicer::grow_db_with_target(std::array<LFT,MAX_SIEVING_DIM> &t_yr, size_t n_per_target){
void RandomizedSlicer::grow_db_with_target(float* t_yr, size_t n_per_target){

    Entry_t input_t;

//input_t.yr = t_yr;

    std::move(t_yr, t_yr+MAX_SIEVING_DIM, input_t.yr.begin());


    this->sieve.recompute_data_for_entry_t<Siever::Recompute::recompute_all>(input_t);

    unsigned long const start = db_t.size();
    unsigned long const N = start+n_per_target;

    db_t.reserve(N);
    cdb_t.reserve(N);
    db_t.resize(N);
    cdb_t.resize(N);

    for( size_t i = start; i < N; i++)
    {
        randomize_target_small_task(input_t);

        int col = 0;

        for ( col = 0; col < 64; ++col)
        {
            if(!uid_hash_table_t.insert_uid(input_t.uid)) continue;
            db_t[i].yr = input_t.yr;

            CompressedEntry ce;
            ce.len = input_t.len;
            ce.c = input_t.c;
            ce.i = i;
            cdb_t[i] = ce;
            break;

        }
        if(col>=64)
        {
            std::cerr << "Error : All new randomizations collide." << std::endl;
            exit(1);
        }
    }
}
