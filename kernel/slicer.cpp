#include "siever.h"
#include "slicer.h"
#include "fht_lsh.h"


void RandomizedSlicer::randomize_target_small_task(Entry_t &t)
{
    size_t fullS = this->sieve.cdb.size();
    size_t max_trial = 2 + std::pow(fullS, .3);
    CompressedEntry* fast_cdb = &(this->sieve.cdb.front());
    //Entry* e_db = &(this->sieve.db.front());
    size_t partial = 5 * std::pow(fullS, .6);

    //std::cout << "fullS " << fullS << std::endl;
    //std::cout << "max_trial " << max_trial << " partial " << partial << std::endl;

    Entry_t tmp;

    size_t i = (rng_t() % partial);
    size_t j = (rng_t() % partial);

    //std::cout << "n = " << this->sieve.n << std:: endl;
    //std::cout << "cdb.size = " << this->sieve.cdb.size() << std:: endl;

    //std::cout << i << " " << j  << std::endl;

    for (unsigned int trial=0;;++trial) {
        ++j;
        unsigned int w1 = 0;
        //unsigned int w2 = 0;


        for (size_t k = 0; k < XPC_WORD_LEN; ++k) {
            w1 += __builtin_popcountl(fast_cdb[i].c[k] ^ fast_cdb[j].c[k]);
        }
        //std::cout << "w1: " << w1 << std::endl;
        if (w1 < XPC_SLICER_SAMPLING_THRESHOLD || w1 > (XPC_BIT_LEN - XPC_SLICER_SAMPLING_THRESHOLD) || trial > max_trial) {
            if (i == j) continue;
            ZT sign = w1 < XPC_SLICER_SAMPLING_THRESHOLD / 2 ? -1 : 1;

            //std::cout << "good w1: " << w1 << std::endl;

            std::array<LFT,MAX_SIEVING_DIM> new_yr = this->sieve.db[this->sieve.cdb[i].i].yr;

            //for( size_t ii = 0; ii < n; ii++ ) {
            //    new_yr[ii] += sign*this->sieve.db[this->sieve.cdb[j].i].yr[ii];
            //}

            //std::cout << "new_yr addition" << std::endl;

            this->sieve.addsub_vec(new_yr, this->sieve.db[this->sieve.cdb[j].i].yr, static_cast<ZT>(sign));

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
void RandomizedSlicer::grow_db_with_target(const double t_yr[], size_t n_per_target){

    Entry_t input_t;

    for(int i = 0; i < MAX_SIEVING_DIM; i++){
        input_t.yr[i] = t_yr[i];
    }

    this->sieve.recompute_data_for_entry_t<Siever::Recompute::recompute_all>(input_t);


    //std::cout << "length: " << input_t.len << " uid: " <<input_t.uid << std::endl;

    unsigned long const start = db_t.size();
    unsigned long const N = start+n_per_target;

    db_t.reserve(N);
    cdb_t.reserve(N);
    db_t.resize(N);
    cdb_t.resize(N);

    std::cout << "input_t.uid:" << input_t.uid << std::endl;


    if(!uid_hash_table_t.insert_uid(input_t.uid)){
        for (size_t ii=0; ii<n; ii++){ std::cout << input_t.yr[ii] << " ";}
        std::cerr << "The original target is already in db" << std::endl;
        exit(1);
    }
    db_t[start] = input_t;
    CompressedEntry ce;
    ce.len = input_t.len;
    ce.c = input_t.c;
    ce.i = start;
    cdb_t[start] = ce;


    for( size_t i = start+1; i < N; i++)
    {

        //std::cout << " " << input_t.len << std::endl;
        int col = 0;

        Entry_t e = input_t;

        for ( col = 0; col < 10; ++col)
        {
            Entry_t tmp = e;
            randomize_target_small_task(tmp);
            //std::cout << col << " tmp.uid:" << tmp.uid << std::endl;

            if(!uid_hash_table_t.insert_uid(tmp.uid)) {
                //for (size_t ii=0; ii<n; ii++){ std::cout << tmp.yr[ii] << " ";}
                continue;
            }
            db_t[i] = tmp;

            CompressedEntry ce;
            ce.len = tmp.len;
            ce.c = tmp.c;
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

    /*
    for( size_t i = start; i < N; i++)
    {
        std::cout << i << ": " << db_t[i].len << " " << db_t[i].uid  << std::endl;
    }
    */
}

// assumes buckets and buckets_index are resized and resetted correctly.
void RandomizedSlicer::slicer_bucketing(const size_t blocks, const size_t multi_hash, const size_t nr_buckets_aim,
                            std::vector<uint32_t> &buckets, std::vector<atomic_size_t_wrapper> &buckets_index)
{
    // init hash
    std::cout << "lsh_seed from sieve: " << this->sieve.lsh_seed << std::endl;
    ProductLSH lsh(n, blocks, nr_buckets_aim, multi_hash, this->sieve.lsh_seed);
    exit(1);
    const size_t nr_buckets = lsh.codesize;
    const size_t S = cdb_t.size();
    size_t bsize = 2 * (S*multi_hash / double(nr_buckets));
    buckets.resize( nr_buckets * bsize );
    buckets_index.resize(nr_buckets);
    for( size_t i = 0; i < nr_buckets; i++ )
        buckets_index[i].val = 0;

    for (size_t t_id = 0; t_id < threads; ++t_id)
    {
        //threadpool.push([this, t_id, multi_hash, &buckets, &buckets_index, &lsh](){
        //    slicer_bucketing_task(t_id, buckets, buckets_index, lsh);
        //});
    }
    threadpool.wait_work();

    for( size_t i = 0; i < nr_buckets; ++i ) {
        // bucket overflow
        if( buckets_index[i].val > bsize ) {
            buckets_index[i].val = bsize;
        }
    }
}

/*
void RandomizedSlicer::slicer_bucketing_task(const size_t t_id,
                                       const std::vector<uint32_t> &buckets,
                                       const std::vector<atomic_size_t_wrapper> &buckets_index, std::vector<QEntry> &t_queue) {

    const size_t nr_buckets = buckets_index.size();
    const size_t bsize = buckets.size() / nr_buckets;

    const uint32_t *const fast_buckets = buckets.data();
    CompressedEntry *const fast_cdb = cdb_t.data();

    const size_t S = cdb_t.size();

    // todo: start insert earlier
    int64_t kk = S - 1 - t_id;

    //TODO:
    LFT lenbound = fast_cdb[std::min(S - 1, size_t(params.bdgl_improvement_db_ratio * S))].len;
    const size_t b_start = t_id;

    size_t B = 0;
    for (size_t b = b_start; b < nr_buckets; b += threads) {
        const size_t i_start = bsize * b;
        const size_t i_end = bsize * b + buckets_index[b].val;

        B += ((i_end - i_start) * (i_end - i_start - 1)) / 2;
        for (size_t i = i_start; i < i_end; ++i) {
            if (kk < .1 * S) break;

            uint32_t bi = fast_buckets[i];
            CompressedEntry *pce1 = &fast_cdb[bi];
            CompressedVector cv = pce1->c;
            for (size_t j = i_start; j < i; ++j) {
                uint32_t bj = fast_buckets[j];
                if (is_reducible_maybe<XPC_THRESHOLD>(cv, fast_cdb[bj].c)) {
                    std::pair<LFT, int> len_and_sign = reduce_to_QEntry(pce1, &fast_cdb[bj]);
                    if (len_and_sign.first < lenbound) {
                        if (kk < .1 * S) break;
                        kk -= threads;

                        t_queue.push_back({pce1->i, fast_cdb[bj].i, len_and_sign.first, (int8_t) len_and_sign.second});
                    }
                }
            }
        }
    }
    //statistics.inc_stats_xorpopcnt_inner(B);
    std::sort(t_queue.begin(), t_queue.end(), &compare_QEntry);
}
*/

bool RandomizedSlicer::bdgl_like_sieve(size_t nr_buckets_aim, const size_t blocks, const size_t multi_hash ){

    std::vector<std::vector<Entry_t>> transaction_db(threads, std::vector<Entry_t>());
    std::vector<uint32_t> buckets;
    std::vector<atomic_size_t_wrapper> buckets_i;
    std::vector<std::vector<QEntry>> t_queues(threads);

    size_t it = 0;
    while( true ) {

        slicer_bucketing(blocks, multi_hash, nr_buckets_aim, buckets, buckets_i);

        if( it > 10000 ) {
            std::cerr << "Couldn't find a close vector after 10000 iterations" << std::endl;
            return false;
        }
        it++;
    }
}
