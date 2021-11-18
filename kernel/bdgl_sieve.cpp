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


#include "siever.h"
#include "fht_lsh.h"
#include <immintrin.h>
#include <stdio.h>
#include <assert.h>
#include <cstring>
#include <algorithm>
#include <vector>
#include <ctime>
#include <future>
#include <chrono> // needed for wall time>
#include <iostream>
#include <iomanip>
#include <numeric>

struct QEntry {
    size_t i,j;
    float len;
    int8_t sign;
};

struct atomic_size_t_wrapper
{
    atomic_size_t_wrapper(): val(0) {}
    atomic_size_t_wrapper(const size_t& v): val(v) {}
    atomic_size_t_wrapper(const atomic_size_t_wrapper& v): val(size_t(v.val)) {}
    std::atomic_size_t val;
    CACHELINE_PAD(pad);
};

inline bool compare_QEntry(QEntry const& lhs, QEntry const& rhs) { return lhs.len > rhs.len; }

std::pair<LFT, int8_t> Siever::reduce_to_QEntry(CompressedEntry *ce1, CompressedEntry *ce2)
{
    LFT inner = std::inner_product(db[ce1->i].yr.begin(), db[ce1->i].yr.begin()+n, db[ce2->i].yr.begin(),  static_cast<LFT>(0.));
    LFT new_l = ce1->len + ce2->len - 2 * std::abs(inner);
    int8_t sign = (inner < 0 ) ? 1 : -1;
    return { new_l, sign };
}


inline int Siever::bdgl_reduce_with_delayed_replace(const size_t i1, const size_t i2, LFT const lenbound, std::vector<Entry>& transaction_db, int64_t& write_index, LFT new_l, int8_t sign)
{
    if (new_l < lenbound)
    {
        UidType new_uid = db[i1].uid;
        if(sign==1)
        {
            new_uid += db[i2].uid;
        }
        else
        {
            new_uid -= db[i2].uid;
        }
        if (uid_hash_table.insert_uid(new_uid))
        {
            std::array<ZT,MAX_SIEVING_DIM> new_x = db[i1].x;
            addsub_vec(new_x,  db[i2].x, static_cast<ZT>(sign));
            int64_t index = write_index--; // atomic and signed!
            if( index >= 0 ) {
                Entry& new_entry = transaction_db[index];
                new_entry.x = new_x;
                recompute_data_for_entry<Recompute::recompute_all_and_consider_otf_lift>(new_entry);
                return 1;
            }
            return -2; // transaction_db full
        }
        else
        {
            // duplicate
            return 0;
        }
    } 
    else if (params.otf_lift && (new_l < params.lift_radius))
    {
        bdgl_lift(i1, i2, new_l, sign);
    }
    return -1;
}

// assumed that sign is known
inline void Siever::bdgl_lift(const size_t i1, const size_t i2, LFT new_l, int8_t sign)
{   
    ZT x[r];
    LFT otf_helper[OTF_LIFT_HELPER_DIM];
    std::fill(x, x+l, 0);
    std::copy(db[i1].x.cbegin(), db[i1].x.cbegin()+n, x+l);
    std::copy(db[i1].otf_helper.cbegin(), db[i1].otf_helper.cbegin()+OTF_LIFT_HELPER_DIM, otf_helper);

    if(sign == 1)
    {
        for(unsigned int i=0; i < n; ++i)
        {
            x[l+i] += db[i2].x[i];
        }
        for(unsigned int i=0; i < OTF_LIFT_HELPER_DIM; ++i)
        {
            otf_helper[i] += db[i2].otf_helper[i];
        }
    }
    else
    {
        for(unsigned int i=0; i < n; ++i)
        {
            x[l+i] -= db[i2].x[i];
        }
        for(unsigned int i=0; i < OTF_LIFT_HELPER_DIM; ++i)
        {
            otf_helper[i] -= db[i2].otf_helper[i];
        }
    }
    lift_and_compare(x, new_l * gh, otf_helper);
}


// Replace the db and cdb entry pointed at by cdb[cdb_index] by e, unless
// length is actually worse
bool Siever::bdgl_replace_in_db(size_t cdb_index, Entry &e)
{
    CompressedEntry &ce = cdb[cdb_index]; // abbreviation

    if (REDUCE_LEN_MARGIN_HALF * e.len >= ce.len)
    {
        uid_hash_table.erase_uid(e.uid);
        return false;
    }
    uid_hash_table.erase_uid(db[ce.i].uid);
    ce.len = e.len;
    ce.c = e.c;
    db[ce.i] = e;
    return true;
}

void Siever::bdgl_bucketing_task(const size_t t_id, std::vector<uint32_t> &buckets, std::vector<atomic_size_t_wrapper> &buckets_index, ProductLSH &lsh)
{
    CompressedEntry* const fast_cdb = cdb.data();
    const size_t S = cdb.size();
    const size_t multi_hash = lsh.multi_hash;
    const unsigned int nr_buckets = buckets_index.size();
    const size_t bsize = buckets.size() / nr_buckets;
    const size_t threads = params.threads;

    uint32_t i_start = t_id;
    int32_t res[multi_hash];
    size_t bucket_index = 0;
    for (uint32_t i = i_start; i < S; i += threads)
    {  
        auto db_index = fast_cdb[i].i;
        lsh.hash( db[db_index].yr.data() , res);
        for( size_t j = 0; j < multi_hash; j++ ) {
            uint32_t b = res[j];
            assert( b < nr_buckets );
            bucket_index = buckets_index[b].val++; // atomic
            if( bucket_index < bsize ) {
                buckets[bsize * b + bucket_index] = i;
            }
        }
    }
}

// assumes buckets and buckets_index are resized and resetted correctly.
void Siever::bdgl_bucketing(const size_t blocks, const size_t multi_hash, const size_t nr_buckets_aim, 
    std::vector<uint32_t> &buckets, std::vector<atomic_size_t_wrapper> &buckets_index)
{
    // init hash
    const int64_t lsh_seed = rng();
    ProductLSH lsh(n, blocks, nr_buckets_aim, multi_hash, lsh_seed);
    const size_t nr_buckets = lsh.codesize;
    const size_t S = cdb.size();
    size_t bsize = 2 * (S*multi_hash / double(nr_buckets));
    buckets.resize( nr_buckets * bsize );
    buckets_index.resize(nr_buckets);
    for( size_t i = 0; i < nr_buckets; i++ )
        buckets_index[i].val = 0;

    for (size_t t_id = 0; t_id < params.threads; ++t_id)
    {
        threadpool.push([this, t_id, multi_hash, &buckets, &buckets_index, &lsh](){
            bdgl_bucketing_task(t_id, buckets, buckets_index, lsh);
        });
    }
    threadpool.wait_work(); 
    
    for( size_t i = 0; i < nr_buckets; ++i ) {
        // bucket overflow
        if( buckets_index[i].val > bsize ) {
            buckets_index[i].val = bsize;
        }
    }
}

void Siever::bdgl_process_buckets_task(const size_t t_id, 
    const std::vector<uint32_t> &buckets, 
    const std::vector<atomic_size_t_wrapper> &buckets_index, std::vector<QEntry> &t_queue)
{

    const size_t nr_buckets = buckets_index.size();
    const size_t bsize = buckets.size() / nr_buckets;

    const uint32_t* const fast_buckets = buckets.data();
    CompressedEntry* const fast_cdb = cdb.data();
    
    const size_t S = cdb.size();
    
    // todo: start insert earlier
    int64_t kk = S-1-t_id;
    
    LFT lenbound = fast_cdb[std::min(S-1, size_t(params.bdgl_improvement_db_ratio * S))].len;
    const size_t b_start = t_id;

    size_t B = 0;
    for (size_t b = b_start; b < nr_buckets; b += params.threads)
    {
        const size_t i_start = bsize * b;
        const size_t i_end = bsize * b + buckets_index[b].val;
        B +=( (i_end - i_start) * (i_end-i_start-1)) / 2;
        for( size_t i = i_start; i < i_end; ++i ) 
        {
            if (kk < .1 * S) break;

            uint32_t bi = fast_buckets[i];
            CompressedEntry *pce1 = &fast_cdb[bi];
            CompressedVector cv = pce1->c;
            for (size_t j = i_start; j < i; ++j)
            {
                uint32_t bj = fast_buckets[j];
                if( is_reducible_maybe<XPC_THRESHOLD>(cv, fast_cdb[bj].c) )
                {
                    std::pair<LFT, int> len_and_sign = reduce_to_QEntry( pce1, &fast_cdb[bj] );
                    if( len_and_sign.first < lenbound)
                    {
                        if (kk < .1 * S) break;
                        kk -= params.threads;
                        
                        statistics.inc_stats_2redsuccess_outer();

                        t_queue.push_back({ pce1->i, fast_cdb[bj].i, len_and_sign.first, (int8_t)len_and_sign.second});
                    } else if( params.otf_lift and len_and_sign.first < params.lift_radius ) {
                        bdgl_lift( pce1->i, fast_cdb[bj].i, len_and_sign.first, len_and_sign.second );
                    }
                }
            }
        }
    }
    statistics.inc_stats_xorpopcnt_inner(B);
    std::sort( t_queue.begin(), t_queue.end(), &compare_QEntry);
}

// Returned queue is sorted
void Siever::bdgl_process_buckets(const std::vector<uint32_t> &buckets, const std::vector<atomic_size_t_wrapper> &buckets_index, 
    std::vector<std::vector<QEntry>> &t_queues)
{
    for (size_t t_id = 0; t_id < params.threads; ++t_id)
    {
        threadpool.push([this, t_id, &buckets, &buckets_index, &t_queues](){bdgl_process_buckets_task(t_id, buckets, buckets_index, t_queues[t_id]);});
    }
    threadpool.wait_work();
}

void Siever::bdgl_queue_dup_remove_task( std::vector<QEntry> &queue) {
    const size_t Q = queue.size();
    for( size_t index = 0; index < Q; index++ ) {
        size_t i1 = queue[index].i;
        size_t i2 = queue[index].j;
        UidType new_uid = db[i1].uid;
        if(queue[index].sign==1)
        {
            new_uid += db[i2].uid;
        }
        else
        {
            new_uid -= db[i2].uid;
        }
        // if already present, use sign as duplicate marker
        if (uid_hash_table.check_uid_unsafe(new_uid) )
            queue[index].sign = 0;
    }
}

void Siever::bdgl_queue_create_task( const size_t t_id, const std::vector<QEntry> &queue, std::vector<Entry> &transaction_db, int64_t &write_index) {
    const size_t S = cdb.size();
    const size_t Q = queue.size();

    const size_t insert_after = S-1-t_id-params.threads*write_index; 
    for(unsigned int index = 0; index < Q; index++ )  {
        // use sign as skip marker
        if( queue[index].sign == 0 ){
            continue;
        }
        bdgl_reduce_with_delayed_replace( queue[index].i, queue[index].j, 
                                                  cdb[std::min(S-1, insert_after+params.threads*write_index)].len / REDUCE_LEN_MARGIN,
                                                  transaction_db, write_index, queue[index].len, queue[index].sign);
        if( write_index < 0 ){
            std::cerr << "Spilling full transaction db" << t_id << " " << Q-index << std::endl;
            break;
        }
    }
}

size_t Siever::bdgl_queue_insert_task( const size_t t_id, std::vector<Entry> &transaction_db, int64_t write_index) {
    const size_t S = cdb.size();
    const size_t insert_after = std::max(int(0), int(int(S)-1-t_id-params.threads*(transaction_db.size()-write_index))); 
    size_t kk = S-1 - t_id;
    for( int i = transaction_db.size()-1; i > write_index and kk >= insert_after; --i )  { 
            if( bdgl_replace_in_db( kk, transaction_db[i] ) ) {
                kk -= params.threads;
            }
    }
    return kk + params.threads;
}

void Siever::bdgl_queue(std::vector<std::vector<QEntry>> &t_queues, std::vector<std::vector<Entry>>& transaction_db ) {
    // clear duplicates read only
    for( size_t t_id = 0; t_id < params.threads; ++t_id ) {
        threadpool.push([this, t_id, &t_queues](){
            bdgl_queue_dup_remove_task(t_queues[t_id]);
        });
    }
    threadpool.wait_work();

    const size_t S = cdb.size(); 
    size_t Q = 0;
    for(unsigned int i = 0; i < params.threads; i++)
        Q += t_queues[i].size();
    size_t insert_after = std::max(0, int(int(S)-Q));
    
    for(unsigned int i = 0; i < params.threads; i++ )
        transaction_db[i].resize(std::min(S-insert_after, Q)/params.threads + 1);

    std::vector<int> write_indices(params.threads, transaction_db[0].size()-1);
    // Prepare transaction DB from queue
    for( size_t t_id = 0; t_id < params.threads; ++t_id ) {
        threadpool.push([this, t_id, &t_queues, &transaction_db,&write_indices](){
            int64_t write_index = write_indices[t_id];
            bdgl_queue_create_task(t_id, t_queues[t_id], transaction_db[t_id], write_index);
            write_indices[t_id] = write_index;
            t_queues[t_id].clear();
        });
    }
    threadpool.wait_work(); 

    // Insert transaction DB
    std::vector<size_t> kk(params.threads);
    for( size_t t_id = 0; t_id < params.threads; ++t_id ) {
        threadpool.push([this, &kk, t_id, &transaction_db,&write_indices](){
            kk[t_id] = bdgl_queue_insert_task(t_id, transaction_db[t_id], write_indices[t_id]);        
        });
    }
    threadpool.wait_work(); 
    size_t min_kk = kk[0];
    size_t inserted = 0;
    for(unsigned int i = 0; i < params.threads; i++ ){
        min_kk = std::min(min_kk, kk[i]);
        inserted += (S-1-i - kk[i]-params.threads)/params.threads;
    }
    status_data.plain_data.sorted_until = min_kk;
}

bool Siever::bdgl_sieve(size_t nr_buckets_aim, const size_t blocks, const size_t multi_hash) {
    switch_mode_to(SieveStatus::plain);
    parallel_sort_cdb();
    statistics.inc_stats_sorting_sieve();
    size_t const S = cdb.size();
    recompute_histo();

    size_t saturation_index = 0.5 * params.saturation_ratio * std::pow(params.saturation_radius, n/2.0);
    if( saturation_index > 0.5 * S ) {
        std::cerr << "Saturation index larger than half of db size" << std::endl;
        saturation_index = std::min(saturation_index, S-1);
    }
    
    std::vector<std::vector<Entry>> transaction_db(params.threads, std::vector<Entry>());
    std::vector<uint32_t> buckets;
    std::vector<atomic_size_t_wrapper> buckets_i;
    std::vector<std::vector<QEntry>> t_queues(params.threads);

    size_t it = 0;
    while( true ) {
        bdgl_bucketing(blocks, multi_hash, nr_buckets_aim, buckets, buckets_i);

        bdgl_process_buckets(buckets, buckets_i, t_queues);

        bdgl_queue(t_queues, transaction_db );

        parallel_sort_cdb();

        if( cdb[saturation_index].len <= params.saturation_radius ) {
            assert(std::is_sorted(cdb.cbegin(),cdb.cend(), compare_CE()  ));
            invalidate_histo();
            recompute_histo();
            return true;
        } 

        if( it > 10000 ) {
            std::cerr << "Not saturated after 10000 iterations" << std::endl;
            return false;
        }

        it++;
    }
}
