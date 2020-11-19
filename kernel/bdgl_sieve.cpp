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

struct lPair {
    size_t i,j;
    int8_t sign;
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
    // only recompute new_l and sign if not passed
    if( new_l <= 0.0 and !(sign==1 or sign==-1) ) {
        LFT const inner = std::inner_product(db[i1].yr.cbegin(), db[i1].yr.cbegin()+n, db[i2].yr.cbegin(), static_cast<LFT>(0.));
        new_l = db[i1].len + db[i2].len - 2 * std::abs(inner);
        sign = inner < 0 ? 1 : -1;
    }

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
            // only recompute new_l and sign if not passed
            if( new_l <= 0.0 ) {
                LFT const inner = std::inner_product(db[i1].yr.cbegin(), db[i1].yr.cbegin()+n, db[i2].yr.cbegin(), static_cast<LFT>(0.));
                if( sign != 0 )
                    new_l = db[i1].len + db[i2].len + sign * 2 * inner;
                else {
                    new_l = db[i1].len + db[i2].len - 2 * std::fabs(inner);
                    sign = (inner<0)?1:-1;
                }
                if( new_l >= lenbound ) {
                    uid_hash_table.erase_uid(new_uid);
                    return -1;
                }
            }
            std::array<ZT,MAX_SIEVING_DIM> new_x = db[i1].x;
            addsub_vec(new_x,  db[i2].x, static_cast<ZT>(sign));
            
            int64_t index = write_index--; // atomic and signed!
            if( index >= 0 ) {
                Entry& new_entry = transaction_db[index];
                new_entry.x = new_x;
                //std::cerr << "Lift general short: " << std::endl;
                recompute_data_for_entry<Recompute::recompute_all_and_consider_otf_lift>(new_entry); // includes recomputing uid !
                // don't think this can happen in this sieve:
                if (UNLIKELY(new_entry.uid != new_uid)) // this may happen only due to data races, if the uids and the x-coos of the points we used were out-of-sync
                {
                    uid_hash_table.erase_uid(new_uid);
                    transaction_db.pop_back();
                    std::cerr << "Data race (" << index << "," << static_cast<ZT>(sign) << "," << new_uid << "," << new_entry.uid << ")" << std::endl;
                    return 0;
                }
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
    return -1;
}

// assumed that sign is known
inline void Siever::bdgl_lift(const size_t i1, const size_t i2, LFT new_l, int8_t sign)
{   
    if( new_l <= 0. ) {
        LFT const inner = std::inner_product(db[i1].yr.cbegin(), db[i1].yr.cbegin()+n, db[i2].yr.cbegin(), static_cast<LFT>(0.));
        new_l = db[i1].len + db[i2].len + 2 * sign * inner;
    }
    
    if (params.otf_lift )
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

template<bool cdb_indices=true>
void Siever::bdgl_bucketing_task(const size_t t_id, std::vector<int> &buckets, std::vector<size_t> &buckets_index, ProductLSH &lsh)
{
    CompressedEntry* const fast_cdb = cdb.data();
    const size_t S = cdb.size();
    const size_t multi_hash = lsh.multi_hash;
    const int nr_buckets = buckets_index.size();
    const size_t bsize = buckets.size() / nr_buckets;
    const size_t threads = params.threads;

    size_t i_start = t_id;
    int32_t res[multi_hash];
    size_t bucket_index = 0;
    for (size_t i = i_start; i < S; i += threads)
    {  
        auto db_index = fast_cdb[i].i;
        lsh.hash( db[db_index].yr.data() , res);
        for( size_t j = 0; j < multi_hash; j++ ) {
            int32_t b = res[j];
            size_t b_abs = std::abs(b)-1;
            bool sign = b < 0;
            assert( b_abs < nr_buckets );
            {
                std::lock_guard<std::mutex> lockguard(bdgl_bucket_mut[ b_abs % BDGL_BUCKET_SPLIT ]);
                bucket_index = buckets_index[b_abs]++; // mutex protected
                // todo: do this without locks
            }
            if( bucket_index < bsize ) {
                if( cdb_indices )
                    buckets[bsize * b_abs + bucket_index] = (sign)?-i:i;
                else
                    buckets[bsize * b_abs + bucket_index] = (sign)?-db_index:db_index;
            }
        }
    }
}

inline bool compare_abs(int const& lhs, int const& rhs) { return std::abs(lhs) < std::abs(rhs); }

void Siever::bdgl_bucketing_sort_task(const size_t t_id, std::vector<int> &buckets, const std::vector<size_t> &buckets_index){
    CompressedEntry* const fast_cdb = cdb.data();
    const size_t nr_buckets = buckets_index.size();
    const size_t bsize = buckets.size() / nr_buckets;
    const size_t threads = params.threads;

    const size_t buckets_per_thread = (nr_buckets + threads-1)/threads;
    const size_t b_start = t_id * buckets_per_thread;
    const size_t b_end = std::min(nr_buckets, (t_id+1)*buckets_per_thread);

    for(size_t b = b_start; b < b_end; b++ ) {
        std::sort( buckets.begin() + b*bsize, buckets.begin() + b*bsize + buckets_index[b], &compare_abs);
        size_t i_start = b*bsize;
        size_t i_end = i_start + buckets_index[b];
        for( size_t i = i_start; i < i_end; i++ ) {
            int bi = buckets[i];
            buckets[i] = (bi > 0 ) ? cdb[std::abs(bi)].i : -cdb[std::abs(bi)].i; 
        }
    }
}

// assumes buckets and buckets_index are resized and resetted correctly.
template<bool cdb_indices=true>
void Siever::bdgl_bucketing(const size_t blocks, const size_t multi_hash, const size_t nr_buckets_aim, 
    std::vector<int> &buckets, std::vector<size_t> &buckets_index)
{
    // init hash
    const int64_t lsh_seed = rng();
    ProductLSH lsh(n, blocks, nr_buckets_aim, multi_hash, lsh_seed, true);
    const size_t nr_buckets = lsh.codesize;
    const size_t S = cdb.size();
    size_t bsize = 2 * (S*multi_hash / double(nr_buckets));
    buckets.resize( nr_buckets * bsize );
    buckets_index.resize(nr_buckets);

    for (size_t t_id = 0; t_id < params.threads; ++t_id)
    {
        threadpool.push([this, t_id, multi_hash, &buckets, &buckets_index, &lsh](){
            bdgl_bucketing_task<true>(t_id, buckets, buckets_index, lsh);
        });
    }
    threadpool.wait_work(); 
    
    // // ----------------check bucketing balance--------------------------
    // const size_t bsize = buckets.size() / nr_buckets;
    // double buckets_avg = 0.;
    // double buckets_min = bsize;
    // double buckets_max = 0.;
    // // buckets_index can have become larger than the bucket size
    
    // for( size_t i = 0; i < nr_buckets; ++i ) {
    //     buckets_avg += buckets_index[i];
    //     buckets_min = std::min( buckets_min, double(buckets_index[i]));
    //     buckets_max = std::max( buckets_max, double(buckets_index[i]));
    //     if( buckets_index[i] > bsize ) {
    //         std::cerr << "Bucket too large by ratio " << i << " " <<  (buckets_index[i] / double(bsize)) << " " << buckets_index[i] << " " << bsize << std::endl;
    //         buckets_index[i] = bsize;
    //     }
    // }
    // buckets_avg /= nr_buckets;
   
    // --------------sort and convert to db indices--------------------
    if(!cdb_indices) {
        for (size_t t_id = 0; t_id < params.threads; ++t_id)
        {
            threadpool.push([this, t_id, &buckets, &buckets_index](){bdgl_bucketing_sort_task(t_id, buckets, buckets_index);});
        }
        threadpool.wait_work(); 
    }
}

void Siever::bdgl_process_buckets_task(const size_t threads, const size_t t_id, 
    const std::vector<int> &buckets, 
    const std::vector<size_t> &buckets_index, std::vector<QEntry> &t_queue)
{

    const size_t nr_buckets = buckets_index.size();
    const size_t bsize = buckets.size() / nr_buckets;

    const int* const fast_buckets = buckets.data();
    CompressedEntry* const fast_cdb = cdb.data();
    
    const size_t S = cdb.size();
    const size_t A = cdb.size();
    
    int64_t kk = A-1-t_id; //std::min(A-1, size_t(params.bgj1_improvement_db_ratio * (S-1)));
    
    LFT lenbound = fast_cdb[kk].len / REDUCE_LEN_MARGIN;

    // TODO: divide work better by looking at bucket sizes
    // doesn't seem nec. so far
    const size_t b_start = t_id;

    size_t B = 0;
    for (size_t b = b_start; b < nr_buckets; b += threads)
    {
        const size_t i_start = bsize * b;
        const size_t i_end = bsize * b + buckets_index[b];
        B +=( (i_end - i_start) * (i_end-i_start-1)) / 2;
        //std::cerr << B << " ";
        for( size_t i = i_start; i < i_end; ++i ) 
        {
            if (kk < .1 * A) break;

            size_t bi = std::abs(fast_buckets[i]);
            CompressedEntry *pce1 = &fast_cdb[bi];
            CompressedVector cv = pce1->c;
            for (size_t j = i_start; j < i; ++j)
            {
                size_t bj = std::abs(fast_buckets[j]);
                if( is_reducible_maybe<XPC_THRESHOLD>(cv, fast_cdb[bj].c) ) // UNLIKELY OR NOT?
                {
                    std::pair<LFT, int> len_and_sign = reduce_to_QEntry( pce1, &fast_cdb[bj] );
                    if( len_and_sign.first < lenbound)
                    {
                        if (kk < .1 * A) break;
                        kk -= threads;
                        
                        statistics.inc_stats_2redsuccess_outer();

                        lenbound = fast_cdb[kk].len / REDUCE_LEN_MARGIN;
                        t_queue.push_back({ pce1->i, fast_cdb[bj].i, len_and_sign.first, len_and_sign.second});
                    } else if( len_and_sign.first < params.lift_radius ) {
                        // on the fly lifting
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
double Siever::bdgl_process_buckets(thread_pool::thread_pool &pool, const size_t threads,
    const std::vector<int> &buckets, 
    const std::vector<size_t> &buckets_index, 
    std::vector<QEntry> &queue)
{
    auto start_processing = std::chrono::steady_clock::now();

    std::vector<std::vector<QEntry>> t_queues(threads);
    for (size_t t_id = 0; t_id < threads; ++t_id)
    {
        pool.push([this, threads, t_id, &buckets, &buckets_index, &t_queues](){bdgl_process_buckets_task(threads, t_id, buckets, buckets_index, t_queues[t_id]);});
    }
    pool.wait_work();

    // TODO: parallel mergetree
    size_t Q = 0;
    for(size_t t_id = 0; t_id < threads; ++t_id)
    {
        Q += t_queues[t_id].size();
    }
    queue.resize(Q);
    Q = 0;
    for(size_t t_id = 0; t_id < threads; ++t_id)
    {
        std::copy(t_queues[t_id].begin(), t_queues[t_id].end(), queue.begin() + Q);
        if( t_id > 0 )
            std::inplace_merge(queue.begin(),queue.begin() + Q, queue.begin() + Q + t_queues[t_id].size(), &compare_QEntry);
        Q += t_queues[t_id].size();
    }

    assert(std::is_sorted(queue.begin(), queue.end(), &compare_QEntry));
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_processing).count();
}

void Siever::bdgl_queue_dup_remove_task( const size_t threads, const size_t t_id, std::vector<QEntry> &queue) {
    const size_t Q = queue.size();
    const size_t Qstart = (t_id*Q)/threads;
    const size_t Qend = ((t_id+1)*Q)/threads;
    for( size_t index = Qstart; index < Qend; index++ ) {
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

void Siever::bdgl_queue_create_task( const size_t threads, const size_t t_id, const std::vector<QEntry> &queue, std::vector<Entry> &transaction_db, int64_t &write_index) {
    const size_t S = cdb.size();
    const size_t Q = queue.size();

    const size_t insert_after = (cdb.size() != cdb.size()) ? cdb.size() : S-1-t_id-threads*write_index; 
    for( int index = t_id; index < Q; index += threads )  {     
        // use sign as skip marker
        if( queue[index].sign == 0 ){
            continue;
        }
        bdgl_reduce_with_delayed_replace( queue[index].i, queue[index].j, 
                                                  cdb[std::min(S-1, insert_after+threads*write_index)].len / REDUCE_LEN_MARGIN,
                                                  transaction_db, write_index, queue[index].len, queue[index].sign);
        if( write_index < 0 ){
            std::cerr << "Spilling full transaction db" << t_id << " " << Q-index << std::endl;
            break;
        }
    }
}

size_t Siever::bdgl_queue_insert_task( const size_t threads, const size_t t_id, std::vector<Entry> &transaction_db, int64_t write_index) {
    const size_t S = cdb.size();
    const size_t insert_after = (cdb.size() != cdb.size()) ? cdb.size() : std::max(int(0), int(int(S)-1-t_id-threads*(transaction_db.size()-write_index))); 
    size_t kk = S-1 - t_id;
    for( int i = transaction_db.size()-1; i > write_index and kk >= insert_after; --i )  { 
            if( bdgl_replace_in_db( kk, transaction_db[i] ) ) {
                kk -= threads;
            }
    }
    return kk + threads;
}

size_t Siever::bdgl_queue(thread_pool::thread_pool &pool, const size_t threads, 
  std::vector<QEntry> &queue, std::vector<std::vector<Entry>>& transaction_db ) {

    auto queue_start = std::chrono::steady_clock::now();

    if( queue.size() == 0 )
        return 0;

    // clear duplicates read only
    for( size_t t_id = 0; t_id < threads; ++t_id ) {
        pool.push([this, threads, t_id, &queue, &transaction_db](){
            bdgl_queue_dup_remove_task(threads, t_id, queue);
        });
    }
    pool.wait_work();


    const size_t S = cdb.size(); 
    size_t Q = queue.size();
    // detect if we need to insert after certain bound
    size_t insert_after = (cdb.size() != cdb.size()) ? cdb.size() : std::max(0, int(int(S)-Q));
    
    
    for( int i = 0; i < threads; i++ )
        transaction_db[i].resize(std::min(S-insert_after, Q)/threads + 1);

    std::vector<int> write_indices(threads, transaction_db[0].size()-1);
    // Prepare transaction DB from queue
   for( size_t t_id = 0; t_id < threads; ++t_id ) {
        pool.push([this, threads, t_id, &queue, &transaction_db,&write_indices](){
            int64_t write_index = write_indices[t_id];
            bdgl_queue_create_task(threads, t_id, queue, transaction_db[t_id], write_index);
            write_indices[t_id] = write_index;
        });
    }
    pool.wait_work(); 
    queue.clear();

    size_t transaction_vecs_used = 0;
    for( int t_id = 0; t_id < threads; t_id++ ) {
        transaction_vecs_used += transaction_db[t_id].size()-1 - write_indices[t_id];
    }
    queue_start = std::chrono::steady_clock::now();

    // Insert transaction DB
    std::vector<size_t> kk(threads);
    for( size_t t_id = 0; t_id < threads; ++t_id ) {
        pool.push([this, &kk, threads, t_id, &transaction_db,&write_indices](){
            kk[t_id] = bdgl_queue_insert_task(threads, t_id, transaction_db[t_id], write_indices[t_id]);        
        });
    }
    pool.wait_work(); 
    size_t min_kk = kk[0];
    size_t inserted = 0;
    for( int i = 0; i < threads; i++ ){
        min_kk = std::min(min_kk, kk[i]);
        inserted += (S-1-i - kk[i]-threads)/threads;
    }
    status_data.plain_data.sorted_until = min_kk;

    /*
    std::cerr << "Seq Part: "
        << double(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - queue_start).count())
        << std::endl;
*/
    return transaction_vecs_used;
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

    std::vector<int> buckets;
    std::vector<size_t> buckets_i;
    std::vector<QEntry> queue;

    size_t it = 0;
    while( true ) {

        // bucketing
        // TODO: make sure the bucketing function can also work directly on the cdb for cpu version
        cdb.resize(S);
        std::copy( cdb.begin(), cdb.begin()+S, cdb.begin());
        
        
        std::fill( buckets_i.begin(), buckets_i.end(), 0);
        
        bdgl_bucketing<true>(blocks, multi_hash, nr_buckets_aim, buckets, buckets_i);

        bdgl_process_buckets(threadpool, params.threads, buckets, buckets_i, queue);

        size_t kk = bdgl_queue( threadpool, params.threads, queue, transaction_db );

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
