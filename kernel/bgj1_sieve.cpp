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
#include "iostream"
#include "fstream"
#include <numeric>
#include <atomic>
#include <thread>
#include <mutex>


/**
    Threaded Bucketed NV Sieve
*/

/**
    bgj1_sieve realizes a threaded bucketed NV sieve.
    The algorithm roughly works as follows:

    It chooses random bucket centers aux from (c)db.
    It then creates a "bucket" of vectors that are (up to sign) close to it. (Bucketing phase)
    Then, it searches for reductions within elements from the bucket.
    Since the bucket centers are lattice points, we may also find reductions
    during the bucketing phase and we use them as well.
    For better concurrency, newly found reductions are first put into a (thread-local) database
    of pending db insertions and only inserted later.
    Insertion is performed by overwriting the (presumed) longest elements from the db (i.e. at the end of cdb).
    After a certain amount of insertions, we resort.
    The parameter alpha controls what vectors we put into a bucket (up to missing vectors due to concurrency issues or imperfect simhashes):
    We put x into the bucket with center aux if |<x, aux>| > alpha * |x| * |aux|
    We do not grow the database inside this algorithm. This has to be done by the caller.
*/


/**
    Siever::bgj1_sieve is what is called from outside.

    This function just sets up "global" parameters for the bgj1_sieve, dispatches to worker threads and cleans up afterwards.
*/
void Siever::bgj1_sieve(double alpha)
{
    CPUCOUNT(100);
    // std::cout << "in bgj1_sieve" << std::endl;
    switch_mode_to(SieveStatus::bgj1);
    assert(alpha >= 0); // negative alpha would work, but not match the documentation.

    size_t const S = cdb.size();
    if(S == 0) return;
    parallel_sort_cdb();
    statistics.inc_stats_sorting_sieve();

    // initialize global variables: GBL_replace_pos is where newly found elements are inserted.
    // This variable is reset after every sort.
    // For better concurrency, when we insert a bunch of vectors, we first (atomically) decrease GBL_replace_pos
    // (this reserves the portion of the database for exclusive access by the current thread)
    // then do the actual work, then (atomically) decrease GBL_replace_done_pos.
    // GBL_replace_done_pos is used to synchronize (c)db writes with resorting.

    GBL_replace_pos = S - 1;
    GBL_replace_done_pos = S - 1;

    // GBL_max_len is the (squared) length bound below which we consider vectors as good enough for
    // potential insertion. This formula ensures some form of meaningful progress:
    // We need to at least improve by a constant fraction REDUCE_LEN_MARGIN
    // and we need to improve the rank (in sorted order) by at least a constant fraction.
    GBL_max_len = cdb[params.bgj1_improvement_db_ratio * (cdb.size()-1)].len / REDUCE_LEN_MARGIN;

    // maximum number of buckets that we try in-between sorts.
    // If we do not trigger sorting after this many buckets, we give up / consider our work done.
    GBL_max_trial = 100 + std::ceil(4 * (std::pow(S, .65)));
    GBL_remaining_trial = GBL_max_trial; // counts down to 0 to realize this limit.

    // points to the beginning of cdb. Note that during sorting, we first sort into a copy
    // and then swap. Publishing the new cdb is then done essentially by atomically changing this pointer.
    GBL_start_of_cdb_ptr = cdb.data();

    // We dispatch to worker threads. UNTEMPLATE_DIM overwrites task by bgj1_sieve_task<dim>
    // i.e. a pre-compiled version of the worker task with the dimension hardwired.
    auto task = &Siever::bgj1_sieve_task<-1>;
    UNTEMPLATE_DIM(&Siever::bgj1_sieve_task, task, n);

    // We terminate bgj1 if we have (roughly) more than the initial value of GBL_saturation_count
    // vectors in our database that are shorter than params.saturation_radius (radius includes gh normalization).
    // We decrease this until 0, at which point we will stop.
    // Remark: When overwriting a vector that was already shorter than the saturation_radius, we might double-count.
    // For parameters in the intended ranges, this is not a problem.
    GBL_saturation_count = std::pow(params.saturation_radius, n/2.) * params.saturation_ratio / 2.;

    // current number of entries below the saturation bound. We use the fact that cdb is sorted.
    size_t cur_sat = std::lower_bound(cdb.cbegin(), cdb.cend(), params.saturation_radius, [](CompressedEntry const &ce, double const &bound){return ce.len < bound; } ) - cdb.cbegin();
    if (cur_sat >= GBL_saturation_count)
        return;
    GBL_saturation_count -= cur_sat;


    for (size_t c = 0; c < params.threads; ++c)
    {
        threadpool.push([this,alpha,task](){((*this).*task)(alpha);});
    }
    threadpool.wait_work();

    invalidate_sorting();
    statistics.inc_stats_sorting_sieve();

    status_data.plain_data.sorted_until = 0;
    // we set histo for statistical purposes
    recompute_histo(); // TODO: Remove?

    return;
}

// Worker task for bgj1 sieve. The parameter tn is equal to n if it is >= 0.
template <int tn>
void Siever::bgj1_sieve_task(double alpha)
{
    ATOMIC_CPUCOUNT(101);

    assert(tn < 0 || static_cast<unsigned int>(tn)==n);

    // statistics are collected per-thread and only merged once at the end. These variables get merged automatically at scope exit.
    #if COLLECT_STATISTICS_XORPOPCNT_PASS || COLLECT_STATISTICS_FULLSCPRODS
        // number of successful xorpopcnt tests in the create-buckets phase
        auto &&local_stat_successful_xorpopcnt_bucketing = merge_on_exit<unsigned long long>( [this](unsigned long long val)
        {
            statistics.inc_stats_xorpopcnt_pass_outer(val);
            statistics.inc_stats_fullscprods_outer(val);
        } );
        // number of successful xorpopcnt tests in the second phase
        auto &&local_stat_successful_xorpopcnt_reds = merge_on_exit<unsigned long long>( [this](unsigned long long val)
        {
            statistics.inc_stats_xorpopcnt_pass_inner(val);
            statistics.inc_stats_fullscprods_inner(val);
        } );
    #endif
    #if COLLECT_STATISTICS_XORPOPCNT || COLLECT_STATISTICS_REDS
        // number of 2-reduction attempts in the second phase
        auto &&local_stat_2red_attempts = merge_on_exit<unsigned long long>([this](unsigned long long val)
        {
            statistics.inc_stats_xorpopcnt_inner(val);
            statistics.inc_stats_2reds_inner(val);
        } );
    #endif
    ENABLE_IF_STATS_REDSUCCESS ( auto &&local_stat_successful_reductions = merge_on_exit<unsigned long>([this](unsigned long val){ statistics.inc_stats_2redsuccess_inner(val); }); )
    ENABLE_IF_STATS_REDSUCCESS ( auto &&local_stat_successful_red_outer  = merge_on_exit<unsigned long>([this](unsigned long val){ statistics.inc_stats_2redsuccess_outer(val); }); )
    ENABLE_IF_STATS_BUCKETS(     auto &&local_stats_number_of_buckets    = merge_on_exit<unsigned long>([this](unsigned long val){ statistics.inc_stats_buckets(val); }); )

    size_t const fullS = cdb.size();
    std::vector<CompressedEntry> bucket; // bucket that this thread uses.

    std::vector<Entry> transaction_db; // stores pending transactions, i.e. points to be added to the DB.
    transaction_db.reserve(params.bgj1_transaction_bulk_size);

    for (; GBL_remaining_trial > 0; GBL_remaining_trial.fetch_sub(1))
    {
        ENABLE_IF_STATS_BUCKETS(++local_stats_number_of_buckets;)
        ///////////// preparing a bucket
        bucket.clear();

        CompressedEntry* const fast_cdb = GBL_start_of_cdb_ptr; // atomic load
        size_t const j_aux = fullS/4 + (rng() % (fullS/4));
        CompressedEntry const aux = fast_cdb[j_aux];
        std::array<LFT,MAX_SIEVING_DIM> const yr1 = db[aux.i].yr;
        CompressedVector const cv = sim_hashes.compress(yr1);

        LFT const alpha_square_times_len = alpha * alpha * aux.len;
        LFT maxlen = GBL_max_len;


        for (size_t j = 0; j < fullS; ++j)
        {

            if (UNLIKELY(is_reducible_maybe<XPC_BUCKET_THRESHOLD>(cv, fast_cdb[j].c)))
            {
                #if COLLECT_STATISTICS_XORPOPCNT_PASS || COLLECT_STATISTICS_FULLSCPRODS
                    ++local_stat_successful_xorpopcnt_bucketing; //adds to successful xorpopcnt and to #scalar product computations
                #endif
                if (UNLIKELY(j == j_aux))
                {
                    statistics.dec_stats_fullscprods_outer();
                    continue;
                }
                LFT const inner = std::inner_product(yr1.begin(), yr1.begin()+(tn < 0 ? n : tn), db[fast_cdb[j].i].yr.begin(),  static_cast<LFT>(0.));

                // Test for reduction while bucketing.
                LFT const new_l = aux.len + fast_cdb[j].len - 2 * std::abs(inner);
                if (UNLIKELY(new_l < maxlen))
                {
                    #if (COLLECT_STATISTICS_OTFLIFTS >= 2) || (COLLECT_STATISTICS_REDSUCCESS >= 2) || (COLLECT_STATISTICS_DATARACES >= 2)
                    if (bgj1_reduce_with_delayed_replace<tn>(aux, fast_cdb[j], maxlen, transaction_db, true))
                    {
                        ENABLE_IF_STATS_REDSUCCESS ( ++ local_stat_successful_red_outer; )
                    }
                    #else
                    bgj1_reduce_with_delayed_replace<tn>(aux, fast_cdb[j], maxlen, transaction_db);
                    #endif
                    if (UNLIKELY(transaction_db.size() > params.bgj1_transaction_bulk_size))
                    {
                        if (bgj1_execute_delayed_replace(transaction_db, false))
                            maxlen = GBL_max_len;
                    }
                }

                // Test for bucketing
                if (UNLIKELY(inner * inner > alpha_square_times_len * fast_cdb[j].len))
                {
                    bucket.push_back(fast_cdb[j]);
                }
            }
        }
        // no-ops if statistics are not actually collected
        statistics.inc_stats_xorpopcnt_outer(fullS );
        statistics.inc_stats_2reds_outer(fullS -1);
        statistics.inc_stats_filter_pass(bucket.size());

        // bucket now contains points that are close to aux, and so are hopefully close to each other (up to sign)

        if (GBL_remaining_trial.load(std::memory_order_relaxed) < 0) break;
        size_t const S = bucket.size();

        ////////////// Sieve the bucket
        maxlen = GBL_max_len;
        ATOMIC_CPUCOUNT(102);
        CompressedEntry const * const fast_bucket = bucket.data();
        for (size_t block = 0; block < S; block+=CACHE_BLOCK)
        {
            for (size_t i = block+1; i < S; ++i)
            {
                // skip it if not up to date
                if (UNLIKELY(fast_bucket[i].c[0] != db[fast_bucket[i].i].c[0]))
                {
                    continue;
                }

                size_t const jmin = block;
                size_t const jmax = std::min(i, block+CACHE_BLOCK);
                CompressedEntry const * const pce1 = &fast_bucket[i];
                uint64_t const * const cv = &(pce1->c.front());
                for (size_t j = jmin; j < jmax; ++j)
                {
                    if (UNLIKELY(is_reducible_maybe<XPC_THRESHOLD>(cv, &fast_bucket[j].c.front())))
                    {
                        #if COLLECT_STATISTICS_XORPOPCNT_PASS || COLLECT_STATISTICS_FULLSCPRODS
                        ++local_stat_successful_xorpopcnt_reds; // adds to successful xorpopcnt and also to scalar product computations (done inside bgj1_reduce_with_delayed_replace)
                        #endif
                        ATOMIC_CPUCOUNT(103);
                        bool const red = bgj1_reduce_with_delayed_replace<tn>(*pce1, fast_bucket[j], maxlen, transaction_db);
                        ENABLE_IF_STATS_REDSUCCESS(if(red) {++local_stat_successful_reductions; } )
                        if (UNLIKELY(red) && transaction_db.size() > params.bgj1_transaction_bulk_size)
                        {
                            if (bgj1_execute_delayed_replace(transaction_db, false))
                            {
                                if (GBL_remaining_trial.load(std::memory_order_relaxed) < 0)
                                {
                                    #if COLLECT_STATISTICS_XORPOPCNT || COLLECT_STATISTICS_REDS
                                    local_stat_2red_attempts += (1 + j - jmin); // statistics for half-finished loop
                                    #endif
                                    return;
                                }
                                maxlen = GBL_max_len;
                            }
                        }
                    }
                }
                #if COLLECT_STATISTICS_XORPOPCNT || COLLECT_STATISTICS_REDS
                local_stat_2red_attempts += jmax - jmin;
                #endif
            }
        }
        bgj1_execute_delayed_replace(transaction_db, false);
    }
    bgj1_execute_delayed_replace(transaction_db, true, true);
}



// Attempt reduction between ce1 and c2. lenbound is a bound on the lenght of the result.
// If successful, we store the result in transaction_db.
// Templated by tn=n to hardwire the dimension.
// reduce_while_bucketing indicates whether we call this function during the bucketing phase (rather than when working inside a bucket)
// This is only relevant for statistics collection.
#if (COLLECT_STATISTICS_OTFLIFTS >= 2) || (COLLECT_STATISTICS_REDSUCCESS >= 2) || (COLLECT_STATISTICS_DATARACES >= 2)
template <int tn>
inline bool Siever::bgj1_reduce_with_delayed_replace(CompressedEntry const &ce1, CompressedEntry const &ce2, LFT const lenbound, std::vector<Entry>& transaction_db, bool const reduce_while_bucketing)
{
#else
template <int tn>
inline bool Siever::bgj1_reduce_with_delayed_replace(CompressedEntry const &ce1, CompressedEntry const &ce2, LFT const lenbound, std::vector<Entry>& transaction_db)
{
    constexpr bool reduce_while_bucketing = true; // The actual value does not matter.
#endif
    // statistics.inc_fullscprods done at call site
    LFT const inner = std::inner_product(db[ce1.i].yr.cbegin(), db[ce1.i].yr.cbegin()+(tn < 0 ? n : tn), db[ce2.i].yr.cbegin(), static_cast<LFT>(0.));
    LFT const new_l = ce1.len + ce2.len - 2 * std::abs(inner);
    int const sign = inner < 0 ? 1 : -1;

    if (UNLIKELY(new_l < lenbound))
    {
        UidType new_uid = db[ce1.i].uid;
        if(inner < 0)
        {
            new_uid += db[ce2.i].uid;
        }
        else
        {
            new_uid -= db[ce2.i].uid;
        }
        if (uid_hash_table.insert_uid(new_uid))
        {
            std::array<ZT,MAX_SIEVING_DIM> new_x = db[ce1.i].x;
            addsub_vec(new_x,  db[ce2.i].x, static_cast<ZT>(sign));
            transaction_db.emplace_back();
            Entry& new_entry = transaction_db.back();
            new_entry.x = new_x;
            recompute_data_for_entry<Recompute::recompute_all_and_consider_otf_lift>(new_entry); // includes recomputing uid !
            if (UNLIKELY(new_entry.uid != new_uid)) // this may happen only due to data races, if the uids and the x-coos of the points we used were out-of-sync
            {
                if (reduce_while_bucketing)
                    statistics.inc_stats_dataraces_2outer();
                else
                    statistics.inc_stats_dataraces_2inner();
                uid_hash_table.erase_uid(new_uid);
                transaction_db.pop_back();
                return false;
            }
            return true;
        }
        else
        {
            if (reduce_while_bucketing)
                statistics.inc_stats_collisions_2outer();
            else
                statistics.inc_stats_collisions_2inner();
            return false;
        }
    }
    else if (params.otf_lift && (new_l < params.lift_radius))
    {
        ZT x[r];
        LFT otf_helper[OTF_LIFT_HELPER_DIM];
        std::fill(x, x+l, 0);
        std::copy(db[ce1.i].x.cbegin(), db[ce1.i].x.cbegin()+(tn < 0 ? n : tn), x+l);
        std::copy(db[ce1.i].otf_helper.cbegin(), db[ce1.i].otf_helper.cbegin()+OTF_LIFT_HELPER_DIM, otf_helper);
        if(sign == 1)
        {
          for(unsigned int i=0; i < (tn < 0 ? n : tn); ++i)
          {
            x[l+i] += db[ce2.i].x[i];
          }
          for(unsigned int i=0; i < OTF_LIFT_HELPER_DIM; ++i)
          {
            otf_helper[i] += db[ce2.i].otf_helper[i];
          }

        }
        else
        {
          for(unsigned int i=0; i < (tn < 0 ? n : tn); ++i)
          {
            x[l+i] -= db[ce2.i].x[i];
          }
          for(unsigned int i=0; i < OTF_LIFT_HELPER_DIM; ++i)
          {
            otf_helper[i] -= db[ce2.i].otf_helper[i];
          }
        }
        if (reduce_while_bucketing)
            statistics.inc_stats_otflifts_2outer();
        else
            statistics.inc_stats_otflifts_2inner();
        lift_and_compare(x, new_l * gh, otf_helper);
//        lift_and_compare(db[ce1.i], sign, &(db[ce2.i]));
    }
    return false;
}


bool Siever::bgj1_execute_delayed_replace(std::vector<Entry>& transaction_db, bool force, bool nosort /* =false*/ )
{
    if (UNLIKELY(!transaction_db.size())) return true;
    ATOMIC_CPUCOUNT(104);

    std::unique_lock<std::mutex> lockguard(GBL_db_mutex, std::defer_lock_t());
    size_t resortthres = params.bgj1_resort_ratio * db.size();
    size_t maxts = resortthres;

    if (UNLIKELY(nosort == true))
    {
        // compute maximum transaction count that can be processed without triggering resort
        maxts = GBL_replace_pos.load();
        // if sort is current happening then return false if force = true or else wait
        while (maxts < resortthres)
        {
            if (!force) return false;
            lockguard.lock(); lockguard.unlock();
            maxts = GBL_replace_pos.load();
        }
        maxts = (maxts - resortthres) / params.threads;
    }

    // maximum size that can be safely processed without out-of-bounds errors
    if (UNLIKELY(transaction_db.size() > maxts))
    {
        std::sort(transaction_db.begin(), transaction_db.end(), [](const Entry& l, const Entry& r){return l.len < r.len;});
        transaction_db.resize(maxts);
    }
    size_t ts = transaction_db.size();
    assert(ts <= maxts);

    size_t rpos = 0;
    LFT oldmaxlen = 0;
    // try to claim a replacement interval in cdb: rpos-ts+1,...,rpos
    while (true)
    {
        LFT maxlen = GBL_max_len.load();
        // prune transaction_db based on current maxlen
        if (LIKELY(maxlen != oldmaxlen))
        {
            for (size_t i = 0; i < transaction_db.size(); )
            {
                if (transaction_db[i].len < maxlen)
                    ++i;
                else
                {
                    if (i != transaction_db.size()-1)
                        std::swap(transaction_db[i], transaction_db.back());
                    transaction_db.pop_back();
                }
            }
            oldmaxlen = maxlen;
            ts = transaction_db.size();
        }
        // check & wait if another thread is in the process of sorting cdb
        while (UNLIKELY((rpos = GBL_replace_pos.load()) < resortthres))
        {
            if (!force) return (ts==0); // Note: ts might have become 0 due to pruning. In this case, we return true
            lockguard.lock(); lockguard.unlock();
        }
        // decrease GBL_replace_pos with #ts if it matches the expected value rpos, otherwise retry
        if (LIKELY(GBL_replace_pos.compare_exchange_weak(rpos, rpos-ts)))
            break;
    }
    // now we can replace cdb[rpos-ts+1,...,rpos]
    // if we need to resort after inserting then already grab the lock so other threads can block
    if (UNLIKELY( rpos-ts < resortthres ))
    {
        lockguard.lock();
        // set GBL_remaining_trial to MAXINT, but if it was negative then keep it negative
        if (UNLIKELY(GBL_remaining_trial.exchange( std::numeric_limits<decltype(GBL_remaining_trial.load())>::max()) < 0))
            GBL_remaining_trial = -1;
    }

    // update GBL_max_len already
    GBL_max_len = cdb[params.bgj1_improvement_db_ratio * (rpos-ts)].len / REDUCE_LEN_MARGIN;

    // we can replace cdb[rpos-ts+1,...,rpos]
    size_t cur_sat = 0;
    for (size_t i = 0; !transaction_db.empty(); ++i)
    {
        if (bgj1_replace_in_db(rpos-i, transaction_db.back())
            && transaction_db.back().len < params.saturation_radius)
        {
            ++cur_sat;
        }
        transaction_db.pop_back();
    }
    statistics.inc_stats_replacements_list(cur_sat);
    // update GBL_saturation_count
    if (UNLIKELY(GBL_saturation_count.fetch_sub(cur_sat) <= cur_sat))
    {
        GBL_saturation_count = 0;
        GBL_remaining_trial = -1;
    }

    // update GBL_replaced_pos to indicate we're done writing to cdb
    GBL_replace_done_pos -= ts;
    // if we don't need to resort then we're done
    if (LIKELY(rpos-ts >= resortthres))
        return true;
    // wait till all other threads finished writing to db/cdb
    // TODO: a smarter way to sleep so it gets activated when other threads have finished
    while (GBL_replace_done_pos.load() != GBL_replace_pos.load()) // Note : The value of GBL_replace_pos actually cannot change here, so the order of the reads does not matter.
        std::this_thread::yield();

    // sorting always needs to happen, threads could be waiting on GBL_replace_pos
    CPUCOUNT(105);
    size_t improvindex = params.bgj1_improvement_db_ratio * (cdb.size()-1);
    if (params.threads == 1)
    {
        std::sort(cdb.begin()+GBL_replace_pos+1, cdb.end(), compare_CE());
        std::inplace_merge(cdb.begin(), cdb.begin()+GBL_replace_pos+1, cdb.end(), compare_CE());
    }
    else
    {
        cdb_tmp_copy=cdb;
        std::sort(cdb_tmp_copy.begin()+GBL_replace_pos+1, cdb_tmp_copy.end(), compare_CE());
        std::inplace_merge(cdb_tmp_copy.begin(), cdb_tmp_copy.begin()+GBL_replace_pos+1, cdb_tmp_copy.end(), compare_CE());
        cdb.swap(cdb_tmp_copy);
        GBL_start_of_cdb_ptr = &(cdb.front());
    }
    statistics.inc_stats_sorting_sieve();
    // reset GBL_remaining_trial to GBL_max_trial, unless it was set to < 0 in the meantime.
    if (UNLIKELY(GBL_remaining_trial.exchange(GBL_max_trial) < 0))
       GBL_remaining_trial = -1;
    GBL_replace_done_pos = cdb.size() - 1;
    GBL_replace_pos = cdb.size() - 1;
    GBL_max_len = cdb[improvindex].len / REDUCE_LEN_MARGIN;
    return true;
}

// Replace the db and cdb entry pointed at by cdb[cdb_index] by e, unless
// length is actually worse
bool Siever::bgj1_replace_in_db(size_t cdb_index, Entry &e)
{
    ATOMIC_CPUCOUNT(106);
    CompressedEntry &ce = cdb[cdb_index]; // abbreviation

    if (REDUCE_LEN_MARGIN_HALF * e.len >= ce.len)
    {
        statistics.inc_stats_replacementfailures_list();
        uid_hash_table.erase_uid(e.uid);
        return false;
    }
    uid_hash_table.erase_uid(db[ce.i].uid);
    if(ce.len < params.saturation_radius) { statistics.inc_stats_dataraces_replaced_was_saturated(); } // saturation count becomes (very) wrong if this happens (often)
    ce.len = e.len;
    ce.c = e.c;
    // TODO: FIX THIS PROPERLY. We have a race condition here.
    // NOTE: we tried std::move here (on single threaded), but it ended up slower...
    db[ce.i] = e;
    return true;
}
