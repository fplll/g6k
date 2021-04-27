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


// Sieve the current database
// The 'queue' is stored at the end of the main list
void Siever::gauss_sieve(size_t max_db_size)
{
    CPUCOUNT(301);
    switch_mode_to(SieveStatus::gauss);
    parallel_sort_cdb();
    statistics.inc_stats_sorting_sieve();
    recompute_histo();
    if (max_db_size==0)
    {
        max_db_size = 4 * std::pow(4./3., n/2.) + 4*n;
    }

    for (unsigned int i = 0; i < size_of_histo; ++i)
    {
        GBL_saturation_histo_bound[i] = std::pow(1. + i * (1./size_of_histo), n/2.) * params.saturation_ratio + 20;
    }
    int iter = 0;

    #if COLLECT_STATISTICS_XORPOPCNT || COLLECT_STATISTICS_REDS
        // number of reduction attempts in the second phase
        auto &&local_stat_2red_attempts = merge_on_exit<unsigned long long>([this](unsigned long long val)
        {
            statistics.inc_stats_xorpopcnt_outer(val);
            statistics.inc_stats_2reds_outer(val);
        } );

        // number of successful xorpopcnt tests in the second phase
        auto &&local_stat_successful_xorpopcnt_reds = merge_on_exit<unsigned long long>( [this](unsigned long long val)
        {
            statistics.inc_stats_xorpopcnt_pass_outer(val);
            statistics.inc_stats_fullscprods_outer(val);
        } );
    #endif

    ENABLE_IF_STATS_REDSUCCESS (auto &&local_stat_successful_2red_outer = merge_on_exit<unsigned long>([this](unsigned long val){ statistics.inc_stats_2redsuccess_outer(val); }); )

    // We treat the main cdb as consisting of two parts: The beginning of cdb consists of elements,
    // where (almost) all possible reductions (except those missed due to SimHashes etc) have
    // already been performed. The end of cdb consists of elements which may still participate in
    // reductions. We need to only compare elements from the beginning part with those from the end
    // part. To change the status of an element wrt this distinction, we perform a swap in cdb.
    size_t queue_begin = status_data.gauss_data.queue_start; // We are guaranteed that all elements
                                                             // cdb[0],...,cdb[queue_begin-1] are
                                                             // already reduced wrt each other.
    // termination condition outer loop
    while(cdb.size() <= max_db_size)
    {
        size_t const old_S = status_data.gauss_data.queue_start;
        if (iter) grow_db(cdb.size()*1.02 + 10);
        ++iter;

        // sort the old and new part separately. Note that (except at the beginning), these also
        // correspond to the distinction explained above. The old part already is sorted: We sort
        // the 'queue'-part; shorter vectors to be processed first
        pa::sort(cdb.begin() + old_S, cdb.end(), compare_CE(), threadpool);
        //CompressedEntry* const fast_cdb = &(cdb.front());
        CompressedEntry* const fast_cdb = cdb.data();
        // while there is no elements in the 'queue'-part of the list
        while (queue_begin < cdb.size())
        {
            CompressedEntry * const pce1 = &fast_cdb[queue_begin];
            auto const cv = &(pce1->c.front() );

            size_t const p_index = queue_begin; // remember p_index for the swap at the end of the for-loop
start_over:
            for (size_t j = 0; j < p_index; ++j) // WHY IS IT p_index NOT queue_begin???
            {
#ifndef NDEBUG
                LFT cv2_vec_len = fast_cdb[j].len;
#endif
                if( UNLIKELY(is_reducible_maybe<XPC_THRESHOLD>(cv,&(fast_cdb[j].c.front()) )))
                {
                    #if COLLECT_STATISTICS_XORPOPCNT_PASS || COLLECT_STATISTICS_FULLSCPRODS
                    ++local_stat_successful_xorpopcnt_reds; // adds to successful xorpopcnt and also to scalar product computations
                    #endif
                    short const which_one = gauss_no_upd_reduce_in_db(pce1, &fast_cdb[j]); // takes values {0,1,2}
                    if(which_one != 0) //actual reduction
                    {
                        ENABLE_IF_STATS_REDSUCCESS(++local_stat_successful_2red_outer;)
                        if(which_one==1) // p was reduced, re-start
                        {
                            //std::cout <<" p  is reduced "<< std::endl;
                            goto start_over;
                        }
                        else if(queue_begin) // some other point 'above' p was reduced, move it to the 'Queue'-part of the list
                        {
                            assert(cv2_vec_len > fast_cdb[j].len );
                            using std::swap;
                            swap(fast_cdb[j],fast_cdb[queue_begin-1]); // Aren't we missing a comparison (p, fast_cdb[queue_begin-1])?
                            queue_begin--;
                        }
                    }
                }
            }
            #if COLLECT_STATISTICS_XORPOPCNT || COLLECT_STATISTICS_REDS
            local_stat_2red_attempts += p_index;
            #endif
            // swap p with queue_begin which now indicates the very last swap
            using std::swap;
            swap(fast_cdb[queue_begin],fast_cdb[p_index]);
            ++queue_begin;
        } //while-loop. Now all vectors in cdb are reduced.
        status_data.gauss_data.list_sorted_until = 0;
        status_data.gauss_data.queue_start = cdb.size();
        status_data.gauss_data.queue_sorted_until = cdb.size();
        parallel_sort_cdb();
        statistics.inc_stats_sorting_sieve();
        status_data.gauss_data.reducedness = 2;
        size_t imin = histo_index(params.saturation_radius);
        unsigned int cumul = 0;
        for (unsigned int i=0 ; i < size_of_histo; ++i)
        {
            cumul += histo[i];
            if (i>=imin && 1.99 * cumul > GBL_saturation_histo_bound[i])
            {
                return;
            }
        }
    }   // outer while-loop (managing incrementing the db)
}


bool Siever::nv_sieve()
{
    CPUCOUNT(304);
    switch_mode_to(SieveStatus::plain);
    parallel_sort_cdb();
    size_t const S = cdb.size();
    CompressedEntry* const fast_cdb = cdb.data();
    recompute_histo();

    while(true)
    {
        size_t kk = S-1;

        for (size_t i = 0; i < S; ++i)
        {
            CompressedEntry *pce1 = &cdb[i];
            CompressedVector cv = pce1->c;
            for (size_t j = 0; j < i; ++j)
            {
                if (UNLIKELY( is_reducible_maybe<XPC_THRESHOLD>(cv, fast_cdb[j].c)))
                {
                    if (reduce_in_db(&cdb[i], &cdb[j], &cdb[kk])) kk--;
                    if (kk < .5 * S) break;
                }
            }
//            STATS(stat_P += i);
            if (kk < .5 * S) break;
        }

        pa::sort(cdb.begin(), cdb.end(), compare_CE(), threadpool);
        status_data.plain_data.sorted_until = cdb.size();

        if (kk > .8 *S) return false;

        size_t imin = histo_index(params.saturation_radius);
        long cumul = 0;

        for (size_t i=0 ; i < size_of_histo; ++i)
        {
            cumul += histo[i];
            if (i>=imin && 1.99 * cumul > std::pow(1. + i* (1./size_of_histo), n/2.) * params.saturation_ratio)
            {
                assert(std::is_sorted(cdb.cbegin(),cdb.cend(), compare_CE()  ));
                return true;
            }
        }
    }
}

// Same as reduce_in_db, but additionally returns which one of {ce1, ce2} was reduced
short Siever::gauss_no_upd_reduce_in_db(CompressedEntry *ce1, CompressedEntry *ce2)
{
    short which_one;
//    STATS(stat_F++);
    CompressedEntry* target_ptr;
    if ( ce1->len < ce2->len)
    {
        target_ptr = ce2;
        which_one = 2;
    }
    else
    {
        target_ptr = ce1;
        which_one = 1;
    }


  LFT inner = std::inner_product(db[ce1->i].yr.begin(), db[ce1->i].yr.begin()+n, db[ce2->i].yr.begin(),  static_cast<LFT>(0.));
  LFT new_l = ce1->len + ce2->len - 2 * std::abs(inner);
  int sign = inner < 0 ? 1 : -1;


  if (REDUCE_LEN_MARGIN * new_l >= target_ptr->len)
  {

    if (params.otf_lift && (new_l < params.lift_radius))
        {
            LFT otf_helper[OTF_LIFT_HELPER_DIM];
            ZT x[r];
            std::fill(x, x+l, 0);
            std::copy(db[ce1->i].x.cbegin(), db[ce1->i].x.cbegin()+n, x+l);
            std::copy(db[ce1->i].otf_helper.cbegin(), db[ce1->i].otf_helper.cbegin()+OTF_LIFT_HELPER_DIM, otf_helper);
            if(sign == 1)
            {
              for(unsigned int i=0; i < n; ++i)
              {
                x[l+i] += db[ce2->i].x[i];
              }
              for(unsigned int i=0; i < OTF_LIFT_HELPER_DIM; ++i)
              {
                otf_helper[i] += db[ce2->i].otf_helper[i];
              }

            }
            else
            {
              for(unsigned int i=0; i < n; ++i)
              {
                x[l+i] -= db[ce2->i].x[i];
              }
              for(unsigned int i=0; i < OTF_LIFT_HELPER_DIM; ++i)
              {
                otf_helper[i] -= db[ce2->i].otf_helper[i];
              }
            }
            // if (reduce_while_bucketing)
            //     statistics.inc_stats_otflifts_2outer();
            // else
            //     statistics.inc_stats_otflifts_2inner();
            lift_and_compare(x, new_l * gh, otf_helper);
        }

    return 0; // no reduction
  }
//  STATS(stat_R++);

  std::array<ZT,MAX_SIEVING_DIM> x_new = db[ce1->i].x;
  addsub_vec(x_new, db[ce2->i].x, static_cast<ZT>(sign));
  auto new_uid = uid_hash_table.compute_uid(x_new);
  if(uid_hash_table.replace_uid(db[target_ptr->i].uid, new_uid) == false)
  {
    return 0;
  }
  else
  {
    histo[histo_index(target_ptr->len)] --;
    db[target_ptr->i].x = std::move(x_new);
    db[target_ptr->i].uid = std::move(new_uid);
    recompute_data_for_entry<Recompute::recompute_all_and_consider_otf_lift & (~Recompute::recompute_uid)>(db[target_ptr->i]);
    target_ptr -> len = db[target_ptr->i].len;
    target_ptr -> c = db[target_ptr ->i].c;
    histo[histo_index(target_ptr ->len)] ++; // doing it only now, to avoid numerical error.
    return which_one;
  }

}


// Attempt reduction inside db
// If ce3 is not null, then we attempt to replace ce3,
// otherwise we attempt to replace the worse of *ce1,*ce2

CompressedEntry* Siever::reduce_in_db(CompressedEntry *ce1, CompressedEntry *ce2, CompressedEntry *target_ptr)
{
//    STATS(stat_F++);

    if (target_ptr == nullptr)
    {
        target_ptr = ce1->len < ce2->len ? ce2 : ce1;
    }

    LFT inner = std::inner_product(db[ce1->i].yr.begin(), db[ce1->i].yr.begin()+n, db[ce2->i].yr.begin(),  static_cast<LFT>(0.));
    LFT new_l = ce1->len + ce2->len - 2 * std::abs(inner);
    int sign = inner < 0 ? 1 : -1;


    if (REDUCE_LEN_MARGIN * new_l >= target_ptr->len)
    {

    if (params.otf_lift && (new_l < params.lift_radius))
        {
            LFT otf_helper[OTF_LIFT_HELPER_DIM];
            ZT x[r];
            std::fill(x, x+l, 0);
            std::copy(db[ce1->i].x.cbegin(), db[ce1->i].x.cbegin()+n, x+l);
            std::copy(db[ce1->i].otf_helper.cbegin(), db[ce1->i].otf_helper.cbegin()+OTF_LIFT_HELPER_DIM, otf_helper);

            if(sign == 1)
            {
              for(unsigned int i=0; i < n; ++i)
              {
                x[l+i] += db[ce2->i].x[i];
              }
              for(unsigned int i=0; i < OTF_LIFT_HELPER_DIM; ++i)
              {
                otf_helper[i] += db[ce2->i].otf_helper[i];
              }

            }
            else
            {
              for(unsigned int i=0; i < n; ++i)
              {
                x[l+i] -= db[ce2->i].x[i];
              }
              for(unsigned int i=0; i < OTF_LIFT_HELPER_DIM; ++i)
              {
                otf_helper[i] -= db[ce2->i].otf_helper[i];
              }
            }
            // if (reduce_while_bucketing)
            //     statistics.inc_stats_otflifts_2outer();
            // else
            //     statistics.inc_stats_otflifts_2inner();
            lift_and_compare(x, new_l * gh, otf_helper);
        }
        return nullptr;
    }
//    STATS(stat_R++);

    std::array<ZT,MAX_SIEVING_DIM> x_new = db[ce1->i].x;
    addsub_vec(x_new, db[ce2->i].x, static_cast<ZT>(sign));
    auto new_uid = uid_hash_table.compute_uid(x_new);
    if(uid_hash_table.replace_uid(db[target_ptr->i].uid, new_uid) == false)
    {
        return nullptr;
    }
    else
    {
        histo[histo_index(target_ptr->len)] --;
        db[target_ptr->i].x = std::move(x_new);
        db[target_ptr->i].uid = std::move(new_uid);
        recompute_data_for_entry<Recompute::recompute_all_and_consider_otf_lift & (~Recompute::recompute_uid)>(db[target_ptr->i]);
        target_ptr -> len = db[target_ptr->i].len;
        target_ptr -> c = db[target_ptr ->i].c;
        histo[histo_index(target_ptr ->len)] ++; // doing it only now, to avoid numerical error.
        return target_ptr;
    }
}
