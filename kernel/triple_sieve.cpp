#include "siever.h"
#include "iostream"
#include "fstream"
#include <numeric>
#include <atomic>
#include <thread>
#include <mutex>


// reduce x1 which has maximal length
// xixj = <xi, xj>
// if new_l < x1->len, replaces x1 in db and cdb by a shorter vector

short Siever::reduce_in_db_which_one_with_len(CompressedEntry *ce1, CompressedEntry *ce2, LFT inner)
{
  short which_one;
//  STATS(stat_F++);
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

  LFT new_l = ce1->len + ce2->len - 2*std::abs(inner);
  if (REDUCE_LEN_MARGIN * new_l >= target_ptr->len)
  {
    return 0; // no reduction
  }
  int sign = inner < 0 ? 1 : -1;
//  STATS(stat_R++);

  std::array<ZT,MAX_SIEVING_DIM> x_new = db[ce1->i].x;
  addmul_vec(x_new, db[ce2->i].x, static_cast<ZT>(sign));
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


bool Siever::reduce_ordered_triple_in_db (CompressedEntry* x1, CompressedEntry* x2, CompressedEntry* x3, LFT x1x2, LFT x1x3, LFT x2x3)
{
    // deduce the correct sign for 3-reduction x1 +\- x2 +\- x3
    int sign_x1x2 = (0 < x1x2) - (x1x2 < 0);
    int sign_x1x3 = (0 < x1x3) - (x1x3 < 0);
    int sign_x2x3 = (0 < x2x3) - (x2x3 < 0);

    LFT new_l = x1->len + x2->len + x3->len + 2 * ( (-sign_x1x2 * x1x2) + (-sign_x1x3 * x1x3)+ (-sign_x2x3 * x2x3) );

    //std::cout << sign_x1x2 << " " << sign_x1x3 << " " << sign_x2x3 << std::endl;
    //std::cout << "3 reduction " << "x1: " << x1->len << " x2: " << x2->len << " x3: " << x3->len <<std::endl;
    //std::cout << "old len: " << x1->len << " new len: " << new_l << std::endl;

    //assert(new_l > 0);
    if (REDUCE_LEN_MARGIN * new_l >= x1->len) // we cannot guarantee that x1 != +/-x2 +/- x3; if this happens, new_l is very close but not equal 0 (and can be of any sign)
    {
//        STATS(stat_lenfail++);
        return false;
    }


    std::array<ZT,MAX_SIEVING_DIM> x_new = db[x1->i].x;
    addmul_vec(x_new, db[x2->i].x, static_cast<ZT>( -sign_x1x2 ));
    addmul_vec(x_new, db[x3->i].x, static_cast<ZT>( -sign_x1x3 ));

    auto new_uid = uid_hash_table.compute_uid(x_new);
    if (std::round(new_l)==0) assert(new_uid == 0);
    if(uid_hash_table.replace_uid(db[x1->i].uid, new_uid) == false)
    {
      return false;
    }
    else
    {
//      STATS(stat_R3++);
      histo[histo_index(x1->len)] --;
      db[x1->i].x = std::move(x_new);
      db[x1->i].uid = std::move(new_uid);
      recompute_data_for_entry<Recompute::recompute_all_and_consider_otf_lift & (~Recompute::recompute_uid)>(db[x1->i]);
      x1 -> len = db[x1->i].len;
      x1 -> c = db[x1 ->i].c;
      histo[histo_index(x1 ->len)] ++; // doing it only now, to avoid numerical error.
      return true;

    }
}


// Deduces with point out of the three {ce1, filtered1, filtered2} has maximal length
// which_one is changed accordingly and reduce_ordered_triple is called
bool Siever::order_and_reduce_triple(CompressedEntry* ce1, FilteredCompressedEntry* filtered1, FilteredCompressedEntry* filtered2, short &which_one)
{
    // we know <c1,filtered1>, <c1, filtered2>
    // remains to compute <filtered1,filtered2>
    LFT c2c3  = std::inner_product(db[filtered1->compressed_copy.i].yr.begin(), db[filtered1->compressed_copy.i].yr.begin()+n, db[filtered2->compressed_copy.i].yr.begin(),  static_cast<LFT>(0.));

    if (filtered1->simhash_flipped * filtered2->simhash_flipped * c2c3 > 0)
    {
//        STATS(stat_signfail++); // many failures come from here. Increasin XPC_THRESHOLD_TRIPLE_INNER_CHECK reduces the number of failures, but also increases RT
        return false;
    }

    // discover the max out of the triple
    //which_one = (ce1->len > filtered1->compressed_copy.len) ? 0 : 1;
    //which_one = filtered2 -> compressed_copy.len > ( (which_one == 0) ? (ce1 ->len) : (filtered1 -> compressed_copy.len)  ) ? 2 : which_one;

    // which_one can be either:
    //  0 -> p is the largest
    //  1 -> x1 is the largest
    //  2 -> x2 is the largest
    //  3 -> either x1 or x2 is the largest; compare in this case
    which_one = filtered1->is_p_shorter + 2*filtered2->is_p_shorter;
    //std::cout << "which_one: " << which_one << std::endl;
    switch(which_one)
    {
      case 0:
      {
        return reduce_ordered_triple_in_db(ce1, &(cdb[filtered1->index_in_cdb]), &(cdb[filtered2->index_in_cdb]), filtered1->inner_prod,filtered2->inner_prod, c2c3);
      }
      case 1:
      {
        return reduce_ordered_triple_in_db(&(cdb[filtered1->index_in_cdb]), ce1, &(cdb[filtered2->index_in_cdb]), filtered1->inner_prod, c2c3,  filtered2->inner_prod);
      }
      case 2:
      {
        return reduce_ordered_triple_in_db(&(cdb[filtered2->index_in_cdb]), ce1, &(cdb[filtered1->index_in_cdb]), filtered2->inner_prod, c2c3, filtered1->inner_prod);
      }
      case 3:
      {
        if (filtered2 -> compressed_copy.len > filtered1->compressed_copy.len)
        {
          which_one = 2;
          return reduce_ordered_triple_in_db(&(cdb[filtered2->index_in_cdb]), ce1, &(cdb[filtered1->index_in_cdb]), filtered2->inner_prod, c2c3, filtered1->inner_prod);
        }
        else
        {
          which_one = 1;
          return reduce_ordered_triple_in_db(&(cdb[filtered1->index_in_cdb]), ce1, &(cdb[filtered2->index_in_cdb]), filtered1->inner_prod, c2c3,  filtered2->inner_prod);
        }
      }
    }
    // should not be reached
    assert(false); // __builtin_unreachable;
}



// Triple-sieve the current database
// Does not use updated!
/*  Internals:

    For all cdb[i], finds all vectors cdb[j] with abd(<cdb[i], cdb[j]>) > 1/3
    and stores all such cdb[j] in filtered_cdb;
    Check cdb[i], filtered_cdb[j], filtered_cdb[k] for 3-reduction
*/

// single-threaded only
void Siever::gauss_triple_sieve_st(size_t max_db_size)
{
    using std::swap;
    switch_mode_to(SieveStatus::gauss);
    if(status_data.gauss_data.reducedness < 3)
        invalidate_sorting();
    status_data.gauss_data.reducedness = 3;
    recompute_histo();

    //size_t S = cdb.size();
    std::vector<FilteredCompressedEntry> filtered_cdb; // filtered vector of compressed points; for triple sieve
    filtered_cdb.reserve(300);

    if (max_db_size==0)
    {
        max_db_size = 4 * std::pow(1.3, n/2.) + 4*n;
    }

    for (unsigned int i = 0; i < size_of_histo; ++i)
    {
        GBL_saturation_histo_bound[i] = std::pow(1. + i * (1./size_of_histo), n/2.) * params.saturation_ratio + 10;
    }
    int iter = 0;

    // We treat the main cdb as consisting of two parts: The beginning of cdb consists of elements, where (almost) all possible reductions (except those missed due to SimHashes etc)
    // have already been performed. The end of cdb consists of elements which may still participate in reductions. We need to only compare elements from the beginning part with those from the end part.
    // To change the status of an element wrt this distinction, we perform a swap in cdb.
    size_t queue_begin = status_data.gauss_data.queue_start; // We are guaranteed that all elements cdb[0],...,cdb[queue_begin-1] are already reduced wrt each other.
    parallel_sort_cdb();


    // termination condition outer loop
    while(cdb.size() <= max_db_size)
    {
        size_t const old_S = status_data.gauss_data.queue_start;
        if (iter) grow_db(cdb.size()*1.02 + 10);
        ++iter;
        pa::sort(cdb.begin() + old_S, cdb.end(), compare_CE(), threadpool);

        CompressedEntry* const fast_cdb = cdb.data();

         // while there is no elements in the 'queue'-part of the list
        while (queue_begin < cdb.size())
        {
            size_t const p_index = queue_begin; // remember p_index for the swap at the end of the for-loop
            CompressedEntry * const pce1 = &fast_cdb[p_index];
start_over:
            // the pointer is non-const; moved inside start_over; otherwise p_entry below becomes invalid if p was previously reduced
            auto p_compressed = pce1->c; // compressed vector of p, make a copy on the stack

            Entry p_entry = db[pce1->i]; // e of p (copied)
            LFT p_len = pce1->len;   // len of p (copied)


            filtered_cdb.clear();
            //for (size_t j = 0; j < queue_begin; ++j)
            for (size_t x1_index = 0; x1_index < queue_begin; ++x1_index) // Note: We use queue_begin here, see comment below when performing 3-red
            {
                // check for 3-reduction
                if( UNLIKELY(is_reducible_maybe<XPC_THRESHOLD_TRIPLE>(&(p_compressed.front()), &(fast_cdb[x1_index].c.front()) ) ) )
                {
                    bool x1_reduced = false; // if fast_cdb[j] will be reduced, flip to true
//                    STATS(++stat_F3);  // statistics collection
                    // compute the inner-product <fast_cdb[i], fast_cdb[j]> exactly
                    LFT const scalar_prod_p_x1 = std::inner_product(p_entry.yr.begin(), p_entry.yr.begin()+n, db[fast_cdb[x1_index].i].yr.begin(),  static_cast<LFT>(0.));
                    LFT const scalar_prod_p_x1_squared = (scalar_prod_p_x1 * scalar_prod_p_x1);

                    // Check for 2-reduction. Since we compute the exact inner product to store in the filtered db in any case,
                    // we can use that directly for the 2-reduction check. There is no reason to use is_reducible_maybe.
                    LFT const x1_len = fast_cdb[x1_index].len;

                    // 2-reduction - check
                    bool const p_shorter_x1 = (p_len < x1_len);
                    LFT new_len = x1_len + p_len - 2*std::abs(scalar_prod_p_x1);
                    assert(new_len > 0);

                    if( new_len * REDUCE_LEN_MARGIN < (p_shorter_x1 ? x1_len : p_len))
                    {
                        short which_one = reduce_in_db_which_one_with_len(pce1, &(fast_cdb[x1_index]), scalar_prod_p_x1);
                        switch(which_one)
                        {
                          case 1:
                          {
                            goto start_over; // x1 was reduced
                          }
                          case 2:
                          {
                            --queue_begin;
                            swap(fast_cdb[x1_index], fast_cdb[queue_begin]);
                            --x1_index; // consider not doing that
                            continue;
                          }
                        }
                      continue; // can only happen because of uid - collision.
                    }
                    // if <fast_cdb[i], fast_cdb[j]> > 1/3 (everything squared and normalized),
                    // consider fast_cdb[j] as a candidate for 3-reduction
                    if (scalar_prod_p_x1_squared > X1X2 * p_len * x1_len ) //X1X2 is a threshold constant
                    {
//                        STATS(stat_exactF++);   // statistics collection

                        // create a filtered point out of fast_cdb[j]
                        FilteredCompressedEntry x1_filtered;
                        x1_filtered.compressed_copy = fast_cdb[x1_index];
                        x1_filtered.inner_prod = scalar_prod_p_x1;
                        x1_filtered.index_in_cdb = x1_index;
                        x1_filtered.simhash_flipped = 1;
                        x1_filtered.is_p_shorter = p_shorter_x1;

                        // flip the bits of the compressed vector if the inner-product with p is negative
                        if (x1_filtered.inner_prod < 0)
                        {
                          for (size_t i = 0; i<XPC_WORD_LEN; i++)
                          {
                            x1_filtered.compressed_copy.c[i] = ~ x1_filtered.compressed_copy.c[i];
                          }
                          x1_filtered.simhash_flipped = - 1;

                        }

                        // go through filtered_cdb to search for a good triple
                        //In case cdb[i] is reduced: we replace it with a new vector and restart
                        //In case filtered1 is reduced: we *do not* add it to filtered_list and simply continue
                        //In case filtered2 (already in the filtered_list) is reduced: replace filtered2 by the new filtered1

//                        size_t const filteredSize = filtered_cdb.size(); // consider using a fixed bound for the loop, however, this prevents deleting (via swap and pop_back() ) elements.
                        for (unsigned int x2_index = 0; x2_index < filtered_cdb.size(); ++x2_index)
                        {

                            //if( UNLIKELY(is_reducible_maybe<XPC_THRESHOLD_TRIPLE>(&(x1_filtered.compressed_copy.c.front() ), &(filtered_cdb[x2_index].compressed_copy.c.front()) )))
                            if( UNLIKELY(is_far_away<XPC_THRESHOLD_TRIPLE_INNER_CHECK>(&(x1_filtered.compressed_copy.c.front() ), &(filtered_cdb[x2_index].compressed_copy.c.front()) )))
                            {
                                short which_one;
                                if(order_and_reduce_triple(pce1, &x1_filtered, &(filtered_cdb[x2_index]), which_one))
                                {
                                    // if cv was max in the triple, restart TODO: WE SHOULD RATHER COMPARE LENGTHS
                                    if (which_one == 0)
                                    {
                                        goto start_over;
                                    }
                                    // if cv2 was max, flag cv2_reduced as invalid to prevent adding it to filtered_cdb
                                    else if (which_one == 1)
                                    {
                                        x1_reduced = true;
                                        --queue_begin;
                                        swap(fast_cdb[x1_index],fast_cdb[queue_begin]);
                                        --x1_index;
                                        break; // terminate loop over x2, the --j means we choose the element just before queue_begin as the next x1
                                    }
                                    // if fit was max in the triple, mark it as invalid
                                    else // if (filtered_cv2.compressed_copy.len != (&(filtered_cdb[fi]))->ce->len)
                                    {
                                        // Note: It is important that this does not destroy index_in_cdb of points inside filtered_cdb.
                                        // We clearly have filtered_cdb[x2_index] affected, this is dealt by the deletion below.
                                        // fast_cdb[queue_begin-1] is guaranteed not to be in filtered list, because that only contains indexes up to j, which in turn is bounded by queue_begin.

                                        auto index_to_mark_updated = filtered_cdb[x2_index].index_in_cdb;
                                        --queue_begin;
                                        swap(fast_cdb[index_to_mark_updated],fast_cdb[queue_begin]);

                                        // We delete filtered_cdb from the filtered list by swapping to the end.

                                        swap(filtered_cdb[x2_index], filtered_cdb[filtered_cdb.size()-1]);
                                        filtered_cdb.pop_back();
                                        --x2_index; // to make the next iteration use the point we swapped in.
                                    }
                                }
                            }
                        }
                        // if cv2 was not reduced, we append filtered_cv2 to filtered_cdb
                        if (!x1_reduced)
                        {
                            filtered_cdb.emplace_back(std::move(x1_filtered));
                        }
                    } // if exact check passes

                } // if 3-reduction POPCNT-check

            } // for-loop over j (i.e. the index of x1)

            // swap p with queue_begin which now indicates the very last swap

            swap(fast_cdb[queue_begin],fast_cdb[p_index]);
            ++queue_begin;
            filtered_cdb.clear();

        } // while (queue_begin < cdb.size())

        pa::sort(cdb.begin(), cdb.end(), compare_CE(), threadpool);
        status_data.gauss_data.list_sorted_until = cdb.size();
        status_data.gauss_data.queue_start = cdb.size();
        status_data.gauss_data.queue_sorted_until = cdb.size();


        size_t imin = histo_index(params.triplesieve_saturation_radius);
        unsigned int cumul = 0;
        for (unsigned int i=0 ; i < size_of_histo; ++i)
        {
            cumul += histo[i];
            if (i>=imin && 1.99 * cumul > GBL_saturation_histo_bound[i])
            {
                return;
            }
        }

    } // while(cdb.size() <= max_db_size)

}
