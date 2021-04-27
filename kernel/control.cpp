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


#include <thread>
#include <mutex>
#include "siever.h"

#include "parallel_algorithms.hpp"
namespace pa = parallel_algorithms;

// reserves size for db and cdb. If called with a larger size than the current capacities, does nothing
void Siever::reserve(size_t const reserved_db_size)
{
    CPUCOUNT(216);
    db.reserve(reserved_db_size);
    cdb.reserve(reserved_db_size);
    // old: cdb_tmp_copy is only needed for bgj1. We delay the reservation until we need it.
    // old: if( sieve_status == SieveStatus::bgj1 )
    // new: cdb_tmp_copy is also used in parallel_sort_cdb and other functions
    cdb_tmp_copy.reserve(reserved_db_size);
}

// switches the current sieve_status to the new one and updates internal data accordingly.
// Note that, depending on the sieve_status, we keep track of the sortedness of cdb in a different way.
// (because for some sieves this encodes work that is already done and we want to keep this across calls)
bool Siever::switch_mode_to(Siever::SieveStatus new_sieve_status)
{
    CPUCOUNT(217);
    static_assert(static_cast<int>(Siever::SieveStatus::LAST) == 4, "You need to update this function");
    if (new_sieve_status == sieve_status)
    {
        return false;
    }
    cdb_tmp_copy.reserve(db.capacity());
    cdb.reserve(db.capacity());
    switch(new_sieve_status)
    {
        case SieveStatus::bgj1 :
            cdb_tmp_copy.reserve(db.capacity());
            [[fallthrough]];

        case SieveStatus::plain :
            if(sieve_status == SieveStatus::gauss || sieve_status == SieveStatus::triple_mt)
            {
                StatusData::Gauss_Data &data = status_data.gauss_data;
                StatusData new_status_data;
                assert(data.list_sorted_until <= data.queue_start);
                assert(data.queue_start       <= data.queue_sorted_until);
                assert(data.queue_sorted_until<= db_size() );
                assert(std::is_sorted(cdb.cbegin(),cdb.cbegin()+data.list_sorted_until, compare_CE()  ) );
                assert(std::is_sorted(cdb.cbegin()+data.queue_start, cdb.cbegin() + data.queue_sorted_until , compare_CE()  ));

                if(data.list_sorted_until == data.queue_start)
                {
                    cdb_tmp_copy.resize(cdb.size());
                    pa::merge(cdb.begin(), cdb.begin()+data.queue_start, cdb.begin()+data.queue_start, cdb.begin()+data.queue_sorted_until, cdb_tmp_copy.begin(), compare_CE(), threadpool);
                    pa::copy(cdb.begin()+data.queue_sorted_until, cdb.end(), cdb_tmp_copy.begin()+data.queue_sorted_until, threadpool);
                    cdb.swap(cdb_tmp_copy);
                    new_status_data.plain_data.sorted_until =  data.queue_sorted_until;
                }
                else
                {
                    new_status_data.plain_data.sorted_until = data.list_sorted_until;
                }
                status_data = new_status_data;
            }
            // switching between bgj1 and plain does nothing, essentially.
            sieve_status = new_sieve_status;
            break;

        // switching from gauss to triple_mt and vice-versa is ill-supported (it works, but it throws away work)
        case SieveStatus::gauss :
            [[fallthrough]];
        case SieveStatus::triple_mt :
            if(sieve_status == SieveStatus::plain || sieve_status == SieveStatus::bgj1)
            {
                StatusData::Plain_Data &data = status_data.plain_data;
                assert(data.sorted_until <= cdb.size());
                assert(std::is_sorted(cdb.cbegin(), cdb.cbegin() + data.sorted_until, compare_CE() ));
                StatusData new_status_data;
                new_status_data.gauss_data.list_sorted_until = 0;
                new_status_data.gauss_data.queue_start = 0;
                new_status_data.gauss_data.queue_sorted_until = data.sorted_until;
                status_data = new_status_data;
            }
            sieve_status = new_sieve_status;
            break;

        default : assert(false); break;
    }
    return true;
}

// use to indicate that the cdb is no longer sorted. In case of gauss sieves that distinguish between list and queue,
// this also sets the length of the list part to 0.
void Siever::invalidate_sorting()
{
    static_assert(static_cast<int>(Siever::SieveStatus::LAST) == 4, "You need to update this function");
    switch(sieve_status)
    {
        case SieveStatus::plain :
            [[fallthrough]];
        case SieveStatus::bgj1 :
            status_data.plain_data.sorted_until = 0;
            break;
        case SieveStatus::gauss :
            [[fallthrough]];
        case SieveStatus::triple_mt :
            status_data.gauss_data.list_sorted_until = 0;
            status_data.gauss_data.queue_start = 0;
            status_data.gauss_data.queue_sorted_until = 0;
            status_data.gauss_data.reducedness = 0;
            break;
        default: assert(false); break;
    }
}

// indicates that histo is invalid.
void Siever::invalidate_histo()
{
    histo_valid = false;
}

// Sets the number of threads our threadpool uses + master thread.
// Note that the threadpool itself uses one less thread, because there is also a master thread.
void Siever::set_threads(unsigned int nr)
{
    assert(nr >= 1);
    threadpool.resize(nr-1);
}

// Loads (full) gso of size full_n. The GSO matrix is passed as an one-dim C-Array.
// Note that this is called from the python layer with an already existing (c)db
// (GSO Updates are breaking encapsulation at the moment)
void Siever::load_gso(unsigned int full_n, double const* mu)
{
    this->full_n = full_n;
    full_muT.resize(full_n);
    full_rr.resize(full_n);

    for (unsigned int i = 0; i < full_n; ++i)
    {
        full_muT[i].resize(full_n);
        for (unsigned int j = 0; j < full_n; ++j)
        {
            full_muT[i][j] = mu[j * full_n + i];
        }
    }
    for (unsigned int i = 0; i < full_n; ++i)
    {
        full_rr[i] = full_muT[i][i];
        full_muT[i][i] = 1;
    }
    invalidate_sorting();
    invalidate_histo();
}

// initializes a local block from [l_,  r_) from the full GSO object.
// This has to be called before we start sieving.
void Siever::initialize_local(unsigned int ll_, unsigned int l_, unsigned int r_)
{
    CPUCOUNT(200);
    
    assert(l_ >= ll_);
    assert(r_ >= l_);
    assert(full_n >= r_);
    // r stays same or increases => keep best lifts
    if (r_ >= r)
    {
        best_lifts_so_far.resize(l_+1);
        for (auto& bl : best_lifts_so_far)
        {
            if (bl.len == 0.) continue;
            bl.x.resize(r_, 0);

            if (!params.lift_unitary_only) continue;
            bool unitary=false;
            for (unsigned int i = l_; i < r_; ++i)
            {
                unitary |= abs(bl.x[i])==1;
            }
            if (unitary) continue;
            bl.x.clear();
            bl.len = 0;
        }
    }
    // r shrinked
    if (r_ < r)
    {
        best_lifts_so_far.clear();
        best_lifts_so_far.resize(l_+1);
    }
    
    for (unsigned int i = 0; i < ll; ++i)
    {
        assert(best_lifts_so_far[i].len == 0);
    }
    
    for (unsigned int i = ll; i < ll_; ++i)
    {
        best_lifts_so_far[i].x.clear();
        best_lifts_so_far[i].len = 0;
    }

    l = l_;
    r = r_;
    n = r_ - l_;
    ll = ll_;

    // std::fill(histo.begin(), histo.end(), 0);
    invalidate_histo();

    //mu.resize(n);
    muT.resize(n);
    rr.resize(n);
    sqrt_rr.resize(n);

    for (unsigned int i = 0; i < n; ++i)
    {
        muT[i].resize(n);
        for (unsigned int j = 0; j < n; ++j)
        {
            muT[i][j] = full_muT[i + l][j + l];
        }
        rr[i] = full_rr[i + l];
        // Note: rr will get normalized by gh below
        // sqrt_rr is set below after normalization
    }

    // Compute the Gaussian Heuristic of the current block
    double const log_ball_square_vol = n * std::log(M_PI) - 2.0 * std::lgamma(n / 2.0 + 1);
    double log_lattice_square_vol = 0;
    for (unsigned int i = 0; i < n; ++i)
    {
        log_lattice_square_vol += std::log(rr[i]);
    }
    gh = std::exp((log_lattice_square_vol - log_ball_square_vol) / (1.0 * n));

    // Renormalize local rr coefficients
    for (unsigned int i = 0; i < n; ++i)
    {
        rr[i] /= gh;
        sqrt_rr[i] = std::sqrt(rr[i]);
    }

    set_lift_bounds();
    sim_hashes.reset_compress_pos(*this);
    uid_hash_table.reset_hash_function(*this);
    invalidate_sorting();
}

// This is run internally after a change of context / GSO.
// It assumes that the uid_hash table is empty and re-inserts every (c)db element into it.
// If collisions are found, they are replaced by fresh samples.
void Siever::refresh_db_collision_checks()
{
    CPUCOUNT(201);

    // db_uid.clear(); initialize_local calls uid_hash_table.reset_hash_function(), which clears the uid database.
    // initialize_local is always called before this function.
    // updated.clear();
    // Run collision checks
    assert(uid_hash_table.hash_table_size() == 0);

    apply_to_all_compressed_entries([this](CompressedEntry &ce)
    {
        // the cdb is not in a valid state, so do not sample by recombination
        int retry = 1;
        while (!uid_hash_table.insert_uid(db[ce.i].uid))
        {
            db[ce.i] = std::move(sample(retry));
            retry += 1;
        }
        ce.len = db[ce.i].len;
        ce.c = db[ce.i].c;
    } );
    invalidate_sorting();
    invalidate_histo();
}


// Increase the current block to the left by lp coefficients, using babai lifting
// Pretty much everything needs to be recomputed
void Siever::extend_left(unsigned int lp)
{
    CPUCOUNT(202);

    assert(lp <= l);
    initialize_local(ll, l - lp, r);

    apply_to_all_entries([lp,this](Entry &e)
        {
          // Padding with lp entries from the left. Note that these will be overwritten by the
          // babai_lifting done in recompute_data_for_entry_babai.
          // e.yr is padded from the right, but it's completely recomputed anyway.
          std::copy_backward(e.x.begin(), e.x.begin()+n-lp, e.x.begin()+n);
          std::fill(e.x.begin(), e.x.begin()+lp, 0);
          std::fill(e.x.begin()+n,e.x.end(),0);
          recompute_data_for_entry_babai<Recompute::babai_only_needed_coos_and_recompute_aggregates | Recompute::recompute_yr>(e,lp);
        } );
    invalidate_sorting();
    invalidate_histo();
    refresh_db_collision_checks();
}

void Siever::shrink_left(unsigned int lp)
{
    CPUCOUNT(204);
    initialize_local(ll, l + lp , r);
    apply_to_all_entries([lp,this](Entry &e)
        {
            std::copy(e.x.begin()+lp,e.x.begin()+lp+n,e.x.begin());
            std::fill(e.x.begin()+n,e.x.end(),0);
            recompute_data_for_entry<Recompute::recompute_yr | Recompute::recompute_len | Recompute::recompute_c | Recompute::recompute_uid | Recompute::consider_otf_lift | Recompute::recompute_otf_helper >(e);
        } );
    invalidate_sorting();
    invalidate_histo();
    refresh_db_collision_checks();
}

// Increase the current block to the left by lp coefficients, appending zeros.
void Siever::extend_right(unsigned int rp)
{
    CPUCOUNT(203);

    initialize_local(ll, l, r + rp);

    apply_to_all_entries([rp,this](Entry &e)
                          {
                            std::fill(e.x.begin()+n,e.x.end(),0);
                            recompute_data_for_entry<Recompute::recompute_yr | Recompute::recompute_len | Recompute::recompute_c | Recompute::recompute_uid | Recompute::consider_otf_lift | Recompute::recompute_otf_helper >(e);
                          } );
    refresh_db_collision_checks();
    invalidate_sorting(); // False positive hash collisions can actually invalidate sorting atm.
    invalidate_histo();
}

template<int tnold>
void Siever::gso_update_postprocessing_task(size_t const start, size_t const end, int const n_old, std::vector<std::array<ZT,MAX_SIEVING_DIM>> const &MT)
{
    ATOMIC_CPUCOUNT(218);
    assert(n_old == (tnold < 0 ? n_old : tnold));
    std::array<ZT,MAX_SIEVING_DIM> x_new; // create one copy on the stack to avoid reallocating memory inside the loop.

    for (size_t i = start; i < end; ++i)
    {
        std::fill(x_new.begin(), x_new.end(), 0);
        for(unsigned int j = 0; j < n; ++j)
        {
            x_new[j] = std::inner_product(db[i].x.begin(), db[i].x.begin()+(tnold<0?n_old:tnold), MT[j].begin(), static_cast<ZT>(0));
        }
        db[i].x = x_new;
        recompute_data_for_entry<Recompute::recompute_all_and_consider_otf_lift>(db[i]);
    }
}

void Siever::gso_update_postprocessing(const unsigned int l_, const unsigned int r_, long const * M)
{
    CPUCOUNT(208);

    const unsigned int n_old = n;

    // save old best lifts in old basis
    std::vector<Entry> oldbestlifts;
    for (auto& bl: best_lifts_so_far)
    {
        if (bl.len<=0.) continue;
        oldbestlifts.emplace_back();
        for (unsigned int j = 0; j < n; ++j)
            oldbestlifts.back().x[j] = bl.x[l + j];
    }

    best_lifts_so_far.clear();
    best_lifts_so_far.resize(l_+1);

    initialize_local(ll, l_, r_);

    std::vector<std::array<ZT,MAX_SIEVING_DIM>> MT;
    MT.resize(n);
    for (unsigned int i = 0; i < n; ++i)
        std::copy(M+(i*n_old), M+(i*n_old)+n_old, MT[i].begin());

    // retry lifting old best lifts under new basis
    for (auto& e : oldbestlifts)
    {
        auto x_new = e.x;
        std::fill(x_new.begin(), x_new.end(), 0);
        for(unsigned int j = 0; j < n; ++j)
        {
            x_new[j] = std::inner_product(e.x.begin(), e.x.begin()+n_old, MT[j].begin(), static_cast<ZT>(0));
        }

        e.x = x_new;
        recompute_data_for_entry<Recompute::recompute_all_and_consider_otf_lift>(e);
    }

    auto task = &Siever::gso_update_postprocessing_task<-1>;
    UNTEMPLATE_DIM(&Siever::gso_update_postprocessing_task, task, n_old);

    size_t const th_n = std::min(params.threads, static_cast<size_t>(1 + db.size() / MIN_ENTRY_PER_THREAD));
    threadpool.run(
        [this, task, MT, n_old](int th_i, int th_n)
        {
            pa::subrange subrange(this->db.size(), th_i, th_n);
            ((*this).*task)(subrange.first(), subrange.last(), n_old, MT);
        }, th_n);
    invalidate_sorting();
    invalidate_histo();
    refresh_db_collision_checks();
}

void Siever::lift_and_replace_best_lift(ZT * const x_full, unsigned int const i)
{
    assert(i >= ll);

    if (params.lift_unitary_only)
    {
      bool unitary = false;
      for (unsigned int ii = l; ii < r; ++ii)
      {
        unitary |= std::abs(x_full[ii]) == 1;
      }
      if (!unitary) return;
    }

    FT len_precise = 0.;
    for(unsigned int j = i; j < r; ++j)
    {
      FT yi = std::inner_product(x_full + j, x_full + r, full_muT[j].cbegin()+j, static_cast<FT>(0.));
      len_precise += yi * yi * full_rr[j];
    }
    if (len_precise >= lift_bounds[i]) return; // for loop over i

    std::lock_guard<std::mutex> lock_best_lifts(global_best_lift_so_far_mutex);

    // Make sure the condition still holds after grabbing the mutex
    if (len_precise < lift_bounds[i])
    {
        best_lifts_so_far[i].len = len_precise;
        best_lifts_so_far[i].x.resize(r);
        std::copy(x_full, x_full+r, &(best_lifts_so_far[i].x[0]));
    }
    set_lift_bounds();

}

void Siever::set_lift_bounds()
{
    assert(ll <= l);
    lift_bounds.resize(l+1);
    FT max_so_far = 0.;
    for (size_t i = ll; i <= l; ++i)
    {
        FT const bli_len = best_lifts_so_far[i].len;
        if (bli_len == 0.)
        {
            lift_bounds[i] = full_rr[i];
        }
        else
        {
            lift_bounds[i] = std::min(bli_len, full_rr[i]);
        }
        max_so_far = std::max(max_so_far, lift_bounds[i] );
    }
    lift_max_bound = max_so_far;
}




// Babai Lift and return the best vector for insertion at each index i in [0 ... r-1]
// (relatively to the full basis). the outputted vector will be expressed in the full gso basis.

void Siever::best_lifts(long* vecs, double* lens)
{
    std::fill(vecs, &vecs[(l+1) * r], 0);
    std::fill(lens, &lens[l+1], 0.);
    if (!params.otf_lift)
    {
        apply_to_all_entries([this](Entry &e) {
            lift_and_compare(e);
        }); 
    }
    for (size_t i = 0; i < l+1; ++i)
    {
        if (best_lifts_so_far[i].len==0.) continue;
        // if (best_lifts_so_far[i].len > delta * full_rr[i]) continue;
        for (size_t j = 0; j < r; ++j)
        {
            vecs[i * r + j] = best_lifts_so_far[i].x[j];
        }
        lens[i] = best_lifts_so_far[i].len;
    }
}

// sorts cdb and only keeps the best N vectors.
void Siever::shrink_db(unsigned long N)
{

    CPUCOUNT(207);
    switch_mode_to(SieveStatus::plain);
    assert(N <= cdb.size());

    if (N == 0)
    {
        cdb.clear();
        db.clear();
        invalidate_sorting();
        invalidate_histo();
        return;
    }

    parallel_sort_cdb();

    std::vector<IT> to_save;
    std::vector<IT> to_kill;
    to_save.resize(cdb.size()-N);
    to_kill.resize(cdb.size()-N);
    std::atomic_size_t to_save_size(0), to_kill_size(0);

    threadpool.run([this,N,&to_save,&to_kill,&to_save_size,&to_kill_size]
        (int th_i, int th_n)
        {
            pa::subrange lowrange(0, N, th_i, th_n), highrange(N, cdb.size(), th_i, th_n);
            auto l_it = this->cdb.begin() + lowrange.first(), l_end = this->cdb.begin() + lowrange.last();
            auto h_it = this->cdb.begin() + highrange.first(), h_end = this->cdb.begin() + highrange.last();
            // scan ranges and performs swaps on the fly
            for (; l_it != l_end; ++l_it)
            {
                if (l_it->i < N)
                    continue;
                for (; h_it != h_end && h_it->i >= N; ++h_it)
                    uid_hash_table.erase_uid(db[h_it->i].uid);
                if (h_it == h_end)
                    break;
                uid_hash_table.erase_uid(db[h_it->i].uid);
                db[h_it->i] = db[l_it->i];
                std::swap(l_it->i, h_it->i);
                ++h_it;
            }
            // remaining parts have to be saved
            std::vector<IT> tmpbuf;
            for (; l_it != l_end; ++l_it)
            {
                if (l_it->i < N)
                    continue;
                tmpbuf.emplace_back(l_it - this->cdb.begin());
            }
            size_t w_idx = to_save_size.fetch_add(tmpbuf.size());
            std::copy(tmpbuf.begin(), tmpbuf.end(), to_save.begin()+w_idx);
            tmpbuf.clear();

            for (; h_it != h_end; ++h_it)
            {
                if (h_it->i >= N)
                {
                    uid_hash_table.erase_uid(db[h_it->i].uid);
                    continue;
                }
                tmpbuf.emplace_back(h_it - this->cdb.begin());
            }
            w_idx = to_kill_size.fetch_add(tmpbuf.size());
            std::copy(tmpbuf.begin(), tmpbuf.end(), to_kill.begin()+w_idx);
            tmpbuf.clear();
        });
    assert(to_kill_size == to_save_size);
    threadpool.run([this,&to_save,&to_kill,&to_save_size,&to_kill_size](int th_i, int th_n)
        {
            std::size_t size = to_save_size;
            pa::subrange subrange(size, th_i, th_n);
            for (auto j : subrange)
            {
                std::size_t k = to_kill[j], s = to_save[j];
                uid_hash_table.erase_uid( db[ cdb[k].i ].uid );
                db[ cdb[k].i ] = db[ cdb[s].i ];
                std::swap(cdb[k].i, cdb[s].i);
            }
        });

    cdb.resize(N);
    db.resize(N);
    assert(std::is_sorted(cdb.begin(), cdb.end(), compare_CE()));
    status_data.plain_data.sorted_until = N;
    invalidate_histo();

}

// sorts the current cdb. We keep track of how far the database is already sorted to avoid resorting
// and to avoid screwing with the gauss sieves (for Gauss sieves cdb is split into a list and a queue which are
// separately sorted)
void Siever::parallel_sort_cdb()
{
    CPUCOUNT(209);
    static_assert(static_cast<int>(SieveStatus::LAST) == 4, "Need to update this function");
    if(sieve_status == SieveStatus::plain || sieve_status == SieveStatus::bgj1)
    {
        StatusData::Plain_Data &data = status_data.plain_data;
        assert(data.sorted_until <= cdb.size());
        assert(std::is_sorted(cdb.cbegin(), cdb.cbegin() + data.sorted_until, compare_CE()  ));
        if(data.sorted_until == cdb.size())
        {
            return; // nothing to do. We do not increase the statistics counter.
        }

        pa::sort(cdb.begin()+data.sorted_until, cdb.end(), compare_CE(), threadpool);
        cdb_tmp_copy.resize(cdb.size());
        pa::merge(cdb.begin(), cdb.begin()+data.sorted_until, cdb.begin()+data.sorted_until, cdb.end(), cdb_tmp_copy.begin(), compare_CE(), threadpool);
        cdb.swap(cdb_tmp_copy);
        data.sorted_until = cdb.size();
        assert(std::is_sorted(cdb.cbegin(), cdb.cend(), compare_CE() ));
        // TODO: statistics.inc_stats_sorting_overhead();
        return;
    }
    else if(sieve_status == SieveStatus::gauss || sieve_status == SieveStatus::triple_mt)
    {
        StatusData::Gauss_Data &data = status_data.gauss_data;
        assert(data.list_sorted_until <= data.queue_start);
        assert(data.queue_start <= data.queue_sorted_until);
        assert(data.queue_sorted_until <= cdb.size());
        assert(std::is_sorted(cdb.cbegin(), cdb.cbegin()+ data.list_sorted_until, compare_CE() )  );
        assert(std::is_sorted(cdb.cbegin()+ data.queue_start, cdb.cbegin() + data.queue_sorted_until, compare_CE() ));
        size_t const unsorted_list_left = data.queue_start - data.list_sorted_until;
        size_t const unsorted_queue_left = cdb.size() - data.queue_sorted_until;
        if ( (unsorted_list_left == 0) && (unsorted_queue_left == 0))
        {
            return; // nothing to do.
        }
        assert(unsorted_list_left + unsorted_queue_left > 0);
        size_t max_threads_list  = (params.threads * unsorted_list_left + unsorted_list_left + unsorted_queue_left - 1) / (unsorted_list_left + unsorted_queue_left);
        size_t max_threads_queue = params.threads - max_threads_list;
        if (unsorted_list_left > 0 && max_threads_list == 0)
            max_threads_list = 1;
        if (unsorted_queue_left > 0 && max_threads_queue == 0)
            max_threads_queue = 1;

        if (unsorted_list_left > 0 && unsorted_queue_left > 0)
        {
            // list range : [0, data.queue_start) of which [0, data.list_sorted_until) is sorted
            cdb_tmp_copy.resize(cdb.size());
            pa::sort(cdb.begin() + data.list_sorted_until, cdb.begin() + data.queue_start, compare_CE(), threadpool);
            pa::merge(cdb.begin(), cdb.begin() + data.list_sorted_until, cdb.begin() + data.list_sorted_until, cdb.begin() + data.queue_start, cdb_tmp_copy.begin(), compare_CE(), threadpool);
            // queue range: [data.queue_start, end) of which [data.queue_start, data.queue_sorted_until) is sorted
            pa::sort(cdb.begin() + data.queue_sorted_until, cdb.end(), compare_CE(), threadpool);
            pa::merge(cdb.begin() + data.queue_start, cdb.begin() + data.queue_sorted_until, cdb.begin() + data.queue_sorted_until, cdb.end(), cdb_tmp_copy.begin()+data.queue_start, compare_CE(), threadpool);
            cdb.swap(cdb_tmp_copy);
        }
        else if (unsorted_list_left > 0)
        {
            // list range : [0, data.queue_start) of which [0, data.list_sorted_until) is sorted
            cdb_tmp_copy.resize(cdb.size());
            pa::sort(cdb.begin() + data.list_sorted_until, cdb.begin() + data.queue_start, compare_CE(), threadpool);
            pa::merge(cdb.begin(), cdb.begin() + data.list_sorted_until, cdb.begin() + data.list_sorted_until, cdb.begin() + data.queue_start, cdb_tmp_copy.begin(), compare_CE(), threadpool);
            if (data.queue_start < cdb.size()/2)
                pa::copy(cdb_tmp_copy.begin(), cdb_tmp_copy.begin()+data.queue_start, cdb.begin(), threadpool);
            else
            {
                pa::copy(cdb.begin()+data.queue_start, cdb.end(), cdb_tmp_copy.begin()+data.queue_start, threadpool);
                cdb.swap(cdb_tmp_copy);
            }
        }
        else if (unsorted_queue_left > 0)
        {
            // list range : [0, data.queue_start) of which [0, data.list_sorted_until) is sorted
            cdb_tmp_copy.resize(cdb.size());
            // queue range: [data.queue_start, end) of which [data.queue_start, data.queue_sorted_until) is sorted
            pa::sort(cdb.begin() + data.queue_sorted_until, cdb.end(), compare_CE(), threadpool);
            pa::merge(cdb.begin() + data.queue_start, cdb.begin() + data.queue_sorted_until, cdb.begin() + data.queue_sorted_until, cdb.end(), cdb_tmp_copy.begin()+data.queue_start, compare_CE(), threadpool);
            if (data.queue_start > cdb.size()/2)
                pa::copy(cdb_tmp_copy.begin()+data.queue_start, cdb_tmp_copy.end(), cdb.begin()+data.queue_start, threadpool);
            else
            {
                pa::copy(cdb.begin(), cdb.begin()+data.queue_start, cdb_tmp_copy.begin(), threadpool);
                cdb.swap(cdb_tmp_copy);
            }
        }
        assert(std::is_sorted(cdb.cbegin(), cdb.cbegin() + data.queue_start, compare_CE()  ));
        assert(std::is_sorted(cdb.cbegin()+ data.queue_start, cdb.cend(), compare_CE()  ));
        data.list_sorted_until = data.queue_start;
        data.queue_sorted_until = cdb.size();
        return;
    }
    else assert(false);
}

void Siever::grow_db_task(size_t start, size_t end, unsigned int large)
{
    for (size_t i = start; i < end; ++i)
    {
        int la = large;
        for (; la < 64; ++la)
        {
            // std::cerr << la << " ";
            Entry e = sample(la);

            if (!uid_hash_table.insert_uid(e.uid)) continue;
            histo[histo_index(e.len)] ++;
            db[i] = e;
            
            CompressedEntry ce;
            ce.len = e.len;
            ce.c = e.c;
            ce.i = i;
            cdb[i] = ce;
            break;            
        }
        // std::cerr << std::endl;
        if (la >= 64)
        {
            std::cerr << "Error : All new sample collide. Oversaturated ?" << std::endl;
            exit(1);
        }
    }
}

void Siever::grow_db(unsigned long N, unsigned int large)
{
    CPUCOUNT(206);

    assert(N >= cdb.size());
    unsigned long const Nt = N - cdb.size();
    unsigned long const S = cdb.size();
    reserve(N);
    cdb.resize(N);
    db.resize(N);

    size_t const th_n = std::min(params.threads, static_cast<size_t>(1 + Nt / MIN_ENTRY_PER_THREAD));

    for (size_t c = 0; c < th_n; ++c)
    {
        threadpool.push([this,c,large, Nt, S, th_n](){this->grow_db_task( S+(c*Nt)/th_n, S+((c+1)*Nt)/th_n, large);});
    }
    threadpool.wait_work();

    // Note :   Validity of histo is unaffected.
    //          Validity of sorting is also unaffected(!) in the sense that sorted_until's remain valid.
}


void Siever::db_stats(long* cumul_histo)
{
    recompute_histo();
    for (size_t i = 0; i < size_of_histo; ++i)
    {
        cumul_histo[i] = histo[i];
        if (i) cumul_histo[i] += cumul_histo[i-1];
    }
}
