# -*- coding: utf-8 -*-

from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdint cimport int16_t, int32_t, uint64_t

cdef extern from "../kernel/siever.h" nogil:

    cdef const long int XPC_WORD_LEN
    cdef const long int XPC_BIT_LEN
    cdef const long int XPC_THRESHOLD
    cdef const long int XPC_SAMPLING_THRESHOLD
    cdef const long int XPC_BUCKET_THRESHOLD

    cdef const long int REDUCE_LEN_MARGIN
    cdef const long int REDUCE_LEN_MARGIN_HALF
    cdef const long int BGJ1_ALPHA_INIT
    cdef const long int BGJ1_ALPHA_STEP
    cdef const long int CACHE_BLOCK

    cdef const int  MAX_SIEVING_DIM

    ctypedef double FT
    ctypedef float  LFT
    ctypedef int16_t ZT
    ctypedef int32_t IT
    ctypedef uint64_t UidType

    # TODO no libcpp.array, but if need be we could define it ourselves
    # typedef std::array<uint64_t, XPC_WORD_LEN> CompressedVector

    cdef cppclass SieveStatistics:
        void clear_statistics()
        void print_statistics(int alg_id)
        string get_statistics_string(int alg)


        bool collect_statistics_reds_total
        bool collect_statistics_2reds_total
        bool collect_statistics_3reds
        bool collect_statistics_2reds_inner
        bool collect_statistics_2reds_outer
        unsigned long get_stats_2reds_inner() const
        unsigned long get_stats_2reds_outer() const
        unsigned long get_stats_3reds() const
        unsigned long get_stats_2reds_total() const
        unsigned long get_stats_reds_total() const

        bool collect_statistics_xorpopcnt_total
        bool collect_statistics_xorpopcnt_inner
        bool collect_statistics_xorpopcnt_outer
        unsigned long long get_stats_xorpopcnt_inner() const
        unsigned long long get_stats_xorpopcnt_outer() const
        unsigned long long get_stats_xorpopcnt_total() const

        bool collect_statistics_xorpopcnt_pass_total
        bool collect_statistics_xorpopcnt_pass_inner
        bool collect_statistics_xorpopcnt_pass_outer
        unsigned long long get_stats_xorpopcnt_pass_inner() const
        unsigned long long get_stats_xorpopcnt_pass_outer() const
        unsigned long long get_stats_xorpopcnt_pass_total() const

        bool collect_statistics_fullscprods_total
        bool collect_statistics_fullscprods_inner
        bool collect_statistics_fullscprods_outer
        unsigned long long get_stats_fullscprods_inner() const
        unsigned long long get_stats_fullscprods_outer() const
        unsigned long long get_stats_fullscprods_total() const

        bool collect_statistics_filter_pass
        unsigned long get_stats_filter_pass() const

        bool collect_statistics_redsuccess_total
        bool collect_statistics_2redsuccess_total
        bool collect_statistics_2redsuccess_inner
        bool collect_statistics_2redsuccess_outer
        bool collect_statistics_3redsuccess
        unsigned long get_stats_2redsuccess_inner() const
        unsigned long get_stats_2redsuccess_outer() const
        unsigned long get_stats_3redsuccess() const
        unsigned long get_stats_2redsuccess_total() const
        unsigned long get_stats_redsuccess_total() const

        bool collect_statistics_dataraces_total
        unsigned long get_stats_dataraces_total() const
        bool collect_statistics_dataraces_2inner
        unsigned long get_stats_dataraces_2inner() const
        bool collect_statistics_dataraces_2outer
        unsigned long get_stats_dataraces_2outer() const
        bool collect_statistics_dataraces_3
        unsigned long get_stats_dataraces_3() const
        bool collect_statistics_dataraces_replaced_was_saturated
        unsigned long get_stats_dataraces_replaced_was_saturated() const
        bool collect_statistics_dataraces_sorting_blocked_cdb
        unsigned long get_stats_dataraces_sorting_blocked_cdb() const
        bool collect_statistics_dataraces_sorting_blocked_db
        unsigned long get_stats_dataraces_sorting_blocked_db() const
        bool collect_statistics_dataraces_get_p_blocked
        unsigned long get_stats_dataraces_get_p_blocked() const
        bool collect_statistics_dataraces_out_of_queue
        unsigned long get_stats_dataraces_out_of_queue() const
        bool collect_statistics_dataraces_insertions
        unsigned long get_stats_dataraces_insertions() const

        bool collect_statistics_collisions_total
        unsigned long get_stats_collisions_total() const
        bool collect_statistics_collisions_2inner
        unsigned long get_stats_collisions_2inner() const
        bool collect_statistics_collisions_2outer
        unsigned long get_stats_collisions_2outer() const
        bool collect_statistics_collisions_nobucket
        unsigned long get_stats_collisions_nobucket() const
        bool collect_statistics_collisions_3
        unsigned long get_stats_collisions_3() const

        bool collect_statistics_otflifts_total
        bool collect_statistics_otflifts_2inner
        bool collect_statistics_otflifts_2outer
        bool collect_statistics_otflifts_3
        unsigned long get_stats_otflifts_2inner() const
        unsigned long get_stats_otflifts_2outer() const
        unsigned long get_stats_otflifts_3() const
        unsigned long get_stats_otflifts_total() const

        bool collect_statistics_replacements_total
        unsigned long get_stats_replacements_total() const
        bool collect_statistics_replacements_list
        unsigned long get_stats_replacements_list() const
        bool collect_statistics_replacements_queue
        unsigned long get_stats_replacements_queue() const
        bool collect_statistics_replacements_large
        unsigned long get_stats_replacements_large() const
        bool collect_statistics_replacements_small
        unsigned long get_stats_replacements_small() const

        bool collect_statistics_replacementfailures_total
        unsigned long get_stats_replacementfailures_total() const
        bool collect_statistics_replacementfailures_queue
        unsigned long get_stats_replacementfailures_queue() const
        bool collect_statistics_replacementfailures_list
        unsigned long get_stats_replacementfailures_list() const
        bool collect_statistics_replacementfailures_prune
        unsigned long get_stats_replacementfailures_prune() const


        bool collect_statistics_sorting_total
        bool collect_statistics_sorting_sieve
        unsigned long get_stats_sorting_total() const
        unsigned long get_stats_sorting_sieve() const

        bool collect_statistics_buckets
        unsigned long get_stats_buckets() const

        bool collect_statistics_memory_buckets
        unsigned long get_stats_memory_buckets() const
        bool collect_statistics_memory_transactions
        unsigned long get_stats_memory_transactions() const
        bool collect_statistics_memory_snapshots
        unsigned long get_stats_memory_snapshots() const

    cdef void show_cpu_stats()

    # Note: x and yr are std::arrays, not std::vectors...
    # This probably works, because it's translated to operator[] somewhere,
    # but the memory layout is very different.
    # Note that exporting Entry, CompressedEntry and db/cdb
    # and only used in itervalues (which is only used for debugging)
    # Consider getting rid of those exports altogether.

    cdef struct Entry:
        vector[ZT] x
        #vector[LFT] yr
        FT len
        #CompressedVector c
        #uint64_t uid

    cdef struct CompressedEntry:
    # CompressedVector c
       IT i
    #  FT len

    cdef struct LiftEntry:
        vector[ZT] x
        FT len

    # internal function, not exported
    # bool compare_CE(CompressedEntry lhs, CompressedEntry rhs)

    cdef cppclass SieverParams:
        unsigned int reserved_n
        size_t reserved_db_size
        size_t threads
        bool sample_by_sums
        bool otf_lift
        double lift_radius
        bool lift_unitary_only
        double saturation_ratio
        double saturation_radius
        double triplesieve_saturation_radius
        double bgj1_improvement_db_ratio
        double bgj1_resort_ratio
        size_t bgj1_transaction_bulk_size
        string simhash_codes_basedir
        double bdgl_improvement_db_ratio

    cdef cppclass Siever:

        ## NOTE: private members are not exported to Python (by design)

        # Global setup methods:

        Siever(const SieverParams &params)
        Siever(const SieverParams &params, unsigned long seed)
        Siever(unsigned int full_n, double* mu, const SieverParams &params)
        Siever(unsigned int full_n, double* mu, const SieverParams &params, unsigned long seed)

        bool set_params(const SieverParams &params)
        SieverParams get_params()
        void load_gso(unsigned int full_n, double* mu)

        # Local setup methods:
        void initialize_local(unsigned int ll_, unsigned int l_, unsigned int r_)
        void extend_left(unsigned int lp)
        void shrink_left(unsigned int lp)
        void extend_right(unsigned int rp)
        void grow_db(unsigned long N, unsigned int large)
        void shrink_db(unsigned long N)

        # Debug only:
        inline bool verify_integrity()

        # Supported Sieves:

        void gauss_sieve(size_t max_db_size)
        void gauss_sieve() # uses default max_db_size
        bool nv_sieve()
        void bgj1_sieve(double alpha)
        void bdgl_sieve(size_t nr_buckets, size_t blocks, size_t multi_hash)

        void hk3_sieve(double alpha)

        void best_lifts(long* vecs, double* lens)
        void db_stats(long* cumul_histo)

        # statistics and histo:
        SieveStatistics statistics
        unsigned int size_of_histo

        void reset_stats()
        size_t db_size()
        size_t histo_index(double l)

        # variables (avoid writing to them from the Python layer, if possible)

        unsigned int full_n
        vector[vector[FT]] full_muT
        vector[FT] full_rr

        unsigned int ll
        unsigned int l
        unsigned int r
        unsigned int n

        void gso_update_postprocessing(const unsigned int l_, const unsigned int r_, long* M)


        vector[vector[FT]] muT
        vector[FT] rr
        # vector[FT] sqrt_rr # will be made private

        FT gh

        vector[Entry] db
        vector[CompressedEntry] cdb

        # best_lifts_so_far is private
