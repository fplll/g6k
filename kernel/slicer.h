//
// Created by Elena Kirshanova on 07/09/2024.
//

#ifndef G6K_HYBRID_SLICER_H
#define G6K_HYBRID_SLICER_H

static constexpr unsigned int XPC_SLICER_SAMPLING_THRESHOLD = 105; // XPC Threshold for iterative slicer sampling //105
static constexpr unsigned int XPC_SLICER_THRESHOLD = 96; // XPC Threshold for iterative slicer sampling

#define REDUCE_DIST_MARGIN 1.02


struct QEntry;
class ProductLSH;

class RandomizedSlicer{

public:
    explicit RandomizedSlicer(Siever &sieve, unsigned long int seed = 0) :
            sieve(sieve), db_t(), cdb_t(), n(0), rng_t(seed), sim_hashes_t(rng_t.rng_nolock())
    {
        /*
        size_t fullS = this->sieve.cdb.size();
        Entry* e_db = &(this->sieve.db.front());
        for (size_t ii = 0; ii<fullS; ii++) {
            for (const auto &e: e_db[ii].c) { std::cout << e << " "; }
            std::cout <<e_db[ii].len << std::endl;
        }
        */
        this->n = this->sieve.n;
        sim_hashes_t.reset_compress_pos(this->sieve);
        uid_hash_table_t.reset_hash_function(this->sieve);
        //std::cout << "initialized randomized slicer" << std::endl;
    }
    ~RandomizedSlicer(){
        this->db_t.clear();
        this->cdb_t.clear();
        this->cdb_t_tmp_copy.clear();
    }

    friend SimHashes;
    friend UidHashTable;

    Siever &sieve;

    CACHELINE_VARIABLE(std::vector<Entry_t>, db_t);             // database of targets
    CACHELINE_VARIABLE(std::vector<CompressedEntry>, cdb_t);  // compressed version, faster access and periodically sorted
    CACHELINE_VARIABLE(std::vector<CompressedEntry>, cdb_t_tmp_copy);
    CACHELINE_VARIABLE(rng::threadsafe_rng, rng_t);

    unsigned int n;

    SimHashes sim_hashes_t; // needs to go after rng!
    UidHashTable uid_hash_table_t; //hash table for db_t -- the database of targets

    //FT dist_sq_bnd = 0;
    size_t threads = 1;


    thread_pool::thread_pool threadpool;
    size_t sorted_until = 0;

    void parallel_sort_cdb();

    void randomize_target_small_task(Entry_t &t);
    void grow_db_with_target(const double t_yr[], size_t n_per_target);

    bool bdgl_like_sieve(size_t nr_buckets_aim, const size_t blocks, const size_t multi_hash, LFT len_bound );
    void slicer_bucketing(const size_t blocks, const size_t multi_hash, const size_t nr_buckets_aim,
                                            std::vector<uint32_t> &buckets, std::vector<atomic_size_t_wrapper> &buckets_index);
    void slicer_bucketing_task(const size_t t_id, std::vector<uint32_t> &buckets, std::vector<atomic_size_t_wrapper> &buckets_index, ProductLSH &lsh);

    void slicer_process_buckets(const std::vector<uint32_t> &buckets, const std::vector<atomic_size_t_wrapper> &buckets_index,
                                std::vector<std::vector<QEntry>> &t_queues);
    void slicer_process_buckets_task(const size_t t_id, const std::vector<uint32_t> &buckets,
                                     const std::vector<atomic_size_t_wrapper> &buckets_index, std::vector<QEntry> &t_queue);
    std::pair<LFT, int8_t> reduce_to_QEntry_t(CompressedEntry *ce1, CompressedEntry *ce2);

    void slicer_queue(std::vector<std::vector<QEntry>> &t_queues, std::vector<std::vector<Entry_t>>& transaction_db );
    void slicer_queue_dup_remove_task( std::vector<QEntry> &queue);

    void slicer_queue_create_task( const size_t t_id, const std::vector<QEntry> &queue, std::vector<Entry_t> &transaction_db, int64_t &write_index);
    inline int slicer_reduce_with_delayed_replace(const size_t i1, const size_t i2, std::vector<Entry_t>& transaction_db, int64_t& write_index, LFT new_l, int8_t sign);
    size_t slicer_queue_insert_task( const size_t t_id, std::vector<Entry_t> &transaction_db, int64_t write_index);
    bool slicer_replace_in_db(size_t cdb_index, Entry_t &e);

    void set_nthreads(size_t nt){ this->threads = nt;}

    //void uninitialize();
};

#endif //G6K_HYBRID_SLICER_H
