//
// Created by Elena Kirshanova on 07/09/2024.
//

#ifndef G6K_HYBRID_SLICER_H
#define G6K_HYBRID_SLICER_H

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
        std::cout << "initialized randomized slicer" << std::endl;
    }

    friend SimHashes;
    friend UidHashTable;

    Siever &sieve;

    CACHELINE_VARIABLE(std::vector<Entry_t>, db_t);             // database of targets
    CACHELINE_VARIABLE(std::vector<CompressedEntry>, cdb_t);  // compressed version, faster access and periodically sorted
    CACHELINE_VARIABLE(rng::threadsafe_rng, rng_t);

    unsigned int n;

    SimHashes sim_hashes_t; // needs to go after rng!
    UidHashTable uid_hash_table_t; //hash table for db_t -- the database of targets
    //Siever* sieve;

    void randomize_target_small_task(Entry_t &t);
    //void grow_db_with_target(std::array<LFT,MAX_SIEVING_DIM> &t_yr, size_t n_per_target);
    void grow_db_with_target(const double t_yr[], size_t n_per_target);


    //FT iterative_slice( std::array<LFT,MAX_SIEVING_DIM>& t_yr, size_t max_entries_used=0);
    //void randomize_target(std::array<LFT, MAX_SIEVING_DIM>& t_yr, size_t k );
    //void randomize_target_small(std::array<LFT, MAX_SIEVING_DIM> &t_yr, unsigned int debug_directives);
    //void randomized_iterative_slice( float* t_yr, size_t max_entries_used=0, size_t samples=1, float dist_sq_bnd=-1.0, unsigned int debug_directives = 873 );

};

#endif //G6K_HYBRID_SLICER_H
