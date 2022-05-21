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


#ifndef G6K_SIEVER_H
#define G6K_SIEVER_H

#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <set>
#include <array>
#include <type_traits>
#include <cstdint>
#include <mutex>
#include <cstring>
#include <limits>
#include "random.hpp"
#include "thread_pool.hpp"
#include "fstream"
#include <condition_variable>
#include "compat.hpp"
#include "statistics.hpp"
#include "g6k_config.h"

using std::size_t;

// Macros
#include "untemplate.mac"   // This defines UNTEMPLATE_DIM, which is used to dispatch a function to
                            // one of many template specializations with hardwired dimension. This
                            // is done for some function calls as an optimization. Note that
                            // UNTEMPLATE_DIM is only active if TEMPLATED_DIM is set, which is
                            // activated by the build environment with ./configure
                            // --enable-templated-dim

// MACRO to enable CPU cycle counters. Use ATOMIC_CPUCOUNT for threaded functions.
// Used counters should be registered in cpuperf.cpp.
// DO NOT ENABLE THIS MANUALLY, use './configure --enable-cpucounters'
/****  PERFORMANCE_COUNTING *****/
#ifdef PERFORMANCE_COUNTING
#include "cpuperformance.hpp"
#define CPUCOUNT(s) cpu::update_performance_counter _cpuperf##s(perfcounters[s]);
#define ATOMIC_CPUCOUNT(s) cpu::update_atomic_performance_counter _atomic_cpuperf##s(atomic_perfcounters[s]);
extern std::vector<uint64_t> perfcounters;
extern std::vector<std::atomic<uint64_t>> atomic_perfcounters;
extern cpu::performance_counter_manager perfmanager;

#else
#define CPUCOUNT(s)
#define ATOMIC_CPUCOUNT(s)
#endif
void show_cpu_stats();

// CACHELINE and EXPECT, (UN)LIKELY macros moved to compat.hpp (to make them available for statistics.hpp)
// MAYBE-TODO: Move all global macros to own file.

/**
    Global constants and typedefs
*/

// Maximum dimension of the local blocks we sieve in.
#ifndef MAX_SIEVING_DIM
#define MAX_SIEVING_DIM 128
#endif

#ifndef XPC_THRESHOLD
#define XPC_THRESHOLD 96
#endif

#ifndef XPC_BUCKET_THRESHOLD
#define XPC_BUCKET_THRESHOLD 102
#endif

#ifndef OTF_LIFT_HELPER_DIM
#define OTF_LIFT_HELPER_DIM 16
#endif

static constexpr unsigned int XPC_WORD_LEN = 4; // number of 64-bit words of simhashes
static constexpr unsigned int XPC_BIT_LEN = 256; // number of bits for simhashes
static constexpr unsigned int XPC_SAMPLING_THRESHOLD = 96; // XPC Threshold for partial sieving while sampling
static constexpr unsigned int XPC_THRESHOLD_TRIPLE = 97; // XPC Threshold for triple sieve
static constexpr unsigned int XPC_THRESHOLD_TRIPLE_INNER_CHECK = 133; // XPC Threshold for triple sieve in the inner-loop
static constexpr float X1X2 = 0.108;  // Threshold to put vector in filtered list ~(1/3)^2

static constexpr unsigned int MIN_ENTRY_PER_THREAD = 100; // factor that determines minimum size of work batch to distribute to a thread.


#define REDUCE_LEN_MARGIN 1.01          // Minimal improvement ratio to trigger a reduction
                                        // (make sure of worthy progress, avoid infinite loops due to numerical-errors)

#define REDUCE_LEN_MARGIN_HALF 1.005    // Minimal improvement ratio to validate a reduction

#define CACHE_BLOCK 512                 // Local loops length for cache-friendlyness. Note that triple_sieve_mt has its separate variable for that.

#define VERBOSE false

typedef float LFT;       // Low Precision floating points for vectors yr
typedef double FT;       // High precision floating points for vectors y
typedef int16_t ZT;      // Integer for vectors x (i.e. coefficients of found vectors wrt the given basis)
typedef uint32_t IT;     // Index type for indexing into the main database of vectors (32 bits for now, limiting db_size to 2^32-1)

typedef std::array<uint64_t, XPC_WORD_LEN> CompressedVector;    // Compressed vector type for XOR-POPCNT

// made a typedef to make it better customizable. (If we increase the size of IT from 32 bits, we
// want to increase this accordingly).

using SimHashDescriptor = unsigned[XPC_BIT_LEN][6]; // typedef to avoid some awkward declarations.

// forward declarations:
struct Entry;
struct QEntry;
struct CompressedEntry;
struct atomic_size_t_wrapper;
class Siever;
class UidHashTable;
class SimHashes;
class ProductLSH;

using UidType = uint64_t; // Type of the hash we are using for collision detection.

/**
    UidHashTable class

    The class UidHashTable encapsulates the hash table that we use to detect collisions.
    Each point that we create has a hash, called uid, of type UidType and we ensure that the uids of
    all point in existance at a given point of time are distinct.

    The UidHashTable class stores the both description of the hash function and the hash tables
    and contains functions to compute hashes and insert / remove them from the database.

    For implementation reasons, our hashes are linear and the hash table takes care about +/- symmetry.
    0 is always a member of the hash table that cannot be removed.

    Usage notes:

    Before any usage, reset_hash_function(siever) has to be run, which sets up a random hash function.
    The hash function assumes that the input has the correct size, which is set during reset_hash_function(siever).
    In particular, the hash function has to be reset after a change of dimension.

    To compute the uid / hash of a given Entry e, use compute_uid(e.x). The hash function operates on
    the x-coos of Entries. It is possible to insert such hashes, checks whether a hash is already
    present and erase hashes. All operations are thread-safe unless explicitly said otherwise.

    uid received by compute_uid should not be used for anything other than for calls into the uid database or
    to make use of linearity of compute_uid.
    We guarantee that the hash function is invariant under negation in the following sense:
    After performing uid = compute_uid(x), uid' = compute_uid(-x) and a successful hash_table.insert(uid),
    hash_table.check(uid') will return true. [ Note: To preserve linearity, we cannot have uid == uid'.]
*/

class UidHashTable
{
public:
    // creates a dummy hash table. Note that reset_hash_function has to be called at least once before it can be used.
  explicit UidHashTable() : db_uid(), n(0), uid_coeffs()
  {
    insert_uid(0);
  }
  // resets the hash_function used. Siever is changed, because it uses it as a randomness source.
  // This also clears the database.
  // NOT THREAD-SAFE (including the randomness calls).
  inline void reset_hash_function(Siever& siever);

  // Compute the uid of x using the current hash function.
  inline UidType compute_uid(std::array<ZT,MAX_SIEVING_DIM> const &x) const;

  inline bool check_uid_unsafe(UidType uid); // checks whether uid is present without locks. Unsafe if other threads are writing to the hash table.
  inline bool check_uid(UidType uid);   // checks whether uid is present in the table. Avoid in multi-threaded contexts.
  inline bool insert_uid(UidType uid);  // inserts uid into the hash table. Return value indicates success (i.e. returns false if uid was present beforehand)
  inline bool erase_uid(UidType uid);   // removes uid from the hash table. return value indicates success (i.e. if it was present at all)
  inline size_t hash_table_size();      // returns the number of stored hashes (excluding 0). NOT THREAD-SAFE

  // If possible, atomically {removes removed_uid and inserts new_uid}:
  // If both new_uid is not yet present and removed_uid is present, performs the change, otherwise does nothing.
  // Return indicates success.
  // If removed_uid == new_uid, returns false [This should only happen with false positive collisions and that way is simpler to implement]
  inline bool replace_uid(UidType removed_uid, UidType new_uid);

private:
    // Note : Implementation is subject to possible changes

    // we split the uid hash table into DB_UID_SPLIT many sub-tables, each with their own mutex.
    static constexpr unsigned DB_UID_SPLIT = 8191;
    // obtain the sub-table for a given uid.
    // Note that since uid is linear, we have to somehow deal with +/- invariance on the inserting / erasure / checking layer.
    // We can either insert both uid and -uid into the database or we preprocess every incoming uid
    // by a +/- invariant function.
    // Since we do not want to lock twice as many mutexes (with a deadlock-avoiding algorithm!),
    // we opt to preprocess all incoming uids rather than insert both +uid and -uid, because we need
    // some form of preprocessing for the sub-table selection anyway.
    // Furthermore, in multi-threaded contexts, we do not use check anyway (but try to insert and
    // insert and handle the case where this fails), so optimizing for checks is pointless.
    static void normalize_uid(UidType &uid)
    {
        static_assert(std::is_unsigned<UidType>::value, "");
        if (uid > std::numeric_limits<UidType>::max()/2  + 1)
        {
            uid = -uid;
        }
    }

    // Note: The splitting is purely to allow better parallelism.
    struct padded_map: std::unordered_set<UidType> { cacheline_padding_t pad; };
    std::array<padded_map, DB_UID_SPLIT> db_uid;     // Sets of the uids of all vectors of the database. db_uid[i] contains only vectors with (normalized) uid % DB_UID_SPLIT == i.
    struct padded_mutex: std::mutex { cacheline_padding_t pad; };
    std::array<padded_mutex, DB_UID_SPLIT> db_mut; // array of mutexes. db_mut[i] protects all access to db_uid[i].    unsigned int n; // dimension of points in the domain of the hash function.
    unsigned int n; // dimension of points in the domain of the hash function.
                    // The hash function only works on vectors of this size
    std::vector<UidType> uid_coeffs;        // description of the hash function.
};


/**
    This class stores the hash function we use to compute simhashes and allows to compute a simhash for a given point.
    Siever stores an object of this type. All db and cdb entries store their simhash in them.
**/
class SimHashes
{
private:
    SimHashDescriptor compress_pos;   // Indices to chose for compression / SimHashes
    unsigned int n;                 // Dimension of the entries on which we compute SimHashes.
    std::mt19937_64 sim_hash_rng;   // we use our own rng, seeded during construction.
                                    //(This is to make simhash selection less random in multithreading and to actually simplify some internal code)
public:
    // constructs a SimHashes objects and stores the seed used to set future hash functions.
    // Note that reset_compress_pos must be called at least once before compress is used.
    explicit SimHashes(uint64_t seed) : sim_hash_rng(seed) {}


    // recomputes the sparse vector defining the current compression function / Simhash.
    // This is called during changes of context / basis switches.
    // Note that this makes a recomputation of all the simhashes stored in db / cdb neccessary.
    inline void reset_compress_pos(Siever const &siever);

    // Compute the compressed representation of an entry.
    // Since it is a function of the normalized GSO coos, we pass the yr - coos.
    inline CompressedVector compress(std::array<LFT,MAX_SIEVING_DIM> const &yr) const;
};

/**
    struct Entry:

    A vector with two representations, and side information.
    This is used to store the actual points of our main database db.

    The "key" entry to Entries is x, which stores the coordinate wrt to the basis B. All other data
    are computed from x. The other data are guaranteed to be consistent with x (and the sieve context
    defining the GSO) except during a change of context.

    When modifying points, there is the recompute_data_for_entry member function template of Siever to
    restore the invariants. Also, we require that the sizes of x and yr are always identical and
    identical to n. Since we use a fixed-length array, this means that only the first n entries are
    meaningful.
**/

struct Entry
{
    std::array<ZT,MAX_SIEVING_DIM> x;       // Vector coordinates in local basis B.
    std::array<LFT,MAX_SIEVING_DIM> yr;     // Vector coordinates in gso basis renormalized by the rr[i] (for faster inner product)
    CompressedVector c;                     // Compressed vector (i.e. a simhash)
    UidType uid;                            // Unique identifier for collision detection (essentially a hash)
    FT len = 0.;                            // (squared) length of the vector, renormalized by the local gaussian heuristic
    std::array<LFT,OTF_LIFT_HELPER_DIM> otf_helper; // auxiliary information to accelerate otf lifting of pairs
};

/**
    Lifted entries. The x coordinates are wrt global basis.
    x is only valid if len > 0.
    If len <= 0. x is typically empty.
*/

struct LiftEntry
{
    std::vector<ZT> x;      // Vector coordinates in basis B if len > 0.
    FT len = 0.;            // (squared) length of the vector, renormalized by the local gaussian heuristic
};


/**
    struct CompressedEntry:

    A compressed vector, and pointer (or index) to the full vector of type Entry.
    This is used to store the compressed points of our main compressed database cdb.
    For the most part of the algorithm, we only access cdb and hence access CompressedEntries.

    Note that CompressedEntries ce are only ever encountered as elements cdb[i]. We have that ce.c
    and ce.len should match db[ce.i].c and db[ce.i].len

    TODO: Rename i into db_index
    TODO: Rename c
    TODO: Rename len into square_len.
*/
struct CompressedEntry {
    CompressedVector c;     // Store the compressed vector here to avoid an indirection
    IT i;                   // Index of the non compressed entry
    LFT len;                // Sorting Value
};

// Define an ordering for sorting Compressed Entries, used for sorting.
struct compare_CE
{
    bool operator()(const CompressedEntry& lhs, const CompressedEntry& rhs) const { return lhs.len < rhs.len; }
};

/**
    An elements in the filtered_list (only used in single-threaded triple sieve)
    TODO: Remove from here and make local
*/

struct FilteredCompressedEntry{
    CompressedEntry compressed_copy;    // copy of an element from cdb
    LFT inner_prod;                     // inner-product of filtered_point with p
    IT index_in_cdb;                    // index of compressed_copy in cdb (for faster access)
    int simhash_flipped;
    bool is_p_shorter;
};

// std:: containers prefer exception guarantees over performance if the move constructor /
// assignment is not noexcept. Whether the implicitly generated functions are noexcept
// depends on various things. Should not be an issue in C++17. Not taking chances : these will
// trigger a compile-time error rather than slow down performance if violated by a change of Entry.
// NOTE: might fail if destructor is non-trivial (GCC bug 51452, LWG issue 2116)
static_assert(std::is_nothrow_move_assignable<Entry>::value, "Entry not nothrow-move-assignable");
static_assert(std::is_nothrow_move_constructible<Entry>::value, "Entry not nothrow-movable");

/**
    This class stores (modifyable) parameters of the Sieve.
    The main Sieve class contains an object of this type.
    This is a separate class to allow a better Python interface.

    TODO: Unify the params for the various sieves and ensure consistency with the Python layer.
    (IMPORTANT!!!)
*/

class SieverParams
{
public:
  unsigned int reserved_n = 0;   // number of coordinates to pre-reserve memory for when
                                 // creating new entries.
                                 // Note that this variable currently is only used on the Python layer.
  size_t reserved_db_size; // number of pre-allocated elements in the databases

  size_t threads = 1;  // ... number of threads

  bool sample_by_sums = true;

  bool otf_lift = true;       // Lift on the fly. If set to false,
                              // best_lifts_so_far is not populated and hence invalid.
  double lift_radius = 1.8;   // Below this radius (wrt to the gh of the context), lift everything
                              // that is visited, even if it is not going to be inserted into the
                              // databases
  bool lift_unitary_only = true;   // Only lift vectors that have a +/- 1 coefficient
                                   // in x

  double saturation_ratio = 0.5;      // Stop the sieve when reaching #vectors/E[#vectors] in
                                      // database at any radius larger than saturation radius
  double saturation_radius = 4./3.;   // Minimum saturation radius
  double triplesieve_saturation_radius = 1.299;

  double bgj1_improvement_db_ratio = .65;  // Only replace cdb[i] when it is better than
                                           // cdb[bgj1_improvement_db_ratio*i]

  double bgj1_resort_ratio = .85;          // Resort the db after having replaced at about index
                                           // bgj1_resort_ratio*|cbd|. For example, the default
                                           // value implies a resort after having replaced 15% of
                                           // vectors in the database

  size_t bgj1_transaction_bulk_size = 0;    // minimal size of transaction_db to launch
                                            // execute_delayed_insert; 0 means AUTO, i.e. 10 +
                                            // 2*threads.

  double bdgl_improvement_db_ratio = .8;

  std::string simhash_codes_basedir = "";  // directory holding spherical codes for simhash.
};

/**
    This is the main class of g6k. It stores the status of the stateful machine and allows
    various sieve algorithms to be run on this state.
*/

class Siever
{
    friend SimHashes;
    friend UidHashTable;
public:

    /**
        Set-up functions:
        Global:
        The constructors construct a Siever object with the given params.
        The version with full_n and mu takes the full dimension of the GSO and the mu-Matrix.
        mu is passed as a 1-dim c-array: full_muT[i][j] = mu[j * full_n + i]

        mu and full_n can also be provided later by calling load_gso.
        Params can be set later / queried by set_params and get_params.

        Local:

        The sieves themselves operate on a local block, which needs to be set up with
        initialize_local.

        TODO: initialize_local screws up invariants if called with an already present db
        This is fine for its internal usage, where this is remedied, but not for external usage.
        Essentially, private / public usage has different undocumented pre- / postconditions.
        Fix this, preferable by making most of the functionality private.

        To extend / shrink the local block to the left / right, use extend_left, extend_right /
        shrink_left / shrink_right.

        To grow / shrink the current database, use grow_db / shrink_db

    */

    explicit Siever(const SieverParams &params, unsigned long int seed = 0) :
      full_n(0), full_muT(), full_rr(), ll(0), l(0), r(-1), n(0),
      muT(), db(), cdb(),
      best_lifts_so_far(), histo(), rng(seed), sim_hashes(rng.rng_nolock())
#ifdef PERFORMANCE_COUNTING
        , _totalcpu(perfcounters[0])
#endif
    {
        set_params(params);
    };

    explicit Siever(unsigned int full_n, double const* mu, const SieverParams &params, unsigned long int seed = 0)
        : Siever(params, seed)
    {
        
        load_gso(full_n, mu);
        r = full_n;
    }

    // sets / gets the current params object of the Sieve.
    bool set_params(const SieverParams &params); // implemented in params.cpp
    SieverParams get_params(); // implemented in params.cpp


    // - setting full dimension and setting full gso
    // Note that this does not set-up the local data, so need to call initialize_local afterwards
    // mu is passed as a 1-DIM c-style array.
    void load_gso(unsigned int full_n, double const* mu); // implemented in control.cpp

    // Local set-up
    // - update the local gso and recompute gaussian_heuristic for renormalization
    // - reset best_lift_so_far if r changed
    // - reset compression and uid functions
    void initialize_local(unsigned int ll_, unsigned int l_, unsigned int r_); // implemented in control.cpp

    // Extend the context to the left (threaded)
    // - change the context
    // - use babai lifting to add lp coordinates on the left. Recompute data to maintain invariants of entries.
    // - cleanup using refresh_db_collision_check
    void extend_left(unsigned int lp); // implemented in control.cpp

    // Shrink the context to the left (threaded)
    // - change the context
    // - shrink all vectors in db/cdb on the left. Recompute data to maintain invariants of entries.
    // - cleanup using refresh_db_collision_check.
    void shrink_left(unsigned int lp); // implemented in control.cpp

    // Extend the context to the right (threaded)
    // - change the context
    // - extend all vectors in db/cdb on the right by padding with zeros. Recompute data to maintain invariants of entries.
    // - cleanup using refresh_db_collision_check.
    //      [Q: Is this even needed? - A: Yes, because the hash function needs to change (due to change of n). Note that we may get (false positive) hash collisions]
    void extend_right(unsigned int rp); // implemented in control.cpp



    // Increase the db size by sampling many new entries (threaded)
    // The new vectors are appended to the current db, leaving the current db elements untouched. Does not sort.
    // (Note: This behaviour is crucial for some of the Gauss sieves)
    // The argument N is the target size.
    // Due to the way current collision detection works, it is unlikely, but possible to end up with a db size that is actually smaller than N.
    // We log to std::cerr if that happens.
    // TODO: Document parameter large
    void grow_db(unsigned long N, unsigned int large = 0); // implemented in control.cpp

     // Sorts and shrink the database, keeping only the N best vectors
    void shrink_db(unsigned long N); // implemented in control.cpp

    // Debug-only function. This makes a self-check of various invariants.
    // Should always return true. Will print (at least) the first problem that was found to std::cerr.
    // test_bijection controls whether we should test that the db indices stored in cdb form a bijection.
    inline bool verify_integrity(bool test_bijection = true); // in db.inl


/**
    Supported Sieve Algorithms.
*/

    // Run the gauss_sieve algorithm on the current db.
    void gauss_sieve(size_t max_db_size=0); // in sieving.cpp

    // Run one loop of the nv_sieve algorithm on the current db.
    // RET: true if the algorithm thinks that it can do more valuable loops.
    bool nv_sieve(); // in sieving.cpp

    // Runs the bgj1 sieve on the current db.
    // The paramter alpha >= 0 controls when to put vectors into a bucket (bgj1 is a bucketed sieve):
    // An entry x is put into the bucket with center c if |<x,c>| > alpha * |x| * |c|
    void bgj1_sieve(double alpha); // in bgj1_sieve.cpp

    bool bdgl_sieve(size_t buckets, size_t blocks, size_t multi_hash); // in bdgl_sieve.cpp

    // runs a multi-threaded gauss-triple-sieve.
    // The parameter alpha has the same meaning as in bgj1.
    void hk3_sieve(double alpha); // in triple_sieve_mt.cpp

/**
    Retrieving data about the sieve:
    TODO: Document!!!
*/
    // Return the best lifts
    // - If otf_lift is on, no heavy work is done here
    // - Otherwise apply lift_and_compare to each vector
    // - Then, just convert best_lifts_so_far and return them
    // - TODO : TREAT THIS TASK for the otf_lift=False case
    void best_lifts(long* vecs, double* lens); // in control.cpp

    void db_stats(long* cumul_histo); // in control.cpp

    // collects various statistics about the sieve. Details about statistics collection are in statistics.hpp
    CACHELINE_VARIABLE(SieveStatistics, statistics);

    static constexpr unsigned int size_of_histo  = 300;

    void reset_stats() { statistics.clear_statistics(); }
    size_t db_size() const { return db.size(); }
    inline size_t histo_index(double l) const; // in siever.inl

/**
    TODO: Global Internal variables. Some of these are written to from the Python layer
    (which is bad from an encapsulation point of view).
*/

public: // TODO: Make more things private and do not export to Python.

    unsigned int full_n;                      // full dimension
    std::vector<std::vector<FT> > full_muT;   // full gso coefficients, size of all vectors is always full_n
    std::vector<FT> full_rr;                  // fulll gso squared norms, size of vector is always full_n

    // Note:  As usual with loops, iterators, ranges in c++, the range considered is [l, r),
    //        i.e. the left boundary in inclusive, the right boundary is exclusive.

    unsigned int ll;                          // left of the lift context
    unsigned int l;                           // current context left position
    unsigned int r;                           // current context right position
    unsigned int n;                           // current context dimension, n = r - l

  
    // gso_update_postprocessing post-processes the database with the change-of-basis transformation M
    // - Thread-safety ensured by each thread working on different data
    // - Matrix M should have dimension old_n * new_n
    // - Apply e.x = e.x * M for all all e in db/cdb
    // - refresh the database (babai_index=0, l_index=n)
    // NOTE: M is passed as a 1-dim C-style array.

    // TODO: Currently, we modify the GSO directly from the python layer
    // and then call gso_update_postprocessing to fix it again.
    // This is bad encapsulation!

    void gso_update_postprocessing(const unsigned int l_, const unsigned int r_, long const *M); // in control.cpp

    std::vector<std::vector<FT>> muT;     // gso coefficients, triangular matrix, current block
                                          // size of all std::vectors here is always n = r - l. It is derived from full_muT.
    std::vector<FT> rr;                   // gso squared norms current block normalized by the current gh. size of rr is always n.
    std::vector<FT> sqrt_rr;              // precomputed value: sqrt_rr[i] == sqrt(rr[i]).

    CACHELINE_VARIABLE(FT, gh);                             // normalization factor for current block
    CACHELINE_VARIABLE(std::vector<Entry>, db);             // database
    CACHELINE_VARIABLE(std::vector<CompressedEntry>, cdb);  // compressed version, faster access and periodically sorted

    // We have the following invariants:
    // (a) Every Entry in db and CompressedEntry in cdb has size n
    // (b) j mapsto cdb[j].i is a bijection, i.e. every element from cdb corresponds to an element from db and vice versa.
    // (We only store the bijection in the cdb -> db direction)
    // (c) If db[i] and cdb[j] are associated, the data stored in them is consistent with each other (and internally, of course), i.e.
    //     cbd[j].c == db[i].c and so on.
    // (d) There is no invalidation of iterators / pointers into db or cdb outside from a call to shrink_db.
    //     [ This is ensured by not changing the size of db and cdb outside of calls to grow_db / shrink_db. Note that grow_db only appends]
    // (e) The uid of every element in db is stored inside the uid_hash_table.
    //     [ But the uid_hash_table may store elements not (yet) in db. ]
    // (f) The uid's of the elements in db do not collide.
    //     [ This is violated and then remedied during context changes and possibly during gso changes. ]

private:

    enum class SieveStatus
    {
        plain = 0,
        bgj1  = 1,
        gauss = 2,
        triple_mt = 3,
        LAST = 4 // last valid option + 1
    } sieve_status = SieveStatus::plain; // for the tagged union below. This indicates some internal status that depends on the algorithm

    union StatusData
    {
        struct Plain_Data // data if SieveStatus == plain or bgj1
        {
            size_t sorted_until = 0; // cdb is sorted until this point
        } plain_data;
        struct Gauss_Data // data if SieveStatus == gauss or triple_mt
        {
            size_t list_sorted_until = 0;
            size_t queue_start = 0;
            size_t queue_sorted_until = 0;
            int reducedness = 0;
        } gauss_data;
        StatusData() : plain_data() {};
    } status_data;

    bool histo_valid = false;

    CACHELINE_VARIABLE(std::vector<CompressedEntry>, cdb_tmp_copy);   // temporary copy for sorting without affecting other threads

    CACHELINE_VARIABLE(std::vector<LiftEntry>, best_lifts_so_far); // vector of size r, containing good (i.e. short) vectors lifts so far.
                                          // More precisely, best_lifts_so_far[i] is an "Entry" of size r, whose GSO-projection onto
                                          // coos [i..r) has good length.
                                          // If unitary_only is set, this is guaranteed to have at least one +/- 1 coo in x.
                                          // len is set to 0 if we have not found anything yet.
                                          // only valid when otf_lift is set to true

    // Note histo is only reliable if histo_valid == true.
    // access histo via the best_lifts function (which is supposed to compute lifts on demand)
    std::array<long, size_of_histo> histo;   // histogram of points in db by length, i.e. the number
                                             // of points with a normalized length between certain
                                             // (disjoint) bounds The bounds are implicitly obtained
                                             // by hist_index(len), which gives the correct index:
                                             // histo[i] counts the number of points e with
                                             // hist_index(e.len) == i in the db. NOTE: histo may or
                                             // may not include points that are commited (like the
                                             // hash db) for better multi-threading.


    SieverParams params;

    UidHashTable uid_hash_table;    // hash table. At any given point of time, this hash table contains precisely the hashes of
                                    // all x-coos of all points in db AND of all points that are already commited for potential inclusion into db.
                                    // The latter is used during grow_db and some sieves with sets of commited vectors given by local variables in their respective implementations.

    // TODO: document
    std::vector<FT> lift_bounds;
    FT lift_max_bound;

    CACHELINE_VARIABLE(rng::threadsafe_rng, rng);
    SimHashes sim_hashes; // needs to go after rng!
#ifdef PERFORMANCE_COUNTING
    cpu::update_performance_counter _totalcpu; // CPU counter that counts the lifetime of the Siever object
#endif


/**
    Global internal functions
*/
    // switches internal status, return value indicates whether it changed.
    bool switch_mode_to(Siever::SieveStatus new_sieve_status); // in control.cpp

    void invalidate_sorting(); // in control.cpp
    void invalidate_histo();   // in control.cpp

    // consider renaming
    void set_lift_bounds(); // in control.cpp

    // pre-allocate db_size slots for db and cdb (and cdb_tmp_copy ???)
    void reserve(size_t reserve_db_size); // in control.cpp

    // Sets the number of threads used. Note that the threadpool uses nr-1 threads. It is intended
    // to use wait_work (as opposed to wait_sleep) to also utilize the control thread.
    void set_threads(unsigned int nr); // in control.cpp


    // cleanup after context change.
    //  - remove all entries with duplicate uids (replacing them with fresh samples).
    //  - Sorts cdb by length
    //  - keeps histo updated
    void refresh_db_collision_checks(); // consider renaming

    // Worker function for gso_update_postprocessing.
    template<int tn>
    void gso_update_postprocessing_task(size_t const start, size_t const end, int const n_old, std::vector<std::array<ZT,MAX_SIEVING_DIM>> const &MT); // in control.cpp

    void parallel_sort_cdb(); // in control.cpp

    // Lift an entry e to all possible positions 0<i<r, and store the result in
    // best_list_so_far[i] when it is improving its length
    // The second version takes a pointer to a buffer of x-coos (which is written to!)
    // and the NON-NORMALIZED length of the entry. Note that e.len is usually normalized by gh.
    // - Thread-safety ensured by a global mutex (the lock is rarely taken by design)

    inline void lift_and_compare(Entry const &e); // in db.inl

    // Note: *x_full is NOT const, just the pointer itself inside the implementation is.
    inline void lift_and_compare(ZT * const x_full, FT len, LFT const * const helper=nullptr);  // in db.inl
    void lift_and_replace_best_lift(ZT * const x_full, unsigned int const i);

    std::mutex global_best_lift_so_far_mutex;

    // Create a new smallish random entry
    // - If large==0, try to sample by combining current vectors
    // - If large >0:
    //   - sample last cordinates at random and babai the first coordinates
    //   - large parameter can be increased if fresh random entries collide with db
    inline Entry sample(unsigned int large=0); // in db.inl

    // worker task for grow_db
    void grow_db_task(size_t start, size_t end, unsigned int large);

/**
    Various functions to insert / replace elements in db / cdb. They are optimized for various
    specific reasons and differ in whether uid should be inserted as well, whether the point should be
    recomputed, what arguments they take... Essentially, these are mostly specific to a given sieve.
*/

    // Insert an entry into cdb and db, growing both. Does nothing else (in particular, no lifting, sorting, histo updates...)
    // *** NOTE: This does NOT insert into uid database and does NOT check collisions, we assume that uid was inserted into the uid_db previously ***
    // Note: This function is deprecated.
    void insert_in_db(Entry &&e); // in db.inl

    // This creates an entry from x and inserts it into db (and cdb) and the hash_table. In case of hash collision with the table,
    // no insertion is performed. Return value indicates success.
    // This function is only used in load_db and might go away.
    // (load_db does not work atm anyway, in particular, we need to
    inline bool insert_in_db_and_uid(std::array<ZT,MAX_SIEVING_DIM> &x); // in db.inl

    // Replace the entry pointed at by cdb[cdb_index] by e, unless
    // length is actually worse. Assumes that e was already added to the hash db.
    // Will remove the corresponding uid from the hash db, depending on whether it was successful.
    // Only used in bgj1, defined in bgj1_sieve.cpp.
    // TODO: Rename to remove the _nohisto.
    bool bgj1_replace_in_db(size_t cdb_index, Entry &e); // in bgj1_sieve.cpp

    // order_and_reduce_triple and reduce_ordered_triple_in_db are specific to single-threaded triple-sieve
    // and are defined in triple_sieve.cpp

    // Attempt a reduction of max{ce1, filtered1->ce, filtered2->ce}
    // in this function the max is deduced and reduce_ordered_triple is called to actually perform the reduction
    bool order_and_reduce_triple(CompressedEntry* ce1, FilteredCompressedEntry* filtered1, FilteredCompressedEntry* filtered2, short &which_one); // in triple_sieve.cpp

    // reduces x1 which is of maximal length
    // x1  = x1 +/- x2 +/- x3
    // the correct signs are decided from the inner-producs xixj
    bool reduce_ordered_triple_in_db (CompressedEntry* x1, CompressedEntry* x2, CompressedEntry* x3, LFT x1x2, LFT x1x3, LFT x2x3); // in triple_sieve.cpp

    // Only used by the plain sieves. In sieving.cpp
    CompressedEntry* reduce_in_db(CompressedEntry *ce1, CompressedEntry *ce2, CompressedEntry *target_ptr=nullptr);

    // returns 0 for no reduction, 1 for reduced ce1, 2 for reduced ce2.
    // Used in gauss_no_upd only. In sieving.cpp
    short gauss_no_upd_reduce_in_db(CompressedEntry *ce1, CompressedEntry *ce2);

    // Used in single-threaded triple_sieve only. In triple_sieve.cpp
    short reduce_in_db_which_one_with_len(CompressedEntry *ce1, CompressedEntry *ce2, LFT new_l);

public: // We use ENABLE_BITOPS_FOR_ENUM(Siever::Recompute) inside siever.inl,
        // which allows bit-operations (&,|,^,~) for Siever::Recompute.
        // This macro only works at namespace scope and requires to declare Recompute as public.
        // (Operator overloading for private nested classes is possible, but rather cumbersome)

    enum class Recompute // used as a bitmask for the template argument to recompute_data_for_entry below
    {
      none = 0,
      recompute_yr = 1,
      recompute_len = 2,
      recompute_c = 4,
      recompute_uid = 8,
      recompute_otf_helper = 16,
      recompute_all = 31,
      consider_otf_lift = 32,
      recompute_all_and_consider_otf_lift = 63,
      babai_only_needed_coos_and_recompute_aggregates = 30 // for _babai variant if rest of yr is valid.
    };


private:
    // Recomputes the desired data from x in the given point. Non-recomputed data is assumed consistent.
    // Which data to recompute is controlled by template parameter (as a bitmask).
    // Note that Siever::Recompute supports &, |, ^, ~ etc.
    // consider_otf_lift only controls whether otf_lift is considered. It still checks for otf_lift == true.

    // Usage:
    // To recompute everything :
    // recompute_data_for_entry<Recompute::recompute_all>(e);
    // To recompute all but uid:
    // recompute_data_for_entry<Recompute::recompute_all & (~Recompute::recompute_uid)>(e);
    // etc.
    // The _babai variant uses babai lifting to update the babai_index many left-most coos.

    template<Recompute what_to_recompute>
    inline void recompute_data_for_entry(Entry &e);

    // uses babai lifting on indices [0,...babai_index -1] prior to recomputations,
    // changing x and yr on those indices (irrespective of the template arg what_to_recompute)
    // IMPORTANT: For this variant, recompute_yr controls whether to recompute yr on the OTHER indices.
    template<Recompute what_to_recompute>
    inline void recompute_data_for_entry_babai(Entry &e, int babai_index);

    // Recomputes the data in histo to make sure they are up to date.
    inline void recompute_histo(); // in db.inl
    // empties histo. -- deleted
    //    inline void clear_histo();

    // these (static) member functions might go into SimHashes.

    // checks whether the simhash-xor-popcount of the arguments is larger (in absolute value) than THRESHOLD.
    // The second variant takes c-arrays of the appropriate length. Prefer the first variant.
    template<unsigned int THRESHOLD>
    inline static bool is_reducible_maybe(const CompressedVector &left, const CompressedVector &right);
    template<unsigned int THRESHOLD>
    inline static bool is_reducible_maybe(const uint64_t *left, const uint64_t *right);

    // one-sided check (no absolute value) for the simhash-xor-popcount
    // used inside triple_sieve
    // The second variant takes c-arrays of the appropriate length. Prefer the first variant.
    template<unsigned int THRESHOLD>
    inline static bool is_far_away(const CompressedVector &left, const CompressedVector &right);
    template<unsigned int THRESHOLD>
    inline static bool is_far_away(const uint64_t *left, const uint64_t *right);

    // applies functor(e) on all e in db in parallel. Functor should have an overloaded operator()
    // that takes Entries as argument. Functor may be (and usually is) a lambda.
    // If we ever move to c++17, replace by std::for_each, std::transform as applicable.
    // Note that if functor is stateful, every thread gets its own copy.
    template<class Functor>
    void apply_to_all_entries(Functor const &functor); // defined in siever.inl

    // applies functor(ce) on all ce in cdb in parallel. Functor should have an overloaded operator()
    // that takes CompressedEntries as argument. Functor may be lambda.
    // If we ever move to c++17, replace by std::for_each, std::transform as applicable.
    // Note that if functor is stateful, every thread gets its own copy.
    template<class Functor>
    void apply_to_all_compressed_entries(Functor const &functor); // defined in siever.inl

    // performs a+= b * c, where a,b are vectors (supposedly of the same size).
    // The num variant only modifies the first num entries of a (it assumes that num <= a.size(), otherwise UB ).
    template <typename Container, typename Container2>
    inline void addmul_vec(Container &a, Container2 const &b, const typename Container::value_type c); // defined in siever.inl
    template <typename Container, typename Container2>
    inline void addsub_vec(Container &a, Container2 const &b, const typename Container::value_type c); // defined in siever.inl

    // currently unused
    template <typename Container, typename Container2>
    inline void addmul_vec(Container &a, Container2 const &b, const typename Container::value_type c, int num); // defined in siever.inl

    /**
        implementation details of hk3_sieve: Everything prefixed by TS_ or hk3_sieve_ belongs to it
        see triple_sieve_mt.cpp for details about how the algorithm works and what the individual functions do.
        The explanations here are only very brief. Note that the algorithm does not use cdb, but rather
        our own std::vector<CompressedEntry>'s, accessed via std::shared_ptr. The pointers point to
        some (possibly outdated) version of cdb (Called cdb snapshots) and the std::shared_ptr manage the memory.
        An important consequence is that DURING THE RUN OF GAUSS_TRIPLE_MT, CDB IS INVALID.
    **/

    // Everything here is implemented in hk3_sieve

//    using TS_CDB_Snapshot_Ptr = std::shared_ptr<std::vector<CompressedEntry>>;

    struct TS_Transaction_DB_Type // (thread-local) database of pending transactions, i.e. points that are to be added to the database.
    {
        std::vector<Entry> new_vals;
        size_t sorted_until = 0; // The first sorted_until values among new_vals are already sorted.
        // forward some interface of std::vector<Entry>, to avoid having to type .new_vals all the time.
        Entry &back() { return new_vals.back(); }
        Entry const &back() const { return new_vals.back(); }
        NODISCARD bool empty() const noexcept { return new_vals.empty(); }
        std::vector<Entry>::size_type size() const noexcept { return new_vals.size(); }
        template<class...Args> auto emplace_back(Args&&... args) -> decltype( new_vals.emplace_back(std::forward<Args>(args)...) )
            { return new_vals.emplace_back(std::forward<Args>(args)...); } // I want C++14 decltype(auto)
        void pop_back() { new_vals.pop_back(); }
    };

    // TS_FilteredCE stores data for an element x1 that was found to be close to +/- p.
    // Rather than storing just x1, we store some additional information that is useful for the specific algorithm.
    struct TS_FilteredCE
    {
        CompressedVector c_proj;                // Compressed vector, possibly of projection to orthog. complement of p and with sign-flip.
        IT db_index;                            // Index of the point in db.
        bool sign_flip;                         // Is the sign of the scalar product <p, db[i]> > 0.
                                                // Note that depending on sign_flip, the meaning of other data changes that incorporates this flip.
        FT len_score;                           // 1/2 p^2 + |db[i]|^2 +/- 2* <p, db[i]>. The sign is + iff <p,db[i]> < 0 (so len_score <= db[i].len)
        // Remark: It would be slightly more efficient to store the uid of 1/2p + /- db[i], but this requires uid's to be computed modulo something odd.
        UidType uid_adjusted;                   // uid of +/- db[i]. With sign as above.
    };
    static constexpr int TS_Cacheline_Opt = 1024; // We process this number of TS_FilteredCE's as a block to aid cache-locality.

    // For our CDB-snapshots, the datatype of choice would be an (atomic) std::shared_pointer.
    // Unfortunately, this turned out to have several issues:
    // 1.) C++11 does not have good atomic suport for shared pointers.
    // 2.) There is no good thread-safe(!) way to query the number of instances a shared pointer.
    //     The issue at hand is that we re-use memory from old snapshots to avoid reallocations.
    //     The existing use_count member of std::shared_pointer does not work because the internal
    //     reference counting may not be synchronized properly for that. This is a known issue,
    //     made by choice to optimize for the (common) case where it is not needed.
    //     The workaround is to make all updates (including most destructors!) under a lock...
    // Because the workaround required some obscure code and made limiting the #snapshots a pain,
    // we roll out (essentially) our own reference-counted objects.

    struct TS_CDB_Snapshot
    {
        std::vector<CompressedEntry> snapshot;
        std::atomic<size_t> ref_count; // counts the number of threads that currently use it.
    };

    void hk3_sieve_task(TS_Transaction_DB_Type &transaction_db, MAYBE_UNUSED unsigned int id, double const alpha); // Main worker function. id is the thread-nr. Used only for debugging.
    std::pair<Entry, size_t> hk3_sieve_get_p(TS_CDB_Snapshot * &thread_local_snapshot, unsigned int const id, TS_Transaction_DB_Type &transaction_db, float &update_len_bound); // obtains a new point p and cdb-range to work with.
    void hk3_sieve_resort(MAYBE_UNUSED unsigned int const id); // resorts the current cdb-snapshot
    void hk3_sieve_init_metainfo(size_t const already_processed, CompressedEntry const * const fast_cdb); // initializes / resets some of the atomic variables. Called after resorting
    float hk3_sieve_update_lenbound(CompressedEntry const * const fast_cdb); // Recomputes the bound that determines when a vector is deemed interesting for insertion.
    size_t hk3_sieve_execute_delayed_insertion(TS_Transaction_DB_Type &transaction_db, float &update_len_bound, MAYBE_UNUSED unsigned int const id); // inserts pending transactions into cdb and db.

    // subroutine for the inner-loop. Templated to simplify the code (we call this with >= 4 different template args)
    template<bool EnforceOrder, class SmallContainer1, class LargeContainer2, class Integer1>
    inline void hk3_sieve_process_inner_batch(TS_Transaction_DB_Type &transaction_db, Entry const &p, SmallContainer1 const &block1, Integer1 const end_block1, LargeContainer2 const &block2, size_t const end_block2, float &local_len_bound, MAYBE_UNUSED unsigned int const id);

    // these functions attempt a reduction between 2 or 3 points and put the result onto transaction db.
    // The versions differ by how we access the points and what we already know about the scalar products / signs used in the reduction.
    bool hk3_sieve_delayed_red_p_db(TS_Transaction_DB_Type &transaction_db, Entry const &p, size_t const x1_index, bool const sign_flip);
    bool hk3_sieve_delayed_2_red_inner(TS_Transaction_DB_Type &transaction_db, size_t const x1_db_index, bool const x1_sign_flip, size_t const x2_db_index, bool const x2_sign_flip, UidType new_uid);
    bool hk3_sieve_delayed_3_red(TS_Transaction_DB_Type &transaction_db, Entry const &p, size_t const x1_index, bool const x1_sign_flip, size_t const x2_index, bool const x2_sign_flip, UidType const new_uid);

    // corresponding to the reduction attempts, we attempt on-the-fly-lifts.
    void hk3_sieve_otflift_p_db(Entry const &p, size_t const db_index, bool const sign_flip, double const believed_len);
    void hk3_sieve_otflift_p_x1_x2(Entry const &p, size_t const x1_db_index, bool const x1_sign_flip, size_t const x2_db_index, bool const x2_sign_flip, double const believed_len);
    void hk3_sieve_otflift_x1_x2(size_t const x1_db_index, bool const x1_sign_flip, size_t const x2_db_index, bool const x2_sign_flip, double const believed_len);

    // snapshot management: thread-local snapshots and TS_latest_cdb_snapshot_p must be managed by
    // these function, which take care of ref-counting. See triple_sieve_mt.cpp for more details.
    void hk3_sieve_release_snapshot(TS_CDB_Snapshot * &thread_local_snapshot, MAYBE_UNUSED unsigned int const id);
    TS_CDB_Snapshot * hk3_sieve_get_latest_snapshot(MAYBE_UNUSED unsigned int const id);
    CompressedEntry * hk3_sieve_get_true_fast_cdb();
    TS_CDB_Snapshot * hk3_sieve_get_free_snapshot(MAYBE_UNUSED unsigned int const id);
    void hk3_sieve_update_latest_cdb_snapshot(TS_CDB_Snapshot * const next_cdb_snapshot_ptr, MAYBE_UNUSED unsigned int const id);
    void hk3_sieve_init_snapshots();
    void hk3_sieve_restore_cdb();

    // Internal data structure. The cdb-snapshots are partially ordered and separated into 5 parts. See hk3_sieve.cpp for documentation.
    // This separation encodes important information for the algorithm.
    CACHELINE_PAD(Pad_queue_mutex); // We want the following to reside close
    std::mutex TS_queue_head_mutex; // protects current_queue_head and TS_unmerged_transactions.
    size_t TS_queue_head;   // index of first unprocessed queue element (elements currently being processed by other threads count as processed)
    std::vector<size_t> TS_unmerged_transactions; // keeps track of the number of unmerged transactions for each thread. Only updated during get_p.
    size_t TS_total_unmerged_transactions;
    // used to communicate to other threads and to the calling "master thread" that we are done and why.
    // Only the first termination condition that we encounter is stored.
    enum class TS_Finished
    {
        running = 0, // still running
        saturated = 1, // we have reached saturation
        out_of_queue = 2, // we have run out of queue to process: queue size is below TS_min_queue_size
        out_of_queue_resume = 3,    // in addition to the above, when we ran out of queue, we had a
                                    // large number of pending db transactions. We might restart the algorithm after merging those.
        sortings_exceeded = 4  // We re-sorted more than TS_max_number_of_sorts many times.
    };
    std::atomic<TS_Finished> TS_finished;

    CACHELINE_VARIABLE(std::atomic<std::make_signed<size_t>::type>, TS_queue_left); // remaing size of the queue (without elements currently being processed)
    std::mutex TS_insertion_selection_mutex; // protects the following two variables:
    size_t TS_insert_queue;         // start of overwritten queue == insertion position +1 when inserting into queue part
    size_t TS_insert_list;          // start of overwritten list  == insertion position +1 when inserting into list  part
    CACHELINE_VARIABLE(std::atomic<size_t>, TS_insertions_started);      // number of times any thread has started insertion of a transaction block for the current snapshot
    CACHELINE_VARIABLE(std::atomic<size_t>, TS_cdb_insertions_finished); // number of times any thread has finished with the cdb part of insertions
    CACHELINE_VARIABLE(std::atomic<size_t>, TS_db_insertions_finished);  // number of times any thread has finished the the (cdb and) db part of insertions.

//    CACHELINE_VARIABLE(std::atomic<int>, TS_queue_list_imbalance);   // stores the (length-of-queue-end - length-of-list-end) * TS_queue_list_imbalance_multiplier.
                                                // If the queue / is empty, that length is taken as TS_outofqueue_len / TS_outoflist_len, which is a
                                                // negative number (TS_outoutqueue_len is more negative than TS_outoflist_len by design)
    CACHELINE_VARIABLE(std::atomic<size_t>, TS_insertions_performed); // number of insertions actually perfomed. Reset to 0 after resorting. Used to determine when to resort.
    CACHELINE_VARIABLE(std::atomic<size_t>, TS_saturated_entries);    // how many db entries with length^2 below the saturation bound do we still need until we are done.
    CACHELINE_VARIABLE(std::atomic<float>, TS_len_bound);   // length bound: we put vectors into transaction_db if they are shorter than this. includes all modifiers such as REDUCE_LEN_MARGIN

    CACHELINE_VARIABLE2(std::atomic_flag, TS_currently_sorting, = ATOMIC_FLAG_INIT); // Is another thread currently sorting the latest CDB snapshot.

//    CACHELINE_PAD(pad_latest_cdb);
    // pointer to the latest cdb_version. The mutex is to make it atomic.
    // Note that std::atomic<std::shared_pointer> is post C++11 / free atomic_* functions for it have various issues.
    // If you use C++20, use std::atomic< > partial specializations for shared_ptr's instead.
//    std::mutex TS_latest_cdb_mutex;
//    TS_CDB_Snapshot_Ptr TS_latest_cdb_version;
//    CACHELINE_PAD(pad_latest_cdb_end);

    CACHELINE_PAD(pad_latest_cdb_snapshot);
    std::mutex TS_latest_cdb_mutex; // protects writes to the following (even though it is atomic).
                                    // Needed to assign from it; it really protects the ref-count.
    std::atomic<TS_CDB_Snapshot *>  TS_latest_cdb_snapshot_p; // points to the most recent snapshot
    CACHELINE_PAD(pad_snapshots_start);
    static constexpr size_t TS_max_snapshots = 3; // maximum number of snapshots in use.
    std::array<TS_CDB_Snapshot, TS_max_snapshots> TS_cdb_snapshots;
    size_t TS_snapshots_used = 0; // how many of the above are actually used. Unfortunately, we cannot use a std::vector here (because the struct contains atomics, preventing anything that could trigger reallocation, such as resize).
    // Used to recycle old snapshots to avoid memory reallocation and limit memory consumption.
    // The maximum number of allocated snapshots is TS_max_snapshots.
    std::mutex TS_free_snapshot_mutex;                  // protects TS_free_snapshot, used with std::condition_variable
    size_t TS_free_snapshot = 0;                            // 1 + some index of an unused snapshot, 0 if we do not know if a snapshot is free
    std::condition_variable TS_wait_for_free_snapshot;  // used to let sorting wait for a snapshot be recycle-able.
    CACHELINE_PAD(pad_snapshots_end);

    CACHELINE_VARIABLE(size_t, TS_start_queue_original); // start of processed queue part, this does not change for a given snapshot. Does not need to be atomic.
    CACHELINE_VARIABLE(std::atomic<size_t>, TS_number_of_sorts);      // number of sortings performed (this also counts the number of snapshots). Does not need to be atomic.

    CACHELINE_VARIABLE(std::condition_variable, TS_wait_for_sorting); // used if a thread has to wait for sorting to finish.

//    CACHELINE_VARIABLE(std::vector<TS_CDB_Snapshot_Ptr>, TS_snapshots);

    // These 3 might go away.
//    static constexpr int TS_queue_list_imbalance_multiplier = 1 << 16;
//    static constexpr int TS_outofqueue_len = - (1 << 24);
//    static constexpr int TS_outoflist_len  = - (1 << 23);

    // TODO: Explain these parameters more
    static double constexpr TS_improvement_db_ratio = .65;
    static double constexpr TS_resort_after = .15;
    static size_t constexpr TS_transaction_bulk_size = 8;
    static size_t constexpr TS_large_transaction_size = 10;
    static size_t constexpr TS_max_extra_queue_size = 20; // If we only have approx. TS_max_extra_queue_size unprocessed and non-overwritten queue elements left, we start countermeasures against queue-exhaustion and trigger sorting.
    static double constexpr TS_default_alpha = 0.315; // default value of alpha if the provided value is <= 0.
    static constexpr int    TS_OUTER_SIMHASH_THRESHOLD = 102; // threshold in the bucketing / filtered-list phase
    static constexpr int    TS_INNER_SIMHASH_THRESHOLD = 140; // threshold in the inner processing phase. Details might change (upper vs. lower bounds are not the same here)
    static constexpr int    TS_min_queue_size = 20; // If after resorting, we have this little queue left, we terminate the algorithm, provided that:
    static constexpr int    TS_min_pending_transactions = 50;  // we have less than TS_min_pending_transactions left.
    static constexpr int    TS_max_number_of_sorts = 100;
    static constexpr bool   TS_perform_inner2red = true;
    static constexpr bool   TS_extra_range       = true;
    static_assert(TS_max_snapshots >= 3, "The algorithm requires at least 3 snapshots to sort in parallel");
    // If a thread holding a local copy of snapshot A, latest snapshot being B, requests sorting and creates snapshot C,
    // we might require 3 snapshots. Updating the local copy from A to B/C would is an option, but
    // that is error-prone: for now, local copies are only changed when processing a new p, which
    // makes the algorithm much easier.
    /** end of Multi-threaded Triple-sieve implementation details **/

    /**
        Implementation details of bgj1 sieve
    */

    // Attempts a reduction / possibly OFT-Lift between ce1 and ce2.
    // lenbound denotes the bound below which we deem the result good enough for reduction.
    // Good enough results are put into transaction_db.
    //
    // Note: lenbound is assumed to be already adjusted by REDUCE_LEN_MARGIN.
    // reduce_while_bucketing indicates that this function is called during the bucketing phase or not.
    // This is only needed for statistical purposes and the paramter is only passed if it is needed.
    // The #if condition is an ugly hack, but this ensures that we have zero performance loss in case we do not collect statistics.
    #if (COLLECT_STATISTICS_OTFLIFTS >= 2) || (COLLECT_STATISTICS_REDSUCCESS >= 2) || (COLLECT_STATISTICS_DATARACES >= 2)
    template <int tn>
    inline bool bgj1_reduce_with_delayed_replace(CompressedEntry const &ce1, CompressedEntry const &ce2, LFT lenbound, std::vector<Entry>& transaction_db, bool reduce_while_bucketing = true); // in bgj1_sieve.cpp
    #else
    template <int tn>
    inline bool bgj1_reduce_with_delayed_replace(CompressedEntry const &ce1, CompressedEntry const &ce2, LFT lenbound, std::vector<Entry>& transaction_db); // in bgj1_sieve.cpp
    #endif
    // Tries to put pending transactions from transaction_db into the database.
    // Note that this can fail for various reasons: Points can be removed from transaction_db without inserting into db,
    // because the points are too long. Furthermore, insertion can fail due to concurrency issues
    // and points may remain in transaction_db (we return and do other useful work rather than wait to acquire a mutex)
    // Return value indicates whether transaction_db is empty after the call.

    // Note: force is supposed enforce acquiring a mutex (so return value should always be true)
    // Currently, execut_delayed_replace_nohisto is only ever called with force == false.
    bool bgj1_execute_delayed_replace(std::vector<Entry>& transaction_db, bool force, bool nosort = false); // in bgj1_sieve.cpp
    template <int tn>
    void bgj1_sieve_task(double alpha); // in bgj1_sieve.cpp

    /**
      Implementation details of bdgl sieve
    */
    inline int bdgl_reduce_with_delayed_replace(const size_t i1, const size_t i2, LFT const lenbound, std::vector<Entry> &transaction_db, int64_t &write_index, LFT new_l = -1.0, int8_t sign = 1);

    inline void bdgl_lift(const size_t i1, const size_t i2, LFT new_l, int8_t sign);

    bool bdgl_replace_in_db(size_t cdb_index, Entry &e);

    void bdgl_bucketing_task(const size_t t_id, 
                             std::vector<uint32_t> &buckets, std::vector<atomic_size_t_wrapper> &buckets_index,
                             ProductLSH &lsh);
    void bdgl_bucketing(const size_t blocks, const size_t multi_hash, const size_t nr_buckets_aim, 
                        std::vector<uint32_t> &buckets, std::vector<atomic_size_t_wrapper> &buckets_index);

    void bdgl_process_buckets_task(const size_t t_id, const std::vector<uint32_t> &buckets, 
                                   const std::vector<atomic_size_t_wrapper> &buckets_index, std::vector<QEntry> &t_queue);
    void bdgl_process_buckets(const std::vector<uint32_t> &buckets, const std::vector<atomic_size_t_wrapper> &buckets_index,
                                std::vector<std::vector<QEntry>> &t_queues);
    
    void bdgl_queue_create_task( const size_t t_id, const std::vector<QEntry> &queue, std::vector<Entry> &transaction_dbi, int64_t &write_index);
    void bdgl_queue_dup_remove_task( std::vector<QEntry> &queue);
    size_t bdgl_queue_insert_task( const size_t t_id, std::vector<Entry> &transaction_dbi, int64_t write_index);
    void bdgl_queue( std::vector<std::vector<QEntry>> &t_queues, std::vector<std::vector<Entry>> &transaction_db);

    std::pair<LFT, int8_t> reduce_to_QEntry(CompressedEntry *ce1, CompressedEntry *ce2);

// previously, these were global variables. TODO: Document / refactor those.
    CACHELINE_VARIABLE(std::atomic_size_t, GBL_replace_pos);
    CACHELINE_VARIABLE(std::atomic_size_t, GBL_replace_done_pos);
    CACHELINE_VARIABLE(std::atomic<LFT>, GBL_max_len); // already adjusted by REDUCE_LEN_MARGIN
    long GBL_max_trial; // maximum number of buckets bgj1 will consider between resorting before it gives up.
    CACHELINE_VARIABLE(std::atomic<std::int64_t>, GBL_remaining_trial);
    CACHELINE_VARIABLE(std::mutex, GBL_db_mutex);
    CACHELINE_VARIABLE(std::atomic<CompressedEntry*>, GBL_start_of_cdb_ptr); // point always either to the start of cdb or to cdb_tmp_copy (due to for sorting)

    // saturation stop conditions
    double GBL_saturation_histo_bound[Siever::size_of_histo]; // used by gauss sieve & triple sieve
    CACHELINE_VARIABLE(std::atomic_size_t, GBL_saturation_count); // used by bgj1 sieve

    thread_pool::thread_pool threadpool;

}; // End of Siever Class definition

// implementation of inline functions that need to go into the header (e.g. all template member functions)
#include "simhash.inl"
#include "siever.inl"
#include "hash_table.inl"
#include "db.inl"

#endif
