#ifndef G6K_STATISTICS_HPP
#define G6K_STATISTICS_HPP

#include "compat.hpp"
#include <atomic>
#include <string>
#include <sstream>
#include <ostream>
#include <type_traits>

#ifndef G6K_SIEVER_H
    #error Do not include siever.inl directly
#endif

/** This file defines the SieveStatistics subclass that
    is responsible for collecting and managing statistics.
    Note that most functionality in this file depend on the
    ENABLE_STATS / ENABLE_EXTENDED_STATS macros, which are set by the build process.
    (use -s resp. -ss to enable it)
    If none is set, most functions in this file are no-ops.
    This is intended to not waste clock cycles on collecting statistics.

    We also define a MergeOnExit class template (and a merge_on_exit helper function)
    that creates a counter that calls a customizable lambda on destruction.
    This is intended to be used to count thread-locally.
*/

/**
    How SieveStatistics works:
    Depending on whether ENABLE_STATS / ENABLE_EXTENDED_STATS are set, we set COLLECT_STATISTICS to a numerical value that
    indicates the level of statistics we collect.
    This is then used to set various COLLECT_STATISTICS_FOO macros.
    ENABLE_STATS / ENABLE_EXTENDED_STATS is set by the build system (rebuild.sh -s resp. rebuild.sh -ss). The COLLECT_STATISTICS_FOO macros can be
    manually set to override individual statistics.
    These macros control which member fields SieveStatistics have.
    We also have ENABLE_IF_STATS_FOO(x) macros that conditionally compile x to simplify some code.

    The (non-static) members of SieveStatistics are all private.
    We have getters and incrementers with names get_stats_FOO and inc_stats_BAR.
    For each get_stats_FOO, there is a constexpr bool collect_statistics_FOO that tells whether the return value of get_stats_FOO is meaningful.

    Note that we have more getters than incrementers, because we also have getters for aggregate statistics.

    Note that which getters/incrementers are available does not depend on the macro settings.
    The functions just don't do anything if the statistics is not collected and will be optimized out.
    We take care that the latter actually happens.
    (This means there is less need for conditional compilation. You can use the incrementers unconditionally.
    ENABLE_IF_STATS_FOO(x) is only needed if you have thread-local variables.)

    On the python layer, statistics are exported as None if collect_stastics_FOO is false.

    We also have a clear_statistics function to reset everything to zero and some rudimentary printing.
    (We also export a string rather than print directly. This is exported to python, so we can use
    python's print rather than c++'s std::cout << )

    IMPORTANT (Checklist): If you add something:
        - Possibly modify the preprocessor instructions if you need a new flag.
            This means adding a COLLECT_STATISTICS_FOO macro and ENABLE_IF_STATS_FOO macro.
        - Add the members variables to the class.
        - Add / Modify the getters / incrementers
        - Add it to clear_statistics
        - Add it to decl.pxd
        - Add it to siever.pyx (as a @property)
        - Optionally: Add it to the print routines, if desired
        - Optionally: Add it to the tracer, if special treatment is neccessary.

*/

#if defined ENABLE_EXTENDED_STATS
    #define COLLECT_STATISTICS 2
#elif defined ENABLE_STATS
    #define COLLECT_STATISTICS 1
#else
    #define COLLECT_STATISTICS 0
#endif

/**
    Define macros COLLECT_STATISTICS_*:
    COLLECT_STATISTICS_* gives a level how fine-grained the statistics are to be collected.
    0 means no colection takes place at all.
    * is one of:
    REDS (reductions)
    XORPOPCNT (xor-pop-count computations)
    XORPOPCNT_PASS ( how often these are "successful" )
    FULLSCPRODS ( full scalar product computations inside the sieve)
    FILTER_PASS ( how often vectors pass our filters (for bucketing / filtered lists)
    REDSUCCESS ( how often we actually successfully (believe to) create a short vector )
    REPLACEMENTS ( db replacements )
    REPLACEMENTFAILURE ( failures for various reasons )
    OTFLIFTS ( lifting attempts )
    COLLISIONS ( hash collisions )
    DATARACES ( data races / various error conditions )
    SORTING (re-sortings inside the sieve)
    BUCKETS (number of buckets considered)
*/

//

// XOR_POPCNT counts the number of simhash-xor-popcnt computations
// 0: no statistics collection
// 1: just total number
// 2: distinguish the different contributions ( creating bucket vs. actual tests in 2-sieve, outer vs. inner loop in 3-sieve. Note that the 2-sieve and 3-sieve cases are the same thing, really.)
#ifndef COLLECT_STATISTICS_XORPOPCNT
#define COLLECT_STATISTICS_XORPOPCNT COLLECT_STATISTICS
#endif

// FULLSCPRODS counts the number of full scalar products during the sieve
// This only includes the sieving operation, not any context-switch operation
// levels are like XORPOPCNT
#ifndef COLLECT_STATISTICS_FULLSCPRODS
#define COLLECT_STATISTICS_FULLSCPRODS COLLECT_STATISTICS
#endif

// XORPOPCNT_PASS counts the number of times a simhash-xor-popcnt test was passed.
// Likely equal to the fullscprods statistics.
// Levels are like XORPOPCNT
#ifndef COLLECT_STATISTICS_XORPOPCNT_PASS
#define COLLECT_STATISTICS_XORPOPCNT_PASS COLLECT_STATISTICS
#endif

// FILTER_PASS counts the number of times a point is actually put in a bucket (2-sieve) or filtered_list (3-sieve)
// 0: Do not collect statistics
// 1: Collect statistics
#ifndef COLLECT_STATISTICS_FILTER_PASS
#if COLLECT_STATISTICS
    #define COLLECT_STATISTICS_FILTER_PASS 1
#else
    #define COLLECT_STATISTICS_FILTER_PASS 0
#endif
#endif

// REDSUCCCESS counts the number of successful reductions
// 0: Do not collect statistics
// 1: Collect statistics
// 2: Distinguish where the successful reduction came from
#ifndef COLLECT_STATISTICS_REDSUCCESS
#define COLLECT_STATISTICS_REDSUCCESS COLLECT_STATISTICS
#endif

// REPLACEMENTS counts the number of successful(!) db replacements done during the actual sieve.
// 0: no statistics collection
// 1: just total number
// 2: distinguish where the replacement vector came from (only meaningful for 3-sieve, atm)
#ifndef COLLECT_STATISTICS_REPLACEMENTS
#define COLLECT_STATISTICS_REPLACEMENTS COLLECT_STATISTICS
#endif

// REPLACEMENTFAILURES counts the number of replacement-failures during the actual sieve
// 0: no statistics collection
// 1: just total number
// 2: differentiate by reason for failure. TODO: Also by origin of vector?
#ifndef COLLECT_STATISTICS_REPLACEMENTFAILURES
#define COLLECT_STATISTICS_REPLACEMENTFAILURES COLLECT_STATISTICS
#endif

// OTFLIFTS counts the number of otflifts that were attempted
// 0: no statistics collection
// 1: just total number
// 2: TODO: differentiate by origin (and success?)
#ifndef COLLECT_STATISTICS_OTFLIFTS
#define COLLECT_STATISTICS_OTFLIFTS COLLECT_STATISTICS
#endif

// COLLISIONS counts the number of collisions we have encountered in the database
// 0: No statistics collections
// 1: just total number
// 2: source of collisions (i.e from where they originate)
#ifndef COLLECT_STATISTICS_COLLISIONS
#define COLLECT_STATISTICS_COLLISIONS COLLECT_STATISTICS
#endif

// REDS counts the number of attempted pairs / triples for which we tried a reduction.
// 0: No statistics collection
// 1: just total number
// 2: Differentiate pairs and triples (and inner/outer pairs). This is only meaningful for 3-sieves.
#ifndef COLLECT_STATISTICS_REDS
#define COLLECT_STATISTICS_REDS COLLECT_STATISTICS
#endif

// DATARACES counts the number of errors due to data races we have to recover from
// 0: No statistics collection
// 1: just total number
// 2: Distinguish source
#ifndef COLLECT_STATISTICS_DATARACES
#define COLLECT_STATISTICS_DATARACES COLLECT_STATISTICS
#endif

// SORTING counts the number of resorts that happen
// 0: No statistics collection
// 1: just total number
// 2: Distinguish source
#ifndef COLLECT_STATISTICS_SORTING
#define COLLECT_STATISTICS_SORTING COLLECT_STATISTICS
#endif

#ifndef COLLECT_STATISTICS_BUCKETS
#if COLLECT_STATISTICS
    #define COLLECT_STATISTICS_BUCKETS 1
#else
    #define COLLECT_STATISTICS_BUCKETS 0
#endif
#endif

#ifndef COLLECT_STATISTICS_MEMORY
#if COLLECT_STATISTICS
    #define COLLECT_STATISTICS_MEMORY 1
#else
    #define COLLECT_STATISTICS_MEMORY 0
#endif
#endif

/**
    ENABLE_IF_STATS_*(x) is equal to x if COLLECT_STATISTICS_* is != 0
    This is intended to make more concise statements in the rest of the code that are conditional
    on statistics collection (such as defining thread-local counters).
    Note that the actual operations on the global statistics objects do not need to be wrapped in such a macro:
    We use incrementers which default to no-ops (and let the compiler optimize away the call)
*/

// Note that ENABLE_IF_STATS_FOO(s) is defined as s and not s; The semicolon has to go inside the argument.
#if COLLECT_STATISTICS_COLLISIONS
    #define ENABLE_IF_STATS_COLLISIONS(s) s
#else
    #define ENABLE_IF_STATS_COLLISIONS(s)
#endif
#if COLLECT_STATISTICS_FULLSCPRODS
    #define ENABLE_IF_STATS_COLLISIONS(s) s
#else
    #define ENABLE_IF_STATS_COLLISIONS(s)
#endif // COLLECT_STATISTICS_FULLSCPRODS
#if COLLECT_STATISTICS_OTFLIFTS
    #define ENABLE_IF_STATS_OTFLIFTS(s) s
#else
    #define ENABLE_IF_STATS_OTFLIFTS(s)
#endif
#if COLLECT_STATISTICS_REDS
    #define ENABLE_IF_STATS_REDS(s) s
#else
    #define ENABLE_IF_STATS_REDS(s)
#endif
#if COLLECT_STATISTICS_REPLACEMENTFAILURES
    #define ENABLE_IF_STATS_REPLACEMENTFAILURES(s) s
#else
    #define ENABLE_IF_STATS_REPLACEMENTFAILURES(s)
#endif
#if COLLECT_STATISTICS_REPLACEMENTS
    #define ENABLE_IF_STATS_REPLACEMENTS(s) s
#else
    #define ENABLE_IF_STATS_REPLACEMENTS(s)
#endif // COLLECT_STATISTICS_REPLACEMENTS
#if COLLECT_STATISTICS_XORPOPCNT
    #define ENABLE_IF_STATS_XORPOPCNT(s) s
#else
    #define ENABLE_IF_STATS_XORPOPCNT(s)
#endif
#if COLLECT_STATISTICS_XORPOPCNT_PASS
    #define ENABLE_IF_STATS_XORPOPCNT_PASS(s) s
#else
    #define ENABLE_IF_STATS_XORPOPCNT_PASS(s)
#endif
#if COLLECT_STATISTICS_FILTER_PASS
    #define ENABLE_IF_STATS_FILTER_PASS(s) s
#else
    #define ENABLE_IF_STATS_FILTER_PASS(s)
#endif

#if COLLECT_STATISTICS_REDSUCCESS
    #define ENABLE_IF_STATS_REDSUCCESS(s) s
#else
    #define ENABLE_IF_STATS_REDSUCCESS(s)
#endif

#if COLLECT_STATISTICS_DATARACES
    #define ENABLE_IF_STATS_DATARACES(s) s
#else
    #define EANBLE_IF_STATS_DATARACES(s)
#endif

#if COLLECT_STATISTICS_SORTING
    #define ENABLE_IF_STATS_SORTING(s) s
#else
    #define ENABLE_IF_STATS_SORTING(s)
#endif

#if COLLECT_STATISTICS_BUCKETS
    #define ENABLE_IF_STATS_BUCKETS(s) s
#else
    #define ENABLE_IF_STATS_BUCKETS(s)
#endif

#if COLLECT_STATISTICS_MEMORY
    #define ENABLE_IF_STATS_MEMORY(s) s
#else
    #define ENABLE_IF_STATS_MEMORY(s)
#endif

/**
    Actual SieveStatistics class.
    It holds the statistics information as private data.
    We have public
        - getter functions get_stats_* to retrieve the data
        - static constexpr bool collect_statistics_* that tell whether the data is meaningful
            (This essentially allows to query what options we compiled with)
        - incrementer functions inc_stats_* to increment data.
        In some cases, we also have dec_stats_* to decrement data.
**/

class SieveStatistics
{

/**
    Data members:   No cacheline separation for now. This may be added later.
    Note1: For efficiency, use local counters and merge at thread termination.
    Note2: To reduce error-proneness, only data that are actually used are defined in a modifyable way.
            Our getters exist unconditionally, but return constant 0. Use collect_statistics_* to check whether the values are meaningful.
            The static constexpr 0's exist mainly to decltype() them, such that the signatures of the getters do not depend on flags.
    Note3: Our getters return non-atomic data types, so Cython and the python layer does not have to deal with atomics.


*/

private:
#if COLLECT_STATISTICS_REDS
    // we always internally distinguish the counts.
    // 2reds_outer counts 2-reduction attempts within directly from db.
    // 2reds_inner counts 2-reduction attempts within a bucket.
    // (So Bgj1 currently only has inner reductions, plain Gauss Sieve has only outer reductions)
    std::atomic_ulong   stats_2reds_outer;
    std::atomic_ulong   stats_2reds_inner;
    std::atomic_ulong   stats_3reds;
#else
    static constexpr unsigned long stats_2reds_outer = 0;
    static constexpr unsigned long stats_2reds_inner = 0;
    static constexpr unsigned long stats_3reds = 0;
#endif

#if   COLLECT_STATISTICS_XORPOPCNT
    std::atomic_ullong stats_xorpopcnt_outer; // bucketing
    std::atomic_ullong stats_xorpopcnt_inner; // actual test
#else
    static constexpr unsigned long long stats_xorpopcnt_outer = 0;
    static constexpr unsigned long long stats_xorpopcnt_inner = 0;
#endif

#if   COLLECT_STATISTICS_XORPOPCNT_PASS
    std::atomic_ullong stats_xorpopcnt_pass_outer; // bucketing
    std::atomic_ullong stats_xorpopcnt_pass_inner; // actual test
#else
    static constexpr unsigned long long stats_xorpopcnt_pass_outer = 0;
    static constexpr unsigned long long stats_xorpopcnt_pass_inner = 0;
#endif

#if   COLLECT_STATISTICS_FULLSCPRODS
    std::atomic_ullong stats_fullscprods_outer; // bucketing
    std::atomic_ullong stats_fullscprods_inner; // actual test
#else
    static constexpr unsigned long long stats_fullscprods_outer = 0;
    static constexpr unsigned long long stats_fullscprods_inner = 0;
#endif

#if COLLECT_STATISTICS_FILTER_PASS
    std::atomic_long stats_filter_pass;
#else
    static constexpr unsigned long stats_filter_pass = 0;
#endif

#if   COLLECT_STATISTICS_REPLACEMENTS
    std::atomic_ulong stats_replacements_list;
    std::atomic_ulong stats_replacements_queue;
    std::atomic_ulong stats_replacements_large; // triple_mt_only
    std::atomic_ulong stats_replacements_small; // triple_mt_only
#else
    static constexpr unsigned long stats_replacements_list = 0;
    static constexpr unsigned long stats_replacements_queue = 0;
    static constexpr unsigned long stats_replacements_large = 0; // triple_mt_only
    static constexpr unsigned long stats_replacements_small = 0; // triple_mt_only
#endif

#if   COLLECT_STATISTICS_REPLACEMENTFAILURES // most of this is only meaningful for triple_sieve_mt
    std::atomic_ulong stats_replacementfailures_list;
    std::atomic_ulong stats_replacementfailures_queue;
    std::atomic_ulong stats_replacementfailures_prune;
#else
    static constexpr unsigned long stats_replacementfailures_list = 0;
    static constexpr unsigned long stats_replacementfailures_queue = 0;
    // static constexpr unsigned long stats_replacementfailures_full = 0;
    static constexpr unsigned long stats_replacementfailures_prune = 0;
#endif

#if COLLECT_STATISTICS_OTFLIFTS
    std::atomic_ulong stats_otflifts_2inner;
    std::atomic_ulong stats_otflifts_2outer;
    std::atomic_ulong stats_otflifts_3;
#else
    static constexpr unsigned long stats_otflifts_2inner = 0;
    static constexpr unsigned long stats_otflifts_2outer = 0;
    static constexpr unsigned long stats_otflifts_3 = 0;
#endif

#if   COLLECT_STATISTICS_COLLISIONS
    std::atomic_ulong stats_collisions_2inner; // collisions encountered while creation new vectors during the main reduction phase.
    std::atomic_ulong stats_collisions_2outer;
    std::atomic_ulong stats_collisions_nobucket; // collisions preventing bucketing
    std::atomic_ulong stats_collisions_3;
#else
    static constexpr unsigned long stats_collisions_2inner = 0;
    static constexpr unsigned long stats_collisions_2outer = 0;
    static constexpr unsigned long stats_collisions_nobucket = 0;
    static constexpr unsigned long stats_collisions_3 = 0;
#endif

#if   COLLECT_STATISTICS_REDSUCCESS
    std::atomic_ulong stats_2redsuccess_inner;
    std::atomic_ulong stats_2redsuccess_outer;
    std::atomic_ulong stats_3redsuccess;
#else
    static constexpr unsigned long stats_2redsuccess_inner = 0;
    static constexpr unsigned long stats_2redsuccess_outer = 0;
    static constexpr unsigned long stats_3redsuccess = 0;
#endif

#if   COLLECT_STATISTICS_DATARACES
    std::atomic_ulong stats_dataraces_2inner;
    std::atomic_ulong stats_dataraces_2outer;
    std::atomic_ulong stats_dataraces_3;
    std::atomic_ulong stats_dataraces_replaced_was_saturated; // replaced db entry was already below saturation threshold. If that happens in bgj1, the saturation count becomes wrong. We track this to detect bugs.
    std::atomic_ulong stats_dataraces_sorting_blocked_cdb;
    std::atomic_ulong stats_dataraces_sorting_blocked_db;
    std::atomic_ulong stats_dataraces_get_p_blocked;
    std::atomic_ulong stats_dataraces_out_of_queue;
    std::atomic_ulong stats_dataraces_insertions;
#else
    static constexpr unsigned long stats_dataraces_2inner = 0;
    static constexpr unsigned long stats_dataraces_2outer = 0;
    static constexpr unsigned long stats_dataraces_3 = 0;
    static constexpr unsigned long stats_dataraces_replaced_was_saturated = 0;
    static constexpr unsigned long stats_dataraces_sorting_blocked_cdb = 0;
    static constexpr unsigned long stats_dataraces_sorting_blocked_db = 0;
    static constexpr unsigned long stats_dataraces_get_p_blocked = 0;
    static constexpr unsigned long stats_dataraces_out_of_queue = 0;
    static constexpr unsigned long stats_dataraces_insertions = 0;
#endif

#if COLLECT_STATISTICS_SORTING
    std::atomic_ulong stats_sorting_sieve;
#else
    static constexpr unsigned long stats_sorting_sieve = 0;
#endif

#if COLLECT_STATISTICS_BUCKETS
    std::atomic_ulong stats_buckets;
#else
    static constexpr unsigned long stats_buckets = 0;
#endif

#if COLLECT_STATISTICS_MEMORY
    std::atomic_ulong stats_memory_buckets;
    std::atomic_ulong stats_memory_transactions;
    std::atomic_ulong stats_memory_snapshots;
#else
    static constexpr unsigned long stats_memory_buckets = 0;
    static constexpr unsigned long stats_memory_transactions = 0;
    static constexpr unsigned long stats_memory_snapshots = 0; // we might meaningfully write 2 here for bgj1
#endif

/**
    To avoid at least some boilerplate, we use macros to create incrementers / getters:
    MAKE_ATOMIC_INCREMENTER(INCNAME,STAT) will create a
    function inc_stats_INCNAME(how_much) that increments stats_STAT by how_much (default:1)
    MAKE_ATOMIC_GETTER(GETTERNAME, STAT) will create a
    getter function get_stats_GETTERNAME() that returns stats_STAT
    MAKE_NOOP_INCREMENTER(INCNAME) create an incrementer that does nothing

    MAKE_INCREMENTER_FOR(INCNAME, STAT, NONTRIVIAL) will alias to MAKE_ATOMIC_GETTER if NONTRIVIAL is small positive preprocessor constant
    and alias MAKE_NOOP_INCREMENTER if NONTRIVIAL is the preprocessor constant 0.
    MAKE_INCREMENTER(NAME,NONTRIVIAL) is the same, but with INCNAME == NAME == STAT
    MAKE_GETTER_FOR(GETTERNAME, STAT, NONTRIVIAL) will create a getter function or a trivial getter function, depending on NONTRIVIAL
    MAKE_GETTER(GETTERNAME, NONTRIVIAL) will do the same, with STAT == GETTERNAME
    MAKE_GETTER_AND_INCREMENTER(NAME, NONTRIVIAL) will create both getter and incrementer, depending on NONTRIVIAL.
*/

#define MAKE_ATOMIC_INCREMENTER(INCNAME, STAT) \
void inc_stats_ ## INCNAME( mystd::decay_t<decltype(stats_##STAT.load())> how_much = 1) noexcept { stats_##STAT.fetch_add(how_much, std::memory_order_relaxed); }

#define MAKE_ATOMIC_DECREMENTER(INCNAME, STAT) \
void dec_stats_ ## INCNAME( mystd::decay_t<decltype(stats_##STAT.load())> how_much = 1) noexcept { stats_##STAT.fetch_sub(how_much, std::memory_order_relaxed); }

#define MAKE_ATOMIC_GETTER(GETTERNAME, STAT) \
FORCE_INLINE mystd::decay_t<decltype(stats_##STAT.load())> get_stats_##GETTERNAME() const noexcept { return stats_##STAT.load(); }

#define MAKE_ATOMIC_SETTER(SETTERNAME, STAT) \
void set_stats_##SETTERNAME(mystd::decay_t<decltype(stats_##STAT.load())> const new_val) noexcept {stats_##STAT.store(new_val); }

#define MAKE_NOOP_INCREMENTER(INCNAME) \
template<class Arg> FORCE_INLINE static void inc_stats_##INCNAME(Arg) noexcept {} \
FORCE_INLINE static void inc_stats_##INCNAME() noexcept {}

#define MAKE_NOOP_DECREMENTER(INCNAME) \
template<class Arg> FORCE_INLINE static void dec_stats_##INCNAME(Arg) noexcept {} \
FORCE_INLINE static void dec_stats_##INCNAME() noexcept {}

#define MAKE_NOOP_SETTER(SETTERNAME) \
template<class Arg> FORCE_INLINE static void set_stats_##SETTERNAME(Arg) noexcept {} \
FORCE_INLINE static void set_stats_##SETTERNAME() noexcept {}


/** Totally evil hackery to work around lack of C++ constexpr if (or #if's inside macro definitions...)
    Some gcc version might actually allow #if's inside macros, but we prefer portability. **/

#define MAKE_INCREMENTER_FOR(INCNAME, STAT, NONTRIVIAL) MAKE_INCREMENTER_AUX(INCNAME, STAT, NONTRIVIAL) // to macro-expand the name "NONTRIVIAL", such that token-pasting in the following macro is done AFTER macro expansion.
#define MAKE_INCREMENTER(INCNAME, NONTRIVIAL) MAKE_INCREMENTER_AUX(INCNAME, INCNAME, NONTRIVIAL)
#define MAKE_INCREMENTER_AUX(INCNAME, STAT, NONTRIVIAL) MAKE_INCREMENTER_##NONTRIVIAL(INCNAME, STAT) // This is evil
#define MAKE_INCREMENTER_0(INCNAME, STAT) MAKE_NOOP_INCREMENTER(INCNAME)
#define MAKE_INCREMENTER_1(INCNAME, STAT) MAKE_ATOMIC_INCREMENTER(INCNAME, STAT)
#define MAKE_INCREMENTER_2(INCNAME, STAT) MAKE_ATOMIC_INCREMENTER(INCNAME, STAT)
#define MAKE_INCREMENTER_3(INCNAME, STAT) MAKE_ATOMIC_INCREMENTER(INCNAME, STAT)
#define MAKE_INCREMENTER_4(INCNAME, STAT) MAKE_ATOMIC_INCREMENTER(INCNAME, STAT)

#define MAKE_DECREMENTER(NAME, NONTRIVIAL) MAKE_DECREMENTER_AUX(NAME, NAME, NONTRIVIAL)
#define MAKE_DECREMENTER_AUX(NAME, STAT, NONTRIVIAL) MAKE_DECREMENTER_##NONTRIVIAL(NAME, STAT)
#define MAKE_DECREMENTER_0(NAME, STAT) MAKE_NOOP_DECREMENTER(NAME)
#define MAKE_DECREMENTER_1(NAME, STAT) MAKE_ATOMIC_DECREMENTER(NAME, STAT)
#define MAKE_DECREMENTER_2(NAME, STAT) MAKE_ATOMIC_DECREMENTER(NAME, STAT)
#define MAKE_DECREMENTER_3(NAME, STAT) MAKE_ATOMIC_DECREMENTER(NAME, STAT)
#define MAKE_DECREMENTER_4(NAME, STAT) MAKE_ATOMIC_DECREMENTER(NAME, STAT)

#define MAKE_GETTER_FOR(GETTERNAME, STAT, NONTRIVIAL) MAKE_GETTER_AUX(GETTERNAME, STAT, NONTRIVIAL)
#define MAKE_GETTER(GETTERNAME, NONTRIVIAL) MAKE_GETTER_AUX(GETTERNAME, GETTERNAME, NONTRIVIAL)
#define MAKE_GETTER_AUX(GETTERNAME, STAT, NONTRIVIAL) MAKE_GETTER_##NONTRIVIAL(GETTERNAME, STAT)
#define MAKE_GETTER_0(GETTERNAME, STAT) \
FORCE_INLINE static constexpr auto get_stats_##GETTERNAME() noexcept -> mystd::remove_cv_t<decltype(stats_##STAT)> { return stats_##STAT; }
#define MAKE_GETTER_1(GETTERNAME, STAT) MAKE_ATOMIC_GETTER(GETTERNAME, STAT)
#define MAKE_GETTER_2(GETTERNAME, STAT) MAKE_ATOMIC_GETTER(GETTERNAME, STAT)
#define MAKE_GETTER_3(GETTERNAME, STAT) MAKE_ATOMIC_GETTER(GETTERNAME, STAT)
#define MAKE_GETTER_4(GETTERNAME, STAT) MAKE_ATOMIC_GETTER(GETTERNAME, STAT)

#define MAKE_SETTER_FOR(SETTERNAME, STAT, NONTRIVIAL) MAKE_SETTER_AUX(SETTERNAME, STAT, NONTRIVIAL)
#define MAKE_SETTER(SETTERNAME, NONTRIVIAL) MAKE_SETTER_AUX(SETTERNAME, SETTERNAME, NONTRIVIAL)
#define MAKE_SETTER_AUX(SETTERNAME, STAT, NONTRIVIAL) MAKE_SETTER_##NONTRIVIAL(SETTERNAME, STAT)
#define MAKE_SETTER_0(SETTERNAME, STAT) MAKE_NOOP_SETTER(SETTERNAME)
#define MAKE_SETTER_1(SETTERNAME, STAT) MAKE_ATOMIC_SETTER(SETTERNAME, STAT)
#define MAKE_SETTER_2(SETTERNAME, STAT) MAKE_ATOMIC_SETTER(SETTERNAME, STAT)
#define MAKE_SETTER_3(SETTERNAME, STAT) MAKE_ATOMIC_SETTER(SETTERNAME, STAT)
#define MAKE_SETTER_4(SETTERNAME, STAT) MAKE_ATOMIC_SETTER(SETTERNAME, STAT)

#define MAKE_GETTER_AND_INCREMENTER(NAME, NONTRIVIAL) \
MAKE_INCREMENTER(NAME, NONTRIVIAL) \
MAKE_GETTER(NAME, NONTRIVIAL)

public:
    static constexpr int collect_statistics_level = COLLECT_STATISTICS;

/** REDS **/
    // collect_statistics_* is set iff get_stats_* works.
    static constexpr bool collect_statistics_reds_total  = (COLLECT_STATISTICS_REDS >= 1);
    static constexpr bool collect_statistics_2reds_total = (COLLECT_STATISTICS_REDS >= 2);
    static constexpr bool collect_statistics_3reds       = (COLLECT_STATISTICS_REDS >= 2);
    static constexpr bool collect_statistics_2reds_inner = (COLLECT_STATISTICS_REDS >= 2);
    static constexpr bool collect_statistics_2reds_outer = (COLLECT_STATISTICS_REDS >= 2);
    MAKE_GETTER_AND_INCREMENTER(2reds_inner, COLLECT_STATISTICS_REDS)
    MAKE_GETTER_AND_INCREMENTER(2reds_outer, COLLECT_STATISTICS_REDS)
    MAKE_GETTER_AND_INCREMENTER(3reds, COLLECT_STATISTICS_REDS)
    unsigned long get_stats_2reds_total() const { return get_stats_2reds_inner() + get_stats_2reds_outer(); }
    unsigned long get_stats_reds_total() const  { return get_stats_2reds_total() + get_stats_3reds(); }

/** XORPOPCNT **/
    static constexpr bool collect_statistics_xorpopcnt_total  = (COLLECT_STATISTICS_XORPOPCNT >= 1);
    static constexpr bool collect_statistics_xorpopcnt_inner = (COLLECT_STATISTICS_XORPOPCNT >= 2);
    static constexpr bool collect_statistics_xorpopcnt_outer = (COLLECT_STATISTICS_XORPOPCNT >= 2);
    MAKE_GETTER_AND_INCREMENTER(xorpopcnt_inner, COLLECT_STATISTICS_XORPOPCNT)
    MAKE_GETTER_AND_INCREMENTER(xorpopcnt_outer, COLLECT_STATISTICS_XORPOPCNT)
    unsigned long long get_stats_xorpopcnt_total() const { return get_stats_xorpopcnt_inner() + get_stats_xorpopcnt_outer(); }

/** XORPOPCNT_PASS **/
    static constexpr bool collect_statistics_xorpopcnt_pass_total = (COLLECT_STATISTICS_XORPOPCNT_PASS >= 1);
    static constexpr bool collect_statistics_xorpopcnt_pass_inner = (COLLECT_STATISTICS_XORPOPCNT_PASS >= 2);
    static constexpr bool collect_statistics_xorpopcnt_pass_outer = (COLLECT_STATISTICS_XORPOPCNT_PASS >= 2);
    MAKE_GETTER_AND_INCREMENTER(xorpopcnt_pass_inner, COLLECT_STATISTICS_XORPOPCNT_PASS)
    MAKE_GETTER_AND_INCREMENTER(xorpopcnt_pass_outer, COLLECT_STATISTICS_XORPOPCNT_PASS)
    unsigned long long get_stats_xorpopcnt_pass_total() const { return get_stats_xorpopcnt_pass_inner() + get_stats_xorpopcnt_pass_outer(); }

/** FULLSCPRODS **/
    static constexpr bool collect_statistics_fullscprods_total = (COLLECT_STATISTICS_FULLSCPRODS >= 1);
    static constexpr bool collect_statistics_fullscprods_inner = (COLLECT_STATISTICS_FULLSCPRODS >= 2);
    static constexpr bool collect_statistics_fullscprods_outer = (COLLECT_STATISTICS_FULLSCPRODS >= 2);
    MAKE_GETTER_AND_INCREMENTER(fullscprods_inner, COLLECT_STATISTICS_FULLSCPRODS)
    MAKE_GETTER_AND_INCREMENTER(fullscprods_outer, COLLECT_STATISTICS_FULLSCPRODS)
    MAKE_DECREMENTER(fullscprods_outer, COLLECT_STATISTICS_FULLSCPRODS)
    unsigned long long get_stats_fullscprods_total() const { return get_stats_fullscprods_inner() + get_stats_fullscprods_outer(); }

/** FILTER_PASS **/
    static constexpr bool collect_statistics_filter_pass = (COLLECT_STATISTICS_FILTER_PASS >= 1);
    MAKE_GETTER_AND_INCREMENTER(filter_pass, COLLECT_STATISTICS_FILTER_PASS)

/** REDSUCCESS  **/
    static constexpr bool collect_statistics_redsuccess_total = (COLLECT_STATISTICS_REDSUCCESS >= 1);
    static constexpr bool collect_statistics_2redsuccess_total= (COLLECT_STATISTICS_REDSUCCESS >= 1);
    static constexpr bool collect_statistics_2redsuccess_inner = (COLLECT_STATISTICS_REDSUCCESS >= 2);
    static constexpr bool collect_statistics_2redsuccess_outer = (COLLECT_STATISTICS_REDSUCCESS >= 2);
    static constexpr bool collect_statistics_3redsuccess= (COLLECT_STATISTICS_REDSUCCESS >= 1);
    MAKE_GETTER_AND_INCREMENTER(2redsuccess_inner, COLLECT_STATISTICS_REDSUCCESS)
    MAKE_GETTER_AND_INCREMENTER(2redsuccess_outer, COLLECT_STATISTICS_REDSUCCESS)
    MAKE_GETTER_AND_INCREMENTER(3redsuccess, COLLECT_STATISTICS_REDSUCCESS)
    unsigned long get_stats_2redsuccess_total() const { return get_stats_2redsuccess_inner() + get_stats_2redsuccess_outer(); }
    unsigned long get_stats_redsuccess_total() const   { return get_stats_2redsuccess_total() + get_stats_3redsuccess(); }

/** DATARACES **/
    static constexpr bool collect_statistics_dataraces_total  =                 (COLLECT_STATISTICS_DATARACES >= 1);
    static constexpr bool collect_statistics_dataraces_2inner =                 (COLLECT_STATISTICS_DATARACES >= 2);
    static constexpr bool collect_statistics_dataraces_2outer =                 (COLLECT_STATISTICS_DATARACES >= 2);
    static constexpr bool collect_statistics_dataraces_3 =                      (COLLECT_STATISTICS_DATARACES >= 2);
    static constexpr bool collect_statistics_dataraces_replaced_was_saturated = (COLLECT_STATISTICS_DATARACES >= 2);
    static constexpr bool collect_statistics_dataraces_sorting_blocked_cdb =    (COLLECT_STATISTICS_DATARACES >= 2);
    static constexpr bool collect_statistics_dataraces_sorting_blocked_db  =    (COLLECT_STATISTICS_DATARACES >= 2);
    static constexpr bool collect_statistics_dataraces_get_p_blocked =          (COLLECT_STATISTICS_DATARACES >= 2);
    static constexpr bool collect_statistics_dataraces_out_of_queue =           (COLLECT_STATISTICS_DATARACES >= 2);
    static constexpr bool collect_statistics_dataraces_insertions =             (COLLECT_STATISTICS_DATARACES >= 2);

    MAKE_GETTER_AND_INCREMENTER(dataraces_2inner,                   COLLECT_STATISTICS_DATARACES)
    MAKE_GETTER_AND_INCREMENTER(dataraces_2outer,                   COLLECT_STATISTICS_DATARACES)
    MAKE_GETTER_AND_INCREMENTER(dataraces_3,                        COLLECT_STATISTICS_DATARACES)
    MAKE_GETTER_AND_INCREMENTER(dataraces_replaced_was_saturated,   COLLECT_STATISTICS_DATARACES)
    MAKE_GETTER_AND_INCREMENTER(dataraces_sorting_blocked_cdb,      COLLECT_STATISTICS_DATARACES)
    MAKE_GETTER_AND_INCREMENTER(dataraces_sorting_blocked_db,       COLLECT_STATISTICS_DATARACES)
    MAKE_GETTER_AND_INCREMENTER(dataraces_get_p_blocked,            COLLECT_STATISTICS_DATARACES)
    MAKE_GETTER_AND_INCREMENTER(dataraces_out_of_queue,             COLLECT_STATISTICS_DATARACES)
    MAKE_GETTER_AND_INCREMENTER(dataraces_insertions,               COLLECT_STATISTICS_DATARACES)
    unsigned long get_stats_dataraces_total() const
    {
        return  get_stats_dataraces_2inner() + get_stats_dataraces_replaced_was_saturated()
              + get_stats_dataraces_2outer() + get_stats_dataraces_3()
              + get_stats_dataraces_sorting_blocked_cdb() + get_stats_dataraces_sorting_blocked_db()
              + get_stats_dataraces_get_p_blocked() + get_stats_dataraces_out_of_queue()
              + get_stats_dataraces_insertions();
    }

/** COLLISIONS **/
    static constexpr bool collect_statistics_collisions_total = (COLLECT_STATISTICS_COLLISIONS >= 1);
    static constexpr bool collect_statistics_collisions_2inner= (COLLECT_STATISTICS_COLLISIONS >= 2);
    static constexpr bool collect_statistics_collisions_2outer= (COLLECT_STATISTICS_COLLISIONS >= 2);
    static constexpr bool collect_statistics_collisions_nobucket = (COLLECT_STATISTICS_COLLISIONS >= 2);
    static constexpr bool collect_statistics_collisions_3 = (COLLECT_STATISTICS_COLLISIONS >= 2);
    MAKE_GETTER_AND_INCREMENTER(collisions_2inner, COLLECT_STATISTICS_COLLISIONS)
    MAKE_GETTER_AND_INCREMENTER(collisions_2outer, COLLECT_STATISTICS_COLLISIONS)
    MAKE_GETTER_AND_INCREMENTER(collisions_nobucket, COLLECT_STATISTICS_COLLISIONS)
    MAKE_GETTER_AND_INCREMENTER(collisions_3, COLLECT_STATISTICS_COLLISIONS)
    MAKE_DECREMENTER(collisions_2outer, COLLECT_STATISTICS_COLLISIONS)
    MAKE_DECREMENTER(collisions_2inner, COLLECT_STATISTICS_COLLISIONS)
    MAKE_DECREMENTER(collisions_3, COLLECT_STATISTICS_COLLISIONS)
    unsigned long get_stats_collisions_total() const
    {
        return get_stats_collisions_2inner() + get_stats_collisions_2outer() + get_stats_collisions_nobucket()
                + get_stats_collisions_3();
    }

/** OTFLIFTS **/
    static constexpr bool collect_statistics_otflifts_total  = (COLLECT_STATISTICS_OTFLIFTS >= 1);
    static constexpr bool collect_statistics_otflifts_2inner = (COLLECT_STATISTICS_OTFLIFTS >= 2);
    static constexpr bool collect_statistics_otflifts_2outer = (COLLECT_STATISTICS_OTFLIFTS >= 2);
    static constexpr bool collect_statistics_otflifts_3 =(COLLECT_STATISTICS_OTFLIFTS >= 2);
    MAKE_GETTER_AND_INCREMENTER(otflifts_2inner, COLLECT_STATISTICS_OTFLIFTS)
    MAKE_GETTER_AND_INCREMENTER(otflifts_2outer, COLLECT_STATISTICS_OTFLIFTS)
    MAKE_GETTER_AND_INCREMENTER(otflifts_3, COLLECT_STATISTICS_OTFLIFTS)
    unsigned long get_stats_otflifts_total() const { return get_stats_otflifts_2inner() + get_stats_otflifts_2outer() + get_stats_otflifts_3(); }

/** REPLACEMENTS */
    static constexpr bool collect_statistics_replacements_total = (COLLECT_STATISTICS_REPLACEMENTS >= 1);
    static constexpr bool collect_statistics_replacements_list  = (COLLECT_STATISTICS_REPLACEMENTS >= 2);
    static constexpr bool collect_statistics_replacements_queue = (COLLECT_STATISTICS_REPLACEMENTS >= 2);
    static constexpr bool collect_statistics_replacements_large = (COLLECT_STATISTICS_REPLACEMENTS >= 2);
    static constexpr bool collect_statistics_replacements_small = (COLLECT_STATISTICS_REPLACEMENTS >= 2);
    MAKE_GETTER_AND_INCREMENTER(replacements_list,  COLLECT_STATISTICS_REPLACEMENTS)
    MAKE_GETTER_AND_INCREMENTER(replacements_queue, COLLECT_STATISTICS_REPLACEMENTS)
    MAKE_GETTER_AND_INCREMENTER(replacements_large, COLLECT_STATISTICS_REPLACEMENTS)
    MAKE_GETTER_AND_INCREMENTER(replacements_small, COLLECT_STATISTICS_REPLACEMENTS)
    // _large  / _small is intentionally not added
    unsigned long get_stats_replacements_total() const { return get_stats_replacements_list() + get_stats_replacements_queue(); }

/** REPLACEMENTFAILURES */
    static constexpr bool collect_statistics_replacementfailures_total =    (COLLECT_STATISTICS_REPLACEMENTFAILURES  >= 1);
    static constexpr bool collect_statistics_replacementfailures_list =     (COLLECT_STATISTICS_REPLACEMENTFAILURES >= 2);
    static constexpr bool collect_statistics_replacementfailures_queue =    (COLLECT_STATISTICS_REPLACEMENTFAILURES >= 2);
    // static constexpr bool collect_statistics_replacementfailures_full =     (COLLECT_STATISTICS_REPLACEMENTFAILURES >= 2);
    static constexpr bool collect_statistics_replacementfailures_prune =    (COLLECT_STATISTICS_REPLACEMENTFAILURES >= 2);
    MAKE_GETTER_AND_INCREMENTER(replacementfailures_list,   COLLECT_STATISTICS_REPLACEMENTFAILURES)
    MAKE_GETTER_AND_INCREMENTER(replacementfailures_queue,  COLLECT_STATISTICS_REPLACEMENTFAILURES)
    // MAKE_GETTER_AND_INCREMENTER(replacementfailures_full,   COLLECT_STATISTICS_REPLACEMENTFAILURES) // deprecated
    MAKE_GETTER_AND_INCREMENTER(replacementfailures_prune,  COLLECT_STATISTICS_REPLACEMENTFAILURES)
    unsigned long get_stats_replacementfailures_total() const
    {
        return get_stats_replacementfailures_list() + get_stats_replacementfailures_queue() + get_stats_replacementfailures_prune();
    }
/** SORTING */
    static constexpr bool collect_statistics_sorting_total = (COLLECT_STATISTICS_SORTING >= 1);
    static constexpr bool collect_statistics_sorting_sieve = (COLLECT_STATISTICS_SORTING >= 2);
    MAKE_GETTER_AND_INCREMENTER(sorting_sieve, COLLECT_STATISTICS_SORTING)
    unsigned long get_stats_sorting_total() const { return get_stats_sorting_sieve(); }

/** BUCKETS */
    static constexpr bool collect_statistics_buckets = (COLLECT_STATISTICS_BUCKETS >= 1);
    MAKE_GETTER_AND_INCREMENTER(buckets, COLLECT_STATISTICS_BUCKETS)

/** MEMORY  */
    static constexpr bool collect_statistics_memory_buckets =       (COLLECT_STATISTICS_MEMORY >= 1);
    static constexpr bool collect_statistics_memory_transactions =  (COLLECT_STATISTICS_MEMORY >= 1);
    static constexpr bool collect_statistics_memory_snapshots =     (COLLECT_STATISTICS_MEMORY >= 1);
    MAKE_GETTER_AND_INCREMENTER(memory_buckets,         COLLECT_STATISTICS_MEMORY)
    MAKE_GETTER_AND_INCREMENTER(memory_transactions,    COLLECT_STATISTICS_MEMORY)
    MAKE_GETTER_AND_INCREMENTER(memory_snapshots,       COLLECT_STATISTICS_MEMORY)
    MAKE_SETTER(memory_buckets,                         COLLECT_STATISTICS_MEMORY)
    MAKE_SETTER(memory_transactions,                    COLLECT_STATISTICS_MEMORY)
    MAKE_SETTER(memory_snapshots,                       COLLECT_STATISTICS_MEMORY)

    inline void clear_statistics() noexcept
    {
    #if COLLECT_STATISTICS_REDS
        stats_2reds_outer = 0;
        stats_2reds_inner = 0;
        stats_3reds = 0;
    #endif

    #if COLLECT_STATISTICS_XORPOPCNT
        stats_xorpopcnt_outer = 0;
        stats_xorpopcnt_inner = 0;
    #endif

    #if COLLECT_STATISTICS_XORPOPCNT_PASS
        stats_xorpopcnt_pass_outer = 0;
        stats_xorpopcnt_pass_inner = 0;
    #endif

    #if COLLECT_STATISTICS_FULLSCPRODS
        stats_fullscprods_outer = 0;
        stats_fullscprods_inner = 0;
    #endif

    #if COLLECT_STATISTICS_FILTER_PASS
        stats_filter_pass = 0;
    #endif

    #if COLLECT_STATISTICS_REPLACEMENTS
        stats_replacements_list = 0;
        stats_replacements_queue = 0;
        stats_replacements_small = 0;
        stats_replacements_large = 0;
    #endif

    #if COLLECT_STATISTICS_REPLACEMENTFAILURES
        stats_replacementfailures_queue = 0;
        stats_replacementfailures_list = 0;
        // stats_replacementfailures_full = 0;
        stats_replacementfailures_prune = 0;

    #endif

    #if COLLECT_STATISTICS_OTFLIFTS
        stats_otflifts_2inner = 0;
        stats_otflifts_2outer = 0;
        stats_otflifts_3 = 0;
    #endif

    #if COLLECT_STATISTICS_COLLISIONS
        stats_collisions_2inner = 0;
        stats_collisions_2outer = 0;
        stats_collisions_nobucket = 0;
        stats_collisions_3 = 0;
    #endif

    #if COLLECT_STATISTICS_REDSUCCESS
        stats_2redsuccess_inner = 0;
        stats_2redsuccess_outer = 0;
        stats_3redsuccess = 0;
    #endif

    #if COLLECT_STATISTICS_DATARACES
        stats_dataraces_2inner = 0;
        stats_dataraces_2outer = 0;
        stats_dataraces_3 = 0;
        stats_dataraces_replaced_was_saturated = 0;
        stats_dataraces_sorting_blocked_cdb = 0;
        stats_dataraces_sorting_blocked_db = 0;
        stats_dataraces_get_p_blocked = 0;
        stats_dataraces_out_of_queue = 0;
        stats_dataraces_insertions = 0;
    #endif

    #if COLLECT_STATISTICS_SORTING
        stats_sorting_sieve = 0;
    #endif

    #if COLLECT_STATISTICS_BUCKETS
        stats_buckets = 0;
    #endif

    #if COLLECT_STATISTICS_MEMORY
        stats_memory_buckets = 0;
        stats_memory_transactions = 0;
        stats_memory_snapshots = 0;
    #endif
    }


    enum class StatisticsOutputForAlg : int
    {
        none = 0,
        bgj1 = 1,
        triple_mt = 2
    };

    void print_statistics(int alg, std::ostream &os = std::cout)
    {
        return print_statistics(static_cast<StatisticsOutputForAlg>(alg), os);
    }

#define STATS_PRINT_IF(NAME, STRING)\
if(collect_statistics_##NAME) { os << STRING << get_stats_##NAME(); }

    void print_statistics(StatisticsOutputForAlg alg, std::ostream &os = std::cout)
    {
        switch(alg)
        {
        case StatisticsOutputForAlg::bgj1 :
            if(collect_statistics_reds_total)
            {
                os << "Reduction attempts: " << get_stats_reds_total();
                STATS_PRINT_IF(2reds_outer, ", while bucketing: ")
                STATS_PRINT_IF(2reds_inner, ", while sieveing: ")
                os << "\n";
            }
            if(collect_statistics_collisions_total)
            {
                os << "Collisions: " << get_stats_collisions_total();
                STATS_PRINT_IF(collisions_2inner,", while bucketing: ")
                STATS_PRINT_IF(collisions_2outer,", while sieving: ")
                os << "\n";
            }
            if(collect_statistics_xorpopcnt_total)
            {
                os << "XOR-Popcounts: " << get_stats_xorpopcnt_total();
                if(collect_statistics_xorpopcnt_outer)
                    os << ", bucketing: " << get_stats_xorpopcnt_outer();
                if(collect_statistics_xorpopcnt_inner)
                    os << ", sieving: " << get_stats_xorpopcnt_inner();
                os << "\n";
            }
            if(collect_statistics_xorpopcnt_pass_total)
            {
                os << "Successful XOR-Popcount: " << get_stats_xorpopcnt_pass_total();
                if(collect_statistics_xorpopcnt_pass_outer)
                    os << ", bucketing: " << get_stats_xorpopcnt_pass_outer();
                if(collect_statistics_xorpopcnt_pass_inner)
                    os << ", sieving: " <<   get_stats_xorpopcnt_pass_inner();
                os << "\n";
            }
            if(collect_statistics_fullscprods_total)
            {
                os << "#full scalar products: " << get_stats_fullscprods_total();
                if(collect_statistics_fullscprods_outer)
                    os << ", bucketing: " << get_stats_fullscprods_outer();
                if(collect_statistics_fullscprods_inner)
                    os << ", bucketing: " << get_stats_fullscprods_inner();
                os << "\n";
            }
            if(collect_statistics_redsuccess_total)
            {
                os << "Successful reductions: " << get_stats_redsuccess_total();
                STATS_PRINT_IF(2redsuccess_outer, ", while bucketing: ")
                STATS_PRINT_IF(2redsuccess_inner, ", while sieving: ")
                os << "\n";
            }
            if(collect_statistics_replacements_total)
            {
                os << "db replacements: " << get_stats_replacements_total();
                os << "\n";
            }
            if(collect_statistics_replacementfailures_total)
            {
                os <<"Replacement failures: " << get_stats_replacementfailures_total();
                os <<"\n";
            }
            if(collect_statistics_otflifts_total)
            {
                os << "Otflift attempts: " << get_stats_otflifts_total();
                STATS_PRINT_IF(otflifts_2outer,", while bucketing: ")
                STATS_PRINT_IF(otflifts_2inner,", while sieving: ")
                os << "\n";
            }
            if(collect_statistics_dataraces_total)
            {
                os << "Total Error events: " << get_stats_dataraces_total();
                STATS_PRINT_IF(dataraces_2outer, ", dataraces in reds while bucketing: ")
                STATS_PRINT_IF(dataraces_2inner, ", dataraces in reds while sieving: ")
                if(collect_statistics_dataraces_replaced_was_saturated)
                    os << ", replacing saturated vector" << get_stats_dataraces_replaced_was_saturated();
                os << "\n";
            }
            if(collect_statistics_sorting_total)
            {
                os << "Sortings performed during sieve: " << get_stats_sorting_total();
                os << "\n";
            }
            if(collect_statistics_buckets)
            {
                os << "Number of buckets considered: " << get_stats_buckets();
                os << "\n";
            }
            break;

        /** triple_mt **/

        case StatisticsOutputForAlg::triple_mt :
            if(collect_statistics_reds_total)
            {
                os << "Total reduction attempts: " << get_stats_reds_total();
                if(collect_statistics_2reds_outer)
                    os <<" outer 2-reds: " << get_stats_2reds_outer();
                if(collect_statistics_2reds_inner)
                    os <<" inner 2-reds: " << get_stats_2reds_inner();
                if(collect_statistics_3reds)
                    os <<" 3-reds: " << get_stats_3reds();
                os << "\n";
            }
            if(collect_statistics_collisions_total)
            {
                os << "Total collisions: " << get_stats_collisions_total();
                STATS_PRINT_IF(collisions_nobucket, ", no bucket: ")
                STATS_PRINT_IF(collisions_2outer, ", outer 2-reds: ")
                STATS_PRINT_IF(collisions_2inner, ", inner 2-reds: ")
                STATS_PRINT_IF(collisions_3, ", 3-reds: ")
                os << "\n";
            }
            if(collect_statistics_xorpopcnt_total)
            {
                os << "Total #XOR-POPCOUNT" << get_stats_xorpopcnt_total();
                STATS_PRINT_IF(xorpopcnt_outer, ", filtering: ")
                STATS_PRINT_IF(xorpopcnt_inner, ", sieving: ")
                os <<"\n";
            }
            if(collect_statistics_xorpopcnt_pass_total)
            {
                os << "Total successful XOR-POPCOUNTS" << get_stats_xorpopcnt_pass_total();
                STATS_PRINT_IF(xorpopcnt_pass_outer, ", filtering: ")
                STATS_PRINT_IF(xorpopcnt_pass_inner, ", sieving: ")
                os << "\n";
            }
            if(collect_statistics_fullscprods_total)
            {
                os << "Total #Scalar Products in Sieve: " << get_stats_fullscprods_total();
                STATS_PRINT_IF(fullscprods_outer, ", filtering: ")
                STATS_PRINT_IF(fullscprods_inner, ", sieving: ")
                os << "\n";
            }
            if(collect_statistics_redsuccess_total)
            {
                os << "Total Successful reductions: " << get_stats_redsuccess_total();
                STATS_PRINT_IF(2redsuccess_outer, ", outer 2-reds: ")
                STATS_PRINT_IF(2redsuccess_inner, ", inner 2-reds: ")
                STATS_PRINT_IF(3redsuccess, ", 3-reds: ")
                os << "\n";
            }
            if(collect_statistics_replacements_total)
            {
                os << "Total #replacements: " << get_stats_replacements_total();
                STATS_PRINT_IF(replacements_queue, ", queue part: ")
                STATS_PRINT_IF(replacements_list, ", list part: ")
                os << "\n";
            }
            if(collect_statistics_replacementfailures_total)
            {
                os <<"Total #replacement failures: " << get_stats_replacementfailures_total();
                STATS_PRINT_IF(replacementfailures_queue, ", queue part: ")
                STATS_PRINT_IF(replacementfailures_list, ", list part: ")
                STATS_PRINT_IF(replacementfailures_prune, ", prune: ")
                os << "\n";
            }
            if(collect_statistics_otflifts_total)
            {
                os << "Total otflift attempts:  " << get_stats_otflifts_total();
                STATS_PRINT_IF(otflifts_2outer, ", bucketing: ")
                STATS_PRINT_IF(otflifts_2inner, ", sieving: ")
                STATS_PRINT_IF(otflifts_3, "triple lifts: ")
                os << "\n";
            }
            if(collect_statistics_dataraces_total)
            {
                os << "Error events: " << get_stats_dataraces_total();
                STATS_PRINT_IF(dataraces_2outer, ", bucketing: ")
                STATS_PRINT_IF(dataraces_2inner, ", sieving: ")
                STATS_PRINT_IF(dataraces_3, "triple: ")
                STATS_PRINT_IF(dataraces_replaced_was_saturated, "over-replace: ")
                os << "\n";
            }
            if(collect_statistics_sorting_sieve)
            {
                os << "Sortings during sieve: " << get_stats_sorting_sieve() << "\n";
            }
            if(collect_statistics_buckets)
            {
                os << "Number of Filtered lists: " << get_stats_buckets() << "\n";
            }
            break;
        default:
                os << "Unrecognized algorithm for statistics output.\n";
            break;
        }

    }

    std::string get_statistics_string(int alg) { return get_statistics_string(static_cast<StatisticsOutputForAlg>(alg)); }
    std::string get_statistics_string(StatisticsOutputForAlg alg)
    {
        std::stringstream out;
        print_statistics(alg, out);
        return out.str();
    }

};

/**
    MergeOnExit<Int,Functor> wraps an Int x (with very limited functionality as of now) and Functor fun
    and call fun(x) upon destruction.
    The intented use case is to create a thread-local counter local_counter_foo
    with a lambda fun = [](Int x){ global_counter_foo += x; }

    To pass parameters, use the merge_on_exit helper function.
    E.g.
    auto &&local_counter_foo = merge_on_exit<int>( [](int x){global_counter_foo += x;} );
    auto &&local_statistic_foo = merge_on_exit<unsigned long> ( [this](unsigned long x) { this->statistics.inc_stats_foo(x); } );
    (where this->statistics is of type SieveStatistics )
    Note that the && is MANDATORY pre C++17 (since MergeOnExit is non-movable, non-copyable, this
    uses lifetime-extension of temporaries).
*/
template<class Integer, class Functor>
class MergeOnExit
{
    static_assert(std::is_integral<Integer>::value, "Wrong template argument");
    // Alas, is_invocable<Functor> is C++17...
    public:
    Functor const fun;
    Integer val;
    constexpr MergeOnExit(Functor fun_) : fun(fun_), val(0) {}
    constexpr MergeOnExit(Functor fun_, Integer val_) : fun(fun_), val(val_) {}
    MergeOnExit &operator=(Integer new_val) { val = new_val; return *this; }
    MergeOnExit &operator=(MergeOnExit const &) = delete;
    MergeOnExit &operator=(MergeOnExit &&) = delete;
    MergeOnExit(MergeOnExit const &) = delete;
    MergeOnExit(MergeOnExit &&) = delete;
    template<class Arg> MergeOnExit &operator+=(Arg &&arg) { val+=(std::forward<Arg>(arg)); return *this; }
    template<class Arg> MergeOnExit &operator-=(Arg &&arg) { val+=(std::forward<Arg>(arg)); return *this; }
    template<class Arg> MergeOnExit &operator*=(Arg &&arg) { val+=(std::forward<Arg>(arg)); return *this; }
    Integer operator++(int) { return val++; }
    MergeOnExit& operator++() { ++val; return *this; }
    Integer operator--(int) { return val--; }
    MergeOnExit& operator--() { --val; return *this; }
    ~MergeOnExit()
    {
        fun(val);
    }
    constexpr operator Integer() const { return val; }
};

// Lacking deduction guides (pre C++17), to deduce template parameters, we use a helper function:
template<class Integer, class Functor> MergeOnExit<Integer, Functor> merge_on_exit(Functor const &fun_) { return {fun_}; }
template<class Integer, class Functor> MergeOnExit<Integer, Functor> merge_on_exit(Functor const &fun_, Integer val_) { return {fun_, val_}; }

#endif // include guard
