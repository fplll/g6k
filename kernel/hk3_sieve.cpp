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
#include <string>

/**
    Multi-threaded Gauss - triple sieve.

    Naming conventions: Global variables and types from Siever that are specific to triple-sieve are
    prefixed TS_, the functions are prefixed with hk3_sieve (which is also the name of the main function).
    The hk3_sieve_ prefix is often dropped in the documentation.

    The multi-threaded triple gauss sieve works (on a very high level) as follows:
    Our database of vectors consists of 2 parts, a LIST and a QUEUE.
    The list part contains vectors where we know that (almost) no reductions are possible among them.
    (More precisely, if each of x1, x2, x3 is from the list part, then we will not find a reduction
    among this triple with our employed heuristics).
    The queue part is the rest of the database.
    Every possible reduction we can find must involve one vector from the queue.
    So the algorithm proceeds by taking an element from the queue part and processing it to check whether it
    can be used in a reduction. After processing, it ends up in the list.

    The processing itself will take an element p from the queue, create a filtered list of points x1
    that are close to +/-p. If p +/- x1 is already short, we call this an "outer" 2-reduction.
    Otherwise, we search for pairs within the filtered list that allow for "inner" 2-reductions (not involving p)
    or 3-reductions (involving p).

    NOTE:   "inner" 2-reductions are only considered if TS_perform_inner2red is set; the same inner 2-reduction
            may be found via many different p's if the algorithm has to consider many different p's.
            In particular, in a regime where the algorithm actually processes all p's, parameter settings
            that are optimized for 3-reductions will lead to many collisions for that reason. Furthermore,
            every inner 2-reduction will eventually be found as an outer 2-reduction anyway, if all p's are processed.
            By contrast, with standard parameters optimized for 2-reductions with a large database size, only a tiny
            fraction of p's are actually processed. Also, our collision detection is cheap, which is why
            TS_perform_inner2red is set to true by default.

    This way, the algorithm is both a 2-sieve and a 3-sieve, allowing a continous time-memory tradeoff.
    If the database size is large, 2-reductions will dominate the algorithm and the algorithm is fast(er),
    otherwise we need 3-reductions. A consequence is that the algorithm has to perform well in different settings,
    which might need tuning of parameters.

    We also support on-the-fly lifts of both pairs and triples.

    ----

    How the datastructures work:

    The database of points consists of a main db (which contains the full entries)
    and a series of cdb-snapshots, which contains compressed entries which index into db.
    The cdb-snapshots are partially sorted and regularly resorted.
    Every time, we resort, we create a brand new cdb-snapshot.
    The cdb-snapshots themselves are stored in TS_cdb_snapshots, which is a fixed-length array
    of std::vector<CompressedEntry>'s together with some reference counters. We use a hand-crafted
    preallocation and memory recycling scheme for those. All accesses (need to) go through dedicated functions
    to handle reference counting, concurrency and memory management correctly.

    After a thread finishes sorting, other threads will eventually learn about the new cdb-snapshot and use that one.
    Threads ideally use the most recent one, but might use outdated data, which causes (recoverable!) errors).

    An individual snapshot is a std::vector (whose size must not change during a triple sieve run).
    It is separated into 5 parts and has the following form
    (Having a handwritten copy of this picture is recommended for reading the actual code):

    +-----------------------------------+---------------------------------------------------------+
    |            ~~LIST~~               |                       ~~QUEUE~~                         |
    | untouched list | overwritten list | processed queue | unprocessed queue | overwritten queue |
    |   SORTED       |  UNSORTED        |         JOINTLY SORTED              |    UNSORTED       |
    +-----------------------------------+-------------------------------------+-------------------+
    ^                ^                  ^                 ^                   ^                   ^
    |                |                  |                 |                   |                   |
    |                |<-   list_left  ->|                 |<-  queue_left   ->|                   |
    |                |                  |                 |                   |                   |
    0         <=insert_list         start_queue        queue_head=>     <=insert_queue         db_size

    The individual 5 parts are all ranges with indices 0, TS_insert_list, TS_start_queue, TS_queue_head, TS_insert_queue, db_size.
    As usual, the left bound is inclusive, the right bound exclusive.
    TS_list_left (currently unused and may be commented out) and TS_queue_left store the size of the
    respective parts as indicated in the picture (up to some multi-threading issues, explained later).
    While these variables conceptually belong to a given cdb-snapshot (of which there may be more than one)
    and should be thought of as part of a cdb-snapshot struct, we actually only (need to) store these
    variables for the most recent snapshot as members of the sieve object.

    Some of the parts are sorted, some are not. The <= and => in the picture denotes how these bounds may
    move during the lifetime of a cdb snapshot.

    Immediately after sorting / at the start of the algorithm, we only have an untouched list and an
    unprocesessed queue. The other parts are empty. (At the very start, the untouched list is empty
    as well unless we carry over information from previous incantations, so we only have the queue part).
    Both parts are individually sorted and at this point, there are no findable reductions that involve only the list part.
    When processing the data, we take the shortest element p from the unprocessed queue part
    (and increment queue_head). We then try to find reductions. After that, p will be part of the processed queue.
    Conceptually, the processed queue belongs to and will eventually be merged with the list part,
    in the sense that all reductions only involving these parts have already been found.
    It's just that we delay inserting it into / merging with the list part, because that requires expensive resorting.
    The actual reductions that are found when processing p will give new (shorter) vectors that are first put on a per-thead transaction_db.
    Elements from such a transaction_db are then used to replace (preferably long) vectors.
    The replacement happen by replacing elements from the (original) list or queue part from the right, i.e.
    at TS_insert_list-1 or TS_insert_queue-1, decrementing TS_insert_list resp. TS_insert_queue.
    The replacement elements then form the overwritten list resp. the overwritten queue.
    (Note: This means that list_left can shrink due to such insertions. queue_left can shrink either by insertions or by choosing a new p)
    These overwritten elements might participate in further reductions, so they should be thought of as "queue" elements,
    and will eventually be merged with the queue. Again, we delay this, because it would require resorting.
    Note: Replacements are allowed to fail in various ways. It may happen that we move TS_insert_list or
    TS_insert_queue without doing an actual replacement. The precise behaviour is subject to change and may have changed
    and the algorithm should be flexible enough not to rely on too many guarantees from the insertion routine.
    (Note that Insertion failures are very expensive for the algorithm if it happens when inserting into the list part,
    because this leads to repeated processing of the very same p)

    If queue_left becomes smaller than TS_max_extra_queue_size or we have made TS_resort_after * db_size many
    (possibly failed) replacements, we trigger resorting. During resorting,
    the merger of the old untouched list and the old processed queue become the new list. The rest becomes the new queue.

    The algorithm terminates when one of the following conditions is true:
        - We have reached saturation
        - The queue is exhausted after resorting
        - We have reached a maximal number of resorts
    If one of these is true, we set the flag TS_finished to an appropriate exit code
    (which indicates the first reason to terminate that was encountered). Individual threads will then finish the next time
    that a new p is due to be processed (This means that processing a given point p is always finished once started).
    After we have finished the threaded parts in this way, we ensure that all pending transactions are merged with the database.
    In the rare case that we finished solely due to queue exhaustion and this final merge step has produced sufficiently many new queue
    elements, we may actually restart another run of the algorithm. (This situation happens, e.g. if the sorting thread gets
    preempted for longer than it takes the other threads to process the whole database, which can indeed happen).

    ---

    Multi-threading considerations

    It helps to have the following mental picture of the algorithm:
    [This model is kind-of accurate from a data-flow point of view, if not from a control-flow point of view]
    Think of db / the current cdb snapshot as a database server, with several clients connected to it.
    The clients contact the server to get a queue element p (together with a range of the database to compare against).
    This defines a work package for the client, who has to find triple reductions involving p
    (finding inner 2reduction, which might not involve p, are fine as well; we use whatever we find).
    The client then locally processes the work package and finds reductions. These reductions are then
    sent back to the server, who changes its database accordingly. Because communcation is costly, we buffer the
    communication with the server and only send our reductions in larger chunks.

    We have 4+1 main functionalites to consider:
    (0) hk3_sieve, which initializes data
    (1) hk3_sieve_get_p, which returns a point p and moves queue_head
    (2) hk3_sieve_task, which processes p, finds otf lifts and reductions, which are put onto transaction_db
    (3) hk3_sieve_resort, which creates a new sorted cdb-snapshot and updates the variables
    (4) hk3_sieve_execute_delayed_insertion, which puts transaction_db elements into the latest cdb-snapshot and db.

    (0) is single-threaded and requires no consideration.
    (2) actually operates completely locally. The compressed entries read might be from a previous snapshot
        and the db elements might be overwritten (and actually garbage due to interleaving writes).
        Our approach here is to simply check in the end if the new vectors are good. This is technically
        UB, but seems to work. It helps that any set of bytes is a valid Entry::x. In the end, we only
        care if the result is short (which we can check thread-locally). The actual insertion will
        eventually perform a (thread-safe) check whether we are indeed improving.
    (3) has to create a new snapshot from the old one. For this purpose, it needs essentially exclusive
        access to all variables that determine the splitting into 5 parts. Consequently, this conflicts
        with (1)get_p and (4)insertions. We also only want one thread sorting, even if multiple theads detect
        that sorting is due.
    (1) needs to increment queue_head without running into insert_queue and conflicting with sorting.
        We also want to allow concurrent calls to get_p. (most of get_p operates under a lock, though)
    (4) insertions potentially conflict with (3) and (1) and with each other. We want to allow concurrent
        insertions, since they are into different parts of the snapshot anyway.

    We use the following mechanisms to tackle these issues (apart from the fact that some variables obviously need to be atomics)

    -   There is a global order of snapshot creation / sortings, so we can talk about "next snapshot" etc.

    -   We use an atomic variable TS_queue_left to synchronize the increments of TS_queue_head and decrements of
        TS_insert_queue.
        Any thread that wants to perform any such increment / decrement must atomically decrement
        TS_queue_left before by the corresponding amount. Only if the result after the atomic
        decrement is still >= 0 is the thread allowed to increment / decrement
        TS_queue_head / TS_insert_queue by at most the given amount. The thread is obliged to increment
        TS_queue_left by an appropriate amount if the value after the decrement was negative
        (or if the thread changed TS_queue_head / TS_insert_queue by less than TS_queue_left).

        Note that there is little happens-before-synchronization beyond atomicity between the modifications of TS_queue_left,
        TS_queue_head and TS_insert_queue and different threads may disagree on the order of modification of
        different variables on non-sequentially-consistent architectures.
        Furthermore, both the back-increments after failed decrements (i.e. those leaving a negative value)
        of TS_queue_left and the modifications of TS_queue_head / TS_insert_queue after successful increments
        of TS_queue_left may be interleaved by other operations.

        The guarantee that we need is that the total set of modifications to TS_queue_head / TS_insert_queue
        is such that every pair of values in the modification order of TS_queue_head and TS_insert_queue that "belongs
        to a given snapshot" satisfies TS_queue_head <= TS_insert_queue.
        This is what the above rules (and the fact that TS_queue_head / TS_insert_queue / TS_queue_left are atomics)
        aim to achieve. Note that this relies on atomicity/mutex-protection of the invidual TS_queue_head / TS_insert_queue,
        which guarantees that there even exists a modification order of these (individual) variables.
        (TS_queue_head and TS_insert_queue are sufficiently protected by mutexes, which is why they do not need to be atomic)

        The synchronization, that is indeed neccessary, is in fact rather indirect:
        In order for "belongs to a given snapshot" to be meaningful, we need to synchronize
        the increments / decrements to TS_queue_head, TS_insert_queue, TS_queue_left with sorting,
        more precisely with the reset of those variable during sorting:
        all increments / decrements must happen-before any such resets for the next snapshots
        and any resets for a given snapshot must happen-before any increments / decrements for this snapshot.
            (Formally, this means that there is a happens-before relationship between any pair
            [increment/decrement to TS_queue_head | TS_insert_queue | TS_queue_left] and
            [reset of TS_queue_head | TS_insert_queue | TS_queue_left]. The latest
            reset of [TS_queue_head | TS_insert_queue | TS_queue_left] that happened-before any increment/decrement
            defines what snapshot an increment/decrement belongs to. Note that the resets occur in triples
            that are associated to a snapshot and the former definition must not depend on which of the three
            reset events we use.
            The increment/decrement to TS_queue_left and the associated back-increment or modification to the
            other variables must belong to the same snapshot.)

        For the increments / decrements during get_p, this is ensured by locking the mutex TS_queue_head_mutex in get_p.
        This mutex is also locked during the reset during resorting.

        For the increments / decrements during during execute_delayed_insertion, this is done via
        TS_insertions_started / TS_cdb_insertions_finished, explained below.

        This solves concurrency issues of (1) with itself and (4).

    -   We use a mutex TS_queue_head_mutex to protect all accessess to TS_queue_head.
        (Writes to TS_queue_left are also only allowed if this mutex is held or the implicit CDB-Mutex is held, see the (1) - (4) issue above)

        This solves concurrency issues (1) - (3)

    -   We use a mutex TS_insertion_selection_mutex to protect TS_insert_queue and TS_insert_list
        (reads / writes are also allowed if the implicit CDB-mutex is exclusively held)
        This allows that the selection where we insert actually appears to be atomic.
        (Note: What this mutex really is about is to protect cdb_snapshot[potential_insertion_position].len during the
               the part of (4) that determines where and how much to insert. For this part of (4), we
               can not guarantee that the ranges are disjoint.)

    -   We use an atomic_flag TS_currently_sorting to make sure only one thread can sort.
        Any thread that want to sort must successfully raise this flag (with it being non-set before)
        before actually calling hk3_sieve_resort. Conversely, any thread that raises this flag
        must actually call hk3_sieve_resort and clear the flag after sorting is finished.
        Threads that want to raise the flag, but fail because it is already set should NOT
        wait until is is cleared, but either assist in sorting or do some other useful work.
        Some logic is in place to encourage that every but the last cdb-snapshot only triggers only one sort
        (i.e. we do not want a cdb-snapshot C to trigger a sort, then after the sorting is done,
        another threads resorts because of some condition that is tied to the by-now outdated snapshot C.
        As this is not fatal, it's enough to make this very unlikely to happen.)

        This settles (non-)concurrency of (3) with itself.

    -   We use TS_insertions_started and TS_cdb_insertions_finished as an (implicit) mutex-like protection for
        the cdb bounds TS_insert_list and TS_insert_queue as well as for actual writes to cdb.
        This essentially consists of an atomic (bool + counter), where the counter is realized as the higher-order bits
        of an single atomic variable.
        A theads that wishes to non-exclusively acquire this locks needs to atomically increment the high-order bits of
        TS_insertions_started, provided the low-order bits of TS_insertions_started are cleared.
        To release, increment TS_cdb_insertions_finished by 1.
        A thread that wishes to exclusively acquire this lock needs to set the low-order bits of TS_insertions_started to 1
        (provided it was 0 before) and then wait until TS_insertions_started / 2^number_of_low_order_bits == TS_insertions_finished.
        (Note that TS_insertions_started / 2^number_of_low_order_bits can not change any longer after setting the low-oder bits)
        To release the exclusive access, clear the lsb again.

        We use this to prevent concurrency issues with (3) and (4):
        We can have multiple insertions threads, which need access to TS_insert_list / TS_insert_queue and write to cdb.
        sorting requires exclusive access to those.

    -   We use TS_insertions_started and TS_db_insertions_finished as an (implicit) mutex-like protection.
        It works similarly to the mechanism before. (Note that we use the same _started variable)
        While it relates db insertions and sorting, this is really about preventing concurrent db writes to the same
        db element (hence the name). Within one given cdb-snapshot, db insertions are guaranteed to be to
        disjoint db locations, since every cdb-location is overwritten at most once.
        (This synchronization is rather indirect and implicit...)
        We have to assure that all db writes for a given cdb-snapshot are finished before we start any
        new writes.

        TODO: Snapshots
*/



// Debug macros:
// #define TS_DEBUG(X) X
#define TS_DEBUG(X)
#define OUTD(STR)
#ifdef TS_SHOW_EXIT_COND
    #define TS_EXIT_COND(STR) std::cout << STR << std::endl;
#else
    #define TS_EXIT_COND(STR)
#endif

#ifdef TS_SHOW_DEBUG_SNAPSHOTS
    #define TS_DEBUG_SNAPSHOTS(X) std::cout << X;
#else
    #define TS_DEBUG_SNAPSHOTS(X)
#endif

/**
        Siever::hk3_sieve is the main function that is called from the Python layer.
        This function does some preprocessing, hands off to the threaded hk3_sieve_task and then
        does some post-processing.
        The parameter alpha controls the bound on the buckets / filtered list.
        It has the same meaning as for bgj1.
*/
void Siever::hk3_sieve(double alpha)
{
    CPUCOUNT(600);
    if(alpha <= 0)
    {
        alpha = TS_default_alpha;
    }

    // Switch mode to a gauss sieve, i.e. the sortedness information keeps track of queue and list parts.
    switch_mode_to(SieveStatus::triple_mt);
    if(status_data.gauss_data.reducedness < 3)
    {
        // we ran a 2-sieve before. In this case, we forget the sortedness information / splitting in list and queue
        // (because the list part is only known not to contain 2-reductions, but might contain 3-reductions
        invalidate_sorting();
    }
    status_data.gauss_data.reducedness = 3;

    size_t const db_size = db.size();

    assert(status_data.gauss_data.queue_start <= db_size);

    parallel_sort_cdb();
    invalidate_histo(); // we do not keep hist updated during the algorithm, so we invalidate it.
    statistics.inc_stats_sorting_sieve();
    auto &already_searched = status_data.gauss_data.queue_start;

    if(db_size == already_searched)
    {
        return;
    }
    // We will trigger termination of the alg once we have this many vectors with len^2 below saturation radius.
    size_t const requested_saturation = std::pow(params.saturation_radius, n/2.) * params.saturation_ratio / 2.;
    bool saturation_unreachable = false; // To avoid multiple errors / warnings.
    if(requested_saturation > db_size)
    {
        std::cerr << "Warning: requested saturation is larger than the database size, hence unachievable.\nYou might want to decrease saturation_radius or saturation_ratio.\n";
        saturation_unreachable = true;
    }
    else if (requested_saturation > (db_size * 3) /4 )
    {
        std::cerr << "Warning: requested saturation is larger than 75% of the database size.\n";
    }
    // Compute the number of vectors that already are shorter than the saturation radius.
    // Since cdb is partially sorted, we may use binary search (via std::lower_bound) on each sorted part to determine the number.
    auto Comp = [](CompressedEntry const &ce, double const &bound){return ce.len < bound; };
    size_t TS_current_saturation = std::lower_bound(cdb.cbegin(), cdb.cbegin()+already_searched, params.saturation_radius, Comp) - cdb.cbegin();
    TS_current_saturation += ( std::lower_bound(cdb.cbegin()+already_searched, cdb.cend(), params.saturation_radius, Comp) - (cdb.cbegin() + already_searched) );

    if(TS_current_saturation >= requested_saturation)
    {
        return;
    }

    // Note that TS_saturated_entries is UNSIGNED. So triggering saturation means that this variable underflows.
    // TODO: Consider changing to a signed variable.
    TS_saturated_entries.store(requested_saturation - TS_current_saturation, std::memory_order_relaxed);

    // We create the transaction db's here at the caller rather than inside the threads; the main reason is
    // that it is simpler to clean-up / resume. Notably, any unmerged transaction persists between calls and we
    // only make a final clean-up in the end (when there are no multiple threads)
    // This avoids the boolean force parameter of the current bgj1 implementation.

    // The padding is just to have each padded_transaction_db[i].t on separate cachelines.

    struct PaddedTransactionDb{ CACHELINE_PAD(foo); TS_Transaction_DB_Type t; CACHELINE_PAD(foo2); };
    std::vector<PaddedTransactionDb> padded_transaction_db(params.threads);

    // We use TS_snapshots rather than cdb. For this, we set up our snapshots data structure.
    // This steals memory from cdb by swapping it with TS_snapshots[0].
    hk3_sieve_init_snapshots();
    /** From this point on, cdb is invalid until hk3_sieve_restore_cdb() is called. */

    TS_finished.store(TS_Finished::running, std::memory_order_relaxed);
    TS_number_of_sorts = 0;
    hk3_sieve_init_metainfo(already_searched, TS_latest_cdb_snapshot_p.load(std::memory_order_relaxed)->snapshot.data());

    TS_insertions_started.store(0, std::memory_order_relaxed);
    TS_cdb_insertions_finished.store(0, std::memory_order_relaxed);
    TS_db_insertions_finished.store(0, std::memory_order_relaxed);
    TS_currently_sorting.clear();

    ENABLE_IF_STATS_MEMORY
    (
        auto old_memory_buckets = statistics.get_stats_memory_buckets();
        statistics.set_stats_memory_buckets(0);
    )

    TS_unmerged_transactions.resize(params.threads);
    std::fill_n(TS_unmerged_transactions.begin(), params.threads, 0);
    TS_total_unmerged_transactions = 0;

    while(TS_finished.load(std::memory_order_relaxed) == TS_Finished::running)
    {
        for (size_t c = 0; c < params.threads; ++c)
        {
            threadpool.push([this, c, &padded_transaction_db, alpha](){this->Siever::hk3_sieve_task(padded_transaction_db[c].t, c, alpha);});
        }
        threadpool.wait_work();
        assert(TS_finished.load(std::memory_order_relaxed) != TS_Finished::running);
        // clean up pending transactions
        for (size_t c = 0; c < params.threads; ++c)
        {
            float dummy; // hk3_sieve_execute_delayed_insertions requires a parameter to update len_bound to the caller. We do not need this info.
            while(!padded_transaction_db[c].t.empty())
            {
                hk3_sieve_execute_delayed_insertion(padded_transaction_db[c].t, dummy, 1000); // 1000 is just a dummy tread-id, useful for debugging.
            }
        }
        hk3_sieve_resort(1000); // 1000 is just a dummy thread-id, useful for debugging.

        switch(TS_finished.load(std::memory_order_relaxed)) // TS_finished contains some exit code why we terminated the threaded part.
        {
            case TS_Finished::out_of_queue:
                if(!saturation_unreachable)
                    std::cerr << "Warning: Triple sieve exited because of empty queue.\n";
                break;
            case TS_Finished::out_of_queue_resume:
                // This exit code is set if we have run out of queue, but had lots of pending transactions.
                // After merging these of transaction_dbs & resorting, we may have many new queue elements
                // and so we actually restart.
                // We test for saturation, because we might have reached saturation in the final merge & resort.
                // (The usual saturation trigger is not set, because TS_finished only records the first reason
                // why we terminate -- we might change that, but this way seems cleaner.)
                if (   (TS_saturated_entries.load(std::memory_order_relaxed) > 0)
                    && (TS_saturated_entries.load(std::memory_order_relaxed) <= requested_saturation))
                {
                    TS_finished.store(TS_Finished::running, std::memory_order_relaxed); // will trigger rerun.
                }
                break;
            case TS_Finished::sortings_exceeded:
                std::cerr << "Warning: Triple sieve exceeded maximum number of resorts\n";
                break;
            case TS_Finished::running:
                assert(false);
                break;
            case TS_Finished::saturated:
                // everything is fine. This is the normal case.
                break;
            default:
                assert(false);
        }
    } // end of while (TS_finished == TS_Finished::running), will restart thread-dispatch if TS_finished was reset in the TS_Finished::out_of_queue_resume case.

    if(TS_snapshots_used > statistics.get_stats_memory_snapshots())
        statistics.set_stats_memory_snapshots(TS_snapshots_used);

    // we restore cdb from TS_snapshots data and put TS_snapshot data in a defined state.
    // (This effectively swaps TS_snapshots[current] with TS_snapshots[0] and then TS_snapshots[0] with cdb)
    hk3_sieve_restore_cdb();
    /** CDB IS VALID AGAIN **/

    assert(cdb.size() == db.size());

    // Update sortedness information that persists across calls.
    status_data.gauss_data.queue_start = TS_start_queue_original;
    status_data.gauss_data.list_sorted_until = TS_start_queue_original;
    status_data.gauss_data.queue_sorted_until = cdb.size();
    assert(std::is_sorted(cdb.cbegin(), cdb.cbegin() + TS_start_queue_original, compare_CE() ));
    assert(std::is_sorted(cdb.cbegin() + TS_start_queue_original, cdb.cend(), compare_CE() ));
    recompute_histo();

ENABLE_IF_STATS_MEMORY
    (
        if(old_memory_buckets > statistics.get_stats_memory_buckets())
            statistics.set_stats_memory_buckets(old_memory_buckets);
        size_t mem_transactions = std::accumulate(padded_transaction_db.cbegin(), padded_transaction_db.cend(), size_t(0), [](size_t const &a, PaddedTransactionDb const &b){ return a + b.t.new_vals.capacity();  }  );
        if(mem_transactions > statistics.get_stats_memory_transactions())
            statistics.set_stats_memory_transactions(mem_transactions);
    )

    if( (TS_saturated_entries <= requested_saturation) && (TS_saturated_entries > 0) && !saturation_unreachable) // indicating no underflow
    {
        // Note: We will also get a warning related to TS_finished from the above switch statement.
        std::cerr << "Warning: Saturation condition could not be reached during triplesieve.\n";
    }

//    statistics.print_statistics(SieveStatistics::StatisticsOutputForAlg::triple_mt);
    return;
}

/**
    hk3_sieve_task is the main worker function of the triple-sieve.
    hk3_sieve will thread-dispatch several workers running this function.

    Roughly speaking (ignoring some multi-threading issues),
    it operates by obtaining elements p from the queue (which describes a "job" for the worker).
    We then process p by first comparing p with every element from the current cdb snapshot (or a subrange)
    to produce a sub-list of the current cdb snapsnot as a local bucket. We then search for reductions within that bucket.

    Newly found reductions are buffered with a transaction_db and regularly merged with the current cdb snapshots.

    For reasons of efficiency, as opposed to a naive for(x1:bucket) { for(x2:bucket) { ... }} inner double-loop,
        (a) the bucket is processed while we create it (this gives potential reductions earlier)
            (i.e. a run of for(x2:bucket) is triggered whenever we add an element to the bucket, modulo (b);
            note that this avoids checking the same pair (x1,x2) twice as (x2,x1) )
        (b) This processing itself is buffered in cache-friently chunks to reduce memory-latency
            (i.e. we swap the order of loops: We collect N many x1's before we do a
            for(x2:bucket) { for(i<=N) {process those x1's with x2 }},
            where N is chose such that the all N x1's should fit into the cache)
    This innermost loop is delegated to a function template hk3_sieve_process_inner_batch(...) to avoid some code duplication.

    Getting p and merging the transactions are delegated to designated functions.

    Params:
    transaction_db is the transaction_db used. (This is created by the calling thread, so unmerged transactions are passed to the caller)
    id is the thread-id (ranging from 0 to #threads-1), used purely for debugging purposes
    alpha is a constant parameter taken from hk3_sieve.
*/

void Siever::hk3_sieve_task(TS_Transaction_DB_Type &transaction_db, MAYBE_UNUSED unsigned int id, double const alpha)
{
    ATOMIC_CPUCOUNT(605);

    // TS_FilteredCE is a bucket element. We use a special-purpose optimized format.

    std::array<TS_FilteredCE, TS_Cacheline_Opt> local_cache_block;  // Ring buffer for putting elements in bucket.
                                                                    // We collect TS_Cacheline_Opt many TS_FilteredCE entries
                                                                    // and then put them into the bucket and process them as a block.
                                                                    // This is purely to optimize memory access patterns.

    // local stastistics: to reduce the number atomic operations,
    // we only merge per-thread statistics into the global statistics at the end.
    ENABLE_IF_STATS_BUCKETS(auto &&local_stat_number_of_filteredlists   = merge_on_exit<unsigned long>([this](unsigned long val){ statistics.inc_stats_buckets(val); }); )
    ENABLE_IF_STATS_REDSUCCESS (auto &&local_stat_successful_2red_outer = merge_on_exit<unsigned long>([this](unsigned long val){ statistics.inc_stats_2redsuccess_outer(val); }); )
    ENABLE_IF_STATS_COLLISIONS( auto &&local_stat_collisions_2outer     = merge_on_exit<unsigned long>([this](unsigned long val){ statistics.inc_stats_collisions_2outer(val); }); )
    ENABLE_IF_STATS_COLLISIONS( auto &&local_stat_collisions_nobucket   = merge_on_exit<unsigned long>([this](unsigned long val){ statistics.inc_stats_collisions_nobucket(val); }); )
    ENABLE_IF_STATS_OTFLIFTS( auto &&local_stat_otflifts_2outer         = merge_on_exit<unsigned long>([this](unsigned long val){ statistics.inc_stats_otflifts_2outer(val); }); )

    #if COLLECT_STATISTICS_XORPOPCNT || COLLECT_STATISTICS_REDS
        // number of 2-reduction attempts in the bucketing phase
        auto &&local_stat_bucketing_tries = merge_on_exit<unsigned long long>([this](unsigned long long val)
        {
            statistics.inc_stats_xorpopcnt_outer(val);
            statistics.inc_stats_2reds_outer(val);
        } );
    #endif
    #if COLLECT_STATISTICS_XORPOPCNT_PASS || COLLECT_STATISTICS_FULLSCPRODS
        auto &&local_stat_successful_xorpopcnt_bucketing = merge_on_exit<unsigned long long>( [this](unsigned long long val)
        {
            statistics.inc_stats_xorpopcnt_pass_outer(val);
            statistics.inc_stats_fullscprods_outer(val);
        } );
    #endif

    // These values do not change during the algorithm.
    auto const n = this->n;
    double const alpha_square = alpha * alpha;
#ifndef NDEBUG
    size_t const db_size = db.size();
#endif
    // This contains the bucket elements of the current run. Note that we put no limit on its size.
    // (It is bounded by db_size, though). We also do not preallocate. Should we?
    std::vector<TS_FilteredCE> bucket;

    // thread_local_snapshot is the latest cdb snapshot that the _task function knows about.
    // It is syncronized with the global latest snapshot (only) during get_p.
    // Note that thread_local_snapshot may be lagging behind the latest global snapshot.
    // During processing a single p, thread_local_snapshot does not change.
    //      (Note that this is true even if the same thread calls execute_delayed_replace, which obtains the latest snapshot as well
    //      This information is not passed to the caller by design, because we do not want to break the loop)

    // Important: The initial value of thread_local_snapshot will be overwritten by the first call to get_p() anyway.
    // Still, we have to initialize it with hk3_sieve_get_latest_snapshot(id), because
    // get_p(thread_lcoal_snapshot, ...)  assumes thread_local_snapshot to be in a valid state (nullptr is NOT valid)
    // to properly maintain its ref-counting.
    // Similarly, we have to pair this call to hk3_sieve_get_latest_snapshot with a release_snapshot call
    // (get_latest_snapshot / release_snapshot are effectively a (manual) constructor / destructor pair.
    // TODO: Replace TS_CDB_Snapshot * by a proper LocalSnapshotPtr class with constructor / destructor.
    // (This is not quite straightforward, because the class would need to contain the Siever's this,
    //  and furthermore, the global latest snapshot would not be of that type. It's ugly in any case.
    //  If we ever use these snapshots in another algorithm, make the appropriate changes. )

    TS_CDB_Snapshot *thread_local_snapshot = hk3_sieve_get_latest_snapshot(id);
    TS_DEBUG_SNAPSHOTS("Thread " + std::to_string(id) + " initialized with snapshot " + std::to_string(thread_local_snapshot-&TS_cdb_snapshots[0]) + "\n")

    // local_len_bound is the length bound for new vectors. Vectors better than this are considered as insertion candidates.
    // Note that the local copy local_len_bound may be lagging behind the global TS_len_bound.
    // However, since we perfom a final check before every actual insertion, this is fine.
    // It is updated during get_p and possibly during execute_delayed_insertion.
    float local_len_bound = TS_len_bound.load();

    while(true) // We exit once get_p return a length < 0 (which is an exit code meaning "Finished")
    {
        // Note that a given bucket is technically local to a single loop iteration. Still, we
        // created it outside the loop and clear in each iteration to avoid memory allocations.
        // (bucket.clear() does not free up memory)
        bucket.clear();

        // Get p_entry from the queue and number_of_x1s_considered.
        // The algorithm will then first construct a filtered list of vectors close to p_entry (only the number_of_x1s_considered first element of the cdb snapshot are considered for this)
        // Futher, this call updates thread_local_snapshot and local_queue_start to their global counterparts.
        auto const bucket_task = hk3_sieve_get_p(thread_local_snapshot, id, transaction_db, local_len_bound);
        Entry const &p_entry = bucket_task.first;
        size_t const &number_of_x1s_considered = bucket_task.second;

        // hk3_sieve_get_p returns negative length to indicate that we should finish. The other return values are meaningless in that case.
        if (UNLIKELY(p_entry.len < 0.))
        {
            break;
        }
        ENABLE_IF_STATS_BUCKETS(++local_stat_number_of_filteredlists;)
        assert(thread_local_snapshot->snapshot.size() == db_size);

        CompressedEntry * const fast_cdb = thread_local_snapshot->snapshot.data();
        // to make absolutely sure the compiler knows this is const within the given scope.
        auto const  p_cv = p_entry.c;
        FT const p_len = p_entry.len;
        FT const p_len_half = 0.5 * p_len;
        FT const p_len_squared_target = p_len * alpha_square; // Note : p_len is already squared
        UidType const p_uid = p_entry.uid;

        // counter inside local_cache_block, for the latter to act as a ring buffer:
        // The next bucket element we find will be put into local_cache_block[block_pos++]
        // Whenever block_pos reaches the size of local_cache_block, we actually merge it with the bucket.
        // and reset block_pos to 0.
        // Note: We deep copy the elements from local_cache_block onto the bucket, so they remain in local_cache_block until overwritten.
        // This is by design.
        uint_fast16_t block_pos = 0;

        static_assert(TS_Cacheline_Opt < std::numeric_limits<decltype(block_pos)>::max(), "TS_Cacheline_Opt+1 must fit into block_pos"); // Not sure about the +1, but don't want surprises if some code changes.

        #if COLLECT_STATISTICS_XORPOPCNT || COLLECT_STATISTICS_REDS
        local_stat_bucketing_tries += number_of_x1s_considered; // tracks XOR-POPCNTS below and outer 2-reds
        #endif
        // main outer loop over x1: The is the bucketing loop, i.e. x1 may or may not be put into the current bucket centered at p.
        for (size_t x1_cdb_index = 0; x1_cdb_index < number_of_x1s_considered; ++x1_cdb_index)
        {
            if (UNLIKELY(is_reducible_maybe<TS_OUTER_SIMHASH_THRESHOLD>(p_cv, fast_cdb[x1_cdb_index].c)))
            {
                #if COLLECT_STATISTICS_XORPOPCNT_PASS || COLLECT_STATISTICS_FULLSCPRODS
                ++local_stat_successful_xorpopcnt_bucketing; // tracks successful XOR-POPCOUNTS and scalar product computations below
                #endif
                // We have a plausible candidate for reduction.
                // Since we need to compute the exact scalar product <p, x1>, where x1 == fast_cdb[i] anyway, we do it right away.
                // This means that the check for 2-reducibility can use the exact scalar product.
                size_t const x1_db_index = fast_cdb[x1_cdb_index].i;
                assert(x1_db_index < db_size);
                LFT const scalar_prod_p_x1 = std::inner_product(p_entry.yr.cbegin(), p_entry.yr.cbegin()+n, db[x1_db_index].yr.cbegin(),  static_cast<LFT>(0.));
                bool const x1_sign_flip = scalar_prod_p_x1 > 0;
                LFT const scalar_prod_p_x1_squared = (scalar_prod_p_x1 * scalar_prod_p_x1);
                // x1_len_score = ||p+/- x1||^2 - 1/2||p||^2.
                // If we put the entry in a bucket, this value is useful in later computations for the 3-sieve.
                LFT const x1_len_score = p_len_half + fast_cdb[x1_cdb_index].len -std::abs(2 * scalar_prod_p_x1); // std::abs() was faster than branching on x1_sign_flip.
                LFT const p_x1_len = p_len_half + x1_len_score; // length^2 of p +/- x1
                if (UNLIKELY(p_x1_len < local_len_bound)) // 2-reduction possible: We try to directly perform an (outer) 2-reduction.
                {
                    ENABLE_IF_STATS_REDSUCCESS(++local_stat_successful_2red_outer;)
                    if (!hk3_sieve_delayed_red_p_db(transaction_db, p_entry, x1_db_index, x1_sign_flip)) // perform 2-reduction
                    {
                        // Note that failures due to data races will decrement the collision counter inside delayed_red_p_db
                        ENABLE_IF_STATS_COLLISIONS(++local_stat_collisions_2outer;)
                    }
                    else if(transaction_db.size() % TS_transaction_bulk_size == 0)
                    {
                        hk3_sieve_execute_delayed_insertion(transaction_db, local_len_bound, id);
                    }
                    // continue; We may or may not proceed with 3-reductions here. Given that collisions are cheap, we proceed.
                }
                else if(params.otf_lift && (p_x1_len < params.lift_radius)) // UNLIKELY??? Not so sure
                {
                    ENABLE_IF_STATS_OTFLIFTS(++local_stat_otflifts_2outer;)
                    hk3_sieve_otflift_p_db(p_entry, x1_db_index, x1_sign_flip, p_x1_len);
                }
                // Check for candidate 3-reduction |<p, x1>| > TS_filter_threshold * |p| * |x1|
                if (scalar_prod_p_x1_squared > p_len_squared_target * fast_cdb[x1_cdb_index].len)
                {
                    if(x1_sign_flip) // This means that we consider p - x1 +/- x2, so we negate c_proj and uid:
                    {
                        // hash collision check: If p - x1 is already in db, then we do not perform 3-reductions.
                        // 2-reductions will find the result already.
                        if(uid_hash_table.check_uid(p_uid  - db[x1_db_index].uid) == true)
                        {
                            ENABLE_IF_STATS_COLLISIONS(++local_stat_collisions_nobucket;)
                            continue;
                        }
                        local_cache_block[block_pos].c_proj       = fast_cdb[x1_cdb_index].c;
                        local_cache_block[block_pos].uid_adjusted = -db[x1_db_index].uid;
                        // bit-wise negate c_proj, corresponding to the CompressedVector of -x1.
                        // Note that negating x1.x and recomputing c may give a different answer in a few bit positions.
                        // (This is because the bits of our simhashes test the sign of <x1,r> for random r, and there is a small change of <x1,r>==0 )
                        std::for_each(local_cache_block[block_pos].c_proj.begin(), local_cache_block[block_pos].c_proj.end(), [](CompressedVector::value_type &x){x=~x;});
                    }
                    else // no sign flip: We consider p + x1 +/- x2
                    {
                        // hash collision check as above
                        if(uid_hash_table.check_uid(p_uid + db[x1_db_index].uid) == true)
                        {
                            ENABLE_IF_STATS_COLLISIONS(++local_stat_collisions_nobucket;)
                            continue;
                        }
                        local_cache_block[block_pos].c_proj = fast_cdb[x1_cdb_index].c;
                        local_cache_block[block_pos].uid_adjusted = db[x1_db_index].uid;
                    }
                    local_cache_block[block_pos].db_index     = x1_db_index;
                    local_cache_block[block_pos].sign_flip    = x1_sign_flip;
                    local_cache_block[block_pos].len_score    = x1_len_score;
                    ++block_pos;
                    if(block_pos == TS_Cacheline_Opt) // block is full, we now actually process it:
                    {
                        block_pos = 0;
                        auto const prev_filter = bucket.size(); // previous size of bucket
                        // copy local_cache_block to end of bucket.
                        bucket.insert(bucket.end(),local_cache_block.cbegin(), local_cache_block.cbegin()+TS_Cacheline_Opt);
                        // compare every element from local_cache_block with everything from bucket (excluding the newly inserted elements)
                        hk3_sieve_process_inner_batch<false>(transaction_db, p_entry, local_cache_block, std::integral_constant<uint_fast16_t,TS_Cacheline_Opt>{}, bucket, prev_filter, local_len_bound, id);
                        // compare every element from local_cache_block with each other (which equals the newly inserted elements of the bucket)
                        hk3_sieve_process_inner_batch<true> (transaction_db, p_entry, local_cache_block, std::integral_constant<uint_fast16_t,TS_Cacheline_Opt>{}, local_cache_block, TS_Cacheline_Opt, local_len_bound, id);
                    } // end of processing cacheline block
                } // end of if (p,x1) are a 3-reduction candidate
            } // end of if (current cdb[x1_cdb_index] is promising wrt the chosen p)
        } // end of x1_cdb_index loop
        // process local_cache_block, even if it is not full. There is no reason to copy local_cache_block into bucket in this case.
        hk3_sieve_process_inner_batch<false>(transaction_db, p_entry, local_cache_block, block_pos, bucket, bucket.size(), local_len_bound, id);
        hk3_sieve_process_inner_batch<true> (transaction_db, p_entry, local_cache_block, block_pos, local_cache_block, block_pos, local_len_bound, id);
        hk3_sieve_execute_delayed_insertion(transaction_db, local_len_bound, id);
    } // end of while(true) loop / loop over p's. We exit if TS_finished gets set
    statistics.inc_stats_memory_buckets(bucket.capacity());

    hk3_sieve_release_snapshot(thread_local_snapshot, id);
    TS_DEBUG_SNAPSHOTS("Thread " + std::to_string(id) + " finished.\n")

    // Try to merge pending transactions already now during multi-threaded stage.
    // This is not guaranteed to process all pending transactions, but
    // the caller hk3_sieve(...) will later clean up those remaining transactions after all threads have finished.
    hk3_sieve_execute_delayed_insertion(transaction_db, local_len_bound, id);

}

/**
    The function mt_init_metainfo resets / set up the various TS_ members after sorting or the start of the algorithm.
    More precisely, it initializes the variables that define the decomposition into list / queue parts.
    Furthermore, it sets up
    - the lenght bound for new insertions
    - data to determine whether the next insertions are to go to the list or queue.
    - TS_insertions_performed (counting the number of insertions since the last resort, used to trigger sorting)
    (NOTE: The latter might go away completely).
    Note that TS_latest_cdb_snapshot_p is not touched. For getting the length bounds, it takes a fast_cdb pointer
    that should equal to the data() entry of the latest / upcoming CDB snapshot.
    (Regarding upcoming: for switching to a new cdb snapshot, this function is called to setup the decomposition into parts
    before changing TS_latest_cdb_snapshot_p to the new snapshot)

    Does NOT set TS_finished

    This function has NO thread-safety measures. Thread-safety is the responsibility of the caller, which means
    that enough locks must be held to prevent any concurrent access to the decomposition.
    (At the moment, this means TS_queue_head_mutex and the implicit TS_insertions_started must be held to block get_p and insertions (sortings are also blocked for several reasons))

    Params: already_processed: The size of the new list part
            fast_cdb: a raw pointer to the latest / upcoming snapshot.
*/

void Siever::hk3_sieve_init_metainfo(size_t const already_processed, CompressedEntry const * const fast_cdb)
{
    ATOMIC_CPUCOUNT(601);

    size_t const db_size = db.size(); // Note: We must not use cdb.size here.
    assert(db_size > TS_max_extra_queue_size+2);
    assert(db_size > 0);
    assert(already_processed <= db_size); // Note : If == holds, we may have to treat things differently.

    TS_queue_head = already_processed;      // next queue index to be processed.
    TS_insert_queue = db_size;                 // when inserting into queue part, we insert at pos TS_insert_queue - 1
    TS_insert_list =already_processed;        // when inserting into list part, we insert at pos TS_insert_list - 1
//    TS_list_left.store(already_processed);
    TS_start_queue_original = already_processed;    // first position of queue (i.e. end of list part)
    TS_queue_left.store(db_size - already_processed); // number of elements in the queue part to be processed.

    // compute the scaled length difference between the end of the queue and the end of the list.
    // This is currently used for the decision where to insert (list or queue) in some cases.
    // Unfortunately, we have to treat the cases where list / queue are empty in a special way.
    // Note that at least one of them is not empty.
//    decltype(TS_queue_list_imbalance.load()) new_imbalance;
//    if(UNLIKELY(already_processed == db_size)) // already_processed == db_size means the queue part is empty. The algorithm is over anyway...
//    {
//        // We make the last element in the list part "infinitely good" (better than any proper length, i.e. negative).
//        // As a consequence, we will never try to insert into the list part or touch TS_insert_list.
//        new_imbalance = TS_outofqueue_len; // Note that this is negative
//    }
//    else
//    {
//        new_imbalance = std::round (TS_queue_list_imbalance_multiplier * fast_cdb[db_size -1].len);
//    }
//    if(already_processed == 0) // list is empty.
//    {
//        new_imbalance -= TS_outoflist_len; // TS_outoflist_len is negative.
//    }
//    else
//    {
//        new_imbalance -= std::round( TS_queue_list_imbalance_multiplier * fast_cdb[already_processed -1].len);
//    }
//    TS_queue_list_imbalance.store(new_imbalance);

    TS_insertions_performed.store(0); // counts the number of insertions we have performed. Is reset to 0 after resorting.

    if (UNLIKELY(already_processed == db_size)) // we are actually done with the sieve in that case.
    {
        TS_len_bound = -1.0; // This prevents any new reductions from being found.
        return;
    }
    hk3_sieve_update_lenbound(fast_cdb); // Note that this function has no internal thread-safety either
}

/**
    This function updates TS_len_bound and returns its value;

    TS_len_bound is the value used to determine whether a possible reduction is an insertion candidate.

    Note that this function has no internal thread-safety measures. This is the responsibility of the caller.
    (Currently, it is only called by init_metainfo, so there are no issues at the moment.)
    IMPORTANT: If we update TS_len_bound more often, we need to address this.

    The current formula takes a weighted sum between the length of a middle list and a middle queue element,
    where the middle list element is list element indexed by TS_improvement_db_ratio * list_size
    (and similarly TS_improvement_db_ratio * queue_size).
    This is intended to mimic what bgj1 does, in the sense that any improvement increases the rank of the cdb element by at
    least a certain factor. Be aware that this is quite heuristical and we do not have as nice guarantees as
    in the bgj1 case (where the resort threshold is tied to the improvement_db_ratio and we do not have
    the list / queue distinction): in particular, we are not even guaranteed that the length bound is monotously
    decreasing during the algorithm.

    NOTE:   We do NOT multiply the obtained length by a factor 1-epsilon. In particular, if all database elements
            have the same length, we might make only minimal progress.

    NOTE: It might be worthwhile to experiment with different bounding functions. Since both list and queue parts are sorted,
    finding the actual TS_improvement_db_ration * db_size's largest element is actually possible in
    O(log db_size) time. (cf. what is used in execute_delayed_insertion)

    Params: fast_cdb : raw pointer to the current / upcoming cdb snapshot's underlying array.
*/
float Siever::hk3_sieve_update_lenbound(CompressedEntry const * const fast_cdb)
{

    // the len bound is determined similarly to what is done in bgj1: We require that we make at least a fixed amount of progress:
    // Notably, we require that if we make a replacement, then
    // a) we improve the length at least by a (small) constant factor and
    // b) the length-rank (i.e. the index of the element if we sorted the lists) improves at least by a certain amount.
    // The latter condition is to ensure that if we have a large spread wrt length, we actually make meaningful improvements.
    // Unfortunately, we do not have a single sorted list, but rather two parts that are individually sorted.
    // What we do instead is we multiply the length rank of each part by a constant factor and then
    // take a weighted average of the resulting lengths.

    ATOMIC_CPUCOUNT(608);

    // The issue with these asserts here is that the correct index that points at the replacement pos is TS_insert_list - 1 resp. TS_insert_queue -1
    // The -1 will cause problems if TS_insert_list == 0 (i.e. empty list part)
    // We need to make sure that corner-cases TS_insert_list == 0, TS_insert_queue == TS_insert_list, TS_insert_queue == db_size work without out-of-bounds access.
    static_assert(TS_improvement_db_ratio < 1., "This does not work");
    static_assert(TS_improvement_db_ratio > 0., "This does not work");

    auto current_queue_left_load = TS_queue_left.load();
    size_t const current_queue_left = (current_queue_left_load > 0) ? current_queue_left_load : 0;
    size_t const db_size = db.size();
    size_t const list_compare_index  = std::floor(TS_improvement_db_ratio * TS_insert_list); // no -1 here. If TS_insert_list == 0, we access the first element of the queue rather than something in the list, but the result appears with weight 0 anyway.
    size_t const queue_compare_index = std::floor( (TS_insert_queue - 1) - (1-TS_improvement_db_ratio) * current_queue_left );
    assert(list_compare_index < db_size);
    assert(queue_compare_index < db_size);
    float const update_len_bound = ( fast_cdb[list_compare_index].len * (db_size - current_queue_left) + fast_cdb[queue_compare_index].len * current_queue_left)/ static_cast<FT>(db_size);
    TS_len_bound.store( update_len_bound);
    return update_len_bound;
}

/**
    This function sorts the current cdb snapshot.
    More precisely, it creates a new, pristine and sorted cdb snapshot from the old one,
    resets the TS_ variables with ..._init_metainfo and then updates the lasted cdb snapshot.

    May set TS_finished if the number of sorts exceeds a threshold / we ran out of queue.

    TS_currently_sorting must be set by the calling thread outside of this function.
    No mutex must be held when calling this function.
    The caller must only have one ref-counted snapshot pointer (or possibly several, but pointing to the same snapshot),
    since otherwise, we might run into a deadlock when trying to create a new snapshot.

    Note that this function crucially relies on the way the cdb snapshots are structured, and on how get_p and execute_delayed work.

    Params: id is just a thread-identifier used solely for debug purposes.
*/

void Siever::hk3_sieve_resort(MAYBE_UNUSED unsigned int const id)
{
    // calling this function is protected by an atomic flag so only one thread may call it.
    CPUCOUNT(602)

    // Obtain cdb and db-insertion locks.
    assert(TS_insertions_started.load() % 2 == 0);
    auto const wait_threshold = TS_insertions_started.fetch_add(1, std::memory_order_seq_cst) / 2;
    // We now reserved 2 locks. This blocks any further insertions from starting.
    // We also know that all pending cdb insertions will have finished once TS_cdb_insertions_finished == wait_threshold
    // We also know that all pending db insertion will have finished once TS_db_insertions_finished == wait_threshold

    statistics.inc_stats_sorting_sieve();
    size_t const db_size        = db.size(); // We must NOT use cdb.size() !

    // Get a raw pointer to the current cdb snapshot's underlying array.
    // Previous writes in previous invocations of sort happened-before because of synchronization via TS_insertions_started
    // and we are the only thread that is allowed to write to the current cdb snapshot.
    CompressedEntry const * const previous_fast_cdb = hk3_sieve_get_true_fast_cdb();

    // Get next snapshot, reusing old memory if possible:
    auto const next_cdb_snapshot_ptr = hk3_sieve_get_free_snapshot(id);
    CompressedEntry * const next_fast_cdb = next_cdb_snapshot_ptr->snapshot.data();

    // spin-wait until cdb writes have finished.
    // TODO: Be more clever here. Use __builtin_ia32_pause(); maybe?
    // Note that we typically do not have to wait (long) here.
    statistics.inc_stats_dataraces_sorting_blocked_cdb(wait_threshold - TS_cdb_insertions_finished);
    while(wait_threshold != TS_cdb_insertions_finished.load()) {}

    // Note that the TS_ variants here actually can not change anymore. This is just to move everything to the stack.
    size_t const insert_queue           = TS_insert_queue;
    size_t const insert_list            = TS_insert_list;
    size_t const start_queue_original   = TS_start_queue_original;

//  Just in case someone starts clang-formating this
//  clang-format off
/** Recall that the current cdb snapshot looks as follows (our off-by one choices are such that in this picture, all ranges are [begin, end) as usual)
|                                         |                                                                                    |
|   OLD LIST (before previous sort)       |                     OLD QUEUE (before previous sort)                               |
|                                         |                                                                                    |
| LIST PART  | points inserted into list  | processed queue elements | unprocessed queue elements | points inserted into queue |
| (untouched)| (need to go into queue)    | (need to go to list)     | (stay in queue)            | (need to go to queue)      |
| (sorted)   | (unsorted)                 |              (sorted as a whole)                      | (unsorted)                 |
0       insert_list             start_queue_original         current_queue_head             insert_queue                    db_size
     I                     II                          III                         IV                            V
**/


// Note: The global TS_queue_head may move forward a bit while we sort. This is by design.
// To avoid unneeded delays, we postpone reading current_queue_head (and separating III from IV) until as late as possible.
//  clang-format on

// We need to copy the current cdb snapshot into the next, reordering parts from (I-II) | (III - IV - V) into (I - III) | (IV - II - V)
// while sorting the individual parts and then merge-sorting (I - III) and (IV - II - V)

// We create variables for each part's size, starting address and address where it needs to go.
// This is just for readability.
    size_t const size_untouched_list = insert_list;                                 // Part  I
    size_t const size_overwritten_list = start_queue_original - insert_list;        // Part  II
    size_t const size_proc_and_unproc_queue = insert_queue - start_queue_original;  // Parts III & IV
 // size_t const size_overwritten_queue = db_size - insert_queue;                   // Part  V (variable is unused, commented out to silence warning)

    CompressedEntry const * const start_old_cdb_untouched_list    = previous_fast_cdb;
    CompressedEntry const * const start_old_cdb_overwritten_list  = previous_fast_cdb + insert_list;
    CompressedEntry const * const start_old_cdb_processed_queue   = previous_fast_cdb + start_queue_original;
//    CompressedEntry const * const start_old_cdb_unprocessed_queue = previous_fast_cdb + current_queue_head; -- TS_queue_head might be changed by another thread.
    CompressedEntry const * const start_old_cdb_overwritten_queue = previous_fast_cdb + insert_queue;
    CompressedEntry const * const end_old_cdb                     = previous_fast_cdb + db_size;

    assert(start_old_cdb_untouched_list <= start_old_cdb_overwritten_list);
    assert(start_old_cdb_overwritten_list <= start_old_cdb_processed_queue);
    assert(start_old_cdb_processed_queue <= start_old_cdb_overwritten_queue);
    assert(start_old_cdb_overwritten_queue <= end_old_cdb);

    // adress where it should go to
    CompressedEntry * const start_new_cdb_untouched_list  = next_fast_cdb;
    CompressedEntry * const start_new_cdb_processed_queue = start_new_cdb_untouched_list + size_untouched_list;
//    CompressedEntry * const start_new_cdb_unprocessed_queue= start_new_cdb_processed_queue + size_processed_queue; // size_processed_queue is yet-unknown
    CompressedEntry * const start_new_cdb_overwritten_list = start_new_cdb_processed_queue + size_proc_and_unproc_queue;
    CompressedEntry * const start_new_cdb_overwritten_queue= start_new_cdb_overwritten_list + size_overwritten_list;
    CompressedEntry * const end_new_cdb                    = next_fast_cdb + db_size;

    assert(start_new_cdb_untouched_list + size_untouched_list <= end_new_cdb);
    assert(start_new_cdb_processed_queue+ size_proc_and_unproc_queue <= end_new_cdb);
    std::copy_n(start_old_cdb_untouched_list,  size_untouched_list,        start_new_cdb_untouched_list);  // copy I
    std::copy_n(start_old_cdb_processed_queue, size_proc_and_unproc_queue, start_new_cdb_processed_queue); // copy III & IV

    // copy & sort II
    assert(start_old_cdb_processed_queue >= start_old_cdb_overwritten_list);
    assert(start_new_cdb_overwritten_queue >= start_new_cdb_overwritten_list);
    assert(start_old_cdb_processed_queue - start_old_cdb_overwritten_list == start_new_cdb_overwritten_queue - start_new_cdb_overwritten_list);
    std::partial_sort_copy( start_old_cdb_overwritten_list, start_old_cdb_processed_queue,
                            start_new_cdb_overwritten_list, start_new_cdb_overwritten_queue, compare_CE());

    // copy & sort V
    assert(end_old_cdb >= start_old_cdb_overwritten_queue);
    assert(end_new_cdb >= start_new_cdb_overwritten_queue);
    assert(end_old_cdb - start_old_cdb_overwritten_queue == end_new_cdb - start_new_cdb_overwritten_queue);
    std::partial_sort_copy(start_old_cdb_overwritten_queue, end_old_cdb,
                           start_new_cdb_overwritten_queue, end_new_cdb, compare_CE());

    // We now have reordered I-II-III-IV-V into I-III-IV-II-V and sorted each indidivual piece.
    // Furthermore (III-IV) is still sorted as a whole

    // Merge-sort II and V
    assert(start_new_cdb_overwritten_list <= start_new_cdb_overwritten_queue);
    assert(start_new_cdb_overwritten_queue <= end_new_cdb);
    std::inplace_merge(start_new_cdb_overwritten_list, start_new_cdb_overwritten_queue, end_new_cdb, compare_CE());

    // separate III from IV, this requires another lock
    { // scope for lock_guard
        std::lock_guard<std::mutex> lock_queue_head (TS_queue_head_mutex);

        assert(TS_queue_left >= 0);

        assert(insert_queue == TS_insert_queue);
        assert(insert_list == TS_insert_list);
        assert(start_queue_original == TS_start_queue_original);

        size_t const current_queue_head = TS_queue_head;
        assert(current_queue_head >= start_queue_original);
        assert(current_queue_head <= insert_queue);
        size_t const size_processed_queue = current_queue_head - start_queue_original;  // size of Part III
    //  Should be correct, but is unused (commented out to silence warnings):
    //    size_t const size_unprocessed_queue = insert_queue - current_queue_head;       // size of Part IV
    //    CompressedEntry const * const start_old_cdb_unprocessed_queue = previous_fast_cdb + current_queue_head;
        CompressedEntry * const start_new_cdb_unprocessed_queue= start_new_cdb_processed_queue + size_processed_queue;
        size_t const new_list_size = size_untouched_list + size_processed_queue;

        assert(start_new_cdb_untouched_list <= start_new_cdb_processed_queue);
        assert(start_new_cdb_processed_queue <= start_new_cdb_unprocessed_queue);
        std::inplace_merge(start_new_cdb_untouched_list, start_new_cdb_processed_queue, start_new_cdb_unprocessed_queue, compare_CE());
        assert(start_new_cdb_unprocessed_queue <= start_new_cdb_overwritten_list);
        assert(start_new_cdb_overwritten_list <= end_new_cdb);
        std::inplace_merge(start_new_cdb_unprocessed_queue, start_new_cdb_overwritten_list, end_new_cdb, compare_CE());

        // finished creating the data of the new snapshot. We now set up the metadata:
        assert(new_list_size <= db_size);
        hk3_sieve_init_metainfo(new_list_size, next_fast_cdb);

        // Check for termination condition: If the new queue size is extremely small, we would immediately re-trigger sorting.
        // In this case, we opt to terminate the algorithm.
        if (UNLIKELY( db_size - new_list_size < TS_min_queue_size) && (TS_finished.load(std::memory_order_relaxed) == TS_Finished::running))
        {
            assert(std::accumulate(TS_unmerged_transactions.cbegin(), TS_unmerged_transactions.cend(),size_t(0)) == TS_total_unmerged_transactions );
            // Note: TS_finished might be set to TS_Finished::saturated by another thread in the meantime.
            // This is actually acceptable, so we do not need compare_exchange_strong.

            // Note: If list size and saturation conditions are set correctly, we should not run out of queue.
            // Still, with a small(ish) total list size and a high number of threads (e.g. in the early pump stages)
            // this can happen due to unfortunate thread scheduling with essentially most good vectors
            // that are supposed to form the new queue being in some unmerged transaction_db's.

            // So if we have a lot of unmerged transactions, we signal all threads to finish, then
            // merge all pending transactions. After that, we may restart the algorithm or switch contexts.
            if(TS_total_unmerged_transactions > TS_min_pending_transactions)
            {
                TS_EXIT_COND("Triple sieve finished: queue size small, but considering resumption")
                TS_finished.store(TS_Finished::out_of_queue_resume, std::memory_order_relaxed);
            }
            else
            {
                TS_EXIT_COND("Triple sieve finished: queue size has become too small");
                TS_finished.store(TS_Finished::out_of_queue, std::memory_order_relaxed);
            }

        }

        // This actually sort-of "publishes" the new snapshot.
        // (Note that other threads can only read it once we give up some locks, so the atomicity is not even needed here.
        // and the publishing only really happens upon lock-release.)
        hk3_sieve_update_latest_cdb_snapshot(next_cdb_snapshot_ptr, id);

        ++TS_number_of_sorts; // should this be done inside the previous call?

        // This is not supposed to happen, but if it does, we bail out. It ensures that the algorithm actually
        // terminates in weird cases.
        if((TS_number_of_sorts > TS_max_number_of_sorts) && (TS_finished.load(std::memory_order_relaxed) == TS_Finished::running))
        {
            TS_EXIT_COND("Triple sieve finished: Maximum number of resorts exceeded.");
            TS_finished.store(TS_Finished::sortings_exceeded, std::memory_order_relaxed);
        }
    }   // release TS_queue_head_mutex. This allows get_p to work again.

    // Notify threads waiting in get_p: They will pick up the new snapshot now.
    // Note that insertions are still blocked.
    TS_wait_for_sorting.notify_all();

    // NOTE: The currently sorting flag is still set, but we have already released TS_queue_head_mutex
    // and we do not know if all pending db insertions have finished.

    // IMPORTANT CONSEQUENCE: In low-dimensional sieves with high number of threads (e.g. at start of pump)
    // there is a chance that held-up db writes will cause a lot of "data corruptions"
    // (i.e. data read from cdb and db is contradictory). The algorithm discards those data, which
    // means that we may lose a substantial number of reductions in such a setting.
    // We might consider moving the wait for db insertions upwards if this is a problem.

    // We now have to wait until all pending db insertions have finished, because otherwise
    // we cannot guarantee that the next set of insertions is disjoint.
    // This will ensure that all threads starting new writes have seen
    // all modifications above and all cdb and db writes related to any cdb-snapshot preciding the current (new) one.

    // Wait for db writes to finish. Note that this will very rarely actually spin.
    statistics.inc_stats_dataraces_sorting_blocked_db(wait_threshold - TS_db_insertions_finished);
    while(wait_threshold != TS_db_insertions_finished.load()) {}

    assert(TS_insertions_started == 2*wait_threshold + 1);
    assert(TS_cdb_insertions_finished == wait_threshold);
    assert(TS_db_insertions_finished == wait_threshold);
    TS_cdb_insertions_finished.store(0);
    TS_db_insertions_finished.store(0);
    TS_insertions_started.store(0, std::memory_order_seq_cst); // This finally allows insertions again. (seq_cst is just for emphasis)

    // Be aware that since we still had TS_currently_sorting set, other threads might have
    // detected a condition that sorting was overdue and failed to start sorting.
    // In particular, these threads might have gone to sleep again to wait on TS_wait_for_sorting.

    TS_currently_sorting.clear();
    TS_wait_for_sorting.notify_all(); // wake up potential sleepers (for the above reason).
}

/** hk3_sieve_execute_delayed_insertion tries to perform insertions of the given transaction_db into the database.
    The return value is the number of actual insertions performed.

    The argument transaction_db is modified accordingly, such that all processed elements are removed from it.
    Note: This function sorts transaction_db and may
        - prune some elements (throw them away)
        - insert some elements into (c)db
        - not process some elements, because the database "is busy" in multi-threaded environments.
    The return value only accounts for the actual insertions. There is no guarantee that transaction_db
    is empty after the call or that the return value is the size-change of transaction_db.

    update_len_bound MAY be updated to reflect TS_len_bound.
    This function may trigger sorting.

    Params:
    transaction_db: transactions to be (possibly) included in the database(s). transaction_db is modified.
    update_len_bound:   parameter that *may* be written to, in order to communicate a new length bound to the caller.
                        Note that the value is never *read* by execute_delayed_insertion, so you can give a dummy variable.
    id: thread-id, only used for debugging.
*/
size_t Siever::hk3_sieve_execute_delayed_insertion(TS_Transaction_DB_Type &transaction_db, float &update_len_bound, MAYBE_UNUSED unsigned int const id)
{
    ATOMIC_CPUCOUNT(603);
    // Change of number of saturated entries that comes from this batch of insertions.
    // We count locally first and perform the atomic increment at the end of the function.
    auto &&new_saturated_entries = merge_on_exit<size_t>([this](size_t const val)
    {
        if(TS_saturated_entries.fetch_sub(val) <= val )
        {
            if (TS_finished.load(std::memory_order_relaxed) == TS_Finished::running)
                TS_finished.store(TS_Finished::saturated, std::memory_order_relaxed);
        }
    });
    size_t const db_size = db.size(); // Note Siever::db_size() is cdb.size() at the moment, which DOES NOT WORK.

    // Sort transaction_db.

    // NOTE:  We sort transaction_db such that the *LARGEST* element is in front, (as opposed to list/queue, where the shortest is in front)
    // This is because we remove the shortest elements first for insertions and the largest elements for pruning.
    // For std::vector, removal of elements at the end are very cheap, but removals at the front means copying the rest of the vector,
    // so we delay all actual removals due to pruning until the end of execute_delayed_insertion, where we do the cleanup
    // in one go and when no longer holding any locks.
    // If we sorted the other way round, we would have to delay the removals due to insertions rather than those
    // due to pruning, which is more annoying to implement. Also, we typically do not prune at all, so we can often skip this altogeher.

    // We store how far an initial segment of transaction_db is already sorted in the transaction_db,
    // because in the case that another thread is sorting, we could enter exectue_delayed_insertion
    // repeatedly (and fail do anything). We do not want to sort almost the same transaction_db over and over in that case.
    assert(transaction_db.sorted_until <= transaction_db.size() );
    if(transaction_db.sorted_until < transaction_db.size() )
    {
        assert(std::is_sorted(transaction_db.new_vals.begin(), transaction_db.new_vals.begin()+transaction_db.sorted_until, [](Entry const &x, Entry const &y){return x.len > y.len;} ));
        std::sort(transaction_db.new_vals.begin()+transaction_db.sorted_until,transaction_db.new_vals.end(), [](Entry const &x, Entry const &y){return x.len > y.len;} );
        std::inplace_merge(transaction_db.new_vals.begin(), transaction_db.new_vals.begin()+transaction_db.sorted_until, transaction_db.new_vals.end(), [](Entry const &x, Entry const &y) {return x.len > y.len;}  );
    }
    size_t number_of_insertions_performed = 0; // we eventually return this to the caller.

    // We will sometimes prune the transaction_db from the left/front end.
    // The actual pruning is delayed until the cleanup phase at the end of the function. (see the comment about sorting order above)
    // transactions_left is the number of non-pruned transactions still in transaction_db and
    // can become smaller than transaction_db.size() iff we perform such pruning.
    // This means that the "real" (i.e. without the pruned elements) start of the transaction_db is at transaction_db[transaction_db.size() - transactions_left]
    size_t transactions_left = transaction_db.size();
    Entry * true_transaction_db = transaction_db.new_vals.data(); // points to the start of the non-pruned transactions.

    // sort_after indicates whether we need to sort afterwards. If yes, this value is >0 and should equal 1 + TS_number_of_sorts.
    // The purpose of including TS_number_of_sorts rather than using only a bool is that we do not immediately sort
    // once we detect that sorting is due and in the meantime some other thread might have finished sorting.
    // In the latter case, we do not sort ourself (if sorting is still really needed, this will be detected soon anyway;
    // note that our checks here are not perfect and may lead very rarely to double-sorting, which is OK).
    size_t sort_after = 0;

    /********************************************************************************************
                    How the insertion algorithm works on high-level & rationale:
    *********************************************************************************************

    We perform the actual overwriting of the current cdb-snapshot +db in several steps
    1.) We determine where and how many elements we insert in one go (alternatively, we prune, and skip 2--4)
    2.) We modify the corresponding TS_ indices to reserve a block inside cdb
    3.) We perform the actual replacements in cdb
    4.) We perform the actual replacements in db

    We may or may not perform 1+2 under a lock to make the modifications atomically
    (either lock or hope for the best && make some tests later, needs experiments)
    The current implementation uses a lock, because there are tricky data races with the cdb-accesses otherwise.

    We wish to basically overwrite appropriate worst vectors from the end of the queue / list, provided
    the vectors in transaction_db are actually shorter.
    A moment's though reveals that the problem we wish to solve here in step 1 is to find the (indices of the)
    t = #transaction_db many longest elements in the union of the 3 ranges overwriteable-queue, overwriteable-list and transaction_db.
    i.e. we want to select X elements from the queue part, Y elements from the list part and Z elements from the transaction_db such that
    these X+Y+Z = t elements are the t longest from their union.
    Then we overwrite Y list elements and X queue elements with the X+Y shortest transaction_db
    elements and prune the remaing Z transaction_db elements.
    Since the 3 ranges are individually sorted, it is enough to determine the numbers X,Y,Z of elements and this can be done in time O(log t).
    (This algorithm is well-known for 2 ranges and is essentially a binary search for the splitting of t into the ranges)
    For the case of 3 ranges, the algorithmic idea can be generalized and restated as follows:
    We know that at least one of the following three options is true for any x+y+z = t+2
    a) At least x of the t longest elements have to be from the queue part (-> We need to insert at least x elements into the queue)
    b) At least y of the t longest elements have to be from the list part (-> We need to insert at least y elements into the list)
    c) At leass z of the t longest elements have to be inside transaction_db (-> We have to prune transaction_db by at least z elements)
    (Proof by contradiction: otherwise, we simultaneously have X <= x-1, Y <= y-1, Z <= y-1, which implies (by adding) X + Y + Z <= t - 1)
    Determining one of the cases (a),(b),(c) that is true can be done by simply
    comparing the x'th largest element from the queue part, the y'th largest element from the list part
    and the z'th largest element from the transaction_db part.
    In each case, the problem can be recursively reduced to one with smaller value of t.

    Note that we do not care how the X+Y shortest elements of transaction_db are distributed to X elements from the queue
    and Y elements from the list in the end (they ultimately end up in the queue after the next resort).
    So rather than determining X, Y, Z completely by a recursive algorithm
    before doing any actual insertions / pruning, we include the actual insertion / pruning in the recursion:

    In cases a) or b), we can assume that the shortest x resp. y elements of transaction_db will end up in the queue resp.
    the list. In case c) we can prune the z longest elements from transaction_db.
    Consequently, we can avoid determining X,Y,Z completely, but rather just find some number
    x,y,z we can be sure to insert / prune.
    Then we actually perform the corresponding action as a "microtransaction" and then start over with a smaller value of t.
    (This makes a difference in multi-threading, since several threads are concurrently performing such actions.
    Also note that we can actually release the implicit locks in between the restarts with smaller t;
    this matters if some thread starts sorting in the meantime. )
    If x,y,z are at least a constant fraction of t, the total cost of Phase 1 + 2 from processing t inital elements is only O(log t).

    Since we do not really care that *exactly* the t longest elements are eliminated, we will put everything into either the
    queue or the list for small values of t (we still may wish to do pruning correctly for list insertions, though).
    "Small values" is generous and can be as large as the
    threshold where we actually trigger execute_delayed_inserion, so we expect the majority of
    insertions to actually use the small-t algorithm.

    **********
    Rationale:
    **********

    Be aware that the reason to use the large-t algorithm at all is to avoid phase 1+2 from potentially becoming a bottleneck:
    Since it is effectively a critical section, if 1+2 becomes a bottleneck due to high number of threads,
    depending on locking strategy used, various types of bad things would happen.
    E.g. if we return to caller after failing to grab the lock for 1+2 (as we do!),
    we would end up producing transactions faster than we can process them.
    We want to avoid such scenarios by design.
    The current design essentially implicitly auto-adjust the "microtransaction" size to the current contention level.
    Furthermore, the neccessity to determine X,Y,Z rather precisely arises from the fact that wrong list insertions in step 2 are
    very costly. Once we reserve a part of the list, we really want to make an actual replacement.
    If we later determine that the new element is actually longer than what we had, we can bail out,
    but this would still turn a list element into a queue element later. Consequently, the algorithm
    would then re-process the same element as p multiple times, which hurts overall performance.
    (In the (plausible!) scenario where the longest list elements are barely longer than the shortest queue
    elements, this phenomenon can be catastrophic: we have no guarantees of any progress at all.
    Avoiding both a bottleneck in 1+2 and list insertion failures basically rules out a naive algorithm.
    */

    // This while loop corresponds to the recursion above, with t == transactions_left, which
    // we hope to decrease in each iteration.
    // This loop might be left by a "goto cleanup;" from a nested loop as well.
    // (the "cleanup:" label is just after the loop.)
    // @goto-dislikers: Unfortunately, "break;" is not an option from within nested loops in C++, which lacks multi-break
    // and introducing booleans and more breaks is less clean than a simple goto.
    while( (transactions_left > 0) && sort_after == 0)
    {
        assert(transaction_db.new_vals.data() + transaction_db.size() == true_transaction_db + transactions_left);
        // TS_insertion_started  / TS_cdb_insertions_finished / TS_db_insertions_finished act as a locking mechanism
        // that allow concurrent insertions and prevent concurrent insertion+sorting.
        // (some limited form of concurrency is possible even for the latter)

        // atomically increment TS_insertions_started by 2 iff it is even, abort if odd.
        // Note that the lsb of TS_insertions_started indicates that some other thread is sorting,
        // whereas the other bits count the number of times we entered the code sections below.
        // TS_(c)db_insertions_finished count how often we left the code sections below.

        // NOTE: The sorting function sets the lsb of TS_insertions_started and then waits with some of its tasks
        // until TS_insertions_started/2 == TS_cdb_insertions_finished.
        // Then it waits with other tasks until TS_insertions_started/2 == TS_db_insertions_finished.
        // After sorting is all done, it clears the lsb of TS_insertions_started.

        { // scope for believed_insertions_started, the code in this scope essentially acquires 2 implicit locks.
            size_t believed_insertions_started = TS_insertions_started.load();
            do
            {
                if(believed_insertions_started % 2 == 1) // lsb of TS_insertions_started indicates that some other thread is sorting
                {
                    // TODO: Help with sorting?
                    assert(sort_after == 0);
                    goto cleanup; // this returns after some post-processing cleanup.
                }
            }
            while( TS_insertions_started.compare_exchange_weak(believed_insertions_started, believed_insertions_started+2, std::memory_order_acq_rel) == false );
            // Note: ABA problem is non-existant here.
        }
        // IMPORTANT:
        // We have acquired 2 locks here. If we leave the while-loop via a
        // break / goto, DO NOT FORGET TO RELEASE THEM.
        // TODO: Make some RAI wrappers for those.
        // Unfortunately, these would need to hold the Siever's this pointer and a released? bool (reference) as a state,
        // making it somewhat ugly.

        // TS_latest_cdb_version cannot change until we increment TS_cdb_insertions_finished.
        // So we use hk3_sieve_get_true_fast_cdb, which gets the latest cdb snapshot's underlying
        // array, bypassing any ref-counts.
        // This pointer is only safe to use until we increment TS_cdb_insertions_finished.
        CompressedEntry * const true_fast_cdb = hk3_sieve_get_true_fast_cdb();

        /**
            Step 1 + 2, determine the actual number of points to insert and and the start of the range.
        */
        // insertion_size is the number of actual insertions we perform in this iteration of the while loop.
        // insertion_size may shrink during the loop for various reasons as we encounter obstacles to insert everything
        // (or do not even try). We start with transactions_left, which is the number of valid transactions we actually have, which is
        // clearly the maximum we could hope for.
        size_t insertion_size = transactions_left;
        // We will (eventually) insert into *insertion_start_ptr, ..., *insertion_end_ptr (right bound is EXCLUSIVE)
        // number of insertions is insertion_size. Note that if insertion_size == 0, the pointers are meaningless.
        // If we count from the right, we go insertion_end_ptr[-1], ..., insertion_end_ptr[-insertion_size] (inclusive)
#ifndef NDEBUG
        CompressedEntry * insertion_start_ptr = nullptr;
#endif
        CompressedEntry * insertion_end_ptr   = nullptr;
        bool choose_queue_to_insert; // indicates whether we insert into the list or the queue.


        { // lock scope for locking TS_insertion_selection_mutex (phase 1 + 2)
            std::unique_lock<std::mutex> const lock_insertion_selection_mutex(TS_insertion_selection_mutex, std::try_to_lock_t{});
            if(!lock_insertion_selection_mutex)
            {
                TS_cdb_insertions_finished.fetch_add(1, std::memory_order_release);
                TS_db_insertions_finished.fetch_add(1, std::memory_order_release);
                goto cleanup;
            }

            auto const known_TS_insert_list  = TS_insert_list; // these used to be atomics
            auto const known_TS_insert_queue = TS_insert_queue;

            // We will jump here after pruning some of the largest elements from our transaction_db, and retry
            // (in this case, we keep the locks)
            try_again_reserve:

            /***********
                Step 1:
            ************/

            // We first select where and how many elements to reserve.
            // The purpose is to first fix the values of choose_queue_to_insert and insertion_size
            if(insertion_size > TS_large_transaction_size) // "large" insertion size, we use the strategy outline above.
            {

                // potential_list_insertions, potential_queue_insertions, potential_pruning correspond
                // to the values of x,y,z in the explanation from the beginning.
                // Unfortunately, we have to make some adjustments, because
                // - these values must only be as large as the corresponding ranges
                // - we want to avoid queue insertions if that would make the resulting size very small.
                // (essentially, if the size is very small, we limit transactions to 1 element;
                // if we reach the small regime for the first time, we back off from some insertions,
                // so that we can make multiple small transactions in that regime. This is to avoid
                // get_p from blocking)
                // - We need to treat the special cases x == 0 or y == 0 differently.

                statistics.inc_stats_replacements_large();

                // We try to insert 40% of the vectors into the list part, but at most as many elements into the list as there are.
                size_t const potential_list_insertions = std::min( (insertion_size+ 3) /4, known_TS_insert_list);
                bool const try_list_insertions = (known_TS_insert_list > 0); // are list insertions even an option?

                // full_queue_left includes both the processed and the unprocessed non-overwritten queue.
                // Note that the "correct" thing to do is to use only the unprocessed non-overwritten queue.
                // However, for our selection of x,y,z, we actually pretend that we can overwrite the processed queue.
                // (Observe that this still gives a correct answer where we can overwrite,
                // because the processed and unprocessed non-overwritten queue are JOINTLY sorted.)
                // It might just happen that we later realize that we can overwrite less than what we had anticipated:
                // If x of the t longest elements among the union of transaction_db, non-overwritten list
                // and (non-overwritten queue + processed queue) are from (non-overwritten queue + processed queue)
                // and x involves r elements from the processed queue,
                // then at least x' = x - r of the t longest element among the union of transaction_db, non-overwritten list
                // and non-overwritten queue are from non-overwritten queue.

                // Since TS_queue_left is an atomic that can change any moment, we have to deal
                // with this problem anyway later by reducing insertion_size, and adjusting already here seems not worthwhile.
                // Also note that if the insertion size later becomes small (or 0) because of this, we need to resort anyway,
                // so we do not get into an infinite loop.
                size_t const full_queue_left = known_TS_insert_queue - TS_start_queue_original;
                size_t const potential_queue_insertions  = std::min( (insertion_size+1)/2, full_queue_left);
                bool const try_queue_insertions = (potential_queue_insertions  > 0); // are queue insertions even an option?

                // bools are considered 0/1. The static_cast's are just to make this explicit.
                size_t const potential_pruning = insertion_size + static_cast<int>(try_list_insertions) + static_cast<int>(try_queue_insertions) - potential_list_insertions - potential_queue_insertions;
                assert(potential_pruning >= 1);
                assert(potential_pruning <= insertion_size);

                // transaction_bound_len, queue_bound_len, list_bound_len are set to the corresponding
                // x/y/z'th largest element of the ranges.

                // -1.0 is just a dummy value that makes the list / queue end infinitely good (thereby preventing overwrites)
                FT const transaction_bound_len = true_transaction_db[potential_pruning-1].len;
                FT const list_bound_len = LIKELY(try_list_insertions) ? true_fast_cdb[known_TS_insert_list - potential_list_insertions].len : -1.0;
                FT const queue_bound_len = LIKELY(try_queue_insertions) ? true_fast_cdb[known_TS_insert_queue - potential_queue_insertions].len : -1.0;

                // We now decide whether to insert into the list or the queue or to prune.
                // in case of pruning, it is enough to adjust transactions_left and insertion_size
                // (the actual processing of transaction_db is done afterwards)
                // Note that we do not release and reacquire locks if we prune (that would be possible, though)
                // and we also do not reload known_TS_insert_list / known_TS_insert_queue.
                if(queue_bound_len > list_bound_len)
                {
                    if(queue_bound_len > transaction_bound_len)
                    {
                        insertion_size = potential_queue_insertions;
                        choose_queue_to_insert = true;
                    }
                    else
                    {
                        // prune: statistics is collected during cleanup phase
                        assert(potential_pruning <= insertion_size);
                        assert(insertion_size <= transactions_left);
                        transactions_left -= potential_pruning;
                        true_transaction_db += potential_pruning;
                        insertion_size    -= potential_pruning;
                        goto try_again_reserve;
                    }
                }
                else
                {
                    if(list_bound_len > transaction_bound_len)
                    {
                        insertion_size = potential_list_insertions;
                        choose_queue_to_insert = false;
                    }
                    else
                    {
                        // prune: statistics is collected during cleanup phase
                        assert(potential_pruning <= insertion_size);
                        assert(insertion_size <= transactions_left);
                        transactions_left -= potential_pruning;
                        insertion_size    -= potential_pruning;
                        true_transaction_db += potential_pruning;
                        goto try_again_reserve;
                    }
                }
            } // end of large-insertion case
            else // if insertion_size is small, we (try to) put everything in one go.
            {
                assert(insertion_size > 0);
                statistics.inc_stats_replacements_small();

                if(UNLIKELY(known_TS_insert_list == 0))
                {
                    choose_queue_to_insert = true;
                }
                else if(UNLIKELY(known_TS_insert_queue == TS_start_queue_original))
                {
                    // extremely unlikely to happen, because this means we exhaust the queue without
                    // even having taken a single p in the current snapshot.
                    choose_queue_to_insert = false;
                }
                else
                {
                    choose_queue_to_insert = (true_fast_cdb[known_TS_insert_queue-1].len > true_fast_cdb[known_TS_insert_list-1].len);
                }

//                choose_queue_to_insert = (TS_queue_list_imbalance.load() > 0);
                if(choose_queue_to_insert)
                {
                    // unneccessary: The logic inside step 2 takes care of this!
                    // (due to the interplay with TS_queue_left)
                    // insertion_size = std::min(insertion_size, currently known queue size);
                    // intentionally keeping this as commented out.
                }
                else // list insertion case: We do not want insertion failures here!
                {
                    // limit insertion_size to the actual list size.
                    if (UNLIKELY(insertion_size > known_TS_insert_list))
                    {
                        insertion_size = known_TS_insert_list;
                        if(UNLIKELY(insertion_size == 0)) // should never happen, but anyway:
                        {
                            TS_cdb_insertions_finished.fetch_add(1, std::memory_order_release);
                            TS_db_insertions_finished.fetch_add(1, std::memory_order_release);
                            goto cleanup;
                        }
                    }
                    // prevent insertion failures: We check whether we should prune rather than list-insert
                    if(UNLIKELY(true_transaction_db[0].len > true_fast_cdb[known_TS_insert_list - insertion_size].len ) )
                    {
                        //back off from some insertions:
                        do
                        {
                            // prune 1 element
                            --insertion_size;
                            ++true_transaction_db;
                            --transactions_left;
                        } while((true_transaction_db[0].len > true_fast_cdb[known_TS_insert_list - insertion_size]. len) && (insertion_size > 0));
                        if(insertion_size == 0)
                        {
                            TS_cdb_insertions_finished.fetch_add(1, std::memory_order_release);
                            TS_db_insertions_finished.fetch_add(1, std::memory_order_release);
                            goto cleanup;
                        }
                    }
                }
                // TODO: Deprecate TS_queue_list_imbalance
            }

            /***********
                Step 2:
            ************/

            // We now have selected choose_queue_to_insert and set a (preliminary) value of insertion_size
            assert(insertion_size <= transactions_left);

            if(choose_queue_to_insert) // we try to insert into queue
            {
                assert(insertion_size > 0);
                // Note that TS_queue_left serves only to synchronize different ways the queue can shrink.
                // If TS_queue_left is positive after decrementing it, the PRE-reservation is deemed successful.
                auto const old_queue_left = TS_queue_left.fetch_sub(insertion_size, std::memory_order_relaxed);
                static_assert(std::is_signed<decltype(old_queue_left)>::value, "TS_queue_left must be signed"); // just to be sure.

                // In this case, the new value old_queue_left - insertion_size is small, and we have to back off from
                // some insertions and pre-reservations.
                if(old_queue_left < static_cast<std::remove_const<decltype(old_queue_left)>::type> (insertion_size + TS_max_extra_queue_size))
                {
                    if(old_queue_left <= 0) // no more queue left, abort. We do not trigger sorting ourselves.
                    {
                        TS_queue_left.fetch_add(insertion_size, std::memory_order_relaxed);
                        TS_cdb_insertions_finished.fetch_add(1, std::memory_order_release);
                        TS_db_insertions_finished.fetch_add(1, std::memory_order_release);
                        goto cleanup;
                    }
                    assert(old_queue_left > 0);
                    assert(insertion_size > 0);

                    if(old_queue_left > static_cast<std::remove_const<decltype(old_queue_left)>::type>(TS_max_extra_queue_size))
                    {
                        // Reduce the insertion size, such that TS_queue_left hits TS_max_extra_queue_size if we had
                        // decremented TS_queue_left by that value. We add back to TS_queue_left accordingly.
                        // (Note that there may be interleaving operations to TS_queue_left, but that is fine)
                        // Note: argument to fetch_add is >= 1
                        TS_queue_left.fetch_add( insertion_size + TS_max_extra_queue_size - old_queue_left, std::memory_order_relaxed );
                        insertion_size = old_queue_left - TS_max_extra_queue_size;
                    }
                    else    // we are probably already in "critical queue size" mode. We only add one element to ensure some progress.
                            // (Note: We will exit the big while-loop anyway, because we detect sorting is due)
                    {
                        TS_queue_left.fetch_add(insertion_size - 1, std::memory_order_relaxed);
                        insertion_size = 1;
                    }
                    // This will trigger sorting later; it also prevents more iterations of the big while-loop.
                    sort_after = 1 + TS_number_of_sorts.load(std::memory_order_relaxed);
                }
                // we successfully decremented TS_queue_left by insertion_size elements, such that the result after decrementing
                // was still positive (well, or at least it's as if...).

                assert(old_queue_left >= 0);

                // This is the new value of known_TS_insert_queue we want to have.
                size_t const insertion_start_index = known_TS_insert_queue - insertion_size;

                // modify_queue_list_imbalance will be (new length of back of queue) - (old length of back of queue)
                // where length of back of queue is the length of the next element to be overwritten.
                // We read the next element before we actually perform the change to TS_insert_queue
                // so that we know no other thread can modify it.
                // decltype(TS_queue_list_imbalance.load()) modify_queue_list_imbalance = TS_outofqueue_len;
                assert(insertion_start_index > TS_start_queue_original);
                // modify_queue_list_imbalance = std::round (TS_queue_list_imbalance_multiplier * true_fast_cdb[insertion_start_index-1].len);
                // assert(TS_insert_queue == known_TS_insert_queue);
                TS_insert_queue = insertion_start_index; // Equivalently: TS_insert_queue-= insertion_size;

                #ifndef NDEBUG
                insertion_start_ptr = true_fast_cdb + insertion_start_index;
                #endif
                insertion_end_ptr = true_fast_cdb + insertion_start_index + insertion_size;

                // Modify TS_queue_list_imbalance:
                // modify_queue_list_imbalance -= std::round(TS_queue_list_imbalance_multiplier * insertion_end_ptr[-1].len);
                // TS_queue_list_imbalance.fetch_add(modify_queue_list_imbalance); // seq_cst !
                assert(insertion_size > 0);

                // end of phase 1 + 2 here
            }
            else // We try to insert insertion_size many elements into the list. This is much easier than the queue.
            {
                assert(insertion_size <= known_TS_insert_list);

                size_t const insertion_start_index = known_TS_insert_list - insertion_size;
                // decltype(TS_queue_list_imbalance.load()) modify_queue_list_imbalance = TS_outoflist_len;
//                if(LIKELY(insertion_start_index > 0))
//                {
//                    modify_queue_list_imbalance = std::round(TS_queue_list_imbalance_multiplier * true_fast_cdb[insertion_start_index-1].len);
//                }
                TS_insert_list = insertion_start_index;

#ifndef NDEBUG
                insertion_start_ptr = true_fast_cdb + insertion_start_index;
#endif
                insertion_end_ptr   = true_fast_cdb + insertion_start_index + insertion_size;

//                modify_queue_list_imbalance -= std::round(TS_queue_list_imbalance_multiplier * insertion_end_ptr[-1].len);
//                TS_queue_list_imbalance.fetch_sub(modify_queue_list_imbalance); // seq_cst
            }
        } // release lock_insertion_selection_mutex

        // If we get here, we have successfully reserved from the list or queue.

        assert(insertion_size > 0); // this should not happen anyway (and it would not be a problem). Code below would handle this correctly.
        if(insertion_size == 0) // to avoid endless loop in case we have overwritten both list and queue.
        {
            TS_cdb_insertions_finished.fetch_add(1, std::memory_order_release);
            TS_db_insertions_finished.fetch_add(1, std::memory_order_release);
            goto cleanup;
        }
        assert(insertion_start_ptr != nullptr);
        assert(insertion_end_ptr != nullptr);
        assert(insertion_start_ptr <= insertion_end_ptr);
        // Even with the above assert this needs a cast
        assert(static_cast<decltype(insertion_size)>(insertion_end_ptr - insertion_start_ptr) == insertion_size);

        // We now perform step 3: Insertion into cdb:
        // We also have exclusive write access to the reserved portion of cdb.
        // Note that we are not guaranteed that the vectors we would overwrite are actually longer, so we need to make some checks first
//        std::vector<IT> db_insertion_positions(insertion_size); // will hold the db indexes of where we insert

        IT db_insertion_positions[insertion_size]; // This uses a gcc extension!
        // We insert shortest from transactions_db first into longest from cdb first.
        // Remember we sorted transaction_db the other way round, so shortest from transaction_db means highest index, longest from cdb also means highest index.

        auto const transactions_end_it = transaction_db.new_vals.cend();
        auto const insertion_size_copy = insertion_size;
        for(size_t i = 1; i <= insertion_size_copy; ++i) // index is from 1 to insertion_size, because we count from the end.
        {
            if(UNLIKELY(insertion_end_ptr[-i].len < transactions_end_it[-i].len ))
            { // What we would overwrite is shorter that what is in transaction_db.
                insertion_size = i - 1; // number of successful insertions we actually performed.
#ifndef NDEBUG
                insertion_start_ptr = insertion_end_ptr - insertion_size;
#endif
                // TODO: If this happens when inserting into the list, take countermeasures.
                if(choose_queue_to_insert)
                    statistics.inc_stats_replacementfailures_queue(insertion_size_copy - insertion_size);
                else
                    statistics.inc_stats_replacementfailures_list(insertion_size_copy - insertion_size);
                break;
            }
            insertion_end_ptr[-i].len = transactions_end_it[-i].len;
            insertion_end_ptr[-i].c   = transactions_end_it[-i].c;
            auto const db_insert = insertion_end_ptr[-i].i;
            db_insertion_positions[i-1] = db_insert;
            for(unsigned int t = 0; t < (sizeof(Entry) + 63) / 64; ++t)
            {
                PREFETCH3(reinterpret_cast<char*>(db.data()+db_insert) + 64*t, 1,0 );
            }
        }

        if (TS_insertions_performed.fetch_add(insertion_size_copy, std::memory_order_relaxed) + insertion_size_copy >= TS_resort_after * db_size)
        {
            sort_after = 1 + TS_number_of_sorts.load(std::memory_order_relaxed);
        }

        TS_cdb_insertions_finished.fetch_add(1, std::memory_order_release);

        // NOTE: If we failed to insert something above, we prune the corresponding number of vectors
        // from transaction_db. These pruned vectors are pruned from the left end of transaction_db
        // and do not need to be the same as the vectors for which insertion failed above.
        transactions_left -= insertion_size_copy; // this is both for successful and unsuccessful cdb replacements
        number_of_insertions_performed += insertion_size;
        true_transaction_db += (insertion_size_copy - insertion_size);

        if(choose_queue_to_insert)
            statistics.inc_stats_replacements_queue(insertion_size);
        else
            statistics.inc_stats_replacements_list(insertion_size);
        // We now have overwritten the relevant parts of cdb. db_insertion_positions marks the db positions we have yet to overwrite.

        // Part IV: Overwrite db:

        for(size_t i = 0; i < insertion_size; ++i)
        {
            MAYBE_UNUSED auto const b = uid_hash_table.erase_uid( db[db_insertion_positions[i]].uid); assert(b);
            if(db[db_insertion_positions[i]].len < params.saturation_radius )
            {
                statistics.inc_stats_dataraces_replaced_was_saturated();
            }
            else if (transactions_end_it[-i-1].len < params.saturation_radius)
            {
                ++new_saturated_entries;
            }
            db[db_insertion_positions[i]] = transactions_end_it[-i-1];
        }
        TS_db_insertions_finished.fetch_add(1, std::memory_order_release); // release the db lock
        transaction_db.new_vals.resize(transaction_db.new_vals.size() - insertion_size ); // transactions_left was already shrunk.
    } // end of while transaction_db not empty

cleanup:

    if(sort_after != 0)
    {
        // If the cdb snapshot has changed, don't resort. Note that this check is not perfect
        // because TS_latest_cdb_version might change between our load and any checks (until some
        // lock acquisition inside sorting).
        // This rare event just adds some inefficiency.
        // (The alternative is to raise the flag before releasing the cdb mutex)
        TS_latest_cdb_mutex.lock();
//        auto const true_fast_cdb_new = TS_latest_cdb_version -> data();
        TS_latest_cdb_mutex.unlock();
        if (sort_after == TS_number_of_sorts + 1)
        {
            if(TS_currently_sorting.test_and_set() == false)
            {
                hk3_sieve_resort(id);
            }
        }
    }

    if(transactions_left < transaction_db.size())
    {
        size_t const to_prune = transaction_db.size() - transactions_left;
        for(unsigned int i = 0; i < to_prune; ++i)
        {
            MAYBE_UNUSED bool const b = uid_hash_table.erase_uid(transaction_db.new_vals[i].uid); assert(b);
        }
        for(unsigned int i = 0; i < transactions_left; ++i)
        {
            transaction_db.new_vals[i] = transaction_db.new_vals[i+to_prune];
        }
        transaction_db.new_vals.resize(transactions_left);
        statistics.inc_stats_replacementfailures_prune(to_prune);
    }
    transaction_db.sorted_until = transaction_db.size();
    update_len_bound = TS_len_bound.load();
    return number_of_insertions_performed;
}

// obtains the next point p to be processed and the number of x1's that are to be compared with it.
// Here, p is the point among the triple that is from the queue.
// p may have negative length, which indicates that the algorithm has finished. In this case, the size_t element is meaningless.
// thread_local_snapshot is the latest CDB snapshot the calling thread knows about, which is also updated.
std::pair<Entry, size_t> Siever::hk3_sieve_get_p(TS_CDB_Snapshot * &thread_local_snapshot, unsigned int const id, TS_Transaction_DB_Type &transaction_db, float &)
{
    ATOMIC_CPUCOUNT(604);
    std::pair<Entry, size_t> return_pair; // return value
    Entry &ret = return_pair.first;
    // NOTE: We may actually unlock (and relock) the mutex prior to destruction.
    // Use of std::unique_lock is mandated because we use a std::condition_variable (which only works well with std::unique_lock)
#if COLLECT_STATISTICS_DATARACES >= 2
    std::unique_lock<std::mutex> lock_queue_head (TS_queue_head_mutex, std::try_to_lock);
    if(!lock_queue_head)
    {
        statistics.inc_stats_dataraces_get_p_blocked();
        lock_queue_head.lock();
    }
#else
    std::unique_lock<std::mutex> lock_queue_head (TS_queue_head_mutex);
#endif
    TS_total_unmerged_transactions += (transaction_db.size() - TS_unmerged_transactions[id]);
    TS_unmerged_transactions[id] = transaction_db.size();
    while(true) // we either use break to finish or continue to try again
    {
        // if we jump here by continue; lock_queue_head is still owned (it may have been released in the meantime)
        if(UNLIKELY(TS_finished.load(std::memory_order_relaxed) != TS_Finished::running))
        {
            ret.len = -1.; // negative value signals "finished" to the caller
            break; // i.e. goto return ret;
        }

        if(thread_local_snapshot != TS_latest_cdb_snapshot_p.load(std::memory_order_relaxed))
        {
            hk3_sieve_release_snapshot(thread_local_snapshot, id);
            thread_local_snapshot = hk3_sieve_get_latest_snapshot(id);
        }

//        thread_local_snapshot = TS_latest_cdb_version;
        // TODO: TS_extra_range should become a Sieve parameter.
        // In fact, We may want to interpolate between these two extreme cases.
        if (TS_extra_range)
        {
            return_pair.second = db.size();
        }
        else
        {
            return_pair.second = TS_queue_head;
        }

        CompressedEntry const * const fast_cdb = thread_local_snapshot -> snapshot.data();

        auto const old_queue_left = TS_queue_left.fetch_sub(1);
        // Note that as long as TS_queue_head_mutex is owned, no other thread can reset TS_queue_left due to sorting.


        if (old_queue_left > static_cast<int>(TS_max_extra_queue_size)) // there was enough space on the queue before we took the value.
        {
            size_t const db_index = fast_cdb[TS_queue_head].i;
            ++TS_queue_head;
            lock_queue_head.unlock(); // We don't need to keep the lock anymore.
            assert(db_index < db.size());
            ret.x = db[db_index].x; // may in very unlucky cases read garbage.

            recompute_data_for_entry<Recompute::recompute_all>(ret); // since we did not lock the db, we recompute.
            break; // i.e. finish;
        }
        else
        {
            if(TS_currently_sorting.test_and_set() == false) // == false means that it was us who set the value to true, so this mean "success".
            {
                // Recall that since we held TS_queue_head_mutex up to this point, no resort has finished in the current while(true) iteration:
                // This also implies that no resort can have has started since, because the TS_currently_sorting flag was clear.
                TS_queue_left.fetch_add(1); // we give back what we took, so other threads may use it while we sort.
                lock_queue_head.unlock();
                hk3_sieve_release_snapshot(thread_local_snapshot, id);
                hk3_sieve_resort(id);
                thread_local_snapshot = hk3_sieve_get_latest_snapshot(id);
#if COLLECT_STATISTICS_DATARACES >= 2
                if (!lock_queue_head.try_lock())
                {
                    statistics.inc_stats_dataraces_get_p_blocked();
                    lock_queue_head.lock();
                }
#else
                lock_queue_head.lock();
#endif
                continue; // try again
            }
            else // someone else is already sorting.
            {
                if(old_queue_left > 0) // we actually took something from the queue, proceed as normal.
                {
                    size_t const db_index = fast_cdb[TS_queue_head].i;
                    ++TS_queue_head;
                    lock_queue_head.unlock(); // We don't need to keep the lock anymore.
                    ret.x = db[db_index].x; // may in very unlucky cases read garbage.
                    recompute_data_for_entry<Recompute::recompute_all>(ret); // since we did not lock the db, we recompute.
                    if(params.threads == 1)
                    {
                        assert(ret.uid == db[db_index].uid);
                        assert(ret.c == db[db_index].c);
                    }
                    break; // finish
                }
                else // too bad, some thread is sorting and no p is left. We wait for sorting to finish
                {
                    TS_queue_left.fetch_add(1);
                    statistics.inc_stats_dataraces_out_of_queue();
                    TS_wait_for_sorting.wait(lock_queue_head); // we could help with sorting rather than wait...
                    continue; // try again
                }
            }
        }
    } // end of while(true) - loop
    return return_pair;
}

// snapshot management:
// Our algorithm processes a series of cdb snapshots, for which we preallocate memory.
// To reduce memory footprint, we limit the maximum number of snapshots that can exits and
// re-use the memory from an old cdb snapshot.
// For this, there exists a (global) repository TS_cdb_snapshots of currently existing snapshots
// (this includes "inactive" snapshots, which can be re-used). Note that, since we cannot reallocate
// TS_cdb_snapshot itself due to atomics, it is a fixed-length array of size TS_max_snapshots, but only the first
// TS_snapshots_used entries are valid (in the sense that the std::vector TS_cdb_snapshot[i].snapshot
// has reserved some memory).

// The datastructures are initialized / de-initialized in the (non-threaded part of) hk3_sieve,
// stealing memory from cdb. Within the threaded part, *all* access to the latest snapshot must go through
// the following functions, with the following usage restrictions:

// A thread that want to get access to the latest cdb snapshot can use either
// hk3_sieve_get_true_fast_cdb(...) or hk3_sieve_get_latest_snapshot(...). These differ
// in usage characteristics and limitations.

// With auto local_ptr = hk3_sieve_get_latest_snapshot(...),
// the calling thread can get a local copy of some cdb snapshot that can be freely used until hk3_sieve_release_snapshot is called.
// Any such call must be paired off with a call to hk3_sieve_release_snapshot(local_ptr) by the same thread,
// which invalidates local_ptr again. local_ptr can then be reassigned with hk3_sieve_get_latest_snapshot(...).
// (This means that to change the thread's local version, we first release the old, then assign some new)

// hk3_sieve_get_true_fast_cdb gives a pointer to the latest cdb snapshot without the need to release and is more efficient.
// This function must only be called and the result used when the implicit TS_insertions_started / TS_cdb_insertions_finished lock is held.
// (Note that the latest cdb snapshot cannot change while this implicit lock is held)

// hk3_sieve_get_free_snapshot will return a new TS_cdb_snapshots of a new snapshot.
// Note that this function might create new snapshots and block if too many snapshots are in use.
// (So the calling thread might need to release its own local_ptr's first before calling to avoid deadlock)

// The result of mt_get_free_snapshot must be used in a subsequent call to hk3_sieve_update_latest_cdb_snapshot by the same thread.
// This atomically published the new free snapshot.
// This pair of calls needs to be done within a critical section and no other snapshot-related functions must be called in between by this thread.
// Between the pair of calls, the thread has exclusive access to the new snapshot.

void Siever::hk3_sieve_init_snapshots()
{
    if(TS_snapshots_used == 0)
    {
        TS_DEBUG_SNAPSHOTS("Creating initial snapshot\n")
        // TS_cdb_snapshot[0] lends ressources from cdb, so we make it empty.
        TS_cdb_snapshots[0].snapshot.clear();
        TS_cdb_snapshots[0].snapshot.shrink_to_fit();
        ++TS_snapshots_used;
    }

    // TS_snapshots[0] is unused AND EMPTY (we swap ressources with cdb)
    assert(TS_snapshots_used > 0);
    assert(TS_cdb_snapshots[0].snapshot.size() == 0);
    // We swap cdb with TS_cb_snapshots[0]:
    TS_cdb_snapshots[0].snapshot.swap(cdb);
    TS_latest_cdb_snapshot_p = &TS_cdb_snapshots[0];
    TS_cdb_snapshots[0].ref_count = 1;
    for(size_t i = 1; i < TS_snapshots_used; ++i)
    {
        assert(TS_cdb_snapshots[i].ref_count == 0);
    }

    if(TS_snapshots_used > 1)
    {
        TS_free_snapshot = 1 + 1; // meaning that TS_cdb_snapshot[1] is unused.
    }
    else
    {
        TS_free_snapshot = 0; // no unused snapshot known.
    }
}

void Siever::hk3_sieve_restore_cdb()
{
    auto const find_latest_snapshot = TS_latest_cdb_snapshot_p.load(std::memory_order_relaxed) - &TS_cdb_snapshots[0];
    assert(find_latest_snapshot >=0);
    assert(static_cast<std::remove_const<mystd::make_unsigned_t<decltype(find_latest_snapshot)>>::type> (find_latest_snapshot) < TS_snapshots_used);
    assert(TS_snapshots_used <= TS_max_snapshots);

    TS_cdb_snapshots[0].snapshot.swap(TS_cdb_snapshots[find_latest_snapshot].snapshot);
    // This is just to ensure that all snapshots but 0 have their ref-count at 0:
    TS_cdb_snapshots[find_latest_snapshot].ref_count.store(TS_cdb_snapshots[0].ref_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
    cdb.swap(TS_cdb_snapshots[0].snapshot);

    for(size_t i = 1; i < TS_snapshots_used; ++i)
    {
        assert(TS_cdb_snapshots[i].ref_count.load(std::memory_order_relaxed) == 0);
    }
}


void Siever::hk3_sieve_release_snapshot(TS_CDB_Snapshot * &thread_local_snapshot, MAYBE_UNUSED unsigned int const id)
{
    TS_DEBUG_SNAPSHOTS( "Thread " + std::to_string(id) + " forgets old snapshot " + std::to_string(thread_local_snapshot-&TS_cdb_snapshots[0]) + "\n" )
    auto const old_ref = thread_local_snapshot->ref_count.fetch_sub(1, std::memory_order_release);
    if(old_ref == 1)
    {
        TS_free_snapshot_mutex.lock();
        if(thread_local_snapshot ->ref_count.load(std::memory_order_relaxed) == 0) // This check IS neccessary!
            TS_free_snapshot = 1 + (thread_local_snapshot - &TS_cdb_snapshots[0]);
        TS_free_snapshot_mutex.unlock();
        TS_wait_for_free_snapshot.notify_one();
        TS_DEBUG_SNAPSHOTS( "Thread " + std::to_string(id) + " marked old snapshot " + std::to_string(thread_local_snapshot - &TS_cdb_snapshots[0]) + " as unused.\n" )
    }
}

auto Siever::hk3_sieve_get_latest_snapshot(MAYBE_UNUSED unsigned int const id) -> TS_CDB_Snapshot *
{
    TS_CDB_Snapshot * ret;
    { // lock_guard scope
        std::lock_guard<std::mutex> lock_latest_snapshot(TS_latest_cdb_mutex);
        ret = TS_latest_cdb_snapshot_p.load(std::memory_order_relaxed);
        MAYBE_UNUSED auto const old_ref_count = ret->ref_count.fetch_add(1, std::memory_order_relaxed);
        assert(old_ref_count > 0); // TS_latest_cdb_snapshot itself adds +1 to the refcount.
    }
    TS_DEBUG_SNAPSHOTS( "Thread " + std::to_string(id) + " learnt new snapshot " + std::to_string(ret-&TS_cdb_snapshots[0]) + "\n" )
    return ret;
}

auto Siever::hk3_sieve_get_true_fast_cdb() -> CompressedEntry *
{
    return TS_latest_cdb_snapshot_p.load(std::memory_order_relaxed)->snapshot.data();
}

void Siever::hk3_sieve_update_latest_cdb_snapshot(TS_CDB_Snapshot * const next_cdb_snapshot_ptr, MAYBE_UNUSED unsigned int const id)
{
    std::lock_guard<std::mutex> lock_latest_snapshot(TS_latest_cdb_mutex);
    // Note: This cannot mark the snapshot as unused, because the caller of resort still has a reference.
    // (unless we are single-threaded, but then we do not care)
    TS_latest_cdb_snapshot_p.load(std::memory_order_relaxed)->ref_count.fetch_sub(1,std::memory_order_relaxed);
    TS_latest_cdb_snapshot_p = next_cdb_snapshot_ptr;
    assert(next_cdb_snapshot_ptr -> ref_count == 1);
//    next_cdb_snapshot_ptr ->ref_count.store(1, std::memory_order_relaxed);
}

auto Siever::hk3_sieve_get_free_snapshot(MAYBE_UNUSED unsigned int const id) -> TS_CDB_Snapshot *
{
    size_t new_snapshot_index; // return value
    // Not that TS_free_snapshot is 1 + index, with 0 meaning "No free snapshot known". So we have to adjust by -1

    // Make make 4 tries to get a new snapshot:
    // 1.) If TS_free_snapshot != 0, we know a free snapshot
    // 2.) Check if any ref-count is 0.
    // 3.) Create a new snapshot, if allowed to do so
    // 4.) Wait until TS_free_snapshot becomes non-zero.

    // Note that after we checked TS_free_snapshot == 0 under a lock in (1), any snapshot that becomes *newly* free will
    // be guaranteed to set TS_free_snapshot to a non-zero value.
    // (2) is neccessary, because TS_free_snapshots only indicates *new* free snapshots
    // (If more than one is free at time, it is not picked up)
    // This ensures that (4) will eventually trigger.

    auto const db_size = db.size();

    TS_free_snapshot_mutex.lock();
    new_snapshot_index = TS_free_snapshot;
    TS_free_snapshot = 0; // either it already was 0, or we set it to zero, meaning "We do not know any more free snapshots".

    if(new_snapshot_index != 0)
    {
        --new_snapshot_index;
        assert(TS_cdb_snapshots[new_snapshot_index].ref_count.load() == 0);
        TS_cdb_snapshots[new_snapshot_index].ref_count.store(1, std::memory_order_relaxed);
        TS_free_snapshot_mutex.unlock();

        TS_DEBUG_SNAPSHOTS( "Sorting by thread " + std::to_string(id) + " (1)-recognized snapshot" + std::to_string(new_snapshot_index) + " as unused\n" )
        assert(new_snapshot_index < TS_snapshots_used);

        TS_cdb_snapshots[new_snapshot_index].snapshot.resize(db_size);
        return &(TS_cdb_snapshots[new_snapshot_index]);
    }
    TS_free_snapshot_mutex.unlock();

    for(size_t i = 0; i < TS_snapshots_used; ++i)
    {
        if(TS_cdb_snapshots[i].ref_count.load(std::memory_order_acquire) ==0 )
        {
            TS_DEBUG_SNAPSHOTS( "Sorting by thread " + std::to_string(id) + " (2)-recognized snapshot" + std::to_string(i) + " as unused\n" )

            TS_free_snapshot_mutex.lock();
            TS_free_snapshot = 0;
            TS_cdb_snapshots[i].ref_count.store(1, std::memory_order_relaxed);
            TS_free_snapshot_mutex.unlock();

            TS_cdb_snapshots[i].snapshot.resize(db_size);


            return &(TS_cdb_snapshots[i]);
        }
    }

    if(TS_snapshots_used < TS_max_snapshots)
    {
        TS_cdb_snapshots[TS_snapshots_used].snapshot.reserve(db.capacity());
        TS_cdb_snapshots[TS_snapshots_used].snapshot.resize(db_size);
        TS_cdb_snapshots[TS_snapshots_used].ref_count.store(1, std::memory_order_relaxed);
        new_snapshot_index = TS_snapshots_used;
        ++TS_snapshots_used;
        TS_DEBUG_SNAPSHOTS( "Sorting by thread " + std::to_string(id) + " created new snapshot #" + std::to_string(new_snapshot_index) + "\n" )
        return &(TS_cdb_snapshots[new_snapshot_index]);
    }

    { // scope for the unique_lock (to release before the resize call)
        std::unique_lock<std::mutex> lock_free_snapshots(TS_free_snapshot_mutex);
        while( TS_free_snapshot == 0 )
        {
            TS_DEBUG_SNAPSHOTS( "Sorting by thread " + std::to_string(id) + " is waiting for free snapshots." )
            TS_wait_for_free_snapshot.wait(lock_free_snapshots);
        }
        TS_DEBUG_SNAPSHOTS( "Waiting for new snapshots finished" )
        new_snapshot_index = TS_free_snapshot - 1;
        TS_free_snapshot = 0;
        TS_cdb_snapshots[new_snapshot_index].ref_count.store(1, std::memory_order_relaxed);
    }
    assert(new_snapshot_index < TS_snapshots_used);
    assert(TS_cdb_snapshots[new_snapshot_index].ref_count.load() == 1);
    TS_cdb_snapshots[new_snapshot_index].snapshot.resize(db_size);
    return &(TS_cdb_snapshots[new_snapshot_index]);
}


// put a new 3-reduction candidate p +/- x1 +/- x2 into transaction db:
// x1 and x2 are determined by x1_index and x2_index (into db) and the signs by x1_sign_flip / x2_sign_flip
// (with sign_flip == true if we have a - sign)
// new_uid is the new_uid of the constructed point. This may be wrong due to races and the function correctly handles this.
// Return value indicates success.
bool Siever::hk3_sieve_delayed_3_red(TS_Transaction_DB_Type &transaction_db, Entry const &p, size_t const x1_index, bool const x1_sign_flip, size_t const x2_index, bool const x2_sign_flip, UidType const new_uid)
{
    ATOMIC_CPUCOUNT(606);
    if(uid_hash_table.insert_uid(new_uid)==false)
    {
        return false;
    }
    // otherwise, we are good to go.
    transaction_db.emplace_back();
    Entry &new_entry = transaction_db.new_vals.back();
    new_entry.x = p.x;
    assert(new_entry.x.size() >= n);
    assert(x1_index < db.size());
    assert(x2_index < db.size());
    if(x1_sign_flip)
    {
        for(unsigned int i = 0; i < n; ++i) { new_entry.x[i] -= db[x1_index].x[i]; }
    }
    else
    {
        for(unsigned int i = 0; i < n; ++i) { new_entry.x[i] += db[x1_index].x[i]; }
    }
    if(x2_sign_flip)
    {
        for(unsigned int i = 0; i < n; ++i) { new_entry.x[i] -= db[x2_index].x[i]; }
    }
    else
    {
        for(unsigned int i = 0; i < n; ++i) { new_entry.x[i] += db[x2_index].x[i]; }
    }
    recompute_data_for_entry<Recompute::recompute_all_and_consider_otf_lift>(new_entry); // includes recomputing uid !
    if (UNLIKELY(new_entry.uid != new_uid))
    {
        statistics.inc_stats_dataraces_3();
        statistics.dec_stats_collisions_3();
        transaction_db.pop_back();
        MAYBE_UNUSED auto b = uid_hash_table.erase_uid(new_uid); assert(b);
        return false; // some data race corrupted uid.
    }
    return true;
}

// Put a new 2-reduction +/- x1 -/+ x2 into transaction_db.
// x1 and x2 are determined by db indexes.
// x1_sign_flip == true means x1 appears with - sign
// x2_sign_flip == true means x2 appears with + sign
// (The default is x1 - x2)
// new_uid is the uid of the new point. It may be wrong due to races. The functions handles this.
// Return value indicates success.
bool Siever::hk3_sieve_delayed_2_red_inner(TS_Transaction_DB_Type &transaction_db, size_t const x1_db_index, bool const x1_sign_flip, size_t const x2_db_index, bool const x2_sign_flip, UidType new_uid)
{
    ATOMIC_CPUCOUNT(609);
    // statistics for collisions are managed by the caller.
    if (uid_hash_table.insert_uid(new_uid)==false)
    {
        return false;
    }

    // Note: new_uid was computed as b1*x1_uid - b2*x2_uid
    // where bi == -1 iff xi_sign_flip is set.
    // We compute the new_entry as x1 - B * x2, where
    // B is -1 iff x1_sign_flip != x2_sign_flip
    // If x1_sign_flip is set, this means that the new_uid we just inserted is the negative
    // of the uid of the new_entry. This is fine with the uid_hash_table, which is +/- invariant, but the UNLIKELY(...) event below would trigger, so we flip uid.

    if(x1_sign_flip)
    {
        new_uid = -new_uid;
    }
    transaction_db.emplace_back();
    Entry &new_entry = transaction_db.back();
    new_entry.x = db[x1_db_index].x;
    if(x1_sign_flip == x2_sign_flip)
    {
        for(unsigned int i = 0; i < n; ++i) { new_entry.x[i] -= db[x2_db_index].x[i]; }
    }
    else
    {
        for(unsigned int i = 0; i < n; ++i) { new_entry.x[i] += db[x2_db_index].x[i]; }
    }
    recompute_data_for_entry<Recompute::recompute_all_and_consider_otf_lift>(new_entry);
    if(UNLIKELY(new_entry.uid != new_uid))
    {
        statistics.inc_stats_dataraces_2inner();
        statistics.dec_stats_collisions_2inner();
        transaction_db.pop_back();
        MAYBE_UNUSED auto b = uid_hash_table.erase_uid(new_uid); assert(b);
        return false; // some data race corrupted uid.
    }
    return true;
}

// Put a new 2-reduction p +/- x1 into transaction_db
// x1 is determined by x1_index (into db) and sign_flip,
// with sign_flip == true for p - db[x1_index].
// (so the default is p + x1)
// uid is not passed to this function, but computed inside.
// Return value indicates success.
bool Siever::hk3_sieve_delayed_red_p_db(TS_Transaction_DB_Type &transaction_db, Entry const &p, size_t const x1_index, bool const sign_flip)
{
    ATOMIC_CPUCOUNT(607);
    UidType new_uid = p.uid;
    assert(x1_index < db.size());
    if(sign_flip) // new point is p - db[x1_index]
    {
        new_uid -= db[x1_index].uid;
        if(uid_hash_table.insert_uid(new_uid))
        {
            transaction_db.emplace_back();
            Entry &new_entry = transaction_db.back();
            new_entry.x = p.x;
            for(unsigned int i = 0; i < n; ++i)
            {
                new_entry.x[i] -= db[x1_index].x[i];
            }
            recompute_data_for_entry<Recompute::recompute_all_and_consider_otf_lift>(new_entry); // includes recomputing uid !
            if (UNLIKELY(new_entry.uid != new_uid))
            {
                statistics.inc_stats_dataraces_2outer();
                statistics.dec_stats_collisions_2outer();
                transaction_db.pop_back();
                MAYBE_UNUSED auto b = uid_hash_table.erase_uid(new_uid); assert(b);
                return false; // some data race corrupted uid.
            }
            return true; // successful
        }
        return false; // collision;
    }
    else // new point is p + db[x1_index]
    {
        new_uid += db[x1_index].uid;
        if(uid_hash_table.insert_uid(new_uid))
        {
            transaction_db.emplace_back();
            Entry &new_entry = transaction_db.back();
            new_entry.x = p.x;
            for(unsigned int i = 0; i < n; ++i)
            {
                new_entry.x[i] += db[x1_index].x[i];
            }
            recompute_data_for_entry<Recompute::recompute_all_and_consider_otf_lift>(new_entry); // includes recomputing uid !
            if (UNLIKELY(new_entry.uid != new_uid))
            {
                statistics.inc_stats_dataraces_2outer();
                statistics.dec_stats_collisions_2outer();
                transaction_db.pop_back();
                MAYBE_UNUSED auto b = uid_hash_table.erase_uid(new_uid); assert(b);
                return false; // some data race corrupted uid.
            }
            return true; // successful
        }
        return false; // collision;
    }
assert(false);
}

namespace g6k
{
// helper for template metaprogramming...
template<class Arg> using Get_Value_Type = typename Arg::value_type;
}

// Process the inner loop of the 3 sieve:
// transaction_db is the current thread's transaction database, used to buffer writes to db
// block1 and block2 are containers of TS_FilteredCE's that contain (at least) end_block1 resp. end_block2 many entries.
// The TS_FilteredCE entries are supposed to be far away from p (i.e. close to -p).
// We look for reductions p+/- x1 +/ x2 where x1 is from block1[0], ... block1[end_block1-1] and
// x2 is from block2[0],...,block2[end_block2-1].
// If TS_Inner2red is set, we also look for reductions x1 +/- x2.
// local_len_bound is a bound on the length^2 of new entries. It gets updated as we write to db.
// If EnforceOrder is set, we only consider x1[i], x2[j] pairs with i < j.
// (This is intended for the case where block1 == block2)

// Note that SmallContainer1 is supposed to be a small cache-optimized block and
// LargeContainer2 is supposed to be larger.
// We use std::array for SmallContainer1 and either std::array or std::vector for LargeContainer2.
// You can use g6k_utility::MaybeFixed or std::integral_constant for Integer1 to pass a compile-time constant for end_block1.
template<bool EnforceOrder, class SmallContainer1, class LargeContainer2, class Integer1>
inline void Siever::hk3_sieve_process_inner_batch(TS_Transaction_DB_Type &transaction_db, Entry const &p, SmallContainer1 const &block1, Integer1 const end_block1, LargeContainer2 const &block2, size_t const end_block2, float &local_len_bound, MAYBE_UNUSED unsigned int const id)
{
    ATOMIC_CPUCOUNT(610);
    // Make sure that the template arguments are good:
    // Some of these checks may be too restrictive; in this case, feel free to change them.
    using ValueType1 = mystd::decay_t<mystd::detected_t<g6k::Get_Value_Type, SmallContainer1>>;
    using ValueType2 = mystd::decay_t<mystd::detected_t<g6k::Get_Value_Type, LargeContainer2>>;
    static_assert(std::is_same<ValueType1, mystd::nonesuch>::value == false, "SmallContainer1 has no ::value_type typedef.");
    static_assert(std::is_same<ValueType2, mystd::nonesuch>::value == false, "LargeContainer2 has no ::value_type typedef.");
    static_assert(std::is_same<ValueType1, TS_FilteredCE>::value == true, "SmallContainer1 is no container of TS_FilteredCE's");
    static_assert(std::is_same<ValueType2, TS_FilteredCE>::value == true, "LargeContainer2 is no container of TS_FilteredCE's");
    static_assert( (EnforceOrder == false) || (std::is_same<SmallContainer1,LargeContainer2>::value == true), "EnforceOrder only makes sense if both containers are the same, really." );
    // This turns an std::integral_constant<UInt> or g6k_utility::Maybe_fixed<Uint> into a Uint
    // and a plain UInt (without ::value_type) into a UInt (i.e. keeps it as it is).
    using Int1 =  mystd::detected_or_t<Integer1, g6k::Get_Value_Type, Integer1>;

    if (EnforceOrder)
    {
        // The static_casts and & are only neccessary to make it compile if EnforceOrder is false :-(
        // C++17 constexpr if would solve this issue.
        assert(static_cast<void const*>(&block1) == static_cast<void const*>(&block2));
    }

    #if COLLECT_STATISTICS_XORPOPCNT_PASS || COLLECT_STATISTICS_FULLSCPRODS
        auto &&local_stat_successful_xorpopcnt_sieving = merge_on_exit<unsigned long long>( [this](unsigned long long val)
        {
            statistics.inc_stats_xorpopcnt_pass_inner(val);
            statistics.inc_stats_fullscprods_inner(val);
        } );
    #endif
    ENABLE_IF_STATS_REDSUCCESS (auto &&local_stat_successful_2red_inner = merge_on_exit<unsigned long>([this](unsigned long val){ statistics.inc_stats_2redsuccess_inner(val); }); )
    ENABLE_IF_STATS_REDSUCCESS (auto &&local_stat_successful_3red       = merge_on_exit<unsigned long>([this](unsigned long val){ statistics.inc_stats_3redsuccess(val); }); )
    ENABLE_IF_STATS_COLLISIONS (auto &&local_stat_collisions_2inner     = merge_on_exit<unsigned long>([this](unsigned long val){ statistics.inc_stats_collisions_2inner(val); }); )
    ENABLE_IF_STATS_COLLISIONS (auto &&local_stat_collisions_3          = merge_on_exit<unsigned long>([this](unsigned long val){ statistics.inc_stats_collisions_3(val); }); )
    ENABLE_IF_STATS_OTFLIFTS   (auto &&local_stat_otflifts_2inner       = merge_on_exit<unsigned long>([this](unsigned long val){ statistics.inc_stats_otflifts_2inner(val); }); )
    ENABLE_IF_STATS_OTFLIFTS   (auto &&local_stat_otflifts_3            = merge_on_exit<unsigned long>([this](unsigned long val){ statistics.inc_stats_otflifts_3(val); }); )

    if (!EnforceOrder)
    {
        statistics.inc_stats_xorpopcnt_inner(end_block2 * static_cast<Int1>(end_block1) );
        statistics.inc_stats_3reds(          end_block2 * static_cast<Int1>(end_block1) );
        if (TS_perform_inner2red)
        {
            statistics.inc_stats_2reds_inner(end_block2 * static_cast<Int1>(end_block1) );
        }
    }
    else
    {
        statistics.inc_stats_xorpopcnt_inner( (end_block2 * (end_block2 -1)) / 2 );
        statistics.inc_stats_3reds(           (end_block2 * (end_block2 -1)) / 2 );
        if (TS_perform_inner2red)
        {
            statistics.inc_stats_2reds_inner( (end_block2 * (end_block2 -1)) / 2 );
        }
    }

    assert(static_cast<Int1>(end_block1) <= TS_Cacheline_Opt);
    std::array<decltype(block1[0].c_proj), TS_Cacheline_Opt> block1_c_proj;
    for(Int1 i = 0; i < static_cast<Int1>(end_block1); ++i)
    {
        block1_c_proj[i] = block1[i].c_proj;
    }
    std::array<decltype(db[0].yr), TS_Cacheline_Opt> block1_yr;
    std::array<decltype(db[0].len), TS_Cacheline_Opt> block1_len;
    for(Int1 i = 0; i < static_cast<Int1>(end_block1); ++i)
    {
        block1_yr[i] = db[block1[i].db_index].yr;
        block1_len[i] = db[block1[i].db_index].len;
    }
    //    std::array<decltype(db[0].len), TS_Cacheline_Opt> block1_len;

    auto const n = this->n;

    for(size_t x2_block_index = 0; x2_block_index < end_block2; ++x2_block_index)
    {
        auto const block2_entry = block2[x2_block_index];
        auto const &x2_db_index = block2_entry.db_index;
        auto const &x2_c_proj   = block2_entry.c_proj;
        auto const x2_yr = db[x2_db_index].yr;
        auto const x2_len = db[x2_db_index].len;
        auto const x2_uid_adjusted = block2[x2_block_index].uid_adjusted;
        auto const x2_uid = db[x2_db_index].uid;
        if(x2_uid != x2_uid_adjusted && x2_uid != -x2_uid_adjusted)
        {
            continue;
        }

        for(Int1 x1_block_index = 0; x1_block_index < static_cast<Int1>(end_block1); ++x1_block_index)
        {
            CPP17CONSTEXPRIF(EnforceOrder)
            {
                if(x1_block_index >= x2_block_index) break;
            }
            // we want to know whether POPCNT(x1 ^ x2) is either > TS_INNER_SIMHASH_THRESHOLD (->3 reduction)
            // or                                                < XPC_THRESHOLD              (->2 reduction)

            static_assert(XPC_THRESHOLD < TS_INNER_SIMHASH_THRESHOLD, "");

            unsigned w = unsigned(0) - XPC_THRESHOLD;
            for (size_t k = 0; k < XPC_WORD_LEN; ++k)
            {
                // NOTE return type of __builtin_popcountl is int not unsigned int
                w += __builtin_popcountl(block1_c_proj[x1_block_index][k] ^ x2_c_proj[k]);
            }

            if(UNLIKELY( w > TS_INNER_SIMHASH_THRESHOLD - XPC_THRESHOLD))
            {
                #if COLLECT_STATISTICS_XORPOPCNT_PASS || COLLECT_STATISTICS_FULLSCPRODS
                ++local_stat_successful_xorpopcnt_sieving;
                #endif
		//        size_t const x2_db_index = block2[x2_block_index].db_index;
                size_t const x1_db_index = block1[x1_block_index].db_index;
                FT const scalar_prod_x1_x2 = std::inner_product(block1_yr[x1_block_index].cbegin(), block1_yr[x1_block_index].cbegin()+n, x2_yr.cbegin(), static_cast<FT>(0));
                // w < LARGECONSTANT means POPCNT(x1^x2) >= XPC_THESHOLD, so no overflow occured and we should perform a 3-reduction.
                // The value of LARGECONSTANT is rather arbitrary. The choice of 2^16 is to allow a smart compiler to translate it as a check whether high-order bytes are zero.
                // TODO: Check whether compilers are smart.
                if(w < 65536)
                {
                    // (p + x1 + x2)^2 = p^2 + x1^2 + x2^2 + 2<x1,x2>  + 2<p,x1>) + 2<p,x2>
                    // = (1/2p^2 + x1^2 + 2<p,x1>) + (1/2p^2 + x2^2 + 2<p,x2>) + 2<x1,x2>
                    // The signs for <p,x1>, <p,x2> are correctly adjusted when storing in bucket / local_cache_block.
                    // The correct sign choice for 2<x1,x2> actually depends on whether x1_sign_flip == x2_sign_flip,
                    // but due to the simhash check it is very likely that the correct sign is - 2abs(<x1,x2>).
                    // Experimentally, std::abs() is faster than a branch; note that we do a final recomputation anyway, because we have to account
                    // for data races.
                    FT const believed_final_len = block1[x1_block_index].len_score + block2_entry.len_score - 2. * std::abs(scalar_prod_x1_x2);
                    if(believed_final_len < local_len_bound)
                    {
                        UidType believed_new_uid = block1[x1_block_index].uid_adjusted + block2_entry.uid_adjusted + p.uid;
                        ENABLE_IF_STATS_REDSUCCESS(++local_stat_successful_3red;)
                        if (!hk3_sieve_delayed_3_red(transaction_db, p, x1_db_index, block1[x1_block_index].sign_flip, x2_db_index, block2_entry.sign_flip, believed_new_uid))
                        {
                            ENABLE_IF_STATS_COLLISIONS(++local_stat_collisions_3;)
                        }
                        else if(transaction_db.size() % TS_transaction_bulk_size == 0)
                        {
                            hk3_sieve_execute_delayed_insertion(transaction_db, local_len_bound, id);
                        }
                    }
                    else if(params.otf_lift && (believed_final_len < params.lift_radius)) // UNLIKELY??? Not so sure
                    {
                        ENABLE_IF_STATS_OTFLIFTS(++local_stat_otflifts_3;)
                        hk3_sieve_otflift_p_x1_x2(p, x1_db_index, block1[x1_block_index].sign_flip, x2_db_index, block2_entry.sign_flip, believed_final_len);
                    }
                }
                else CPP17CONSTEXPRIF (TS_perform_inner2red) // w is huge due to underflow, i.e. x1 - x2 (after sign-adjustment) is a possible 2 reduction
                {
                    FT const believed_final_len = block1_len[x1_block_index] + x2_len - 2. * std::abs(scalar_prod_x1_x2);
                    if(believed_final_len < local_len_bound)
                    {
                        UidType believed_new_uid = block1[x1_block_index].uid_adjusted - block2_entry.uid_adjusted;
                        ENABLE_IF_STATS_REDSUCCESS(++local_stat_successful_2red_inner;)
                        if (!hk3_sieve_delayed_2_red_inner(transaction_db, x1_db_index, block1[x1_block_index].sign_flip, x2_db_index, block2_entry.sign_flip, believed_new_uid))
                        {
                            ENABLE_IF_STATS_COLLISIONS(++local_stat_collisions_2inner;)
                        }
                        else if(transaction_db.size() % TS_transaction_bulk_size == 0)
                        {
                            hk3_sieve_execute_delayed_insertion(transaction_db, local_len_bound, id);
                        }
                    }
                    else if (params.otf_lift && (believed_final_len < params.lift_radius))
                    {
                        ENABLE_IF_STATS_OTFLIFTS(++local_stat_otflifts_2inner;)
                        hk3_sieve_otflift_x1_x2(x1_db_index, block1[x1_block_index].sign_flip, x2_db_index, block2_entry.sign_flip, believed_final_len );
                    }
                }
            }
        }
    }
}

void Siever::hk3_sieve_otflift_p_db(Entry const &p, size_t const db_index, bool const sign_flip, double const believed_len)
{
    ATOMIC_CPUCOUNT(611);
    ZT x_full[r];
    LFT otf_helper[OTF_LIFT_HELPER_DIM];
    std::fill(x_full, x_full+l, static_cast<ZT>(0));
    // x_full[l],...,x_full[r-1] = p_entry.x[0] +/- db[x1_db_index].x[0], ..., p_entry.x[n-1] +/- db[x1_db_index].x[n-1]
    if(sign_flip)
    {
        std::transform(p.x.cbegin(), p.x.cbegin()+n, db[db_index].x.cbegin(), x_full+l, std::minus<ZT>{});
        std::transform(p.otf_helper.cbegin(), p.otf_helper.cbegin()+OTF_LIFT_HELPER_DIM, db[db_index].otf_helper.cbegin(), otf_helper, std::minus<LFT>{});
    }
    else
    {
        std::transform(p.x.cbegin(), p.x.cbegin()+n, db[db_index].x.cbegin(), x_full+l, std::plus<ZT>{});
        std::transform(p.otf_helper.cbegin(), p.otf_helper.cbegin()+OTF_LIFT_HELPER_DIM, db[db_index].otf_helper.cbegin(), otf_helper, std::plus<LFT>{});
    }
    lift_and_compare(x_full, believed_len * gh, otf_helper);
}

void Siever::hk3_sieve_otflift_p_x1_x2(Entry const &p, size_t const x1_db_index, bool const x1_sign_flip, size_t const x2_db_index, bool const x2_sign_flip, double const believed_len)
{
    ATOMIC_CPUCOUNT(612);
    ZT x_full[r];
    LFT otf_helper[OTF_LIFT_HELPER_DIM];
    std::fill(x_full, x_full+l, static_cast<ZT>(0));
    if(x1_sign_flip)
    {
        std::transform(p.x.cbegin(), p.x.cbegin()+n, db[x1_db_index].x.cbegin(), x_full+l, std::minus<ZT>{});
        std::transform(p.otf_helper.cbegin(), p.otf_helper.cbegin()+OTF_LIFT_HELPER_DIM, db[x1_db_index].otf_helper.cbegin(), otf_helper, std::minus<LFT>{});
    }
    else
    {
        std::transform(p.x.cbegin(), p.x.cbegin()+n, db[x1_db_index].x.cbegin(), x_full+l, std::plus<ZT>{});
        std::transform(p.otf_helper.cbegin(), p.otf_helper.cbegin()+OTF_LIFT_HELPER_DIM, db[x1_db_index].otf_helper.cbegin(), otf_helper, std::plus<LFT>{});
    }
    if(x2_sign_flip)
    {
        for(unsigned int i = 0; i < n; ++i)
            (x_full+l)[i] -= db[x2_db_index].x[i];
        for(unsigned int i=0; i < OTF_LIFT_HELPER_DIM; ++i)
            otf_helper[i] -= db[x2_db_index].otf_helper[i];
    }
    else
    {
        for(unsigned int i = 0; i < n; ++i)
            (x_full+l)[i] += db[x2_db_index].x[i];
        for(unsigned int i=0; i < OTF_LIFT_HELPER_DIM; ++i)
            otf_helper[i] += db[x2_db_index].otf_helper[i];
    }
    lift_and_compare(x_full, believed_len * gh, otf_helper);
}

void Siever::hk3_sieve_otflift_x1_x2(size_t const x1_db_index, bool const x1_sign_flip, size_t const x2_db_index, bool const x2_sign_flip, double const believed_len)
{
    ATOMIC_CPUCOUNT(613);
    ZT x_full[r];
    LFT otf_helper[OTF_LIFT_HELPER_DIM];
    std::fill(x_full, x_full+l, static_cast<ZT>(0));
    if(x1_sign_flip == x2_sign_flip)
    {
        std::transform(db[x1_db_index].x.cbegin(), db[x1_db_index].x.cbegin()+n, db[x2_db_index].x.cbegin(), x_full+l, std::minus<ZT>{});
        std::transform(db[x1_db_index].otf_helper.cbegin(), db[x1_db_index].otf_helper.cbegin()+OTF_LIFT_HELPER_DIM, db[x2_db_index].otf_helper.cbegin(), otf_helper, std::minus<LFT>{});
    }
    else
    {
        std::transform(db[x1_db_index].x.cbegin(), db[x1_db_index].x.cbegin()+n, db[x2_db_index].x.cbegin(), x_full+l, std::plus<ZT>{});
        std::transform(db[x1_db_index].otf_helper.cbegin(), db[x1_db_index].otf_helper.cbegin()+OTF_LIFT_HELPER_DIM, db[x2_db_index].otf_helper.cbegin(), otf_helper, std::plus<LFT>{});
    }
    lift_and_compare(x_full, believed_len * gh, otf_helper);
}
