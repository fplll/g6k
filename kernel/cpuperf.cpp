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

#ifdef PERFORMANCE_COUNTING
std::vector<uint64_t> perfcounters(1000,0);
std::vector<std::atomic<uint64_t>> atomic_perfcounters(1000);
cpu::performance_counter_manager perfmanager;

struct cpuperf_init_t {
    cpuperf_init_t()
    {
        /* siever.h */
        perfmanager.add_performance_counter(perfcounters[0], "Siever lifetime");

        /* BGJ1 */
        perfmanager.add_performance_counter(perfcounters[100], "BGJ1_sieve");
        perfmanager.add_performance_counter(atomic_perfcounters[101], "BGJ1_sieve_task (T)");
        perfmanager.add_performance_counter(atomic_perfcounters[102], "BGJ1_sieve_mainloop (T)");
        perfmanager.add_performance_counter(atomic_perfcounters[103], "BGJ1_sieve_postprocess (T)"); // after having found a likely reduction
        perfmanager.add_performance_counter(atomic_perfcounters[104], "bgj1_execute_delayed_replace (T)");
        perfmanager.add_performance_counter(perfcounters[105], "execute_delayed_replace_sort");
        perfmanager.add_performance_counter(atomic_perfcounters[106], "bgj1_replace_in_db (T)");

        /* Overhead */
        perfmanager.add_performance_counter(perfcounters[200], "initialize_local");
        perfmanager.add_performance_counter(perfcounters[201], "refresh_db_collision_checks");
        perfmanager.add_performance_counter(perfcounters[202], "extend_left");
        perfmanager.add_performance_counter(perfcounters[203], "extend_right");
        perfmanager.add_performance_counter(perfcounters[204], "shrink_left");
        // perfmanager.add_performance_counter(perfcounters[205], "shrink_right");
        perfmanager.add_performance_counter(perfcounters[206], "growdb");
        perfmanager.add_performance_counter(perfcounters[207], "shrinkdb");
        perfmanager.add_performance_counter(perfcounters[208], "gsopostprocessing");
        perfmanager.add_performance_counter(perfcounters[209], "parallel_sort_cdb");
        perfmanager.add_performance_counter(perfcounters[210], "recompute_histo");
        perfmanager.add_performance_counter(perfcounters[211], "lift_and_compare_prepare");
        perfmanager.add_performance_counter(perfcounters[212], "lift_and_compare");
        perfmanager.add_performance_counter(atomic_perfcounters[213], "sample (T)");
        perfmanager.add_performance_counter(atomic_perfcounters[214], "Recompute (T)");
        perfmanager.add_performance_counter(atomic_perfcounters[215], "Recompute & babai (T)");
        perfmanager.add_performance_counter(perfcounters[216], "reserve");
        perfmanager.add_performance_counter(perfcounters[217], "switch_mode");
        perfmanager.add_performance_counter(atomic_perfcounters[218], "gsopostprocess_task (T)");

        perfmanager.add_performance_counter(atomic_perfcounters[250], "compute_uid (T)");
        perfmanager.add_performance_counter(atomic_perfcounters[251], "insert_uid (T)");
        perfmanager.add_performance_counter(atomic_perfcounters[252], "check_uid (T)");
        perfmanager.add_performance_counter(atomic_perfcounters[253], "erase_uid (T)");
        perfmanager.add_performance_counter(atomic_perfcounters[254], "replace_uid (T)");

        perfmanager.add_performance_counter(atomic_perfcounters[260], "compress to simhash (T)");

        // plain Sieves
        perfmanager.add_performance_counter(perfcounters[301], "plain gauss sieve");
        // 300, 302, 303 were deleted
        perfmanager.add_performance_counter(perfcounters[304], "nv-sieve");

        /* for temporary usage */
        perfmanager.add_performance_counter(perfcounters[501], "TMP1");
        perfmanager.add_performance_counter(perfcounters[502], "TMP2");
        perfmanager.add_performance_counter(perfcounters[503], "TMP3");
        /* triple_sieve_mt.cpp */
        perfmanager.add_performance_counter(perfcounters[600], "TS_triple_sieve");
        perfmanager.add_performance_counter(atomic_perfcounters[601], "TS_init_metainfo (T)");
        perfmanager.add_performance_counter(perfcounters[602], "TS_resort");
        perfmanager.add_performance_counter(atomic_perfcounters[603], "TS_execute insertions (T)");
        perfmanager.add_performance_counter(atomic_perfcounters[604], "TS_get_p (T)");
        perfmanager.add_performance_counter(atomic_perfcounters[605], "TS_triple_sieve_mt_task (T)");
        perfmanager.add_performance_counter(atomic_perfcounters[606], "TS_delayed_3_reds (T)");
        perfmanager.add_performance_counter(atomic_perfcounters[607], "TS_delayed_2_reds (T)");
        perfmanager.add_performance_counter(atomic_perfcounters[608], "TS_update_len_bound (T)");
        perfmanager.add_performance_counter(atomic_perfcounters[609], "TS_delayed_2_reds_inner (T)");
        perfmanager.add_performance_counter(atomic_perfcounters[610], "TS_inner_batch (T)");
        perfmanager.add_performance_counter(atomic_perfcounters[611], "TS_outer_lifts (T)");
        perfmanager.add_performance_counter(atomic_perfcounters[612], "TS_triple_lifts (T)");
        perfmanager.add_performance_counter(atomic_perfcounters[613], "TS_inner_lifts (T)");

        perfmanager.add_performance_counter(atomic_perfcounters[680], "TS_triple_sieve (T)");
        perfmanager.add_performance_counter(atomic_perfcounters[681], "TS_triple_sieve (T)");
        perfmanager.add_performance_counter(atomic_perfcounters[682], "TS_triple_sieve (T)");

        perfmanager.add_performance_counter(perfcounters[690], "TS_triple_sieve");
        perfmanager.add_performance_counter(perfcounters[691], "TS_triple_sieve");
        perfmanager.add_performance_counter(perfcounters[692], "TS_triple_sieve");

    }
} cpuperf_init;

void show_cpu_stats()
{
    uint64_t now = cpu::cpu_timestamp();
    uint64_t total = *perfmanager._counters[0];
    if (total > uint64_t(1)<<57)
        total += now;
    for (unsigned i = 0; i < perfmanager._counters.size(); ++i)
    {
        uint64_t cnt = *perfmanager._counters[i];
        if (0 == cnt)
            continue;
        if (cnt > uint64_t(1)<<57)
            cnt += now;
        std::cout << i << "\t: " << perfmanager._descriptions[i] << std::endl;
        std::cout << i << "\t: " << double(cnt)*100.0/double(total) << " %, \t cycles=" << cnt << std::endl;
    }
    for (unsigned i = 0; i < perfmanager._atomic_counters.size(); ++i)
    {
        uint64_t cnt = *perfmanager._atomic_counters[i];
        if (0 == cnt)
            continue;
        if (cnt > uint64_t(1)<<57)
            cnt += now;
        std::cout << i + perfmanager._counters.size() << "\t: " << perfmanager._atomic_descriptions[i] << std::endl;
        std::cout << i + perfmanager._counters.size() << "\t: " << double(cnt)*100.0/double(total) << " %, \t cycles=" << cnt << std::endl;
    }
}

#else
void show_cpu_stats()
{
}
#endif
