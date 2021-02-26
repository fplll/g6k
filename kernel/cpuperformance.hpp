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


/*********************************************************************************\
*                                                                                 *
* https://github.com/cr-marcstevens/snippets/tree/master/cxxheaderonly            *
*                                                                                 *
* cpuperformance.hpp - A header only C++ light-weight aids for CPU cycle counting *
* Copyright (c) 2017 Marc Stevens                                                 *
*                                                                                 *
* MIT License                                                                     *
*                                                                                 *
* Permission is hereby granted, free of charge, to any person obtaining a copy    *
* of this software and associated documentation files (the "Software"), to deal   *
* in the Software without restriction, including without limitation the rights    *
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell       *
* copies of the Software, and to permit persons to whom the Software is           *
* furnished to do so, subject to the following conditions:                        *
*                                                                                 *
* The above copyright notice and this permission notice shall be included in all  *
* copies or substantial portions of the Software.                                 *
*                                                                                 *
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR      *
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        *
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE     *
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER          *
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   *
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE   *
* SOFTWARE.                                                                       *
*                                                                                 *
\*********************************************************************************/

#ifndef CPUPERFORMANCE_HPP
#define CPUPERFORMANCE_HPP


#include <vector>
#include <string>
#include <iostream>
#include <cstdio>
#include <atomic>

#ifndef __GNUC__
#include <intrin.h>
#endif

namespace cpu {

	inline uint64_t cpu_timestamp()
	{
#ifdef __GNUC__
		uint32_t highpart, lowpart;
		asm volatile("rdtsc": "=d"(highpart), "=a"(lowpart));
		return (uint64_t(highpart) << 32) | uint64_t(lowpart);
#else
		return __rdtsc();
#endif

	}

	inline void start_update_counter(std::atomic<uint64_t> &performancecounter)
	{
        performancecounter.fetch_sub(cpu_timestamp(), std::memory_order_relaxed);
	}
	inline void end_update_counter(std::atomic<uint64_t> &performancecounter)
	{
        performancecounter.fetch_add(cpu_timestamp(), std::memory_order_relaxed);
	}

	inline void start_update_counter(uint64_t& performancecounter)
	{
		performancecounter -= cpu_timestamp();
	}
	inline void end_update_counter(uint64_t& performancecounter)
	{
		performancecounter += cpu_timestamp();
	}

	class update_performance_counter {
		uint64_t& _counter;
	public:
		update_performance_counter(uint64_t& performance_counter)
			: _counter(performance_counter)
		{
			_counter -= cpu_timestamp();
		}
		~update_performance_counter()
		{
			_counter += cpu_timestamp();
		}
	};

	class update_atomic_performance_counter {
        std::atomic<uint64_t>& _counter;
    public:
        update_atomic_performance_counter(std::atomic<uint64_t> &performance_counter)
            : _counter(performance_counter)
        {
            _counter.fetch_sub(cpu_timestamp(), std::memory_order_relaxed);
        }
        ~update_atomic_performance_counter()
        {
            _counter.fetch_add(cpu_timestamp(), std::memory_order_relaxed);
        }
	};

	class performance_counter_manager {
	public:
		std::vector<uint64_t*> _counters;
		std::vector<std::atomic<uint64_t>*> _atomic_counters;
		std::vector<std::string> _descriptions;
		std::vector<std::string> _atomic_descriptions;

		void add_performance_counter(uint64_t& counter, const std::string& description)
		{
			_counters.push_back(&counter);
			_descriptions.push_back(description);
		}

		void add_performance_counter(std::atomic<uint64_t> &atomic_counter, std::string const &description)
		{
            _atomic_counters.push_back(&atomic_counter);
            _atomic_descriptions.push_back(description);
		}

		void show_results()
		{
			for (unsigned i = 0; i < _counters.size(); ++i)
			{
				std::cout << "Counter " << i << ": \t" << _descriptions[i] << std::endl;
				std::cout << "Counter " << i << ": \tValue = " << (*_counters[i]) << std::endl;
			}
			for (unsigned i = 0; i < _atomic_counters.size(); ++i)
			{
                std::cout << "Counter " << i+_counters.size() << ": \t" << _atomic_descriptions[i] << std::endl;
                std::cout << "Counter " << i+_counters.size() << ": \tValue = " << (_atomic_counters[i]->load()) << std::endl;
			}
		}
	};
} // namespace cpu


#endif // CPUPERFORMANCE_HPP
