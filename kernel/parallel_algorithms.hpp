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


/***************************************************************************************\
*                                                                                      *
* https://github.com/cr-marcstevens/snippets/tree/master/cxxheaderonly                 *
*                                                                                      *
* parallel_algorithms.hpp - A header only C++ light-weight parallel algorithms library *
* Copyright (c) 2020 Marc Stevens                                                      *
*                                                                                      *
* MIT License                                                                          *
*                                                                                      *
* Permission is hereby granted, free of charge, to any person obtaining a copy         *
* of this software and associated documentation files (the "Software"), to deal        *
* in the Software without restriction, including without limitation the rights         *
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell            *
* copies of the Software, and to permit persons to whom the Software is                *
* furnished to do so, subject to the following conditions:                             *
*                                                                                      *
* The above copyright notice and this permission notice shall be included in all       *
* copies or substantial portions of the Software.                                      *
*                                                                                      *
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR           *
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,             *
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE          *
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER               *
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,        *
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE        *
* SOFTWARE.                                                                            *
*                                                                                      *
\**************************************************************************************/

#ifndef PARALLEL_ALGORITHMS_HPP
#define PARALLEL_ALGORITHMS_HPP

#include "thread_pool.hpp"
#include <cassert>

/*************************** example usage ***************************************\
grep "^//test.cpp" parallel_algoritms.hpp -A34 > test.cpp
g++ -std=c++11 -o test test.cpp -pthread -lpthread

//test.cpp:
#include "thread_pool.hpp"
#include <iostream>

int main()
{
	// use main thread also as worker using wait_work(), so init 1 less in thread pool
	// (alternatively use wait_sleep() and make threads for all logical hardware cores)
    thread_pool::thread_pool tp(std::thread::hardware_concurrency() - 1);

}

\************************* end example usage *************************************/

namespace parallel_algorithms {

#ifndef PA_PARTITION_CHUNKSIZE
#define PA_PARTITION_CHUNKSIZE   2048
#endif
#ifndef PA_NTH_ELEMENT_CHUNKSIZE
#define PA_NTH_ELEMENT_CHUNKSIZE 2048
#endif
#ifndef PA_SORT_CHUNKSIZE
#define PA_SORT_CHUNKSIZE        2048
#endif

	/*
		// basic iterator: an std::size_t integer that behaves as an iterator
		class range_iterator {
		public:
			range_iterator(std::size_t i = 0);
			// comparison operators: == != < <= > >=
			// reference operators : * -> []
			// add / sub           : ++ ++(int) -- --(int) + - += -=
		};
	*/
	/*
		// represents the i-th subrange of a range divides into n equal size range
		class subrange {
		public:
			subrange(std::size_t first, std::size_t last, std::size_t i, std::size_t n); // for range [first,last)
			subrange(std::size_t size = 0, std::size_t i = 0, std::size_t n = 1);        // for range [0, size)
			range_iterator begin(); // iterator to begin of subrange
			range_iterator end();   // iterator to end of subrange
			std::size_t first();    // index of begin of subrange
			std::size_t last();     // index of end of subrange
		};
	*/
	/*
		// behaves as std::partition(first, last, Pred) in a parallel implementation using a given threadpool
		template<typename RandIt, typename Pred, typename Threadpool>
		RandIt partition(RandIt first, RandIt last, Pred pred, Threadpool& threadpool, const std::size_t chunksize = PA_PARTITION_CHUNKSIZE);
	*/
	/*
		// behaves as std::nth_element(first, nth, last, cf) in a parallel implementation using a given threadpool
		template<typename RandIt, typename Compare, typename Threadpool>
		void nth_element(RandIt first, RandIt nth, RandIt last, Compare cf, Threadpool& threadpool, std::size_t chunksize = PA_NTH_ELEMENT_CHUNKSIZE);

		// behaves as std::nth_element(first, nth, last) in a parallel implementation using a given threadpool
		template<typename RandIt, typename Threadpool>
		void nth_element(RandIt first, RandIt nth, RandIt last, Threadpool& threadpool, std::size_t chunksize = PA_NTH_ELEMENT_CHUNKSIZE);
	*/
	/*
		// behaves as std::merge(first1, last1, first2, last2, dest, cf) in a parallel implementation using a given threadpool
		template<typename RandIt1, typename RandIt2, typename OutputIt, typename Compare, typename Threadpool>
		OutputIt merge(RandIt1 first1, RandIt1 last1, RandIt2 first2, RandIt2 last2, OutputIt dest, Compare cf, Threadpool& threadpool);

		// behaves as std::merge(first1, last1, first2, last2, dest) in a parallel implementation using a given threadpool
		template<typename RandIt1, typename RandIt2, typename OutputIt, typename Threadpool>
		OutputIt merge(RandIt1 first1, RandIt1 last1, RandIt2 first2, RandIt2 last2, OutputIt dest, Threadpool& threadpool);
	*/
	/*
		// behaves as std::sort(first, last, cf) in a parallel implementation using a given threadpool
		template<typename RandIt, typename Compare, typename Threadpool>
		void sort(RandIt first, RandIt last, Compare cf, Threadpool& threadpool, const std::size_t chunksize = PA_SORT_CHUNKSIZE);

		// behaves as std::sort(first, last) in a parallel implementation using a given threadpool
		template<typename RandIt, typename Threadpool>
		void sort(RandIt first, RandIt last, Threadpool& threadpool, const std::size_t chunksize = PA_SORT_CHUNKSIZE);
	*/
	/*
		// behaves as std::copy(first, last, dest) in a parallel implementation using a given threadpool
		template<typename RandIt, typename OutputIt, typename Threadpool>
		OutputIt copy(RandIt first, RandIt last, OutputIt dest, Threadpool& threadpool);
		// behave as std::move(first, last, dest) in a parallel implementation using a given threadpool
		template<typename RandIt, typename OutputIt, typename Threadpool>
		OutputIt move(RandIt first, RandIt last, OutputIt dest, Threadpool& threadpool);
	*/


	// basic iterator: an std::size_t integer that behaves as an iterator
	class range_iterator
		: public std::iterator<std::random_access_iterator_tag, const std::size_t>
	{
		std::size_t _i;

	public:
		range_iterator(std::size_t i = 0): _i(i) {}

		bool operator== (const range_iterator& r) const { return _i == r._i; }
		bool operator!= (const range_iterator& r) const { return _i != r._i; }
		bool operator<  (const range_iterator& r) const { return _i <  r._i; }
		bool operator<= (const range_iterator& r) const { return _i <= r._i; }
		bool operator>  (const range_iterator& r) const { return _i >  r._i; }
		bool operator>= (const range_iterator& r) const { return _i >= r._i; }
		
		std::size_t operator*() const { return _i; }
		const std::size_t* operator->() const { return &_i; }
		std::size_t operator[](std::size_t n) const { return _i + n; }
		
		range_iterator& operator++() { ++_i; return *this; }
		range_iterator& operator--() { --_i; return *this; }
		range_iterator& operator+=(std::size_t n) { _i += n; return *this; }
		range_iterator& operator-=(std::size_t n) { _i -= n; return *this; }

		range_iterator operator++(int) { range_iterator copy(_i); ++_i; return copy; }
		range_iterator operator--(int) { range_iterator copy(_i); --_i; return copy; }
		
		range_iterator operator+(std::size_t n) const { return range_iterator(_i + n); }
		range_iterator operator-(std::size_t n) const { return range_iterator(_i - n); }
		std::size_t operator-(const range_iterator& r) const { return _i - r._i; }
	};
	
	// represents the i-th subrange of a range divides into n equal size range
	class subrange {
	public:
		subrange(std::size_t first, std::size_t last, std::size_t i, std::size_t n)
		{
			assert(i < n);
			assert(first <= last);
			std::size_t dist = last-first;
			std::size_t div = dist/n, rem = dist%n;
			_begin = range_iterator(first + i*div + std::min(i,rem));
			_end   = range_iterator(first + (i+1)*div + std::min(i+1,rem));
		}
		subrange(std::size_t size = 0, std::size_t i = 0, std::size_t n = 1)
			: subrange(0, size, i ,n)
		{
		}
		range_iterator begin() const { return _begin; }
		range_iterator end() const { return _end; }
		std::size_t first() const { return *_begin; }
		std::size_t last() const { return *_end; }
	private:
		range_iterator _begin, _end;
	};


	// behaves as std::partition(first, last, Pred) in a parallel implementation using a given threadpool
	template<typename RandIt, typename Pred, typename Threadpool>
	RandIt partition(RandIt first, RandIt last, Pred pred, Threadpool& threadpool, const std::size_t chunksize = PA_PARTITION_CHUNKSIZE)
	{
		const std::size_t dist = last-first;

		int nr_threads = std::min<int>(threadpool.size()+1, dist/(chunksize*2) );
		if (nr_threads <= 2)
			return std::partition(first, last, pred);

		typedef std::pair<std::size_t,std::size_t> size_pair;
		std::vector< size_pair > low_false_interval(nr_threads, size_pair(dist,dist));
		std::vector< size_pair > high_true_interval(nr_threads, size_pair(dist,dist));

		// we know there is enough room to give two chunks per thread (one at the begin, one at the end) at the start
		// low and high point to the *beginning* of the next available chunk, so low<=high
		std::atomic_size_t low(nr_threads*chunksize), high(dist - ((nr_threads+1)*chunksize));
		std::atomic_size_t usedchunks(2*nr_threads);
		const std::size_t availablechunks=dist/chunksize;

		// each thread processes on a 'low' and a 'high' chunk, obtaining a new chunk whenever one is fully processed.
		threadpool.run([&,first,last,dist,pred,chunksize,availablechunks](int thi, int thn)
			{
				if (thn != nr_threads)
					throw std::runtime_error("thn != nr_threads");
				std::size_t mylow = thi*chunksize, myhigh = dist-(thi+1)*chunksize;
				auto lowfirst=first+mylow, lowlast=lowfirst+chunksize, lowit=lowfirst;
				auto highfirst=first+myhigh, highlast=highfirst+chunksize, highit=highfirst;

				while (true)
				{
					for (; lowit != lowlast; ++lowit)
					{
						if (true == pred(*lowit))
							continue;
						for (; highit != highlast && false == pred(*highit); ++highit)
							;
						if (highit == highlast)
							break;
						std::iter_swap(lowit,highit);
						++highit;
					}
					if (lowit == lowlast)
					{
						if (usedchunks.fetch_add(1) < availablechunks)
						{
							mylow = low.fetch_add(chunksize);
							lowit = lowfirst = first+mylow; lowlast = lowfirst+chunksize;
						} else
							break;
					}
					if (highit == highlast)
					{
						if (usedchunks.fetch_add(1) < availablechunks)
						{
							myhigh = high.fetch_sub(chunksize);
							highit = highfirst = first+myhigh; highlast = highfirst+chunksize;
						} else
							break;
					}
				}

				if (lowit != lowlast)
				{
					auto lm = std::partition(lowit, lowlast, pred);
					low_false_interval[thi] = size_pair(lm-first, lowlast-first);
				} else
					low_false_interval[thi] = size_pair(0,0);

				if (highit != highlast)
				{
					auto hm = std::partition(highit, highlast, pred);
					high_true_interval[thi] = size_pair(highit-first, hm-first);
				} else
					high_true_interval[thi] = size_pair(0,0);
			}, nr_threads);

		assert(low <= high+chunksize);

		std::sort( low_false_interval.begin(), low_false_interval.end(), [](size_pair l,size_pair r){return l.first < r.first;});
		std::sort( high_true_interval.begin(), high_true_interval.end(), [](size_pair l,size_pair r){return l.first < r.first;});

		// current status:
		// on range [0,mid)  : pred(*x)=true unless x in some low_todo_interval
		// on range [mid,end): pred(*x)=false unless x in some high_todo_interval
		std::size_t mid = std::partition(first+low, first+high+chunksize, pred) - first;

		// compute the final middle
		std::size_t realmid = mid;
		for (auto& be : low_false_interval)
			realmid -= be.second - be.first;
		for (auto& be : high_true_interval)
			realmid += be.second - be.first;

		// compute the remaining intervals to swap
		std::vector< size_pair > toswap_false, toswap_true;
		std::sort( low_false_interval.begin(), low_false_interval.end() );
		std::sort( high_true_interval.begin(), high_true_interval.end() );
		
		std::size_t lowdone = 0;
		for (auto& be : low_false_interval)
		{
			assert(be.first <= be.second);
			if (be.first == be.second)
				continue;
			// [lowdone, be.first): pred=true
			// [be.first, be.second): pred=false
			// case1: we have to swap [lowdone, be.first) intersect [realmid,be.first)
			if (realmid < be.first && lowdone < be.first)
				toswap_true.emplace_back(std::max(lowdone,realmid),be.first);
			// case2: we have to swap [be.first, be.second) intersect [be.first, realmid)
			if (be.first < realmid)
				toswap_false.emplace_back(be.first,std::min(be.second,realmid));
			lowdone = be.second;
		}
		// [lowdone,mid): pred=true
		// case3: we have to swap [lowdone,mid) intersect [realmid,mid)
		if (realmid < mid && lowdone < mid)
			toswap_true.emplace_back(std::max(lowdone,realmid),mid);

		std::size_t highdone = mid;
		for (auto& be : high_true_interval)
		{
			assert(be.first <= be.second);
			if (be.first == be.second)
				continue;
			// [highdone,be.first): pred=false
			// [be.first, be.second): pred=true
			// case4: we have to swap [highdone,be.first) intersect [highdone,realmid)
			if (highdone < realmid && highdone < be.first)
				toswap_false.emplace_back(highdone, std::min(be.first, realmid));
			// case5: we have to swap [be.first, be.second) intersect [realmid, be.second)
			if (realmid < be.second)
				toswap_true.emplace_back(std::max(be.first,realmid), be.second);
			highdone = be.second;
		}
		// [highdone,last): pred=false
		if (highdone < realmid)
			toswap_false.emplace_back(highdone, realmid);

		// swap the remaining intervals
		while (!toswap_false.empty() && !toswap_true.empty())
		{
			auto& swf = toswap_false.back();
			auto& swt = toswap_true.back();
			assert(swf.first <= swf.second);
			assert(swt.first <= swt.second);
			std::size_t count = std::min(swf.second-swf.first, swt.second-swt.first);
			std::swap_ranges(first+swf.first, first+(swf.first+count), first+swt.first);
			swf.first += count;
			swt.first += count;
			if (swf.first == swf.second)
				toswap_false.pop_back();
			if (swt.first == swt.second)
				toswap_true.pop_back();
		}
		assert(toswap_false.empty() && toswap_true.empty());
		return first+realmid;
	}

	// behaves as std::nth_element(first, nth, last, cf) in a parallel implementation using a given threadpool
	template<typename RandIt, typename Compare, typename Threadpool>
	void nth_element(RandIt first, RandIt nth, RandIt last, Compare cf, Threadpool& threadpool, std::size_t chunksize = PA_NTH_ELEMENT_CHUNKSIZE)
	{
		typedef typename std::iterator_traits<RandIt>::difference_type difference_type;
		typedef typename std::iterator_traits<RandIt>::value_type value_type;
		while (true)
		{
			assert(first <= nth && nth < last);
			
			difference_type dist = last - first;
			if (dist <= chunksize*4)
			{
				std::nth_element(first, nth, last, cf);
				return;
			}

			// select a small constant number of elements
			const std::size_t selectionsize = 7;
			assert(selectionsize <= dist);
			RandIt selit = first;
			for (std::size_t i = 0; i < selectionsize; ++i,++selit)
				std::iter_swap(selit, first + (rand()%dist));
			std::sort(first, selit, cf);
			
			// pick median as pivot and move to end
			RandIt pivot = last-1;
			std::iter_swap(first+selectionsize/2, pivot);
			auto mid = partition(first, pivot, [cf,pivot](const value_type& r){ return cf(r, *pivot); }, threadpool, chunksize);
			
			if (nth < mid)
				last = mid;
			else
				first = mid;
		}
	}

	// behaves as std::nth_element(first, nth, last) in a parallel implementation using a given threadpool
	template<typename RandIt, typename Threadpool>
	void nth_element(RandIt first, RandIt nth, RandIt last, Threadpool& threadpool, std::size_t chunksize = PA_NTH_ELEMENT_CHUNKSIZE)
	{
		typedef typename std::iterator_traits<RandIt>::value_type value_type;
		nth_element(first, nth, last, std::less<value_type>(), threadpool, chunksize);
	}


	// behaves as std::merge(first1, last1, first2, last2, dest, cf) in a parallel implementation using a given threadpool
	template<typename RandIt1, typename RandIt2, typename OutputIt, typename Compare, typename Threadpool>
	OutputIt merge(RandIt1 first1, RandIt1 last1, RandIt2 first2, RandIt2 last2, OutputIt dest, Compare cf, Threadpool& threadpool)
	{
		typedef typename std::iterator_traits<OutputIt>::difference_type difference_type;
		const std::size_t minchunksize = 4096;

		difference_type size1 = last1-first1, size2=last2-first2;
		if (size1+size2 < difference_type(2*minchunksize))
			return std::merge(first1, last1, first2, last2, dest, cf);
		if (size1 < size2)
			return merge(first2, last2, first1, last1, dest, cf, threadpool);

		const std::size_t threads = std::min(threadpool.size()+1, size1/minchunksize);

		threadpool.run([=](int thi, int thn)
			{
				subrange iv1(size1, thi, thn);
				RandIt1 iv1first=first1 + iv1.first(), iv1last=first1 + iv1.last();
				RandIt2 iv2first=first2, iv2last=last2;
				if (thi>0)
					iv2first=std::lower_bound(first2, last2, *iv1first, cf);
				OutputIt d = dest + (iv1.first() + (iv2first-first2));
				if (iv2first == iv2last)
				{
					std::move(iv1first, iv1last, d);
					return;
				}
				while (true)
				{
					if (cf(*iv2first, *iv1first))
					{
						*d = std::move(*iv2first); ++d; ++iv2first;
						if (iv2first == iv2last)
						{
							std::move(iv1first, iv1last, d);
							return;
						}
					} else
					{
						*d = std::move(*iv1first); ++d; ++iv1first;
						if (iv1first == iv1last)
						{
							if (thi+1 < thn)
							{
								for (; iv2first != iv2last && cf(*iv2first, *iv1last); ++iv2first,++d)
									*d = std::move(*iv2first);
							} else
								std::move(iv2first, iv2last, d);
							return;
						}
					}
				}
			}, threads);
		return dest+(size1+size2);
	}

	// behaves as std::merge(first1, last1, first2, last2, dest) in a parallel implementation using a given threadpool
	template<typename RandIt1, typename RandIt2, typename OutputIt, typename Threadpool>
	OutputIt merge(RandIt1 first1, RandIt1 last1, RandIt2 first2, RandIt2 last2, OutputIt dest, Threadpool& threadpool)
	{
		typedef typename std::iterator_traits<OutputIt>::value_type value_type;
		return merge(first1, last1, first2, last2, dest, std::less<value_type>(), threadpool);
	}


	// behaves as std::sort(first, last, cf) in a parallel implementation using a given threadpool
	template<typename RandIt, typename Compare, typename Threadpool>
	void sort2(RandIt first, RandIt last, Compare cf, Threadpool& threadpool, const std::size_t chunksize = PA_SORT_CHUNKSIZE)
	{
		typedef thread_pool::barrier barrier;

		const std::size_t dist = last-first;
		std::size_t nr_threads = std::min(threadpool.size()+1, dist/chunksize);
		if (nr_threads <= 1)
		{
			std::sort(first, last, cf);
			return;
		}

		std::vector<subrange> thrange(nr_threads);
		for (size_t i = 0; i < nr_threads; ++i)
			thrange[i] = subrange(dist, i, nr_threads);

		int windowsize = 1;
		while (windowsize < nr_threads)
			windowsize *= 2;

		while (windowsize > 1
			&& (((dist/nr_threads)*windowsize)/(3*chunksize)) > (nr_threads/windowsize))
		{
			for (int th = 0; th < nr_threads; th += windowsize)
			{
				int fth = th;
				int mth = std::min<int>(nr_threads-1, th+windowsize/2);
				int lth = std::min<int>(nr_threads-1, th+windowsize-1);
				nth_element(first + thrange[th].first(), first + thrange[mth].first(), first + thrange[lth].last(), cf, threadpool, chunksize);
			}
			windowsize /= 2;
		}

		barrier barriers(nr_threads);
		threadpool.run(
			[=,&barriers,&thrange](int thi, int thn)
			{
				for (int w = windowsize; w > 1; w /= 2)
				{
					if ((thi % w) == 0)
					{
						int fth = thi;
						int mth = std::min<int>(thn-1, thi+w/2);
						int lth = std::min<int>(thn-1, thi+w-1);
						std::nth_element(first + thrange[fth].first(), first + thrange[mth].first(), first+thrange[lth].last(), cf);
					}
					barriers.wait();
				}
				std::sort(first+thrange[thi].first(), first+thrange[thi].last(), cf);
			}, nr_threads);
	}

	// behaves as std::sort(first, last) in a parallel implementation using a given threadpool
	template<typename RandIt, typename Threadpool>
	void sort2(RandIt first, RandIt last, Threadpool& threadpool, const std::size_t chunksize = PA_SORT_CHUNKSIZE)
	{
		typedef typename std::iterator_traits<RandIt>::value_type value_type;
		sort2(first, last, std::less<value_type>(), threadpool, chunksize);
	}

	// behaves as std::sort(first, last, cf) in a parallel implementation using a given threadpool
	template<typename RandIt, typename Compare, typename Threadpool>
	void sort3(RandIt first, RandIt last, Compare cf, Threadpool& threadpool, const std::size_t chunksize = PA_SORT_CHUNKSIZE)
	{
		typedef typename std::iterator_traits<RandIt>::value_type value_type;

		const std::size_t dist = last-first;
		std::size_t nr_threads = std::min(threadpool.size()+1, dist/chunksize);
		if (nr_threads <= 2)
		{
			std::sort(first, last, cf);
			return;
		}

		typedef std::pair<std::size_t,std::size_t> size_pair;
		std::vector<size_pair> ranges;

		ranges.emplace_back(0, dist);

		while (ranges.size() < nr_threads)
		{
			size_pair largest = ranges.back();

			RandIt rf = first + largest.first;
			RandIt rl = rf + largest.second;

			// select a small constant number of elements
			const std::size_t selectionsize = 7;
			RandIt selit = rf;
			for (std::size_t i = 0; i < selectionsize; ++i,++selit)
				std::iter_swap(selit, rf + (rand() % largest.second));
			std::sort(rf, selit, cf);

			// pick median as pivot and move to end
			RandIt pivot = rl-1;
			std::iter_swap(rf + selectionsize/2, pivot);
			auto mid = partition(rf, pivot, [cf,pivot](const value_type& r){ return cf(r, *pivot); }, threadpool, chunksize);

			// update ranges
			ranges.pop_back();
			size_pair subrange1(largest.first, mid-rf);
			size_pair subrange2(subrange1.first+subrange1.second, largest.second-subrange1.second);
			ranges.insert(
				std::upper_bound(ranges.begin(),ranges.end(),subrange1,[](const size_pair& l, const size_pair& r){return l.second < r.second;})
				, subrange1);
			ranges.insert(
				std::upper_bound(ranges.begin(),ranges.end(),subrange2,[](const size_pair& l, const size_pair& r){return l.second < r.second;})
				, subrange2);
		}

		threadpool.run(
			[=](int thi)
			{
				std::sort(first + ranges[thi].first, first + (ranges[thi].first+ranges[thi].second), cf);
/* // attempt to more evenly divide the sorting space
				if (ranges.size() == 2*nr_threads)
					std::sort(first + ranges[2*thn-thi-1].first, first + (ranges[2*thn-thi-1].first+ranges[2*thn-thi-1].second), cf);
*/
			}
			, nr_threads);
	}

	// behaves as std::sort(first, last) in a parallel implementation using a given threadpool
	template<typename RandIt, typename Threadpool>
	void sort3(RandIt first, RandIt last, Threadpool& threadpool, const std::size_t chunksize = PA_SORT_CHUNKSIZE)
	{
		typedef typename std::iterator_traits<RandIt>::value_type value_type;
		sort3(first, last, std::less<value_type>(), threadpool, chunksize);
	}

	// behaves as std::sort(first, last, cf) in a parallel implementation using a given threadpool
	template<typename RandIt, typename Compare, typename Threadpool>
	void sort(RandIt first, RandIt last, Compare cf, Threadpool& threadpool, const std::size_t chunksize = PA_SORT_CHUNKSIZE)
	{
		sort3(first, last, cf, threadpool, chunksize);
	}

	// behaves as std::sort(first, last) in a parallel implementation using a given threadpool
	template<typename RandIt, typename Threadpool>
	void sort(RandIt first, RandIt last, Threadpool& threadpool, const std::size_t chunksize = PA_SORT_CHUNKSIZE)
	{
		typedef typename std::iterator_traits<RandIt>::value_type value_type;
		sort3(first, last, std::less<value_type>(), threadpool, chunksize);
	}

	// behaves as std::copy(first, last, dest) in a parallel implementation using a given threadpool
	template<typename RandIt, typename OutputIt, typename Threadpool>
	OutputIt copy(RandIt first, RandIt last, OutputIt dest, Threadpool& threadpool)
	{
		const std::size_t dist = last-first;
		if (dist < 8192)
			return std::copy(first, last, dest);

		int nr_threads = std::min<int>(threadpool.size()+1, dist/2048);
		threadpool.run( [=](int thi, int thn)
			{
				subrange sr(dist, thi, thn);
				std::copy(first+sr.first(), first+sr.last(), dest+sr.first());
			}, nr_threads);
		return dest + dist;
	}

	// behave as std::move(first, last, dest) in a parallel implementation using a given threadpool
	template<typename RandIt, typename OutputIt, typename Threadpool>
	OutputIt move(RandIt first, RandIt last, OutputIt dest, Threadpool& threadpool)
	{
		const std::size_t dist = last-first;
		if (dist < 8192)
			return std::move(first, last, dest);

		int nr_threads = std::min<int>(threadpool.size()+1, dist/2048);
		threadpool.run( [=](int thi, int thn)
			{
				subrange sr(dist, thi, thn);
				std::move(first+sr.first(), first+sr.last(), dest+sr.first());
			}, nr_threads);
		return dest + dist;
	}

} // namespace parallel_algorithms

#endif // PARALLEL_ALGORITHMS
