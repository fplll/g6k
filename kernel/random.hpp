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


#ifndef SIEVER_RANDOM_HPP
#define SIEVER_RANDOM_HPP

#include <cstdint>
#include <random>
#include <mutex>

namespace rng {

	class threadsafe_rng {
	public:
		typedef typename std::mt19937_64 base_rng_t;
		typedef typename base_rng_t::result_type result_type;

		threadsafe_rng(result_type seed = 0)
			: _rng(seed)
		{
			if (seed != 0)
				return;
			std::random_device rd;
			_rng.seed(rd());
		}

		static constexpr result_type min() { return base_rng_t::min(); }
		static constexpr result_type max() { return base_rng_t::max(); }

		void seed(result_type seed = 0)
		{
			if (seed == 0)
			{
				std::random_device rd;
				seed = rd();
			}
			_rng.seed(seed);
		}

		result_type operator()()
		{
			std::lock_guard<std::mutex> lockguard(_mut);
			return _rng();
		}

		result_type rng()
		{
			std::lock_guard<std::mutex> lockguard(_mut);
			return _rng();
		}

		result_type rng_nolock()
		{
			return _rng();
		}

	private:
		std::mutex _mut;
		base_rng_t _rng;
	};

} // namespace rng

#endif // SIEVER_RANDOM_HPP
