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
