#!/bin/bash

enable_cpucounters=0
enable_stats=0
enable_ndebug=0
maxsievingdim=128
enable_ggdb=0
enable_jobs=4
enable_templated_dim=0

while [[ $# -gt 0 ]]; do
	case "$1" in
		-m|--maxsievingdim)
			maxsievingdim=$2
			shift
			;;
		-j|--jobs)
			jobs=$2
			shift
			;;
		-g|--ggdb)
			enable_ggdb=1
			;;
		-c|--cpucounters)
			enable_cpucounters=1
			;;
		-s|--stats)
			enable_stats=1
			;;
                -ss|--extended_stats)
                        enable_stats=2
                        ;;
		-t|--templated_dim)
			enable_templated_dim=1
			;;
		-f|--fast)
			enable_ndebug=1
			enable_cpucounters=0
			enable_stats=0
			enable_templated_dim=1
			;;
		--build-threshold)
			build_threshold=$2
			shift
			;;
		--sieve-threshold)
			sieve_threshold=$2
			shift
			;;
		*)
			;;
	esac
	shift
done

EXTRAFLAGS=""
if [ ${enable_ggdb} -eq 1 ]; then
	EXTRAFLAGS="$EXTRAFLAGS -g -ggdb"
fi
if [ ${enable_cpucounters} -eq 1 ]; then
	EXTRAFLAGS="$EXTRAFLAGS -DPERFORMANCE_COUNTING"
fi
if [ ${enable_stats} -ge 1 ]; then
	EXTRAFLAGS="$EXTRAFLAGS -DENABLE_STATS"
fi
if [ ${enable_stats} -ge 2 ]; then
	EXTRAFLAGS="$EXTRAFLAGS -DENABLE_EXTENDED_STATS"
fi 
if [ ${enable_ndebug} -eq 1 ]; then
	EXTRAFLAGS="$EXTRAFLAGS -DNDEBUG"
fi
if [ ${enable_templated_dim} -eq 1 ]; then
	EXTRAFLAGS="$EXTRAFLAGS -DTEMPLATED_DIM"
fi
if [ ${maxsievingdim} -gt 0 ]; then
	EXTRAFLAGS="$EXTRAFLAGS -DMAX_SIEVING_DIM=${maxsievingdim}"
fi
if [ "$build_threshold" != "" ]; then
	EXTRAFLAGS="$EXTRAFLAGS -DXPC_BUCKET_THRESHOLD=${build_threshold}"
fi
if [ "$sieve_threshold" != "" ]; then
	EXTRAFLAGS="$EXTRAFLAGS -DXPC_THRESHOLD=${sieve_threshold}"
fi
export EXTRAFLAGS

make -C kernel clean || exit 1
make -C kernel -j ${jobs}

rm g6k/siever.so `find g6k -name "*.pyc"`
python setup.py clean
python setup.py build_ext -j ${jobs} --inplace || exit 1
