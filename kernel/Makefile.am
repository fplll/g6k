lib_LTLIBRARIES=libg6k.la

libg6k_la_LDFLAGS=-no-undefined -version-info @G6K_LT_CURRENT@:@G6K_LT_REVISION@:@G6K_LT_AGE@ $(PTHREAD_CFLAGS)
libg6k_la_LIBADD=$(LIBQD_LIBS)
# We need this all to be on one line, setup.py reads this. Yep, it's a HACK
libg6k_la_SOURCES=sieving.cpp control.cpp bgj1_sieve.cpp bdgl_sieve.cpp hk3_sieve.cpp fht_lsh.cpp params.cpp cpuperf.cpp slicer.cpp
include_HEADERS=compat.hpp cpuperformance.hpp db.inl fht_lsh.h hash_table.inl parallel_algorithms.hpp random.hpp siever.h siever.inl simhash.inl statistics.hpp thread_pool.hpp untemplate.mac slicer.h
#libg6k_la_SOURCES = slicer.cpp
#include_HEADERS = slicer.h