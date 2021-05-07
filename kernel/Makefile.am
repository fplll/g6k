lib_LTLIBRARIES=libg6k.la

libg6k_la_LDFLAGS=-no-undefined -version-info @G6K_LT_CURRENT@:@G6K_LT_REVISION@:@G6K_LT_AGE@ $(PTHREAD_CFLAGS)
libg6k_la_LIBADD=$(LIBQD_LIBS)
# We need this all to be on one line, setup.py reads this. Yep, it's a HACK
libg6k_la_SOURCES=sieving.cpp control.cpp bgj1_sieve.cpp bdgl_sieve.cpp hk3_sieve.cpp fht_lsh.cpp params.cpp cpuperf.cpp
