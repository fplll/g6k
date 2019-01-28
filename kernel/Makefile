CXXFLAGS += -fPIC -Ofast -march=native -ftree-vectorize -funroll-loops -std=c++11 -pthread -Wall -Wextra $(EXTRAFLAGS)
LDFLAGS += -shared -pthread
LIBADD += -lpthread
OBJ = sieving.o control.o bgj1_sieve.o triple_sieve.o params.o cpuperf.o triple_sieve_mt.o
HEADERS = siever.h siever.inl hash_table.inl db.inl simhash.inl

all: G6Klib.so

G6Klib.so: $(OBJ)
	$(CXX) $(LDFLAGS)  $(LIBADD) $^ -o $@

%.o: %.cpp siever.h siever.inl
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	-rm *.o *.so
