from full_sieve import full_sieve_kernel
from g6k import SieverParams


def test_full_sieve():
    full_sieve_kernel(50, SieverParams(), 1)
