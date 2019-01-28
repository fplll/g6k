from svp_challenge import asvp_kernel
from g6k import SieverParams


def test_svp_challenge():
    asvp_kernel(50, SieverParams(load_matrix=None,
                                 challenge_seed=0,
                                 verbose=True), 1)
