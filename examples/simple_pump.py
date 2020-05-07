

# A simple insertion policy: choose the leftmost candidate, no further than aux.
def scoring(i, nlen, olen, aux):
    return -i if i >= aux else 9999

# A simple implementation of the pump, with minimal amount of parameters.
# Significant loss of performances to be expected compared to the one from
# ../g6k/algorithms/pump.p

def pump(g6k, kappa, blocksize, dim4free):
    # The right end of the sieve context (constant through the pump)
    r = kappa+blocksize
    # The left end of the sieve context (at the top of the pump)
    l = kappa+dim4free

    # clear the database
    g6k.shrink_db(0)
    # make sure the current block is reasonably reduced
    g6k.lll(kappa, r)
    # Set up G6K, with an initial sieve context of dim 20, and a lift context up to kappa
    g6k.initialize_local(kappa, max(r-20, l), r)

    # Sieve
    g6k()

    # Pump up (Progressive Sieve)
    while g6k.l > l:
        # Extend the lift context to the left
        g6k.extend_left(1)
        # Sieve
        g6k()

    # pump down
    aux = kappa
    while g6k.l < r-20:
        # Insert the leftmost candidate if we have one
        ii = g6k.insert_best_lift(scoring, aux)
        # If no insertion happened
        if ii is None:
            # Force decrease of the sieve context
            g6k.shrink_left(1)
        else:
            # Forbid future insertion at before the inserted position 
            # (we don't want to break previously done work)
            aux = ii+1