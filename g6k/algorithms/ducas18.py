# -*- coding: utf-8 -*-
"""
A reimplementation of SubSieve+ from [Ducas18]
"""


def ducas18(g6k, tracer, goal):
    """
    :param g6k: The g6k object to work with
    :param tracer: A tracer for g6k
    :param goal: Targer length for the shortest vector
    """
    m = g6k.M.d
    d = m/4 + 1

    with tracer.context(("ducas18", "kappa:%d beta:%d f:%d" % (0, m, 0))):

        with g6k.temp_params(otf_lift=False, saturation_ratio=.5, sample_by_sums=False):
            while g6k.M.get_r(0, 0) > goal:
                d -= 1
                with tracer.context(("SubSieve+", "kappa:%d beta:%d f:%d" % (0, m, d))):
                    # Subsieve+ from [Ducas 18]
                    g6k.lll(0, m)
                    g6k.initialize_local(d, (m+d) / 2)
                    g6k.shrink_db(0)

                    while g6k.r < m:
                        with tracer.context(("progressieve-step", "l:%d r:%d n:%d" % (g6k.l, g6k.r, g6k.n))):
                            g6k.extend_right(1)
                            g6k(alg="gauss", tracer=tracer)

                    g6k.extend_left(d)

                    while g6k.l < m/2:
                        with tracer.context(("insertion-step", "l:%d r:%d n:%d" % (g6k.l, g6k.r, g6k.n))):
                            inserted = g6k.insert_best_lift(lambda i, nle, ole, aux: i == g6k.l)
                            if inserted is None:
                                g6k.shrink_left(1)

    return d
