# -*- coding: utf-8 -*-
####
#
#   Copyright (C) 2018-2021 Team G6K
#
#   This file is part of G6K. G6K is free software:
#   you can redistribute it and/or modify it under the terms of the
#   GNU General Public License as published by the Free Software Foundation,
#   either version 2 of the License, or (at your option) any later version.
#
#   G6K is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with G6K. If not, see <http://www.gnu.org/licenses/>.
#
####


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
