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
Statistics for sieving.
"""

from __future__ import absolute_import
from __future__ import print_function
from fpylll.tools.bkz_stats import Tracer, Node, Accumulator, OrderedDict, pretty_dict
from fpylll.tools.bkz_stats import dummy_tracer # noqa
from fpylll.tools.quality import basis_quality
from g6k.siever import Siever
import time
try:
    from time import process_time  # Python 3
except ImportError:
    from time import clock as process_time  # Python 2
import logging


class SieveTreeTracer(Tracer):
    def __init__(self, instance, root_label="sieve", start_clocks=False, verbosity=False):
        """
        Create a new tracer instance.

        :param instance: sieve-like or BKZ-like object instance
        :param root_label: label to give to root node
        :param start_clocks: start tracking time for the root node immediately

        """
        Tracer.__init__(self, instance)
        self.trace = Node(root_label)
        self.current = self.trace
        self.verbosity = int(verbosity)
        if start_clocks:
            self.reenter()

    recognized_sieves = {"bgj1", "hk3", "gauss", "nv", "bdgl"}

    @classmethod
    def is_sieve_node(cls, label):
        return (isinstance(label, str) and label in cls.recognized_sieves) or \
            (isinstance(label, tuple) and label[0] in cls.recognized_sieves)

    def enter(self, label, **kwds):
        """Enter new context with label

        :param label: label

        """
        self.current = self.current.child(label)
        self.reenter()

    def reenter(self, **kwds):
        """Reenter current context, i.e. restart clocks

        """
        node = self.current
        node.data["cputime"]  = node.data.get("cputime",  0) + Accumulator(-process_time(), repr="sum", count=False)
        node.data["walltime"] = node.data.get("walltime", 0) + Accumulator(-time.time(),  repr="sum", count=False)

    def exit(self, **kwds):
        """
        By default CPU and wall time are recorded.  More information is recorded for sieve labels.
        """
        node = self.current

        node.data["cputime"] += process_time()
        node.data["walltime"] += time.time()

        self.instance.M.update_gso()

        if self.is_sieve_node(node.label):
            if isinstance(self.instance, Siever):
                instance = self.instance
            else:
                instance = self.instance.sieve

            node.data["|db|"] = Accumulator(len(instance), repr="max") + node.data.get("|db|", None)

            # determine the type of sieve:

            # idstring should be among SieveTreeTraces.recognized_sieves or "all".
            # This is used to look up what statistics to include in Siever.all_statistics

            if isinstance(node.label, str):
                idstring = node.label
            elif isinstance(node.label, tuple):
                idstring = node.label[0]
            else:
                idstring = "all"
                logging.warning("Unrecognized algorithm in Tracer")

            for key in Siever.all_statistics:
                # Siever.all_statistics[key][3] is a list of algorithms for which the statistic
                # indexed by key is meaningful instance.get_stat(key) will return None if support for
                # the statistics was not compiled in Siever.all_statistics[key][1] is a short string
                # that identifies the statistic
                if ((idstring == "all") or (idstring in Siever.all_statistics[key][3])) and (instance.get_stat(key) is not None):
                    if(len(Siever.all_statistics[key]) <= 4):
                        node.data[Siever.all_statistics[key][1]] = Accumulator(0, repr="sum")
                    else:
                        node.data[Siever.all_statistics[key][1]] = Accumulator(0, repr=Siever.all_statistics[key][4])
                    node.data[Siever.all_statistics[key][1]] += node.data.get(Siever.all_statistics[key][1], None)

            try:
                i, length, v = (instance.best_lifts())[0]
                if i == 0:
                    node.data["|v|"] = length
                else:
                    self.instance.update_gso(0, self.instance.full_n)
                    node.data["|v|"] = self.instance.M.get_r(0, 0)
            except (IndexError, AttributeError):
                node.data["|v|"] = None

        data = basis_quality(self.instance.M)
        for k, v in data.items():
            if k == "/":
                node.data[k] = Accumulator(v, repr="max")
            else:
                node.data[k] = Accumulator(v, repr="min")

        if kwds.get("dump_gso", node.level <= 1):
            node.data["r"] =  self.instance.M.r()

        verbose_labels = ["tour", "prog_tour"]

        if self.verbosity and node.label[0] in verbose_labels:
            report = OrderedDict()
            report["i"] = node.label[1]
            report["cputime"] = node["cputime"]
            report["walltime"] = node["walltime"]
            try:
                report["preproc"] = node.find("preprocessing", True)["cputime"]
            except KeyError:
                pass
            try:
                report["svp"] = node.find("sieve", True)["cputime"]
                # TODO: re-implement
                # report["sieve sat"] = node.find("sieve", True)["saturation"]
            except KeyError:
                pass

            report["r_0"] = node["r_0"]
            report["/"] = node["/"]

            print(pretty_dict(report))

        self.current = self.current.parent
        return self.trace
