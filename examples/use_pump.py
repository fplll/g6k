#!/usr/bin/env python

from fpylll import IntegerMatrix
try:
	from g6k import Siever
except ImportError:
	raise ImportError("g6k not installed. Please run './setup.py' from ..")
from simple_pump import pump


n = 80

A = IntegerMatrix.random(n, "qary", k=40, bits=20)
g6k = Siever(A)

print("Squared length of the 5 first GS vectors")
# Do a Workout (pumps of increasing effort), and printout basis shape at each step.
for dim4free in range(50, 15, -1):
    pump(g6k, 0, n, dim4free)
    print("dims4free: %d/80 \t %.3e\t%.3e\t%.3e\t%.3e\t%.3e\t"%tuple([dim4free]+[g6k.M.get_r(i, i) for i in range(5)]))

