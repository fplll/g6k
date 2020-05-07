from fpylll import IntegerMatrix
from g6k import Siever
from simple_pump import pump

n = 60

A = IntegerMatrix.random(n, "qary", k=25, bits=20)
g6k = Siever(A)

# Do a Workout (pumps of increasing effort), and printout basis shape at each step.
for dim4free in range(30, 10, -1):
	pump(g6k, 0, n, dim4free)
	print([g6k.M.get_r(i, i) for i in range(5)])