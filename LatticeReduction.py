from fpylll import BKZ as BKZ_FPYLLL, GSO, IntegerMatrix, FPLLL, config
from fpylll.algorithms.bkz2 import BKZReduction
FPLLL.set_precision(240)

try:
  from g6k import Siever, SieverParams
  from g6k.algorithms.bkz import pump_n_jump_bkz_tour
  from g6k.utils.stats import dummy_tracer
except ImportError:
  raise ImportError("g6k not installed")

BKZ_SIEVING_CROSSOVER = 55

class LatticeReduction:

  def __init__(
    self,
    basis, #lattice basis to be reduced
    threads_bkz = 1
  ):

    B = IntegerMatrix.from_matrix(basis, int_type="long")

    if B.nrows <= 160:
      float_type = "long double"
    elif B.nrows <= 450:
      float_type = "dd" if config.have_qd else "mpfr"
    else:
      float_type = "mpfr"

    M = GSO.Mat(B, float_type=float_type,
      U=IntegerMatrix.identity(B.nrows, int_type=B.int_type),
      UinvT=IntegerMatrix.identity(B.nrows, int_type=B.int_type))

    M.update_gso()

    self.__bkz = BKZReduction(M)

    params_sieve = SieverParams()
    params_sieve['threads'] = threads_bkz

    self.__g6k = Siever(M, params_sieve)

    self.basis = M.B
    self.gso = M

  def BKZ(self, beta, tours=2): #tours=8

    par = BKZ_FPYLLL.Param(
      beta,
      strategies=BKZ_FPYLLL.DEFAULT_STRATEGY,
      max_loops=tours,
      flags=BKZ_FPYLLL.MAX_LOOPS
    )

    if beta <=  BKZ_SIEVING_CROSSOVER: #65:
      self.__bkz(par) #bkz-enum is faster this way
    else:
        for t in range(tours): #pnj-bkz is oblivious to ntours
            pump_n_jump_bkz_tour(self.__g6k, dummy_tracer, beta)
