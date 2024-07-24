from fpylll import BKZ as BKZ_FPYLLL, GSO, IntegerMatrix, FPLLL
from fpylll.algorithms.bkz2 import BKZReduction

try:
  from g6k import Siever, SieverParams
  from g6k.algorithms.bkz import pump_n_jump_bkz_tour
  from g6k.utils.stats import dummy_tracer
except ImportError:
  print("g6k is not installed!")
  pass
  # raise ImportError("g6k not installed")

class LatticeReduction:

  def __init__(
    self,
    basis, #lattice basis to be reduced
  ):

    B = IntegerMatrix.from_matrix(basis, int_type="long")

    if B.nrows <= 160:
      float_type = "long double"
    elif B.nrows <= 450:
      float_type = "dd"
    else:
      float_type = "mpfr"

    M = GSO.Mat(B, float_type=float_type,
      U=IntegerMatrix.identity(B.nrows, int_type=B.int_type),
      UinvT=IntegerMatrix.identity(B.nrows, int_type=B.int_type))

    M.update_gso()

    self.__bkz = BKZReduction(M)

    # params_sieve = SieverParams()
    # params_sieve['threads'] = 32
    #
    # self.__g6k = Siever(M, params_sieve)
    #
    self.basis = M.B
    self.gso = M

  def BKZ(self, beta, tours=5):

    par = BKZ_FPYLLL.Param(
      beta,
      strategies=BKZ_FPYLLL.DEFAULT_STRATEGY,
      max_loops=tours,
      flags=BKZ_FPYLLL.AUTO_ABORT|BKZ_FPYLLL.MAX_LOOPS|BKZ_FPYLLL.GH_BND
    )

    self.__bkz(par)

    # for t in range(tours):
    #   if beta <= 65:
    #     self.__bkz(par)
    #   else:
    #     pump_n_jump_bkz_tour(self.__g6k, dummy_tracer, beta, self._callback)
