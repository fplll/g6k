from fpylll import *
from math import log
from copy import copy

load("../framework/proba_utils.sage")
load("../framework/utils.sage")
load("../framework/geometry.sage")
load("../framework/DBDD_generic.sage")


def cannonical_direction_only(fn):
    def decorated(self, *args, **kwargs):
        if not is_cannonical_direction(args[0]):
            raise InvalidHint(
                "Input hint vector must have a cannonical direction")
        return fn(self, *args, **kwargs)
    return decorated


class DBDD_predict_diag(DBDD_generic):
    """
    This class defines all the elements defining a DBDD instance in a diagonal
    prediction mode
    """

    def __init__(self, B, S, mu, u=None, homogeneous=False, verbosity=1, D=None, Bvol=None):
        """constructor that builds a DBDD instance from a lattice, mean, sigma
        and a target
        ;min_dim: Number of coordinates to find to consider the problem solved
        :B: Basis of the lattice
        :S: The Covariance matrix (Sigma) of the uSVP solution
        :mu: The expectation of the uSVP solution
        :u: The unique vector to be found (optinal, for verification purposes)
        :fp_type: Floating point type to use in FPLLL ("d, ld, dd, qd, mpfrX")
        """
        self.homogeneous=homogeneous
        self.verbosity = verbosity
        if not is_diagonal(S):
            raise ValueError("given Σ not diagonal")

        self.S = np.array([RR(S[i, i]) for i in range(S.nrows())])
        self.PP = 0 * self.S  # Span of the projections so far (orthonormal)
        # Orthogonal span of the intersection so far so far (orthonormal)
        self.PI = 0 * self.S
        self.PI[-1] = 1
        self.u = u
        self.u_original = u
        self._dim = S.nrows()
        self.projections = 0
        self.Bvol = Bvol or logdet(B)
        assert check_basis_consistency(B, D, Bvol)
        self.save = {"save": None}
        self.can_constraints = S.nrows() * [1]
        self.can_constraints[-1] = None
        self.estimate_attack(silent=True)

    def dim(self):
        return self._dim

    def S_diag(self):
        return copy(self.S)

    def volumes(self):
        Bvol = self.Bvol
        Svol = sum([log(abs(x)) for x in (self.S + self.PP + self.PI)])
        dvol = Bvol - Svol / 2.
        return (Bvol, Svol, dvol)


    @cannonical_direction_only
    @hint_integration_wrapper(force=True)
    def integrate_perfect_hint(self, V, l):
        self.homogeneize(V, l)
        i, vi = cannonical_param(V)
        self.can_constraints[i] = None
        if self.PI[i]:
            raise RejectedHint("Redundant hint, Rejected.")
        if self.PP[i]:
            raise InvalidHint("This direction has been projected out.")

        assert(vi)
        self.PI[i] = 1
        self.S[i] = 0
        self.Bvol += log(vi)
        self._dim -= 1

    @cannonical_direction_only
    @hint_integration_wrapper(force=True)
    def integrate_modular_hint(self, V, l, k, smooth=True):
        self.homogeneize(V, l)

        i, vi = cannonical_param(V)
        f = (k / vi).numerator()
        self.can_constraints[i] = lcm(f, self.can_constraints[i])
        # vs = vi * self.S[i]
        den = vi**2 * self.S[i]
        if den == 0:
            raise RejectedHint("Redundant hint, Rejected.")
        if self.PP[i]:
            raise InvalidHint("This direction has been projected out.")

        if not smooth:
            raise NotImplementedError()

        self.Bvol += log(k)

    @cannonical_direction_only
    @hint_integration_wrapper(force=True)
    def integrate_approx_hint(self, V, l, variance, aposteriori=False):
        self.homogeneize(V, l)

        if variance < 0:
            raise InvalidHint("variance must be non-negative !")
        if variance == 0:
            raise InvalidHint("variance=0 : must use perfect hint !")

        i, vi = cannonical_param(V)
        vs = vi * self.S[i]
        if self.PP[i]:
            raise InvalidHint("This direction has been projected out.")

        if not aposteriori:
            d = vs * vi
            self.S[i] -= (1 / (variance + d) * vs) * vs
        else:
            if not vs:
                raise RejectedHint("0-Eigenvector of Σ forbidden,")
            den = vi**2
            self.S[i] = variance / den

    @hint_integration_wrapper()
    def integrate_approx_hint_fulldim(self, center, covariance, aposteriori=False):
        # Using http://www.cs.columbia.edu/~liulp/pdf/linear_normal_dist.pdf
        # with A = Id
        if self.homogeneous:
            raise NotImplementedError()

        if not is_diagonal(covariance):
            raise ValueError("given covariance not diagonal")

        if not aposteriori:
            d = len(self.S) - 1

            for i in range(d):
                if covariance[i, i] == 0:
                    raise InvalidHint("Covariances not full dimensional")

            for i in range(d):
                self.S[i] -= self.S[i]**2 / (self.S[i] + covariance[i, i])

            # F = (self.S + block4(covariance, zero.T, zero, vec([1]))).inverse()
            # F[-1,-1] = 0
            # self.S -= self.S * F * self.S
        else:
            raise NotImplementedError()

    @cannonical_direction_only
    @hint_integration_wrapper(force=False)
    def integrate_short_vector_hint(self, V):
        i, vi = cannonical_param(V)

        vi -= vi * self.PP[i]
        den = vi**2
        if den == 0:
            raise InvalidHint("Redundant hint,")

        viPI = vi * self.PI[i]
        if viPI**2:
            raise InvalidHint("Not in Span(Λ),")

        if vi % self.can_constraints[i]:
            raise InvalidHint("Not in Λ,")

        self.projections += 1
        self.Bvol -= log(RR(den)) / 2.
        self._dim -= 1
        self.PP[i] = 1
        self.S[i] = 0

    def attack(self):
        self.logging("Can't run the attack in simulation.", style="WARNING")
        return None
