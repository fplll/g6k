from fpylll import *
from math import log

load("../framework/DBDD_generic.sage")
load("../framework/proba_utils.sage")


class DBDD_predict(DBDD_generic):
    """
    This class defines all the elements defining a DBDD instance
    in a prediction mode
    """

    def __init__(self, B, S, mu, u=None, homogeneous=False, verbosity=1, D=None, Bvol=None):

        self.Bvol = Bvol or logdet(B)
        assert check_basis_consistency(B, D, Bvol)
        self.verbosity = verbosity
        self.S = S
        self._dim = S.nrows()
        self.PP = 0 * S  # Span of the projections so far (orthonormal)
        # Orthogonal span of the intersection so far so far (orthonormal)
        self.homogeneous = homogeneous

        if homogeneous and mu is not None:
            if scal(mu * mu.T) > 0:
                raise InvalidArgument("Homogeneous instances must have mu=0")

        self.PI = 0 * S
        if not homogeneous:
            self.PI[-1, -1] = 1

        self.u = u
        self.u_original = u
        self.projections = 0
        self.save = {"save": None}
        self.can_constraints = S.nrows() * [1]
        self.can_constraints[-1] = None
        self.estimate_attack(silent=True)

    def dim(self):
        return self._dim

    def S_diag(self):
        return [self.S[i, i] for i in range(self.S.nrows())]

    def volumes(self):
        Bvol = self.Bvol
        Svol = logdet(self.S + self.PP + self.PI)
        dvol = Bvol - Svol / 2.
        return (Bvol, Svol, dvol)


    @not_after_projections
    @hint_integration_wrapper(force=True)
    def integrate_perfect_hint(self, v, l):
        V = self.homogeneize(v, l)
        
        for i in range(self.S.nrows()):
            if V[0, i]:
                self.can_constraints[i] = None
        
        V -= V * self.PI
        den = scal(V * V.T)
        
        
        if den == 0:
            raise RejectedHint("Redundant hint")
        
        self.PI += V.T * (V / den)
        
        VS = V * self.S
        self.Bvol += log(den) / 2
        self._dim -= 1

        den = scal(VS * V.T)
        self.S -= VS.T * (VS / den)

    @not_after_projections
    @hint_integration_wrapper(force=True)
    def integrate_modular_hint(self, v, l, k, smooth=True):
        V = self.homogeneize(v, l)
        for i in range(self.S.nrows()):
            if V[0, i] and self.can_constraints[i] is not None:
                f = (k / V[0, i]).numerator()
                self.can_constraints[i] = lcm(f, self.can_constraints[i])

        VS = V * self.S
        den = scal(VS * V.T)
        if den == 0:
            raise RejectedHint("Redundant hint")

        if not smooth:
            raise NotImplementedError()

        self.Bvol += log(k)

    @not_after_projections
    @hint_integration_wrapper(force=True)
    def integrate_approx_hint(self, v, l, variance, aposteriori=False):
        if variance < 0:
            raise InvalidHint("variance must be non-negative !")
        if variance == 0:
            raise InvalidHint("variance=0 : must use perfect hint !")
        # Only to check homogeneity
        self.homogeneize(v, l)

        V = self.homogeneize(v, 0)
        if not aposteriori:
            VS = V * self.S
            d = scal(VS * V.T)
            self.S -= (1 / (variance + d) * VS.T) * VS

        else:
            VS = V * self.S
            if not scal(VS * VS.T):
                raise RejectedHint("0-Eigenvector of Σ forbidden,")

            den = scal(VS * V.T)
            self.S += (((variance - den) / den**2) * VS.T ) * VS

    @not_after_projections
    @hint_integration_wrapper()
    def integrate_approx_hint_fulldim(self, center,
                                      covariance, aposteriori=False):
        # Using http://www.cs.columbia.edu/~liulp/pdf/linear_normal_dist.pdf
        # with A = Id
        if not aposteriori:
            d = self.S.nrows() - 1
            if self.S.rank() != d or covariance.rank() != d:
                raise InvalidHint("Covariances not full dimensional")

            zero = vec(d * [0])
            F = (self.S + block4(covariance, zero.T, zero, vec([1]))).inverse()
            F[-1, -1] = 0

            self.S -= self.S * F * self.S
        else:
            raise NotImplementedError()

    @hint_integration_wrapper(force=False)
    def integrate_short_vector_hint(self, v):
        V = self.homogeneize(v, 0)
        V -= V * self.PP
        den = scal(V * V.T)

        if den == 0:
            raise InvalidHint("Projects to 0,")

        VPI = V * self.PI
        if scal(VPI * VPI.T):
            raise InvalidHint("Not in Span(Λ),")

        if is_cannonical_direction(v):
            i, vi = cannonical_param(V)
            if vi % self.can_constraints[i]:
                raise InvalidHint("Not in Λ,")
        else:
            if self.verbosity:
                self.logging("Not sure if in Λ,",
                             style="WARNING", newline=False)

        self.projections += 1
        self.Bvol -= log(RR(den)) / 2.
        self._dim -= 1
        self.PP += V.T * (V / den)

        # This is slower, but equivalent:
        # PV = identity_matrix(self.S.ncols()) - projection_matrix(V)
        # X = PV.T * self.S * PV
        R = (self.S * V.T) * (V / den)
        self.S -= R
        L = (V.T / den) * (V * self.S)
        self.S -= L

    def attack(self):
        self.logging("Can't run the attack in simulation.", style="WARNING")
        return None
