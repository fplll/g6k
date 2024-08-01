from fpylll import *

# current version fpylll raises a lot of Deprecation warnings. Silence that.
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

load("../framework/proba_utils.sage")
load("../framework/utils.sage")
load("../framework/geometry.sage")


# Issues with hint that are caught, and only raise a warning
class RejectedHint(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message

# Issues with hint that are caught, and only raise a warning


class InvalidHint(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


def not_after_projections(fn):
    def decorated(self, *args, **kwargs):
        # if self.projections:
        #     self.logging("You can't integrate more hints after projection. Sorry.", )
        #     raise ValueError
        return fn(self, *args, **kwargs)
    return decorated


def hint_integration_wrapper(force=False,
                             non_primitive_action=None,
                             requires=[],
                             invalidates=[],
                             catch_invalid_hint=True,
                             estimate=True):
    def decorator(fn):
        def decorated(self, *args, **kwargs):

            if self.verbosity:
                self.logging(fn.__name__.replace("_", " "),
                             style='ACTION', priority=1, newline=False)
                if fn.__name__ == "integrate_perfect_hint":
                    self.logging(hint_to_string(
                        args[0], args[1]), style='DATA',
                        priority=2, newline=False)

                if fn.__name__ == "integrate_modular_hint":
                    self.logging("(smooth)" if kwargs.get(
                        "smooth", True) else "(nonsmooth)",
                        priority=1, newline=False)
                    self.logging(hint_to_string(
                        args[0], args[1]) + " MOD %d" % args[2], style='DATA',
                        priority=2, newline=False)

                if fn.__name__ == "integrate_approx_hint":
                    self.logging("(aposteriori)" if kwargs.get(
                        "aposteriori", False) else "(conditionning)",
                        priority=1, newline=False)
                    self.logging(hint_to_string(
                        args[0], args[1]) + " + χ(σ²=%.3f)" % args[2],
                        style='DATA', priority=2, newline=False)

                if fn.__name__ == "integrate_short_vector_hint":
                    self.logging(hint_to_string(
                        args[0], None, lit="c") + "∈ Λ", style='DATA',
                        priority=2, newline=False)

            if "primal" in requires and self.B is None:
                self.D = eliminate_linear_dependencies(self.D, dim=self.dim())
                self.B = dual_basis(self.D)
            if "dual" in requires and self.D is None:
                self.B = eliminate_linear_dependencies(self.B, dim=self.dim())
                self.D = dual_basis(self.B)

            if "force" in kwargs:
                _force = kwargs["force"]
                del kwargs["force"]
            else:
                _force = force

            if "estimate" in kwargs:
                _estimate = kwargs["estimate"]
                del kwargs["estimate"]
            else:
                _estimate = estimate

            if (not _force) or _estimate:
                self.stash()

            if "catch_invalid_hint" in kwargs:
                _catch_invalid_hint = kwargs["catch_invalid_hint"]
                del kwargs["catch_invalid_hint"]
            else:
                _catch_invalid_hint = catch_invalid_hint

            if "non_primitive_action" in kwargs:
                _non_primitive_action = kwargs["non_primitive_action"]
                del kwargs["non_primitive_action"]
            else:
                _non_primitive_action = non_primitive_action

            try:
                if _non_primitive_action is not None:
                    self.test_primitive_dual(concatenate(
                        args[0], -args[1]), _non_primitive_action)
                fn(self, *args, **kwargs)
                self.beta = None
            except RejectedHint as err:
                logging(str(err) + ", Rejected.", style="REJECT", newline=True)
                return False
            except InvalidHint as err:
                if _catch_invalid_hint:
                    logging(str(err) + ", Invalid.",
                            style="REJECT", newline=True)
                    return False
                else:
                    raise err

            if "primal" in invalidates:
                self.B = None
            if "dual" in invalidates:
                self.D = None

            if (not _force) or _estimate:
                return self.undo_if_unworthy(_force)
            else:
                self.logging("", newline=True)
                return True

        return decorated
    return decorator


class DBDD_generic:
    """
    This class defines all the elements defining a DBDD instance
    """
    def __init__():
        raise NotImplementedError(
            "The generic class is not meant to be used directly.")

    def leak(self, v):
        value = scal(self.u * concatenate([v, [0]]).T)
        return value

    def stash(self):
        if self.beta is None:
            self.estimate_attack(silent=True)
        for key, val in self.__dict__.items():
            if key != "save":
                self.save[key] = copy(val)

    def pop(self):
        for key, val in self.save.items():
            if key != "save":
                self.__dict__[key] = val

    def undo_if_unworthy(self, force):
        self.estimate_attack(silent=True)
        if (-self.beta, self.delta)\
                <= (-self.save["beta"], self.save["delta"]):
            if force:
                self.logging("\t Unworthy hint, Forced it.", style="REJECT")
                return True
            else:
                self.logging("\t Unworthy hint, Rejected.", style="REJECT")
                self.pop()
                return False

        self.logging("\t Worthy hint !", style="ACCEPT", newline=False)
        self.logging("dim=%3d, δ=%.8f, β=%3.2f" %
                     (self.dim(), self.delta, self.beta), style="VALUE")
        return True

    def logging(self, message, priority=1, style='NORMAL', newline=True):
        if priority > self.verbosity:
            return
        logging(message, style=style, newline=newline)

    def check_solution(self, solution):
        """ Checks wether the solution is correct
        If the private attributes of the instance are not None,
        the solution is compared to them. It outputs True
        if the solution is indeed the same as the private s and e,
        False otherwise.
        If the private e and s are not stored, we check that the
        solution is small enough.
        :solution: a vector
        """

        if self.u is not None:
            if self.circulant:
                return (sorted(self.u.list()) == sorted(solution.list())) or (sorted(self.u.list()) == sorted((- solution).list())) 
            return (self.u == solution or self.u == - vec(solution))

        if scal(solution * solution.T) > 1.2 * self.expected_length:
            return False
        if self.u is None:
            return True

        if self.verbosity:
            self.logging("Found an incorrect short solution.",
                         priority=-1, style="WARNING")
        return False

    def homogeneize(self, v, l):
        if self.homogeneous and l!=0:
            raise InvalidHint("This hint is not homogeneous.")
        if self.homogeneous:
            return vec(v)
        else:
            return concatenate(v, -l)

    @not_after_projections
    @hint_integration_wrapper()
    def integrate_perfect_hint(self, v, l):
        raise NotImplementedError("This method is not generic.")

    @not_after_projections
    @hint_integration_wrapper()
    def integrate_modular_hint(self, v, l, k, smooth=True):
        raise NotImplementedError("This method is not generic.")

    @not_after_projections
    @hint_integration_wrapper()
    def integrate_approx_hint(self, v, l, variance, aposteriori=False):
        raise NotImplementedError("This method is not generic.")

    @not_after_projections
    @hint_integration_wrapper()
    def integrate_approx_hint_fulldim(self, center, covariance):
        raise NotImplementedError("This method is not generic.")

    @hint_integration_wrapper()
    def integrate_short_vector_hint(self, v):
        raise NotImplementedError("This method is not generic.")

    def volumes(self):
        raise NotImplementedError("This method is not generic.")

    def dim(self):
        raise NotImplementedError("This method is not generic.")

    def test_primitive_dual(self, V, action):
        raise NotImplementedError("This method is not generic.")

    def estimate_attack(self, probabilistic=False, tours=1, silent=False,
        ignore_lift_proba=False, lift_union_bound=False, number_targets=1):
        """ Assesses the complexity of the lattice attack on the instance.
        Return value in Bikz
        """
        (Bvol, Svol, dvol) = self.volumes()
        dim_ = self.dim()
        beta, delta = compute_beta_delta(
            dim_, dvol, probabilistic=probabilistic, tours=tours, verbose=not silent,
            ignore_lift_proba=ignore_lift_proba, number_targets=number_targets, lift_union_bound=lift_union_bound)

        self.dvol = dvol
        self.delta = delta
        self.beta = beta

        if self.verbosity and not silent:
            self.logging("      Attack Estimation     ", style="HEADER")
            self.logging("ln(dvol)=%4.7f \t ln(Bvol)=%4.7f \t ln(Svol)=%4.7f \t"
                         % (dvol, Bvol, Svol) +
                         "δ(β)=%.6f" % compute_delta(beta),
                         style="DATA", priority=2)
            if delta is not None:
                self.logging("dim=%3d \t δ=%.6f \t β=%3.2f " %
                             (dim_, delta, beta), style="VALUE")
            else:
                self.logging("dim=%3d \t \t \t β=%3.2f " %
                             (dim_, beta), style="VALUE")

            self.logging("")
        return (beta, delta)

    def attack(self, beta_max=None, beta_pre=None, randomize=False, tours=1):
        raise NotImplementedError(
            "The generic class is not meant to be used directly.")

    def S_diag(self):
        raise NotImplementedError(
            "The generic class is not meant to be used directly.")

    def integrate_q_vectors(self, q, min_dim=0, report_every=1, indices=None):
        self.logging("      Integrating q-vectors     ", style="HEADER")
        Sd = self.S_diag()
        n = len(Sd)
        I = []
        J = []
        M = q * identity_matrix(n - 1 + self.homogeneous)
        it = 0
        verbosity = self.verbosity
        if indices is None:
            indices = range(n - 1 + self.homogeneous)
        while self.dim() > min_dim:
            if (it % report_every == 0) and report_every > 1:
                self.logging("[...%d]" % report_every, newline=False)
            Sd = self.S_diag()
            L = [(Sd[i], i) for i in indices if i not in I]
            if len(L) == 0:
                break
            _, i = max(L)
            I += [i]
            try:
                didit = self.integrate_short_vector_hint(
                    vec(M[i]), catch_invalid_hint=False)
                if not didit:
                    break
                J += [i]
            except InvalidHint as err:
                self.logging(str(err) + ", Invalid.",
                             style="REJECT", priority=1, newline=True)
            it += 1
            self.verbosity = verbosity if (it % report_every == 0) else 0
        self.verbosity = verbosity
        return [vec(M[i]) for i in J]
