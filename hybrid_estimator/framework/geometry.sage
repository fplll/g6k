from numpy.linalg import inv as np_inv
# from numpy.linalg import slogdet as np_slogdet
from numpy import array
import numpy as np


def dual_basis(B):
    """
    Compute the dual basis of B
    """
    return B.pseudoinverse().transpose()


def projection_matrix(A):
    """
    Construct the projection matrix orthogonally to Span(V)
    """
    S = A * A.T
    return A.T * S.inverse() * A


def project_against(v, X):
    """ Project matrix X orthonally to vector v"""
    # Pv = projection_matrix(v)
    # return X - X * Pv
    Z = (X * v.T) * v / scal(v * v.T)
    return X - Z


# def make_primitive(B, v):
#     assert False
#     # project and Scale v's in V so that each v
#     # is in the lattice, and primitive in it.
#     # Note: does not make V primitive as as set of vector !
#     # (e.g. linear dep. not eliminated)
#     PB = projection_matrix(B)
#     DT = dual_basis(B).T
#     v = vec(v) * PB
#     w = v * DT
#     den = lcm([x.denominator() for x in w[0]])
#     num = gcd([x for x in w[0] * den])
#     if num==0:
#         return None
#     v *= den/num
#     return v


def vol(B):
    return sqrt(det(B * B.T))


def project_and_eliminate_dep(B, W):
    # Project v on Span(B)
    PB = projection_matrix(B)
    V = W * PB
    rank_loss = V.nrows() - V.rank()

    if rank_loss > 0:
        print("WARNING: their were %d linear dependencies out of %d " %
              (rank_loss, V.nrows()))
        V = V.LLL()
        V = V[rank_loss:]

    return V


def is_cannonical_direction(v):
    v = vec(v)
    return sum([x != 0 for x in v[0]]) == 1


def cannonical_param(v):
    v = vec(v)
    assert is_cannonical_direction(v)
    i = [x != 0 for x in v[0]].index(True)
    return i, v[0, i]


def eliminate_linear_dependencies(B, nb_dep=None, dim=None):
    """
    Transform a lattice generator set into a lattice basis
    :B: Generator set of the lattice
    :nb_dep: Numbers of linear dependencies in B (optional)
    :dim: The rank of the lattice (optional)
    """

    # Get the number of dependencies, if possible
    nrows = B.nrows()
    if (nb_dep is None) and (dim is not None):
        nb_dep = nrows - dim
    assert (dim is None) or (nb_dep + dim == nrows)

    if nb_dep is None or nb_dep > 0:
        # Remove dependencies
        B = B.LLL()
        nb_dep = min([i for i in range(nrows) if not B[i].is_zero()]) \
                if nb_dep is None else nb_dep
        B = B[nb_dep:]

    return B


def lattice_orthogonal_section(D, V, assume_full_rank=False, output_basis=True):
    """
    Compute the intersection of the lattice L(B)
    with the hyperplane orthogonal to Span(V).
    (V can be either a vector or a matrix)
    INPUT AND OUTPUT DUAL BASIS
    :assume_full_rank: if True, assume V is already in Span(B),
            to avoid projection computation (for optimization purpose)
    :output_basis: if False, return only a lattice generator set instead of a basis,
            to avoid expensive process of eliminating linear dependencies with LLL
            (for optimization purpose)
    Algorithm:
    - project V onto Span(B)
    - project the dual basis onto orth(V)
    - eliminate linear dependencies (LLL)
    - go back to the primal.
    """

    if not assume_full_rank:
        V = project_and_eliminate_dep(D, V)
    r = V.nrows()

    # Project the dual basis orthogonally to v
    PV = projection_matrix(V)
    D = D - D * PV

    # Eliminate linear dependencies
    if output_basis:
        D = eliminate_linear_dependencies(D, nb_dep=r)

    # Go back to the primal
    return D


def lattice_project_against(B, V, assume_full_rank=False, assume_belonging=False, output_basis=True):
    """
    Compute the projection of the lattice L(B) orthogonally to Span(V). All vectors if V
    (or at least their projection on Span(B)) must belong to L(B).
    :assume_full_rank: if True, assume V is already in Span(B),
            to avoid projection computation (for optimization purpose)
    :assume_belonging: if True, assume V is already in L(B),
            to avoid the verification (for optimization purpose)
    :output_basis: if False, return only a lattice generator set instead of a basis,
            to avoid expensive process of eliminating linear dependencies with LLL
            (for optimization purpose)
    Algorithm:
    - project V onto Span(B)
    - project the basis onto orth(V)
    - eliminate linear dependencies (LLL)
    """
    # Project v on Span(B)
    if not assume_full_rank:
        V = project_and_eliminate_dep(B, V)
    r = V.nrows()

    # Check that V belongs to L(B)
    if not assume_belonging:
        D = dual_basis(B)
        M = D * V.T
        if not lcm([x.denominator() for x in M.list()]) == 1:
            raise ValueError("Not in the lattice")

    # Project the basis orthogonally to v
    PV = projection_matrix(V)
    B = B - B * PV

    # Eliminate linear dependencies
    if output_basis:
        B = eliminate_linear_dependencies(B, nb_dep=r)

    # Go back to the primal
    return B


def lattice_modular_intersection(D, V, k, assume_full_rank=False, output_basis=True):
    """
    Compute the intersection of the lattice L(B) with
    the lattice {x | x*V = 0 mod k}
    (V can be either a vector or a matrix)
    :assume_full_rank: if True, assume V is already in Span(B),
            to avoid projection computation (for optimization purpose)
    :output_basis: if False, return only a lattice generator set instead of a basis,
            to avoid expensive process of eliminating linear dependencies with LLL
            (for optimization purpose)
    Algorithm:
    - project V onto Span(B)
    - append the equations in the dual
    - eliminate linear dependencies (LLL)
    - go back to the primal.
    """
    # Project v on Span(B)
    if not assume_full_rank:
        V = project_and_eliminate_dep(D, V)
    r = V.nrows()
    # append the equation in the dual
    V /= k
    # D = dual_basis(B)
    D = D.stack(V)

    # Eliminate linear dependencies
    if output_basis:
        D = eliminate_linear_dependencies(D, nb_dep=r)

    # Go back to the primal
    return D


def is_diagonal(M):
    if M.nrows() != M.ncols():
        return False
    A = M.numpy()
    return np.all(A == np.diag(np.diagonal(A)))


def logdet(M, exact=False):
    """
    Compute the log of the determinant of a large rational matrix,
    tryping to avoid overflows.
    """
    if not exact:
        MM = array(M, dtype=float)
        _, l = slogdet(MM)
        return l

    a = abs(M.det())
    l = 0

    while a > 2**32:
        l += RR(32 * ln(2))
        a /= 2**32

    l += ln(RR(a))
    return l


def degen_inverse(S, B=None):
    """ Compute the inverse of a symmetric matrix restricted
    to its span
    """
    # Get an orthogonal basis for the Span of B

    if B is None:
        # Get an orthogonal basis for the Span of B
        V = S.echelon_form()
        V = V[:V.rank()]
        P = projection_matrix(V)
    else:
        P = projection_matrix(B)

    # make S non-degenerated by adding the complement of span(B)
    C = identity_matrix(S.ncols()) - P
    Sinv = (S + C).inverse() - C

    assert S * Sinv == P, "Consistency failed (probably not your fault)."
    assert P * Sinv == Sinv, "Consistency failed (probably not your fault)."

    return Sinv


def degen_logdet(S, B=None):
    """ Compute the determinant of a symmetric matrix
    sigma (m x m) restricted to the span of the full-rank
    rectangular (k x m, k <= m) matrix V
    """
    # Get an orthogonal basis for the Span of B
    if B is None:
        # Get an orthogonal basis for the Span of B
        V = S.echelon_form()
        V = V[:V.rank()]
        P = projection_matrix(V)
    else:
        P = projection_matrix(B)

    # Check that S is indeed supported by span(B)
    assert (S - P.T * S * P).norm() < 1e-10

    # make S non-degenerated by adding the complement of span(B)
    C = identity_matrix(S.ncols()) - P
    l3 = logdet(S + C)
    return l3


def square_root_inverse_degen(S, B=None):
    """ Compute the determinant of a symmetric matrix
    sigma (m x m) restricted to the span of the full-rank
    rectangular (k x m, k <= m) matrix V
    """
    if B is None:
        # Get an orthogonal basis for the Span of B
        V = S.echelon_form()
        V = V[:V.rank()]
        P = projection_matrix(V)
    else:
        P = projection_matrix(B)

    # make S non-degenerated by adding the complement of span(B)
    C = identity_matrix(S.ncols()) - P
    S_inv = np_inv(array((S + C), dtype=float))
    S_inv = array(S_inv, dtype=float)
    L_inv = cholesky(S_inv)
    L_inv = round_matrix_to_rational(L_inv)
    L = L_inv.inverse()

    return L, L_inv


def check_basis_consistency(B=None, D=None, Bvol=None):
    """ Check if the non-null parameters are consistent between them
    """
    try:
        active_basis = B or D
        if active_basis:
            assert (B is None) or (D is None) or (D.T * B == identity_matrix(active_basis.nrows()))
            if Bvol is not None:
                read_Bvol = logdet(active_basis)
                read_Bvol *= (-1)**(active_basis!=B)
                assert abs(Bvol - read_Bvol) < 1e-6
        return True

    except AssertionError:
        return False


def build_substitution_matrix(V, pivot=None, output_extra_data=True):
    """ Compute the substitution matrix Γ of a linear system X⋅M^T = 0 where we know X⋅V^T.
    After substitution, the smaller linear system X'⋅M'^T = 0 will verify X = X'⋅Γ^T.
    """
    dim = V.ncols()

    # Find a pivot for V
    _, pivot = V.nonzero_positions()[0] if (pivot is None) else (None, pivot)
    assert V[0,pivot] != 0, 'The value of the pivot must be non-zero.'

    # Normalize V according to the pivot
    V1 = - V[0,:pivot] / V[0,pivot]
    V2 = - V[0,pivot+1:] / V[0,pivot]

    # Build the substitution matrix
    Gamma = zero_matrix(QQ, dim,dim-1)
    Gamma[:pivot,:pivot] = identity_matrix(pivot)
    Gamma[pivot,:pivot] = V1
    Gamma[pivot,pivot:] = V2
    Gamma[pivot+1:,pivot:] = identity_matrix(dim-pivot-1)

    if not output_extra_data:
        return (Gamma, None)

    # Compute efficiently
    #  - the determinant of (Gamma.T * Gamma)
    #  - the 'pseudo_inv' := (Gamma.T * Gamma).inv()
    det = 1 + scal(V1*V1.T) + scal(V2*V2.T)
    pseudo_inv = zero_matrix(QQ, dim-1,dim-1)
    pseudo_inv[:pivot,:pivot] = identity_matrix(pivot) - V1.T*V1 / det
    pseudo_inv[pivot:,pivot:] = identity_matrix(dim-pivot-1) - V2.T*V2 / det
    pseudo_inv[:pivot,pivot:] = - V1.T*V2 / det
    pseudo_inv[pivot:,:pivot] = - V2.T*V1 / det

    return (Gamma, (det, pseudo_inv))
