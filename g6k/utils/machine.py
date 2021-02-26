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
General Sieving Kernel Machine

..  note :: The paper writes ``a,b`` for applying for ``a`` then ``b``.  This module uses function composition
notation and write ``(b*a)``

"""

from __future__ import absolute_import
import sys
from g6k import Siever
from six.moves import range

#
# Base Instruction
#


class IdentityInstruction(object):
    def __mul__(left, right):
        if not issubclass(left.__class__, IdentityInstruction) \
           or not issubclass(right.__class__, IdentityInstruction):
            raise TypeError("Can only multiply instructions, but got %s, %s"%(type(left), type(right)))
        return ProductInstruction(left, right)

    def __pow__(self, exponent):
        return PowerInstruction(self, exponent)

    def __call__(self, state):
        return state

    def __repr__(self):
        return self.__class__.__name__

#
# Composition Instructions
#


class ProductInstruction(IdentityInstruction):
    """
    Multiplication is function composition, thus::

        >>> from g6k.utils.util import load_svpchallenge_and_randomize
        >>> from g6k.utils.machine import M
        >>> A, _ = load_svpchallenge_and_randomize(60)
        >>> state = (M.Reset(0, 0, 60) * M.Initialize)(A)

    is equivalent to::

        >>> state = M.Initialize(A)
        >>> state = M.Reset(0, 0, 60)(state)

    """
    def __init__(self, *args):
        self.functions = args

    def __call__(self, state):
        # TODO: Write as square-and-multiply?
        for function in reversed(self.functions):
            state = function(state)
        return state

    def __repr__(self):
        return "(" + "*".join(repr(f) for f in self.functions) + ")"


class PowerInstruction(IdentityInstruction):
    def __init__(self, function, exponent):
        self.function = function
        self.exponent = exponent

    def __call__(self, state):
        for i in range(self.exponent):
            state = self.function(state)
        return state

    def __repr__(self):
        return "(%s**(%d))"%(self.function, self.exponent)

#
# Basic Instruction Set
#


class Initialize(IdentityInstruction):
    """
    Initialize the machine with a basis::

        >>> from g6k.utils.util import load_svpchallenge_and_randomize
        >>> from g6k.utils.machine import M
        >>> A, _ = load_svpchallenge_and_randomize(60)
        >>> state = M.Initialize(A)

    """
    def __call__(self, B, params=None):
        return Siever(B, params=params)


class Reset(IdentityInstruction):
    """
    Empty database, and set (ℓ', ℓ, r)::

        >>> from g6k.utils.util import load_svpchallenge_and_randomize
        >>> from g6k.utils.machine import M
        >>> A, _ = load_svpchallenge_and_randomize(60)
        >>> state = (M.Reset(0, 0, 60) * M.Initialize)(A)
        >>> state.l
        0


    """
    def __init__(self, lprime, l, r):
        self.lprime = lprime
        self.l = l  # noqa
        self.r = r

    def __repr__(self):
        return "%s(%d,%d,%d)"%(self.__class__.__name__,
                               self.lprime, self.l, self.r)

    def __call__(self, state):
        state.initialize_local(self.lprime, self.l, self.r)
        return state


class S(IdentityInstruction):
    """
    Sieve using some unspecified sieving algorithm::

        >>> from g6k.utils.util import load_svpchallenge_and_randomize
        >>> from g6k.utils.machine import M
        >>> A, _ = load_svpchallenge_and_randomize(60)
        >>> state = (M.S * M.Reset(0, 10, 40) * M.Initialize)(A)

    """
    def __call__(self, state):
        state()
        return state


class EL(IdentityInstruction):
    """
    Extend left::

        >>> from g6k.utils.util import load_svpchallenge_and_randomize
        >>> from g6k.utils.machine import M
        >>> A, _ = load_svpchallenge_and_randomize(60)
        >>> state = (M.EL * M.S * M.Reset(0, 10, 40) * M.Initialize)(A)
        >>> state.l, state.r
        (9, 40)

    """
    def __call__(self, state):
        state.extend_left()
        return state


class ER(IdentityInstruction):
    """
    Extend right::

        >>> from g6k.utils.util import load_svpchallenge_and_randomize
        >>> from g6k.utils.machine import M
        >>> A, _ = load_svpchallenge_and_randomize(60)
        >>> state = (M.ER * M.S * M.Reset(0, 10, 40) * M.Initialize)(A)
        >>> state.l, state.r
        (10, 41)

    """

    def __call__(self, state):
        state.extend_right()
        return state


class SL(IdentityInstruction):
    """
    Shrink left::

        >>> from g6k.utils.util import load_svpchallenge_and_randomize
        >>> from g6k.utils.machine import M
        >>> A, _ = load_svpchallenge_and_randomize(60)
        >>> state = (M.SL * M.S * M.Reset(0, 10, 40) * M.Initialize)(A)
        >>> state.l, state.r
        (11, 40)


    """
    def __call__(self, state):
        state.shrink_left()
        return state


class I(IdentityInstruction): # noqa
    """
    Insert the best lift: choose the best insertion candidate any selection/score function, and
    insert it::

        >>> from g6k.utils.util import load_svpchallenge_and_randomize
        >>> from g6k.utils.machine import M
        >>> A, _ = load_svpchallenge_and_randomize(60)
        >>> state = (M.I * M.S * M.Reset(0, 10, 40) * M.Initialize)(A)
        >>> state.l, state.r
        (11, 40)

    """
    def __init__(self, kappa=None):
        self.kappa = kappa

    def __call__(self, state):
        if self.kappa is None:
            state.insert_best_lift()
        else:
            state.insert_best_lift(scoring=lambda i, nl, ol, aux: int(i == self.kappa))
        return state

    def __repr__(self):
        if self.kappa is None:
            return self.__class__.__name__
        else:
            return "%s(%d)"%(self.__class__.__name__, self.kappa)

#
# Higher-Level Operations
#


class ProgressiveSieve(IdentityInstruction):
    """
    The progressive-sieving strategy from Ducas'18::

        >>> from g6k.utils.util import load_svpchallenge_and_randomize
        >>> from g6k.utils.machine import M
        >>> A, _ = load_svpchallenge_and_randomize(40)
        >>> state = M.ProgressiveSieve(M.Initialize(A))

    """
    def __call__(self, state):
        d = state.M.d
        return (I() * ((S()*ER())**d) * Reset(0, 0, 0))(state)


class SubSieve(IdentityInstruction):
    """
    The SubSieve from Ducas'18::

        >>> from g6k.utils.util import load_svpchallenge_and_randomize
        >>> from g6k.utils.machine import M
        >>> A, _ = load_svpchallenge_and_randomize(40)
        >>> state = M.SubSieve(f=5)(M.Initialize(A))

    """
    def __init__(self, f):
        self.f = f

    def __call__(self, state):
        d, f = state.M.d, self.f
        state = Reset(0, 0, 0)(state)
        state = ((S()*ER())**(d-f))(state)
        for kappa in range(d-f):
            state = I(kappa=kappa)(state)
        return state

    def __repr__(self):
        return "%s(%d)"%(self.__class__.__name__, self.f)


class LeftProgressiveSieve(IdentityInstruction):
    """
    Left progressive-sieving strategy::

        >>> from g6k.utils.util import load_svpchallenge_and_randomize
        >>> from g6k.utils.machine import M
        >>> A, _ = load_svpchallenge_and_randomize(40)
        >>> state = M.LeftProgressiveSieve(M.Initialize(A))

    """
    def __call__(self, state):
        d = state.M.d
        return (I() * ((S()*EL())**d) * Reset(0, d, d))(state)


class LeftSubSieve(IdentityInstruction):
    """
    Left SubSieve::

        >>> from g6k.utils.util import load_svpchallenge_and_randomize
        >>> from g6k.utils.machine import M
        >>> A, _ = load_svpchallenge_and_randomize(40)
        >>> state = M.LeftSubSieve(f=5)(M.Initialize(A))

    """
    def __init__(self, f):
        self.f = f

    def __call__(self, state):
        d, f = state.M.d, self.f
        state = Reset(0, d, d)(state)
        state = ((S()*EL())**(d-f))(state)
        for kappa in range(d-f):
            state = I(kappa=kappa)(state)
        return state


class Pump(IdentityInstruction):
    """
    Pump::

        >>> from g6k.utils.util import load_svpchallenge_and_randomize
        >>> from g6k.utils.machine import M
        >>> A, _ = load_svpchallenge_and_randomize(40)
        >>> state = M.Pump(0, 10, 30, 1)(M.Initialize(A))
        >>> state = M.Pump(0, 10, 30, 0)(M.Initialize(A))

    """
    def __init__(self, lprime, l, r, s):   # noqa
        self.lprime = lprime
        self.l = l  # noqa
        self.r = r
        self.s = s

    def __call__(self, state):
        lprime, l, r, s = self.lprime, self.l, self.r, self.s # noqa

        R = Reset(lprime, r, r)
        PumpUp = ((S()*EL())**(r-l))
        PumpDown = ((S()**s) * I())**(r-l)

        F = PumpDown * PumpUp * R
        return F(state)

    def __repr__(self):
        return "Pump(%d,%d,%d,%d)"%(self.lprime, self.l, self.r, self.s)

#
# Syntactic Sugar
#


class Machine:
    def __getattr__(self, name):
        attr = getattr(sys.modules[__name__], name)
        try:
            return attr()
        except TypeError:
            return attr

    def __dir__(self):
        l = []
        for k in dir(sys.modules[__name__]):
            v = getattr(sys.modules[__name__], k)
            try:
                if IdentityInstruction in v.__mro__:
                    l.append(k)
            except AttributeError:
                pass

        return list(self.__class__.__dict__.keys()) + l

    def __repr__(self):
        return "G6K Namespace"

M = Machine()  # noqa
