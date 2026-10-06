"""Core implementation of the extended Euclidean algorithm.

The extended Euclidean algorithm computes the greatest common divisor (gcd) of
two integers a and b, together with Bézout coefficients x and y such that

    a*x + b*y = gcd(a, b)

These coefficients are what make the algorithm useful beyond a plain gcd: they
let you compute modular inverses and solve linear Diophantine equations.

Design decisions
----------------
* Gcd is returned as a non-negative int. The textbook recursive algorithm
  produces a sign that depends on the branch taken; normalising to a
  non-negative value makes the identity a*x + b*y = gcd(a, b) hold for every
  pair of inputs, including those involving negative numbers, and matches the
  behaviour of :func:`math.gcd`.
* Inputs must be ints. Python's ``math.gcd`` silently coerces floats that
  happen to be integral, but accepting floats here would invite surprises
  (e.g. ``1e20`` is integral but cannot be split into Bézout coefficients
  losslessly). Rejecting non-int ints keeps the contract crisp.
* Both zero inputs are rejected. gcd(0, 0) is conventionally 0 in Python, but
  the Bézout identity degenerates: 0*x + 0*y = 0 is satisfied by every pair of
  coefficients, so there is no meaningful answer to return. Rather than pick an
  arbitrary convention that some callers would have to undo, we raise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ExtendedGcdResult:
    """Result of :func:`extended_gcd`.

    Attributes
    ----------
    gcd:
        Non-negative greatest common divisor of the inputs.
    x:
        Bézout coefficient for the first input ``a``.
    y:
        Bézout coefficient for the second input ``b``.

    These satisfy ``a * x + b * y == gcd``.
    """

    gcd: int
    x: int
    y: int


def extended_gcd(a: int, b: int) -> ExtendedGcdResult:
    """Return the gcd and Bézout coefficients of ``a`` and ``b``.

    The returned :class:`ExtendedGcdResult` always satisfies
    ``a * result.x + b * result.y == result.gcd`` and ``result.gcd >= 0``.

    Raises
    ------
    TypeError
        If either input is not an :class:`int`. Subclasses of ``int`` (such as
        :class:`bool`) are accepted because they are genuine ints; this mirrors
        :func:`math.gcd`.
    ValueError
        If both inputs are zero, since no canonical Bézout coefficients exist.
    """
    # bool is a subclass of int, so isinstance(True, int) is True. We allow it
    # because it behaves as 0/1 in arithmetic, matching math.gcd.
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError(
            "extended_gcd arguments must be int, got "
            f"{type(a).__name__} and {type(b).__name__}"
        )

    if a == 0 and b == 0:
        raise ValueError("extended_gcd(0, 0) is undefined: no canonical Bézout coefficients")

    # Iterative version avoids Python recursion limits for very large inputs.
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1

    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t

    gcd, x, y = old_r, old_s, old_t

    # The iterative loop yields a gcd whose sign tracks the inputs. Normalise
    # to a non-negative value by flipping all three when it is negative; this
    # preserves the Bézout identity a*x + b*y = gcd.
    if gcd < 0:
        gcd, x, y = -gcd, -x, -y

    return ExtendedGcdResult(gcd=gcd, x=x, y=y)


def mod_inverse(a: int, m: int) -> int:
    """Return the modular multiplicative inverse of ``a`` modulo ``m``.

    The result ``x`` satisfies ``(a * x) % m == 1`` and is always in the range
    ``0 <= x < m``.

    Raises
    ------
    TypeError
        If either input is not an :class:`int`.
    ValueError
        If ``m <= 1`` (no invertible residue class can exist) or if ``a`` and
        ``m`` are not coprime (so no inverse exists).
    """
    if not isinstance(a, int) or not isinstance(m, int):
        raise TypeError(
            "mod_inverse arguments must be int, got "
            f"{type(a).__name__} and {type(m).__name__}"
        )

    # Modulus 0 or 1 cannot host a multiplicative group. We also reject
    # negative moduli rather than silently negating them: callers who pass a
    # negative modulus almost always have a bug, and a clear error beats a
    # surprising result.
    if m <= 1:
        raise ValueError(f"modulus must be > 1, got {m}")

    result = extended_gcd(a % m, m)

    if result.gcd != 1:
        raise ValueError(
            f"{a} has no inverse modulo {m}: gcd({a}, {m}) = {result.gcd}"
        )

    # result.x is the Bézout coefficient satisfying (a % m) * x + m * y = 1,
    # so (a % m) * x ≡ 1 (mod m). Reduce into [0, m) for a canonical return.
    return result.x % m


def solve_linear_diophantine(a: int, b: int, c: int) -> Tuple[int, int]:
    """Return one integer solution ``(x, y)`` to ``a * x + b * y == c``.

    A linear Diophantine equation ``a * x + b * y = c`` has integer solutions
    if and only if ``gcd(a, b)`` divides ``c``. When a solution exists, this
    function returns one particular solution; the full family is

        x = x0 + k * (b / g)
        y = y0 - k * (a / g)

    for any integer ``k``, where ``g = gcd(a, b)`` and ``(x0, y0)`` is the
    returned solution.

    Returns
    -------
    (int, int)
        A particular solution ``(x0, y0)``.

    Raises
    ------
    TypeError
        If any input is not an :class:`int`.
    ValueError
        If ``a == 0`` and ``b == 0`` (the equation is either trivially true
        for every pair or has no solution, so a single answer is not
        meaningful) or if no solution exists (``gcd(a, b)`` does not divide
        ``c``).
    """
    if not isinstance(a, int) or not isinstance(b, int) or not isinstance(c, int):
        raise TypeError(
            "solve_linear_diophantine arguments must be int, got "
            f"{type(a).__name__}, {type(b).__name__}, {type(c).__name__}"
        )

    if a == 0 and b == 0:
        raise ValueError(
            "solve_linear_diophantine with a == 0 and b == 0 has no "
            "canonical single solution"
        )

    result = extended_gcd(a, b)
    g = result.gcd

    if c % g != 0:
        raise ValueError(
            f"no integer solution: gcd({a}, {b}) = {g} does not divide {c}"
        )

    factor = c // g
    return result.x * factor, result.y * factor
