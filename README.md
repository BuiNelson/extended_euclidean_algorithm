# Extended Euclidean Algorithm

A tiny Python library that computes the Bézout coefficients and greatest common divisor of two integers, plus convenience helpers for modular inverses and linear Diophantine equations.

```python
from extended_euclidean_algorithm import extended_gcd, mod_inverse, solve_linear_diophantine

r = extended_gcd(240, 46)
# r.gcd == 2, r.x == -9, r.y == 47, and 240*(-9) + 46*47 == 2

inv = mod_inverse(3, 7)        # 5, because (3 * 5) % 7 == 1

x, y = solve_linear_diophantine(4, 6, 2)  # one solution to 4x + 6y == 2
```

## Why this exists

The standard library gives you `math.gcd`, but not the Bézout coefficients. Those coefficients are what you need to compute modular inverses by hand or to parametrise the solutions of `a*x + b*y = c`. This library fills that gap with one focused function (`extended_gcd`) and two thin helpers built on top of it (`mod_inverse`, `solve_linear_diophantine`).

The trade-off: inputs are restricted to genuine `int`s. `math.gcd` quietly coerces integral floats; this library rejects them, because Bézout coefficients for non-integer values are not well-defined and silent coercion hides bugs.

## Edge cases worth knowing

- `extended_gcd(0, 0)` raises `ValueError`. The gcd is conventionally 0, but every pair `(x, y)` satisfies `0*x + 0*y = 0`, so there is no canonical coefficient pair to return.
- The gcd is always returned non-negative, even for negative inputs. This matches `math.gcd` and keeps the identity `a*x + b*y = gcd` holding in every case.
- `mod_inverse` rejects moduli `<= 1` and negative moduli rather than silently negating them; a negative modulus almost always indicates a caller bug.

## Exports

- `extended_gcd(a, b) -> ExtendedGcdResult` — returns a frozen dataclass with `.gcd`, `.x`, `.y`.
- `mod_inverse(a, m) -> int` — returns `x` in `[0, m)` with `(a * x) % m == 1`.
- `solve_linear_diophantine(a, b, c) -> tuple[int, int]` — returns one particular `(x, y)` solution; raises `ValueError` when no solution exists.
- `ExtendedGcdResult` — the result dataclass, exported so callers can type-check or deconstruct results.
