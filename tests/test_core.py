import math
import unittest

from extended_euclidean_algorithm.core import (
    ExtendedGcdResult,
    extended_gcd,
    mod_inverse,
    solve_linear_diophantine,
)


class ExtendedGcdTests(unittest.TestCase):
    def test_basic_coprime(self):
        r = extended_gcd(3, 5)
        self.assertIsInstance(r, ExtendedGcdResult)
        self.assertEqual(r.gcd, 1)
        self.assertEqual(3 * r.x + 5 * r.y, 1)

    def test_basic_non_coprime(self):
        r = extended_gcd(12, 18)
        self.assertEqual(r.gcd, 6)
        self.assertEqual(12 * r.x + 18 * r.y, 6)

    def test_one_zero(self):
        r = extended_gcd(17, 0)
        self.assertEqual(r.gcd, 17)
        self.assertEqual(17 * r.x + 0 * r.y, 17)

    def test_first_zero(self):
        r = extended_gcd(0, 23)
        self.assertEqual(r.gcd, 23)
        self.assertEqual(0 * r.x + 23 * r.y, 23)

    def test_both_zero_raises(self):
        with self.assertRaises(ValueError):
            extended_gcd(0, 0)

    def test_negative_inputs(self):
        r = extended_gcd(-12, 18)
        self.assertEqual(r.gcd, 6)
        self.assertEqual((-12) * r.x + 18 * r.y, 6)

    def test_both_negative(self):
        r = extended_gcd(-12, -18)
        self.assertEqual(r.gcd, 6)
        self.assertEqual((-12) * r.x + (-18) * r.y, 6)

    def test_negative_result_normalized(self):
        r = extended_gcd(-17, 0)
        self.assertGreaterEqual(r.gcd, 0)
        self.assertEqual(r.gcd, 17)

    def test_type_error_float(self):
        with self.assertRaises(TypeError):
            extended_gcd(3.0, 5)

    def test_type_error_str(self):
        with self.assertRaises(TypeError):
            extended_gcd("3", 5)

    def test_matches_math_gcd(self):
        for a, b in [(0, 1), (1, 0), (7, 13), (100, 75), (-100, 75), (-100, -75), (1, 1)]:
            with self.subTest(a=a, b=b):
                r = extended_gcd(a, b)
                self.assertEqual(r.gcd, math.gcd(a, b))
                self.assertEqual(a * r.x + b * r.y, r.gcd)

    def test_large_inputs(self):
        a = 10**18 + 3
        b = 10**18 + 7
        r = extended_gcd(a, b)
        self.assertEqual(r.gcd, math.gcd(a, b))
        self.assertEqual(a * r.x + b * r.y, r.gcd)


class ModInverseTests(unittest.TestCase):
    def test_basic_inverse(self):
        x = mod_inverse(3, 7)
        self.assertEqual((3 * x) % 7, 1)
        self.assertTrue(0 <= x < 7)

    def test_known_pair(self):
        self.assertEqual(mod_inverse(3, 11), 4)

    def test_non_coprime_raises(self):
        with self.assertRaises(ValueError):
            mod_inverse(4, 8)

    def test_modulus_one_raises(self):
        with self.assertRaises(ValueError):
            mod_inverse(1, 1)

    def test_modulus_zero_raises(self):
        with self.assertRaises(ValueError):
            mod_inverse(1, 0)

    def test_modulus_negative_raises(self):
        with self.assertRaises(ValueError):
            mod_inverse(3, -7)

    def test_negative_base(self):
        x = mod_inverse(-3, 7)
        self.assertEqual((-3 * x) % 7, 1)
        self.assertTrue(0 <= x < 7)

    def test_type_error(self):
        with self.assertRaises(TypeError):
            mod_inverse(3.0, 7)


class SolveDiophantineTests(unittest.TestCase):
    def test_has_solution(self):
        # 3x + 5y = 1; gcd(3,5)=1 divides 1.
        x, y = solve_linear_diophantine(3, 5, 1)
        self.assertEqual(3 * x + 5 * y, 1)

    def test_scaled_solution(self):
        # 4x + 6y = 2; gcd=2 divides 2.
        x, y = solve_linear_diophantine(4, 6, 2)
        self.assertEqual(4 * x + 6 * y, 2)

    def test_no_solution_raises(self):
        # gcd(4,6)=2 does not divide 3.
        with self.assertRaises(ValueError):
            solve_linear_diophantine(4, 6, 3)

    def test_a_zero(self):
        # 0*x + 5*y = 10 -> y=2, x can be anything (we pick 0 via gcd).
        x, y = solve_linear_diophantine(0, 5, 10)
        self.assertEqual(0 * x + 5 * y, 10)

    def test_b_zero(self):
        # 7*x + 0*y = 21 -> x=3.
        x, y = solve_linear_diophantine(7, 0, 21)
        self.assertEqual(7 * x + 0 * y, 21)

    def test_both_zero_raises(self):
        with self.assertRaises(ValueError):
            solve_linear_diophantine(0, 0, 5)

    def test_negative_coefficients(self):
        x, y = solve_linear_diophantine(-4, 6, 2)
        self.assertEqual((-4) * x + 6 * y, 2)

    def test_type_error(self):
        with self.assertRaises(TypeError):
            solve_linear_diophantine(1, 2, 3.0)


if __name__ == "__main__":
    unittest.main()
