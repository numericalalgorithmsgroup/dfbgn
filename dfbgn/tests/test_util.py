"""

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

The development of this software was sponsored by NAG Ltd. (http://www.nag.co.uk)
and the EPSRC Centre For Doctoral Training in Industrially Focused Mathematical
Modelling (EP/L015803/1) at the University of Oxford. Please contact NAG for
alternative licensing.

"""

# Ensure compatibility with Python 2
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import unittest

from dfbgn.hessian import Hessian
from dfbgn.util import *


def array_compare(x, y, thresh=1e-14):
    return np.max(np.abs(x - y)) < thresh


class TestSumsq(unittest.TestCase):
    def runTest(self):
        n = 10
        x = np.sin(np.arange(n))
        normx = np.sum(x**2)
        self.assertAlmostEqual(normx, sumsq(x), msg='Wrong answer')


class TestEval(unittest.TestCase):
    def runTest(self):
        objfun = lambda x : np.array([10*(x[1]-x[0]**2), 1-x[0]])
        x = np.array([-1.2, 1.0])
        fvec, f = eval_least_squares_objective(objfun, x)
        self.assertTrue(np.all(fvec == objfun(x)), 'Residuals wrong')
        self.assertAlmostEqual(f, sumsq(fvec), msg='Sum of squares wrong')


class TestModelValue(unittest.TestCase):
    def runTest(self):
        n = 5
        A = np.arange(n ** 2, dtype=np.float).reshape((n, n))
        H = np.sin(A + A.T)  # force symmetric
        hess = Hessian(n, vals=H)
        vec = np.exp(np.arange(n, dtype=np.float))
        g = np.cos(3*np.arange(n, dtype=np.float) - 2.0)
        mval = np.dot(g, vec) + 0.5 * np.dot(vec, np.dot(H, vec))
        self.assertAlmostEqual(mval, model_value(g, hess, vec), msg='Wrong value')


# class TestRandom(unittest.TestCase):
#     def runTest(self):
#         n = 3
#         lower = -10.0 * np.ones((n,))
#         upper = 10.0 * np.ones((n,))
#         num_pts = 2 * n + 4
#         delta = 1.0
#         dirns = random_orthog_directions_within_bounds(num_pts, delta, lower, upper)
#         for i in range(num_pts):
#             self.assertTrue(np.linalg.norm(dirns[i, :]) <= delta + 1e-10, "Unconstrained: dirn %i too long" % i)
#             self.assertTrue(np.all(dirns[i, :] >= lower), "Direction %i below lower bound" % i)
#             self.assertTrue(np.all(dirns[i, :] <= upper), "Direction %i above upper bound" % i)
#         for i in range(n):
#             self.assertTrue(array_compare(dirns[i, :], -dirns[n+i,:]), "Second set should be -ve first set")
#         for i in range(2*n-1):
#             self.assertTrue(abs(np.dot(dirns[i, :], dirns[i+1, :])) < 1e-10, "First 2n directions should be orthog")
#
#
# class TestRandomBox(unittest.TestCase):
#     def runTest(self):
#         n = 5
#         lower = np.array([-10.0, -0.1, -0.5, 0.0, -1.0])
#         upper = np.array([10.0, 10.0, 0.2, 10.0, 0.0])
#         num_pts = 2 * n + 4
#         delta = 1.0
#         dirns = random_orthog_directions_within_bounds(num_pts, delta, lower, upper)
#         # print(dirns)
#         for i in range(num_pts):
#             # self.assertTrue(np.linalg.norm(dirns[i, :]) <= delta + 1e-10, "Unconstrained: dirn %i too long" % i)
#             self.assertTrue(np.all(dirns[i, :] >= lower), "Direction %i below lower bound" % i)
#             self.assertTrue(np.all(dirns[i, :] <= upper), "Direction %i above upper bound" % i)
#         # for i in range(n):
#         #     self.assertTrue(array_compare(dirns[i, :], -dirns[n+i,:]), "Second set should be -ve first set")
#         # for i in range(2*n-1):
#         #     self.assertTrue(abs(np.dot(dirns[i, :], dirns[i+1, :])) < 1e-10, "First 2n directions should be orthog")
#
#
# class TestRandomShort(unittest.TestCase):
#     def runTest(self):
#         n = 3
#         lower = -10.0 * np.ones((n,))
#         upper = 10.0 * np.ones((n,))
#         num_pts = 2 * n + 4
#         delta = 1.0
#         dirns = random_orthog_directions_within_bounds(num_pts, delta, lower, upper, with_neg_dirns=False)
#         for i in range(num_pts):
#             self.assertTrue(np.linalg.norm(dirns[i, :]) <= delta + 1e-10, "Unconstrained: dirn %i too long" % i)
#             self.assertTrue(np.all(dirns[i, :] >= lower), "Direction %i below lower bound" % i)
#             self.assertTrue(np.all(dirns[i, :] <= upper), "Direction %i above upper bound" % i)
#         for i in range(n-1):
#             self.assertTrue(abs(np.dot(dirns[i, :], dirns[i+1, :])) < 1e-10, "First n directions should be orthog")
#         # print(dirns)
#         # self.assertTrue(False, "bad")
#
#
# class TestRandomBoxShort(unittest.TestCase):
#     def runTest(self):
#         n = 5
#         lower = np.array([-10.0, -0.1, -0.5, 0.0, -1.0])
#         upper = np.array([10.0, 10.0, 0.2, 10.0, 0.0])
#         num_pts = 2 * n + 4
#         delta = 1.0
#         dirns = random_orthog_directions_within_bounds(num_pts, delta, lower, upper, with_neg_dirns=False)
#         # print(dirns)
#         for i in range(num_pts):
#             # self.assertTrue(np.linalg.norm(dirns[i, :]) <= delta + 1e-10, "Unconstrained: dirn %i too long" % i)
#             self.assertTrue(np.all(dirns[i, :] >= lower), "Direction %i below lower bound" % i)
#             self.assertTrue(np.all(dirns[i, :] <= upper), "Direction %i above upper bound" % i)
#         # for i in range(n):
#         #     self.assertTrue(array_compare(dirns[i, :], -dirns[n+i,:]), "Second set should be -ve first set")
#         # for i in range(2*n-1):
#         #     self.assertTrue(abs(np.dot(dirns[i, :], dirns[i+1, :])) < 1e-10, "First 2n directions should be orthog")
#         # self.assertTrue(False, "bad")


class TestRandomNotOrthog(unittest.TestCase):
    def runTest(self):
        n = 3
        lower = -10.0 * np.ones((n,))
        upper = 10.0 * np.ones((n,))
        num_pts = 2 * n + 4
        delta = 1.0
        dirns = random_directions_within_bounds(num_pts, delta, lower, upper)
        for i in range(num_pts):
            self.assertTrue(np.linalg.norm(dirns[i, :]) <= delta + 1e-10, "Unconstrained: dirn %i too long" % i)
            self.assertTrue(np.all(dirns[i, :] >= lower), "Direction %i below lower bound" % i)
            self.assertTrue(np.all(dirns[i, :] <= upper), "Direction %i above upper bound" % i)
        # print(dirns)
        # self.assertTrue(False, "bad")


class TestRandomNotOrthogBox(unittest.TestCase):
    def runTest(self):
        n = 5
        lower = np.array([-10.0, -0.1, -0.5, 0.0, -1.0])
        upper = np.array([10.0, 10.0, 0.2, 10.0, 0.0])
        num_pts = 2 * n + 4
        delta = 1.0
        dirns = random_directions_within_bounds(num_pts, delta, lower, upper)
        # print(dirns)
        for i in range(num_pts):
            self.assertTrue(np.linalg.norm(dirns[i, :]) <= delta + 1e-10, "Unconstrained: dirn %i too long" % i)
            self.assertTrue(np.all(dirns[i, :] >= lower), "Direction %i below lower bound" % i)
            self.assertTrue(np.all(dirns[i, :] <= upper), "Direction %i above upper bound" % i)
        # self.assertTrue(False, "bad")


class TestRandomBoxOrthogSubspace(unittest.TestCase):
    def runTest(self):
        n = 5
        lower = -1e20 * np.ones((n,))
        upper = 1e20 * np.ones((n,))
        num_pts = 3
        delta = 2.8
        np.random.seed(0)
        # 1. Basic version - no bounds, no Q
        dirns = random_orthog_directions_new_subspace_within_bounds(num_pts, delta, lower, upper, Q=None)
        for i in range(num_pts):
            self.assertTrue(np.all(dirns[i, :] >= lower), "Direction %i below lower bound" % i)
            self.assertTrue(np.all(dirns[i, :] <= upper), "Direction %i above upper bound" % i)
            for j in range(num_pts):
                # Check orthogonality & length of directions
                self.assertAlmostEqual(np.dot(dirns[i, :], dirns[j,:]), delta**2 if i==j else 0.0, msg='Orthog/size lost (i,j) = (%g,%g)' % (i, j))

        # 2. No bounds, but with Q
        A = np.arange(2 * n).reshape((n, 2))
        Q, _ = np.linalg.qr(A, mode='reduced')  # make directions orthonormal
        dirns = random_orthog_directions_new_subspace_within_bounds(num_pts, delta, lower, upper, Q=Q)
        for i in range(num_pts):
            self.assertTrue(np.all(dirns[i, :] >= lower), "Direction %i below lower bound" % i)
            self.assertTrue(np.all(dirns[i, :] <= upper), "Direction %i above upper bound" % i)
            self.assertTrue(array_compare(np.dot(Q.T, dirns[i, :]), np.zeros((2,))), 'Direction %g not orthogonal to Q' % i)
            for j in range(num_pts):
                # Check orthogonality & length of directions
                self.assertAlmostEqual(np.dot(dirns[i, :], dirns[j, :]), delta ** 2 if i == j else 0.0, msg='Orthog/size lost (i,j) = (%g,%g)' % (i, j))

        # 3. With bounds (Q irrelevant)
        lower = np.array([-10.0, -0.1, -0.5, 0.0, -1.0])
        upper = np.array([10.0, 10.0, 0.2, 10.0, 0.0])
        num_pts = n - 3
        delta = 1.0
        dirns = random_orthog_directions_new_subspace_within_bounds(num_pts, delta, lower, upper, Q=None)
        for i in range(num_pts):
            self.assertTrue(np.linalg.norm(dirns[i, :]) <= delta + 1e-10, "Unconstrained: dirn %i too long" % i)
            self.assertTrue(np.all(dirns[i, :] >= lower), "Direction %i below lower bound" % i)
            self.assertTrue(np.all(dirns[i, :] <= upper), "Direction %i above upper bound" % i)

        # 4. With bounds and Q (but Q still irrelevant)
        dirns = random_orthog_directions_new_subspace_within_bounds(num_pts, delta, lower, upper, Q=Q)
        for i in range(num_pts):
            self.assertTrue(np.linalg.norm(dirns[i, :]) <= delta + 1e-10, "Unconstrained: dirn %i too long" % i)
            self.assertTrue(np.all(dirns[i, :] >= lower), "Direction %i below lower bound" % i)
            self.assertTrue(np.all(dirns[i, :] <= upper), "Direction %i above upper bound" % i)

if __name__ == '__main__':
    unittest.main()