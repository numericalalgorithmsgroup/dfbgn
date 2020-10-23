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

from dfbgn.model import *


def array_compare(x, y, thresh=1e-14):
    return np.max(np.abs(x - y)) < thresh


def overall_qr_error(A, Q, R):
    # Check various properties of QR factorisation
    m, n = A.shape
    if m < n:
        raise RuntimeError("overall_qr_error not designed for m < n case (A.shape = %s)" % A.shape)
    factorisation_error = np.linalg.norm(A - Q.dot(R))
    if factorisation_error > 1e-10:
        print("- Factorisation error = %g" % factorisation_error)
    Q_orthog_error = np.linalg.norm(np.eye(n) - Q.T.dot(Q))
    if Q_orthog_error > 1e-10:
        print("- Q orthogonal error = %g" % Q_orthog_error)
    R_triang_error = 0.0
    if R_triang_error > 1e-10:
        print("- R upper triangular error = %g" % R_triang_error)
    if R.shape != (n,n):
        print(" - R has wrong shape %s, expect (%g,%g)" % (R.shape, n, n))
    return max(factorisation_error, Q_orthog_error, R_triang_error)


def rosenbrock(x):
    # x0 = np.array([-1.2, 1.0])
    return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])


def powell_singular(x):
    # x0 = np.array([3.0, -1.0, 0.0, 1.0])
    fvec = np.zeros((4,))

    fvec[0] = x[0] + 10.0 * x[1]
    fvec[1] = np.sqrt(5.0) * (x[2] - x[3])
    fvec[2] = (x[1] - 2.0 * x[2]) ** 2
    fvec[3] = np.sqrt(10.0) * (x[0] - x[3]) ** 2

    return fvec


def sumsq(x):
    return np.dot(x, x)


def generic_model_check(tester, model, p, pts, objfun, kopt, mystr="", thresh=1e-14, interp_thresh=1e-14):
    xopt = pts[kopt]
    # All the testing of a model to make sure it's sensible
    tester.assertEqual(model.p, p, msg="Wrong p %s" % mystr)
    for k, xk in enumerate(pts):
        tester.assertTrue(array_compare(model.xpt(k), xk, thresh=thresh), msg="Point %g wrong %s" % (k, mystr))
        tester.assertTrue(array_compare(model.rvec(k), objfun(xk), thresh=thresh), msg="rvec %g wrong %s" % (k, mystr))
        tester.assertAlmostEqual(model.fval(k), sumsq(objfun(xk)), msg="fval %g wrong %s" % (k, mystr))
    tester.assertEqual(model.kopt, kopt, msg="Wrong kopt %s" % mystr)
    tester.assertTrue(array_compare(model.xopt(), xopt, thresh=thresh), msg="Wrong xopt %s" % mystr)
    tester.assertTrue(array_compare(model.ropt(), objfun(xopt), thresh=thresh), msg="Wrong ropt %s" % mystr)
    tester.assertAlmostEqual(model.fopt(), sumsq(objfun(xopt)), msg="Wrong fopt %s" % mystr)
    tester.assertFalse(model.factorisation_current, msg="Wrong factorisation_current %s" % mystr)
    is_ok, cvec, J = model.interpolate_mini_models(jac_in_full_space=True)
    tester.assertTrue(is_ok, msg="Full interp failed %s" % mystr)
    tester.assertAlmostEqual(overall_qr_error(model.directions_from_xopt().T, model.Q, model.R), 0.0, msg="Bad QR factorisation %s" % mystr)  # factorisation done by now
    for k, xk in enumerate(pts):
        tester.assertTrue(array_compare(cvec + J.dot(xk - xopt), objfun(xk), thresh=interp_thresh), msg="Full interp wrong for k=%g %s" % (k, mystr))
    return


class TestInterpFullDim(unittest.TestCase):
    def runTest(self):
        # Full-dimensional model for Rosenbrock
        objfun = rosenbrock
        args = ()  # optional arguments for objfun
        scaling_changes = None
        n, m = 2, 2
        x0 = np.array([-1.2, 1.0])
        r0 = objfun(x0)
        xl = -1e20 * np.ones((n,))
        xu = 1e20 * np.ones((n,))
        model = InterpSet(n, x0, r0, xl, xu)
        self.assertEqual(model.n, n, msg="Wrong n after setup")
        self.assertEqual(model.m, m, msg="Wrong m after setup")
        self.assertEqual(model.p, n, msg="Wrong p after setup")
        self.assertTrue(array_compare(model.xopt(), x0), msg='Wrong xopt after setup')
        self.assertTrue(array_compare(model.ropt(), objfun(x0)), msg='Wrong ropt after setup')
        self.assertAlmostEqual(model.fopt(), sumsq(objfun(x0)), msg='Wrong fopt after setup')
        # Now add better point
        nf = 1
        maxfun = 10
        delta = 1.0
        np.random.seed(0)
        model.initialise_interp_set(delta, objfun, args, scaling_changes, nf, maxfun)
        self.assertTrue(np.all(np.isfinite(model.points)), msg="model.points still has infinite values after init")
        self.assertTrue(array_compare(model.points[0, :], x0), msg="Wrong points[0, :] after init")
        # Extracted manually, given np.random.seed(0)
        # np.set_printoptions(formatter={'float': lambda x: "{0:0.15f}".format(x)})
        # print(model.points[1,:] - x0)
        # print(model.points[2,:] - x0)
        x1 = x0 + delta * np.array([-0.874428832365212, -0.485153807702683])
        x2 = x0 + delta * np.array([-0.485153807702683, 0.874428832365212])
        self.assertAlmostEqual(np.dot(x1 - x0, x2 - x0), 0.0, msg="Directions not orthogonal")
        L = np.zeros((n, n))
        L[:, 0] = x1 - x0
        L[:, 1] = x2 - x0
        dists = np.array([0.0, delta, delta])  # distances to xopt

        # Test data is ok
        for k, xk in enumerate([x0, x1, x2]):
            # print(model.xpt(k), xk)
            self.assertTrue(array_compare(model.xpt(k), xk), msg="Point %g wrong after init" % k)
            self.assertTrue(array_compare(model.rvec(k), objfun(xk)), msg="rvec %g wrong after init" % k)
            self.assertAlmostEqual(model.fval(k), sumsq(objfun(xk)), msg="fval %g wrong after init" % k)
        self.assertAlmostEqual(model.fbeg, sumsq(objfun(x0)), msg="Wrong fbeg after init")
        self.assertEqual(model.kopt, 0, msg="Wrong kopt after init")
        self.assertTrue(array_compare(model.xopt(), x0), msg="Wrong xopt after init")
        self.assertTrue(array_compare(model.ropt(), objfun(x0)), msg="Wrong ropt after init")
        self.assertAlmostEqual(model.fopt(), sumsq(objfun(x0)), msg="Wrong fopt after init")
        self.assertFalse(model.factorisation_current, msg="Wrong factorisation_current after init")

        # Test distances and directions ok
        self.assertTrue(array_compare(model.directions_from_xopt(), L.T), msg="Wrong dirns after init")
        self.assertTrue(array_compare(model.distances_to_xopt(), dists), msg="Wrong distances after init")
        self.assertTrue(array_compare(model.distances_to_xopt(include_kopt=False), dists[1:]), msg="Wrong distances (no xopt) after init")

        # Test factorisation
        model.factorise_system()
        self.assertTrue(model.factorisation_current, msg="Wrong factorisation_current after QR")
        self.assertAlmostEqual(overall_qr_error(L, model.Q, model.R), 0.0, msg="Bad QR factorisation")

        # Test interpolation
        for m1 in range(m):
            c, g = model.build_single_model('residual', m1, gradient_in_full_space=True)  # model based at xopt=x0
            c1, g1 = model.build_single_model('residual', m1, gradient_in_full_space=False)  # model based at xopt=x0
            self.assertTrue(array_compare(g, model.project_to_full_space(g1)), msg="Interp %g project wrong" % m1)
            self.assertTrue(array_compare(g1, model.project_to_reduced_space(g)), msg="Interp %g project wrong 2" % m1)

            for k, xk in enumerate([x0, x1, x2]):
                self.assertAlmostEqual(c + g.dot(xk - x0), objfun(xk)[m1], msg="Interp %g wrong for k=%g" % (m1, k))
                self.assertAlmostEqual(c1 + g1.dot(model.Q.T.dot(xk - x0)), objfun(xk)[m1], msg="Interp %g wrong for k=%g (with Q)" % (m1, k))

        # Test Lagrange polynomials
        for kbase in range(n+1):
            c, g = model.build_single_model('lagrange', kbase, gradient_in_full_space=True)  # model based at xopt=x0
            c1, g1 = model.build_single_model('lagrange', kbase, gradient_in_full_space=False)  # model based at xopt=x0
            self.assertAlmostEqual(c, 1.0 if kbase==model.kopt else 0.0, msg="Wrong c for Lagrange interp %g" % kbase)
            self.assertAlmostEqual(c1, 1.0 if kbase == model.kopt else 0.0, msg="Wrong c for Lagrange interp %g (with Q)" % kbase)
            self.assertTrue(array_compare(g, model.project_to_full_space(g1)), msg="Lagrange %g project wrong" % kbase)
            self.assertTrue(array_compare(g1, model.project_to_reduced_space(g)), msg="Lagrange %g project wrong 2" % kbase)

            for k, xk in enumerate([x0, x1, x2]):
                self.assertAlmostEqual(c + g.dot(xk - x0), 1.0 if k==kbase else 0.0, msg="Lagrange interp %g wrong for k=%g" % (kbase, k))
                self.assertAlmostEqual(c1 + g1.dot(model.Q.T.dot(xk - x0)), 1.0 if k == kbase else 0.0, msg="Lagrange interp %g wrong for k=%g (with Q)" % (kbase, k))

            # We also have these wrapped functions for Lagrange polynomials (similar to interpolate_mini_models below)
            c2, g2 = model.lagrange_poly(kbase, gradient_in_full_space=True)
            c3, g3 = model.lagrange_poly(kbase, gradient_in_full_space=False)
            self.assertAlmostEqual(c, c2, msg="Wrong c for wrapped Lagrange %g" % kbase)
            self.assertAlmostEqual(c1, c3, msg="Wrong c for wrapped Lagrange %g (with Q)" % kbase)
            self.assertTrue(array_compare(g, g2), msg="Wrong g for wrapped Lagrange %g" % kbase)
            self.assertTrue(array_compare(g1, g3), msg="Wrong g for wrapped Lagrange %g (with Q)" % kbase)

        # Test full model interpolation
        is_ok, cvec, J = model.interpolate_mini_models(jac_in_full_space=True)
        is_ok1, cvec1, J1 = model.interpolate_mini_models(jac_in_full_space=False)
        self.assertTrue(is_ok, msg="Full interp failed")
        self.assertTrue(is_ok1, msg="Full interp failed (with Q)")
        self.assertTrue(array_compare(J.T, model.project_to_full_space(J1.T)), msg="Full interp J project wrong")
        self.assertTrue(array_compare(J1.T, model.project_to_reduced_space(J.T)), msg="Full interp J project wrong 2")
        for k, xk in enumerate([x0, x1, x2]):
            self.assertTrue(array_compare(cvec + J.dot(xk-x0), objfun(xk)), msg="Full interp wrong for k=%g" % k)
            self.assertTrue(array_compare(cvec1 + J1.dot(model.Q.T.dot(xk - x0)), objfun(xk)), msg="Full interp wrong for k=%g (with Q)" % k)

        # self.assertFalse(True, msg="abc dummy")


class TestInterpReducedDimCoord(unittest.TestCase):
    def runTest(self):
        # Reduced-dimensional model for Powell Singular
        objfun = powell_singular
        args = ()  # optional arguments for objfun
        scaling_changes = None
        n, m = 4, 4
        p = 2
        x0 = np.array([3.0, -1.0, 0.0, 1.0])
        r0 = objfun(x0)
        xl = -1e20 * np.ones((n,))
        xu = 1e20 * np.ones((n,))
        model = InterpSet(p, x0, r0, xl, xu)
        self.assertEqual(model.n, n, msg="Wrong n after setup")
        self.assertEqual(model.m, m, msg="Wrong m after setup")
        self.assertEqual(model.p, p, msg="Wrong p after setup")
        self.assertTrue(array_compare(model.xopt(), x0), msg='Wrong xopt after setup')
        self.assertTrue(array_compare(model.ropt(), objfun(x0)), msg='Wrong ropt after setup')
        self.assertAlmostEqual(model.fopt(), sumsq(objfun(x0)), msg='Wrong fopt after setup')
        # Now add better point
        nf = 1
        maxfun = 10
        delta = 1.0
        np.random.seed(0)
        model.initialise_interp_set(delta, objfun, args, scaling_changes, nf, maxfun, use_coord_directions=True)
        self.assertTrue(np.all(np.isfinite(model.points)), msg="model.points still has infinite values after init")
        self.assertTrue(array_compare(model.points[0, :], x0), msg="Wrong points[0, :] after init")
        # Extracted manually, given np.random.seed(0)
        # np.set_printoptions(formatter={'float': lambda x: "{0:0.15f}".format(x)})
        # print(model.points[1,:] - x0)
        # print(model.points[2,:] - x0)
        x1 = x0 + delta * np.array([0.0, 0.0, 1.0, 0.0])
        x2 = x0 + delta * np.array([0.0, 0.0, 0.0, 1.0])
        self.assertAlmostEqual(np.dot(x1 - x0, x2 - x0), 0.0, msg="Directions not orthogonal")
        self.assertAlmostEqual(np.linalg.norm(x1 - x0), delta, msg="Direction 1 length wrong")
        self.assertAlmostEqual(np.linalg.norm(x2 - x0), delta, msg="Direction 2 length wrong")
        kopt = np.argmin(np.array([sumsq(objfun(x0)), sumsq(objfun(x1)), sumsq(objfun(x2))]))

        # Test data is ok
        for k, xk in enumerate([x0, x1, x2]):
            # print(model.xpt(k), xk)
            self.assertTrue(array_compare(model.xpt(k), xk), msg="Point %g wrong after init" % k)
            self.assertTrue(array_compare(model.rvec(k), objfun(xk)), msg="rvec %g wrong after init" % k)
            self.assertAlmostEqual(model.fval(k), sumsq(objfun(xk)), msg="fval %g wrong after init" % k)
        self.assertAlmostEqual(model.fbeg, sumsq(objfun(x0)), msg="Wrong fbeg after init")
        self.assertEqual(model.kopt, kopt, msg="Wrong kopt after init")
        xopt = model.xpt(kopt)
        self.assertTrue(array_compare(model.xopt(), xopt), msg="Wrong xopt after init")
        self.assertTrue(array_compare(model.ropt(), objfun(xopt)), msg="Wrong ropt after init")
        self.assertAlmostEqual(model.fopt(), sumsq(objfun(xopt)), msg="Wrong fopt after init")
        self.assertFalse(model.factorisation_current, msg="Wrong factorisation_current after init")

        # Test distances and directions ok
        L = np.zeros((n, p+1))
        L[:, 0] = x0 - xopt
        L[:, 1] = x1 - xopt
        L[:, 2] = x2 - xopt
        dists = np.linalg.norm(L, axis=0)  # norm of each column
        L = np.delete(L, kopt, axis=1)  # delete kopt-th column
        self.assertTrue(array_compare(model.directions_from_xopt(), L.T), msg="Wrong dirns after init")
        self.assertTrue(array_compare(model.distances_to_xopt(), dists), msg="Wrong distances after init")
        self.assertTrue(array_compare(model.distances_to_xopt(include_kopt=False), np.delete(dists,kopt)), msg="Wrong distances (no xopt) after init")

        # Test factorisation
        model.factorise_system()
        self.assertTrue(model.factorisation_current, msg="Wrong factorisation_current after QR")
        self.assertAlmostEqual(overall_qr_error(L, model.Q, model.R), 0.0, msg="Bad QR factorisation")

        # Test interpolation
        for m1 in range(m):
            c, g = model.build_single_model('residual', m1, gradient_in_full_space=True)  # model based at xopt=x0
            c1, g1 = model.build_single_model('residual', m1, gradient_in_full_space=False)  # model based at xopt=x0
            self.assertTrue(array_compare(g, model.project_to_full_space(g1)), msg="Interp %g project wrong" % m1)
            self.assertTrue(array_compare(g1, model.project_to_reduced_space(g)), msg="Interp %g project wrong 2" % m1)

            for k, xk in enumerate([x0, x1, x2]):
                self.assertAlmostEqual(c + g.dot(xk - xopt), objfun(xk)[m1], msg="Interp %g wrong for k=%g" % (m1, k))
                self.assertAlmostEqual(c1 + g1.dot(model.Q.T.dot(xk - xopt)), objfun(xk)[m1], msg="Interp %g wrong for k=%g (with Q)" % (m1, k))

        # Test Lagrange polynomials
        for kbase in range(p + 1):
            c, g = model.build_single_model('lagrange', kbase, gradient_in_full_space=True)  # model based at xopt=x0
            c1, g1 = model.build_single_model('lagrange', kbase, gradient_in_full_space=False)  # model based at xopt=x0
            self.assertAlmostEqual(c, 1.0 if kbase == model.kopt else 0.0, msg="Wrong c for Lagrange interp %g" % kbase)
            self.assertAlmostEqual(c1, 1.0 if kbase == model.kopt else 0.0, msg="Wrong c for Lagrange interp %g (with Q)" % kbase)
            self.assertTrue(array_compare(g, model.project_to_full_space(g1)), msg="Lagrange %g project wrong" % kbase)
            self.assertTrue(array_compare(g1, model.project_to_reduced_space(g)), msg="Lagrange %g project wrong 2" % kbase)

            for k, xk in enumerate([x0, x1, x2]):
                self.assertAlmostEqual(c + g.dot(xk - xopt), 1.0 if k == kbase else 0.0, msg="Lagrange interp %g wrong for k=%g" % (kbase, k))
                self.assertAlmostEqual(c1 + g1.dot(model.Q.T.dot(xk - xopt)), 1.0 if k == kbase else 0.0, msg="Lagrange interp %g wrong for k=%g (with Q)" % (kbase, k))

            # We also have these wrapped functions for Lagrange polynomials (similar to interpolate_mini_models below)
            c2, g2 = model.lagrange_poly(kbase, gradient_in_full_space=True)
            c3, g3 = model.lagrange_poly(kbase, gradient_in_full_space=False)
            self.assertAlmostEqual(c, c2, msg="Wrong c for wrapped Lagrange %g" % kbase)
            self.assertAlmostEqual(c1, c3, msg="Wrong c for wrapped Lagrange %g (with Q)" % kbase)
            self.assertTrue(array_compare(g, g2), msg="Wrong g for wrapped Lagrange %g" % kbase)
            self.assertTrue(array_compare(g1, g3), msg="Wrong g for wrapped Lagrange %g (with Q)" % kbase)

        # Test full model interpolation
        is_ok, cvec, J = model.interpolate_mini_models(jac_in_full_space=True)
        is_ok1, cvec1, J1 = model.interpolate_mini_models(jac_in_full_space=False)
        self.assertTrue(is_ok, msg="Full interp failed")
        self.assertTrue(is_ok1, msg="Full interp failed (with Q)")
        self.assertTrue(array_compare(J.T, model.project_to_full_space(J1.T)), msg="Full interp J project wrong")
        self.assertTrue(array_compare(J1.T, model.project_to_reduced_space(J.T)), msg="Full interp J project wrong 2")
        for k, xk in enumerate([x0, x1, x2]):
            self.assertTrue(array_compare(cvec + J.dot(xk - xopt), objfun(xk)), msg="Full interp wrong for k=%g" % k)
            self.assertTrue(array_compare(cvec1 + J1.dot(model.Q.T.dot(xk - xopt)), objfun(xk)), msg="Full interp wrong for k=%g (with Q)" % k)

        # self.assertFalse(True, msg="abc dummy")


class TestInterpReducedDimOrthog(unittest.TestCase):
    def runTest(self):
        # Reduced-dimensional model for Powell Singular
        objfun = powell_singular
        args = ()  # optional arguments for objfun
        scaling_changes = None
        n, m = 4, 4
        p = 2
        x0 = np.array([3.0, -1.0, 0.0, 1.0])
        r0 = objfun(x0)
        xl = -1e20 * np.ones((n,))
        xu = 1e20 * np.ones((n,))
        model = InterpSet(p, x0, r0, xl, xu)
        self.assertEqual(model.n, n, msg="Wrong n after setup")
        self.assertEqual(model.m, m, msg="Wrong m after setup")
        self.assertEqual(model.p, p, msg="Wrong p after setup")
        self.assertTrue(array_compare(model.xopt(), x0), msg='Wrong xopt after setup')
        self.assertTrue(array_compare(model.ropt(), objfun(x0)), msg='Wrong ropt after setup')
        self.assertAlmostEqual(model.fopt(), sumsq(objfun(x0)), msg='Wrong fopt after setup')
        # Now add better point
        nf = 1
        maxfun = 10
        delta = 1.0
        np.random.seed(0)
        model.initialise_interp_set(delta, objfun, args, scaling_changes, nf, maxfun)
        self.assertTrue(np.all(np.isfinite(model.points)), msg="model.points still has infinite values after init")
        self.assertTrue(array_compare(model.points[0, :], x0), msg="Wrong points[0, :] after init")
        # Extracted manually, given np.random.seed(0)
        # np.set_printoptions(formatter={'float': lambda x: "{0:0.15f}".format(x)})
        # print(model.points[1,:] - x0)
        # print(model.points[2,:] - x0)
        x1 = x0 + delta * np.array([-0.606484744183092, -0.336492087249845, -0.642070192810455, -0.326642308643197])
        x2 = x0 + delta * np.array([-0.083779357809710, -0.866769259973379, 0.480508138821155, 0.103942280602377])
        self.assertAlmostEqual(np.dot(x1-x0, x2-x0), 0.0, msg="Directions not orthogonal")
        self.assertAlmostEqual(np.linalg.norm(x1 - x0), delta, msg="Direction 1 length wrong")
        self.assertAlmostEqual(np.linalg.norm(x2 - x0), delta, msg="Direction 2 length wrong")
        kopt = np.argmin(np.array([sumsq(objfun(x0)), sumsq(objfun(x1)), sumsq(objfun(x2))]))

        # Test data is ok
        for k, xk in enumerate([x0, x1, x2]):
            # print(model.xpt(k), xk)
            self.assertTrue(array_compare(model.xpt(k), xk), msg="Point %g wrong after init" % k)
            self.assertTrue(array_compare(model.rvec(k), objfun(xk)), msg="rvec %g wrong after init" % k)
            self.assertAlmostEqual(model.fval(k), sumsq(objfun(xk)), msg="fval %g wrong after init" % k)
        self.assertAlmostEqual(model.fbeg, sumsq(objfun(x0)), msg="Wrong fbeg after init")
        self.assertEqual(model.kopt, kopt, msg="Wrong kopt after init")
        xopt = model.xpt(kopt)
        self.assertTrue(array_compare(model.xopt(), xopt), msg="Wrong xopt after init")
        self.assertTrue(array_compare(model.ropt(), objfun(xopt)), msg="Wrong ropt after init")
        self.assertAlmostEqual(model.fopt(), sumsq(objfun(xopt)), msg="Wrong fopt after init")
        self.assertFalse(model.factorisation_current, msg="Wrong factorisation_current after init")

        # Test distances and directions ok
        L = np.zeros((n, p+1))
        L[:, 0] = x0 - xopt
        L[:, 1] = x1 - xopt
        L[:, 2] = x2 - xopt
        dists = np.linalg.norm(L, axis=0)  # norm of each column
        L = np.delete(L, kopt, axis=1)  # delete kopt-th column
        self.assertTrue(array_compare(model.directions_from_xopt(), L.T), msg="Wrong dirns after init")
        self.assertTrue(array_compare(model.distances_to_xopt(), dists), msg="Wrong distances after init")
        self.assertTrue(array_compare(model.distances_to_xopt(include_kopt=False), np.delete(dists,kopt)), msg="Wrong distances (no xopt) after init")

        # Test factorisation
        model.factorise_system()
        self.assertTrue(model.factorisation_current, msg="Wrong factorisation_current after QR")
        self.assertAlmostEqual(overall_qr_error(L, model.Q, model.R), 0.0, msg="Bad QR factorisation")

        # Test interpolation
        for m1 in range(m):
            c, g = model.build_single_model('residual', m1, gradient_in_full_space=True)  # model based at xopt=x0
            c1, g1 = model.build_single_model('residual', m1, gradient_in_full_space=False)  # model based at xopt=x0
            self.assertTrue(array_compare(g, model.project_to_full_space(g1)), msg="Interp %g project wrong" % m1)
            self.assertTrue(array_compare(g1, model.project_to_reduced_space(g)), msg="Interp %g project wrong 2" % m1)

            for k, xk in enumerate([x0, x1, x2]):
                self.assertAlmostEqual(c + g.dot(xk - xopt), objfun(xk)[m1], msg="Interp %g wrong for k=%g" % (m1, k))
                self.assertAlmostEqual(c1 + g1.dot(model.Q.T.dot(xk - xopt)), objfun(xk)[m1], msg="Interp %g wrong for k=%g (with Q)" % (m1, k))

        # Test Lagrange polynomials
        for kbase in range(p + 1):
            c, g = model.build_single_model('lagrange', kbase, gradient_in_full_space=True)  # model based at xopt=x0
            c1, g1 = model.build_single_model('lagrange', kbase, gradient_in_full_space=False)  # model based at xopt=x0
            self.assertAlmostEqual(c, 1.0 if kbase == model.kopt else 0.0, msg="Wrong c for Lagrange interp %g" % kbase)
            self.assertAlmostEqual(c1, 1.0 if kbase == model.kopt else 0.0, msg="Wrong c for Lagrange interp %g (with Q)" % kbase)
            self.assertTrue(array_compare(g, model.project_to_full_space(g1)), msg="Lagrange %g project wrong" % kbase)
            self.assertTrue(array_compare(g1, model.project_to_reduced_space(g)), msg="Lagrange %g project wrong 2" % kbase)

            for k, xk in enumerate([x0, x1, x2]):
                self.assertAlmostEqual(c + g.dot(xk - xopt), 1.0 if k == kbase else 0.0, msg="Lagrange interp %g wrong for k=%g" % (kbase, k))
                self.assertAlmostEqual(c1 + g1.dot(model.Q.T.dot(xk - xopt)), 1.0 if k == kbase else 0.0, msg="Lagrange interp %g wrong for k=%g (with Q)" % (kbase, k))

            # We also have these wrapped functions for Lagrange polynomials (similar to interpolate_mini_models below)
            c2, g2 = model.lagrange_poly(kbase, gradient_in_full_space=True)
            c3, g3 = model.lagrange_poly(kbase, gradient_in_full_space=False)
            self.assertAlmostEqual(c, c2, msg="Wrong c for wrapped Lagrange %g" % kbase)
            self.assertAlmostEqual(c1, c3, msg="Wrong c for wrapped Lagrange %g (with Q)" % kbase)
            self.assertTrue(array_compare(g, g2), msg="Wrong g for wrapped Lagrange %g" % kbase)
            self.assertTrue(array_compare(g1, g3), msg="Wrong g for wrapped Lagrange %g (with Q)" % kbase)

        # Test full model interpolation
        is_ok, cvec, J = model.interpolate_mini_models(jac_in_full_space=True)
        is_ok1, cvec1, J1 = model.interpolate_mini_models(jac_in_full_space=False)
        self.assertTrue(is_ok, msg="Full interp failed")
        self.assertTrue(is_ok1, msg="Full interp failed (with Q)")
        self.assertTrue(array_compare(J.T, model.project_to_full_space(J1.T)), msg="Full interp J project wrong")
        self.assertTrue(array_compare(J1.T, model.project_to_reduced_space(J.T)), msg="Full interp J project wrong 2")
        for k, xk in enumerate([x0, x1, x2]):
            self.assertTrue(array_compare(cvec + J.dot(xk - xopt), objfun(xk)), msg="Full interp wrong for k=%g" % k)
            self.assertTrue(array_compare(cvec1 + J1.dot(model.Q.T.dot(xk - xopt)), objfun(xk)), msg="Full interp wrong for k=%g (with Q)" % k)

        # self.assertFalse(True, msg="abc dummy")


class TestChangeAppendRemove(unittest.TestCase):
    def runTest(self):
        # Reduced-dimensional model for Powell Singular
        objfun = powell_singular
        args = ()  # optional arguments for objfun
        scaling_changes = None
        n, m = 4, 4
        p = 2
        x0 = np.array([3.0, -1.0, 0.0, 1.0])
        r0 = objfun(x0)
        xl = -1e20 * np.ones((n,))
        xu = 1e20 * np.ones((n,))
        model = InterpSet(p, x0, r0, xl, xu)
        nf = 1
        maxfun = 10
        delta = 1.0
        np.random.seed(0)
        model.initialise_interp_set(delta, objfun, args, scaling_changes, nf, maxfun)
        # From above...
        x1 = x0 + delta * np.array([-0.606484744183092, -0.336492087249845, -0.642070192810455, -0.326642308643197])
        x2 = x0 + delta * np.array([-0.083779357809710, -0.866769259973379, 0.480508138821155, 0.103942280602377])

        # Step 1 - add a new point
        x3 = x0 + np.array([1.0, 1.0, 1.0, -1.0])
        r3 = objfun(x3)
        model.append_point(x3, r3)
        p = 3
        kopt = 0  # new point is worse
        pts = [x0, x1, x2, x3]
        generic_model_check(self, model, p, pts, objfun, kopt, mystr="after append")

        # Step 2 - add a new, better point
        x4 = np.array([1.0, 0.0, 0.0, 0.0])
        r4 = objfun(x4)
        model.append_point(x4, r4)
        p = 4
        kopt = 4  # new point is best so far
        pts = [x0, x1, x2, x3, x4]
        generic_model_check(self, model, p, pts, objfun, kopt, mystr="after append 2")

        # Step 3 - remove an old point
        model.remove_point(2, check_not_kopt=True)
        p = 3
        kopt = 3  # new point is best so far
        pts = [x0, x1, x3, x4]
        generic_model_check(self, model, p, pts, objfun, kopt, mystr="after remove 1", interp_thresh=1e-12)

        # Step 3a - remove the best point (fails)
        self.assertRaises(AssertionError, model.remove_point, 3)

        # Step 4 - change a point (not an improvement)
        x5 = 3.0 * np.ones((n,))
        r5 = objfun(x5)
        model.change_point(0, x5, r5)
        p = 3
        kopt = 3  # new point is best so far
        pts = [x5, x1, x3, x4]
        generic_model_check(self, model, p, pts, objfun, kopt, mystr="after change 1", interp_thresh=1e-12)

        # Step 4 - change a point (improvement)
        x6 = np.array([0.0, 0.1, 0.0, 0.0])
        r6 = objfun(x6)
        model.change_point(2, x6, r6)
        p = 3
        kopt = 2  # new point is best so far
        pts = [x5, x1, x6, x4]
        generic_model_check(self, model, p, pts, objfun, kopt, mystr="after change 2", interp_thresh=1e-12)

        # self.assertFalse(True, msg="abc dummy")

if __name__ == '__main__':
    unittest.main()