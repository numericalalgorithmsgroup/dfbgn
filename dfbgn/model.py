"""
Model
====

Maintain a class which represents an interpolating set, and its corresponding linear models
for each residual.
This class should calculate the various geometric quantities of interest to us.


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

import logging
import numpy as np
import scipy.linalg as linalg

from .exit_information import *
from .util import sumsq, random_orthog_directions_new_subspace_within_bounds, eval_least_squares_objective, remove_scaling

__all__ = ['InterpSet', 'project_to_full_space', 'project_to_reduced_space']


# TODO  =====================
# TODO  1. Build rest of solver + tests (must call initialise_interp_set immediately after constructor)
# TODO  2. Do QR updating, drop factorisation_current flag? (write unit tests for this too!)
# TODO  =====================

class InterpSet(object):
    def __init__(self, pinit, x0, r0, xl, xu, n=None, m=None, abs_tol=1e-12, rel_tol=1e-20):
        if n is None:
            n = len(x0)
        if m is None:
            m = len(r0)
        assert 1 <= pinit <= n, "Jacobian size p must be in [1..n], got p=%g" % pinit
        assert x0.shape == (n,), "x0 has wrong shape (got %s, expect (%g,))" % (str(x0.shape), n)
        assert xl.shape == (n,), "xl has wrong shape (got %s, expect (%g,))" % (str(xl.shape), n)
        assert xu.shape == (n,), "xu has wrong shape (got %s, expect (%g,))" % (str(xu.shape), n)
        assert r0.shape == (m,), "r0 has wrong shape (got %s, expect (%g,))" % (str(r0.shape), m)
        self.n = n
        self.m = m
        self.p = pinit

        # Interpolation points
        self.xl = xl
        self.xu = xu
        self.points = np.inf * np.ones((self.p + 1, n))  # interpolation points
        self.points[0, :] = x0
        self.point_ages = np.zeros((self.p + 1,), dtype=int)  # how many iterations has each point been here for?

        # Function values
        self.rvals = np.inf * np.ones((self.p + 1, m))  # residuals for each xpt
        self.rvals[0, :] = r0.copy()
        self.fvals = np.inf * np.ones((self.p + 1,))  # overall objective value for each xpt
        self.fvals[0] = sumsq(r0)
        self.kopt = 0  # index of current iterate (should be best value so far)
        self.fbeg = self.fvals[0]  # f(x0), saved to check for sufficient reduction

        # Termination criteria
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol

        # Saved point - always check this value before quitting solver
        self.xsave = None
        self.rsave = None
        self.fsave = None

        # Factorisation of interpolation matrix
        self.factorisation_current = False  # TODO to remove?
        self.Q = None
        self.R = None

    def xopt(self):
        return self.xpt(self.kopt)

    def ropt(self):
        return self.rvec(self.kopt)

    def fopt(self):
        return self.fval(self.kopt)

    def xpt(self, k):
        assert 0 <= k <= self.p, "Invalid index %g" % k
        return np.minimum(np.maximum(self.xl, self.points[k, :]), self.xu)

    def rvec(self, k):
        assert 0 <= k <= self.p, "Invalid index %g" % k
        return self.rvals[k, :]

    def fval(self, k):
        assert 0 <= k <= self.p, "Invalid index %g" % k
        return self.fvals[k]

    def initialise_interp_set(self, delta, objfun, args, scaling_changes, nf, maxfun, use_coord_directions=False,
                              box_bound_thresh=0.01, full_x_thresh=6, check_for_overflow=True):
        assert delta > 0.0, "delta must be strictly positive"

        # Called upon initialisation only
        x0 = self.xpt(0)

        # Get directions
        dirns = np.zeros((self.p, self.n))  # each row is an offset from x0
        if use_coord_directions:
            idx_choices = np.random.choice(self.n, size=self.p, replace=False)
            for i in range(self.p):
                idx = idx_choices[i]
                # Decide whether to do +delta or -delta (usually via a coin toss)
                upper_gap = self.xu[idx] - x0[idx]
                lower_gap = x0[idx] - self.xl[idx]
                if min(lower_gap, upper_gap) <= box_bound_thresh * delta:
                    # If very close to boundary on at least one side, just go in the larger direction we can step
                    step = -min(delta, lower_gap) if lower_gap > upper_gap else min(delta, upper_gap)
                else:
                    step = min(delta, upper_gap) if np.random.random() >= 0.5 else -min(delta, lower_gap)

                dirns[i, idx] = step
        else:
            dirns = random_orthog_directions_new_subspace_within_bounds(self.p, delta, self.xl - x0, self.xu - x0, Q=None,
                                                                        box_bound_thresh=box_bound_thresh)

        # Evaluate objective at these points
        exit_info = None
        for i in range(self.p):
            x = x0 + dirns[i, :]  # point to evaluate

            if nf >= maxfun:
                exit_info = ExitInformation(EXIT_MAXFUN_WARNING, "Objective has been called MAXFUN times")
                break  # quit

            nf += 1
            r, f = eval_least_squares_objective(objfun, remove_scaling(x, scaling_changes), args=args,
                                                eval_num=nf, full_x_thresh=full_x_thresh, check_for_overflow=check_for_overflow)

            if sumsq(r) < self.min_objective_value():
                self.save_point(x, r)
                exit_info = ExitInformation(EXIT_SUCCESS, "Objective is sufficiently small")
                break  # quit

            self.points[i + 1, :] = x
            self.rvals[i + 1, :] = r
            self.fvals[i + 1] = sumsq(r)
            self.point_ages[i + 1] = 0

        # Choose kopt as best value so far
        self.kopt = np.argmin(self.fvals)

        self.factorisation_current = False  # TODO build initial QR here?
        return exit_info, nf

    def directions_from_xopt(self):
        dirns = self.points - self.xopt()  # subtract from each row of matrix (numpy does automatically)
        return np.delete(dirns, self.kopt, axis=0)  # drop kopt-th entry / kopt-th row

    def distances_to_xopt(self, include_kopt=True):
        dirns = self.points - self.xopt()  # subtract from each row of matrix (numpy does automatically)
        distances = np.linalg.norm(dirns, axis=1)  # norm of each row
        if include_kopt:
            return distances
        else:
            return np.delete(distances, self.kopt)  # drop kopt-th entry

    def change_point(self, k, x, rvec, check_not_kopt=True):
        # Update point k to x (w.r.t. xbase), with residual values fvec
        assert 0 <= k <= self.p, "Invalid index %g" % k
        if check_not_kopt:
            assert k != self.kopt, "Cannot remove current iterate from interpolation set"

        self.points[k, :] = x
        self.rvals[k, :] = rvec
        self.fvals[k] = sumsq(rvec)
        self.point_ages[k] = 0
        self.factorisation_current = False  # TODO update QR!

        if self.fvals[k] < self.fvals[self.kopt]:
            self.kopt = k  # TODO different update of QR
        return

    def append_point(self, x, rvec):
        assert self.p < self.n, "Cannot append points to full-dimensional interpolation set"
        self.points = np.append(self.points, x.reshape((1, self.n)), axis=0)  # append row to xpt
        self.rvals = np.append(self.rvals, rvec.reshape((1, self.m)), axis=0)  # append row to fval_v
        f = sumsq(rvec)
        self.fvals = np.append(self.fvals, f)  # append entry to fvals
        self.point_ages = np.append(self.point_ages, 0)  # append 0 to point_ages
        self.p += 1

        if f < self.fopt():
            self.kopt = self.p  # TODO different QR update!

        self.factorisation_current = False  # TODO update QR!
        return

    def remove_point(self, k, check_not_kopt=True):
        assert 0 <= k <= self.p, "Invalid index %g" % k
        assert self.p >= 1, "Need to keep at least one point (iterate) in interpolation set"
        if check_not_kopt:
            assert k != self.kopt, "Cannot remove current iterate from interpolation set"

        self.points = np.delete(self.points, k, axis=0)  # delete row
        self.rvals = np.delete(self.rvals, k, axis=0)
        self.fvals = np.delete(self.fvals, k)
        self.point_ages = np.delete(self.point_ages, k)
        self.p -= 1

        # Even if k!=kopt, need to do this (e.g. deleted point before kopt)
        self.kopt = np.argmin(self.fvals)  # make sure kopt is always the best value we have

        self.factorisation_current = False  # TODO update QR!
        return

    def save_point(self, x, rvec):
        f = sumsq(rvec)
        if self.fsave is None or f <= self.fsave:
            self.xsave = x.copy()
            self.rsave = rvec.copy()
            self.fsave = f
            # self.jacsave = self.model_jac.copy()
            return True
        else:
            return False  # this value is worse than what we have already - didn't save

    def get_final_results(self):
        # Return x and fval for optimal point (either from xsave+fsave or kopt)
        if self.fsave is None or self.fopt() <= self.fsave:  # optimal has changed since xsave+fsave were last set
            return self.xopt(), self.ropt(), self.fopt()
        else:
            return self.xsave, self.rsave, self.fsave

    def min_objective_value(self):
        # Get termination criterion for f small: f <= abs_tol or f <= rel_tol * f0
        return max(self.abs_tol, self.rel_tol * self.fbeg)

    def factorise_system(self):  # TODO don't need now (but keep for occasional resetting to avoid errors accumulating?)
        if not self.factorisation_current:
            if self.p > 0:
                dirns = self.directions_from_xopt()  # size (p, n)
                self.Q, self.R = linalg.qr(dirns.T, mode='economic')  # Q is n*p, R is p*p
            else:
                self.Q, self.R = None, None
            self.factorisation_current = True  # TODO need this variable if updating?
        return

    def build_single_model(self, interp_vals, interp_idx, gradient_in_full_space=False):
        assert interp_vals == 'residual' or interp_vals == 'lagrange'  # what type of interpolation to do?
        interp_resid = (interp_vals == 'residual')
        vals_to_interpolate = None  # move scope outside, will be vector of length p+1
        if interp_resid:
            # Which residual term to interpolate
            assert 0 <= interp_idx <= self.m - 1, "build_single_model for residuals: interp_idx must be in [0..m-1], got %g" % interp_idx
            vals_to_interpolate = self.rvals[:, interp_idx]
        else:
            # Which point to build Lagrange polynomial for
            assert 0 <= interp_idx <= self.p, "build_single_model for Lagrange: interp_idx must be in [0..p], got %g" % interp_idx
            vals_to_interpolate = np.zeros((self.p + 1,))
            vals_to_interpolate[interp_idx] = 1.0  # rest are zero

        self.factorise_system()  # ensure factorisation up-to-date  # TODO needed?

        c = vals_to_interpolate[self.kopt]
        rhs = np.delete(vals_to_interpolate - c, self.kopt)  # drop kopt-th entry
        g = linalg.solve_triangular(self.R, rhs, trans='T')  # R.T \ rhs -> gradient in reduced space

        # model based at xopt
        if gradient_in_full_space:
            return c, self.project_to_full_space(g)
        else:
            return c, g

    def interpolate_mini_models(self, jac_in_full_space=False):
        c = np.zeros((self.m,))
        J = np.zeros((self.m, self.n if jac_in_full_space else self.p))

        try:
            for i in range(self.m):
                c[i], J[i, :] = self.build_single_model('residual', i, gradient_in_full_space=jac_in_full_space)
        except:
            return False, None, None  # flag error

        if not (np.all(np.isfinite(c)) and np.all(np.isfinite(J))):
            return False, None, None  # flag error

        return True, c, J  # model based at xopt

    def lagrange_poly(self, k, gradient_in_full_space=False):
        assert 0 <= k <= self.p, "Invalid index %g" % k
        c, g = self.build_single_model('lagrange', k, gradient_in_full_space=gradient_in_full_space)
        return c, g  # based at xopt

    def poisedness_of_each_point(self, delta=None, d=None):
        # Return the poisedness of each point in the interpolation set
        # if delta is set, then calculate the maximum of |L(xopt+s)| over all ||s||<=delta for each point
        # if d is set, calculate |L(xopt+d)| for each point
        assert (delta is not None or d is not None) and (delta is None or d is None), "Must specify exactly one of delta and d"
        poisedness = np.zeros((self.p + 1,))
        for k in range(self.p + 1):
            c, g = self.lagrange_poly(k, gradient_in_full_space=False) # based at xopt
            if delta is not None:
                normg = np.linalg.norm(g)
                # Maximiser/minimiser of (linear) Lagrange poly in B(xopt, delta) is s = +/- delta/||g|| * g
                # Value is c +/- delta*||g||
                poisedness[k] = max(abs(c + delta * normg), abs(c - delta * normg))
            else:
                poisedness[k] = abs(c + np.dot(g, d))
        return poisedness

    def poisedness_in_reduced_space(self, delta):
        # Calculate poisedness constant in ball around xopt (everything in reduced space)
        # lmax = None
        # for k in range(self.p + 1):
        #     c, g = self.lagrange_poly(k, gradient_in_full_space=False)
        #     # Maximiser/minimiser of (linear) Lagrange poly in B(xopt, delta) is s = +/- delta/||g|| * g
        #     # Value is c +/- delta*||g||
        #     normg = np.linalg.norm(g)
        #     l = max(abs(c + delta*normg), abs(c - delta*normg))
        #     if lmax is None or l > lmax:
        #         lmax = l
        # return lmax
        return np.max(self.poisedness_of_each_point(delta=delta))

    def project_to_full_space(self, x):
        assert self.factorisation_current, "Cannot project, factorisation is invalid"
        return project_to_full_space(self.Q, x)

    def project_to_reduced_space(self, x):
        assert self.factorisation_current, "Cannot project, factorisation is invalid"
        return project_to_reduced_space(self.Q, x)


def project_to_full_space(Q, x):
    return Q.dot(x)


def project_to_reduced_space(Q, x):
    return Q.T.dot(x)