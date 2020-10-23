"""
Various useful functions


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
import sys


__all__ = ['sumsq', 'eval_least_squares_objective', 'model_value',
           'random_directions_within_bounds', 'apply_scaling', 'remove_scaling',
           'random_orthog_directions_new_subspace_within_bounds']


def sumsq(x):
    # There are several ways to calculate sum of squares of a vector:
    #   np.dot(x,x)
    #   np.sum(x**2)
    #   np.sum(np.square(x))
    #   etc.
    # Using the timeit routine, it seems like dot(x,x) is ~3-4x faster than other methods
    return np.dot(x, x)


def eval_least_squares_objective(objfun, x, args=(), verbose=True, eval_num=0, pt_num=None, full_x_thresh=6, check_for_overflow=True):
    # Evaluate least squares function
    fvec = objfun(x, *args)

    if check_for_overflow:
        try:
            if np.max(np.abs(fvec)) >= np.sqrt(sys.float_info.max):
                f = sys.float_info.max
            else:
                f = sumsq(fvec)  # objective = sum(ri^2) [no 1/2 factor at front]
        except OverflowError:
            f = sys.float_info.max
    else:
        f = sumsq(fvec)

    if verbose:
        if len(x) < full_x_thresh:
            if pt_num is not None:
                logging.info("Function eval %i at point %i has f = %.15g at x = " % (eval_num, pt_num, f) + str(x))
            else:
                logging.info("Function eval %i has f = %.15g at x = " % (eval_num, f) + str(x))
        else:
            if pt_num is not None:
                logging.info("Function eval %i at point %i has f = %.15g at x = [...]" % (eval_num, pt_num, f))
            else:
                logging.info("Function eval %i has f = %.15g at x = [...]" % (eval_num, f))

    return fvec, f


def model_value(g, hess, s):
    # Calculate model value (s^T * g + 0.5* s^T * H * s) = s^T * (gopt + 0.5 * H*s)
    assert g.shape == s.shape, "g and s have incompatible sizes"
    Hs = hess.vec_mul(s)
    return np.dot(s, g + 0.5*Hs)


def get_scale(dirn, delta, lower, upper):
    scale = delta
    for j in range(len(dirn)):
        if dirn[j] < 0.0:
            scale = min(scale, lower[j] / dirn[j])
        elif dirn[j] > 0.0:
            scale = min(scale, upper[j] / dirn[j])
    return scale


def random_orthog_directions_new_subspace_within_bounds(num_pts, delta, lower, upper, Q=None, box_bound_thresh=0.01):
    # Generate num_pts random directions d1, d2, ...
    # so that lower <= d1 <= upper and ||d1|| ~ delta [perhaps not equal if constraint active]
    # Q is n*p matrix with orthonormal columns
    # Want to generate the points such that d is orthogonal to each column of Q (if possible)
    # Need num_pts <= n-p to maintain orthogonality
    # If within box_bound_thresh*delta of any bound, drop orthogonality requirement (too hard to do properly)
    n = len(lower)
    if Q is not None:
        p = Q.shape[1]
        assert Q.shape == (n, p), "Q must have n rows"
    else:
        p = 0
    assert lower.shape == (n,), "lower must be a vector"
    assert upper.shape == (n,), "lower and upper have incompatible sizes"
    assert np.min(upper) >= -1e-15, "upper must be non-negative"
    assert np.max(lower) <= 1e-15, "lower must be non-positive"
    assert np.min(upper - lower) > 0.0, "upper must be > lower"
    assert delta > 0, "delta must be strictly positive"
    assert num_pts > 0, "num_pts must be strictly positive"
    assert num_pts <= n - p, "num_pts must be <= n-p (p=number of columns of Q)"

    results = np.zeros((num_pts, n))  # save space for results

    # Check if near the boundary
    if np.min(upper) < box_bound_thresh*delta or np.max(lower) > box_bound_thresh*delta:
        return random_directions_within_bounds(num_pts, delta, lower, upper)

    # Otherwise, we are in the strict interior
    A = np.random.normal(size=(n, num_pts))
    if Q is not None:
        A = A - np.dot(Q, np.dot(Q.T, A))  # make orthogonal to columns of Q
    A_Q, _ = np.linalg.qr(A, mode='reduced')  # make directions orthonormal
    for i in range(num_pts):
        scale = get_scale(A_Q[:,i], delta, lower, upper)  # from above boundary check, scale should be >= 0.01
        results[i, :] = np.maximum(np.minimum(scale * A_Q[:, i], upper), lower)  # double-check bounds satisfied

    return results


def random_directions_within_bounds(num_pts, delta, lower, upper):
    # Generate num_pts random directions d1, d2, ...
    # so that lower <= d1 <= upper and ||d1|| ~ delta [perhaps not equal if constraint active]
    # Directions should be completely random (as much as possible while staying within bounds)
    n = len(lower)
    assert lower.shape == (n,), "lower must be a vector"
    assert upper.shape == (n,), "lower and upper have incompatible sizes"
    assert np.min(upper) >= -1e-15, "upper must be non-negative"
    assert np.max(lower) <= 1e-15, "lower must be non-positive"
    assert np.min(upper - lower) > 0.0, "upper must be > lower"
    assert delta > 0, "delta must be strictly positive"
    assert num_pts > 0, "num_pts must be strictly positive"
    results = np.zeros((n, num_pts))  # save space for results
    # Find the active set
    idx_l = (lower == 0)
    idx_u = (upper == 0)
    active = np.logical_or(idx_l, idx_u)
    # inactive = np.logical_not(active)
    nactive = np.sum(active)
    # ninactive = n - nactive
    idx_active = np.where(active)[0]  # indices of active constraints
    for i in range(num_pts):
        dirn = np.random.normal(size=(n,))
        for j in range(nactive):
            idx = idx_active[j]
            sign = 1.0 if idx_l[idx] else -1.0  # desired sign of direction shift
            if dirn[idx]*sign < 0.0:
                dirn[idx] *= -1.0
        dirn = dirn / np.linalg.norm(dirn)
        scale = get_scale(dirn, delta, lower, upper)
        results[:, i] = dirn * scale
    # Finally, scale by delta and make sure everything is within bounds
    for i in range(num_pts):
        results[:, i] = np.maximum(np.minimum(results[:, i], upper), lower)
    return results.T


def apply_scaling(x_raw, scaling_changes):
    if scaling_changes is None:
        return x_raw
    shift, scale = scaling_changes
    return (x_raw - shift) / scale


def remove_scaling(x_scaled, scaling_changes):
    if scaling_changes is None:
        return x_scaled
    shift, scale = scaling_changes
    return shift + x_scaled * scale

