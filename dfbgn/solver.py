"""
Main solver
===========

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
# import scipy.optimize as sp_opt
import warnings

from .diagnostic_info import *
from .exit_information import *
from .hessian import Hessian
from .model import *
from .params import ParameterList
from .trust_region import trsbox
from .util import sumsq, apply_scaling, remove_scaling, eval_least_squares_objective, random_orthog_directions_new_subspace_within_bounds


__all__ = ['solve']


def update_tr(delta, rho, ratio, norm_sk, norm_gk, norm_gk_Hk, params):
    update_method = params("tr_radius.update_method")
    if update_method == 'bobyqa':
        if ratio < params("tr_radius.eta1"):  # ratio < 0.1
            iter_type = ITER_ACCEPTABLE if ratio > 0.0 else ITER_UNSUCCESSFUL
            delta = min(params("tr_radius.gamma_dec") * delta, norm_sk)

        elif ratio <= params("tr_radius.eta2"):  # 0.1 <= ratio <= 0.7
            iter_type = ITER_SUCCESSFUL
            delta = max(params("tr_radius.gamma_dec") * delta, norm_sk)
            # if norm_gk >= 10*delta:  # try a very successful update
            #     delta = min(max(params("tr_radius.gamma_inc") * delta, params("tr_radius.gamma_inc_overline") * norm_sk),
            #         params("tr_radius.delta_max"))

        else:  # (ratio > eta2 = 0.7)
            iter_type = ITER_VERY_SUCCESSFUL
            delta = min(max(params("tr_radius.gamma_inc") * delta, params("tr_radius.gamma_inc_overline") * norm_sk),
                        params("tr_radius.delta_max"))

    elif update_method == 'bobyqa2':  # TODO slight variation for very successful steps
        if ratio < params("tr_radius.eta1"):  # ratio < 0.1
            iter_type = ITER_ACCEPTABLE if ratio > 0.0 else ITER_UNSUCCESSFUL
            delta = min(params("tr_radius.gamma_dec") * delta, norm_sk)

        elif ratio <= params("tr_radius.eta2"):  # 0.1 <= ratio <= 0.7
            iter_type = ITER_SUCCESSFUL
            delta = max(params("tr_radius.gamma_dec") * delta, norm_sk)

        else:  # (ratio > eta2 = 0.7)
            iter_type = ITER_VERY_SUCCESSFUL
            delta_test = min(params("tr_radius.gamma_inc") * delta, params("tr_radius.gamma_inc_overline") * norm_sk)
            # Take delta_test and cap/floor between (current) delta and delta_max
            delta = min(max(delta, delta_test), params("tr_radius.delta_max"))

    elif update_method == 'scipy':
        # Method from scipy.optimize.leastsq: scipy.optimize._lsq.common.py -> function update_tr_radius (line 224)
        if ratio < params("tr_radius.eta1"):  # ratio < 0.1
            iter_type = ITER_ACCEPTABLE if ratio > 0.0 else ITER_UNSUCCESSFUL
            delta = params("tr_radius.gamma_dec") * delta

        elif ratio <= params("tr_radius.eta2"):  # 0.1 <= ratio <= 0.7
            iter_type = ITER_SUCCESSFUL
            delta = delta  # no change

        else:  # (ratio > eta2 = 0.7)
            iter_type = ITER_VERY_SUCCESSFUL
            if norm_sk >= 0.95 * delta:  # only increase delta if near trust region boundary
                delta = min(params("tr_radius.gamma_inc") * delta, params("tr_radius.delta_max"))
            else:
                delta = delta  # no change if at interior point

    elif update_method == 'test':  # modified BOBYQA version
        if ratio < params("tr_radius.eta1"):  # ratio < 0.1  # TODO this has been modified
            iter_type = ITER_ACCEPTABLE if ratio > 0.0 else ITER_UNSUCCESSFUL
            if ratio < 0:
                delta = min(params("tr_radius.gamma_dec") * delta, norm_sk)
            else:
                delta = delta  # don't decrease delta when ratio > 0

        elif ratio <= params("tr_radius.eta2"):  # 0.1 <= ratio <= 0.7    # TODO this has been modified
            iter_type = ITER_SUCCESSFUL
            # delta = max(params("tr_radius.gamma_dec") * delta, norm_sk)
            delta = delta  # don't decrease delta when ratio > 0

        else:  # (ratio > eta2 = 0.7)
            iter_type = ITER_VERY_SUCCESSFUL
            delta = min(max(params("tr_radius.gamma_inc") * delta, params("tr_radius.gamma_inc_overline") * norm_sk),
                        params("tr_radius.delta_max"))

    elif update_method == 'storm':
        if ratio < params("tr_radius.eta1"):  # ratio < 0.1
            iter_type = ITER_ACCEPTABLE if ratio > 0.0 else ITER_UNSUCCESSFUL
            delta = min(params("tr_radius.gamma_dec") * delta, norm_sk)

        else:  # (ratio > eta1 = 0.1)
            iter_type = ITER_VERY_SUCCESSFUL if ratio > params("tr_radius.eta2") else ITER_SUCCESSFUL
            if norm_gk_Hk >= params("tr_radius.eta2") * delta:
                delta = min(params("tr_radius.gamma_inc") * delta, params("tr_radius.delta_max"))
            else:
                delta = min(params("tr_radius.gamma_dec") * delta, norm_sk)  # shrink TR radius

    else:
        raise RuntimeError("Unknown parameter tr_radius.update_method: %s" % update_method)

    if params("tr_radius.use_rho"):
        delta = max(delta, rho)
    return delta, iter_type


def done_with_current_rho(rho, last_successful_iter, current_iter, diffs, crvmin, in_safety=False):
    # crvmin comes from trust region step

    # Wait at least 3 iterations between reductions of rho
    if current_iter <= last_successful_iter + 5:
        return False

    errbig = max(diffs)
    frhosq = 0.125 * rho ** 2
    # if in_safety and crvmin > 0.0 and errbig > frhosq * crvmin:
    #     logging.debug("Not reducing because of this (crvmin = %g)" % crvmin)
    #     return False

    # Otherwise...
    return True


def reduce_rho(delta, rho, rhoend, params):
    if delta > 1.5*rho:  # nothing needed if delta > rho
        return delta, rho

    alpha1 = params("tr_radius.alpha1")
    alpha2 = params("tr_radius.alpha2")
    ratio = rho / rhoend
    if ratio <= 16.0:
        new_rho = rhoend
    elif ratio <= 250.0:
        new_rho = np.sqrt(ratio) * rhoend  # geometric average of rho and rhoend
    else:
        new_rho = alpha1 * rho

    new_delta = max(alpha2 * rho, new_rho)  # self.rho = old rho
    return new_delta, new_rho


def build_fixed_block_model(model, fixed_block, delta, objfun, args, scaling_changes, nf, maxfun, params):
    # Grow the block size until it reaches a desired value
    exit_info = None

    if model.p > fixed_block:
        # Remove the correct number of points
        while model.p > fixed_block:
            k = np.argmax(model.distances_to_xopt(include_kopt=True))  # include_kopt, otherwise index mismatch
            model.remove_point(k, check_not_kopt=True)

    while model.p < fixed_block:
        # Find a new point orthogonal to existing directions
        model.factorise_system()  # now can get model.Q (None if no directions stored currently, which is ok for next line)
        d = random_orthog_directions_new_subspace_within_bounds(1, delta, model.xl - model.xopt(), model.xu - model.xopt(),
                                                                Q=model.Q, box_bound_thresh=params("geometry.direcion_box_bound_thresh"))[0, :]
        xnew = model.xopt() + d

        # Evaluate objective at xnew
        nf += 1
        rnew, fnew = eval_least_squares_objective(objfun, remove_scaling(xnew, scaling_changes), args=args, eval_num=nf,
                                                  full_x_thresh=params("logging.n_to_print_whole_x_vector"),
                                                  check_for_overflow=params("general.check_objfun_for_overflow"))

        if fnew <= model.min_objective_value():
            model.save_point(xnew, rnew)  # save, since this point was an improvement
            exit_info = ExitInformation(EXIT_SUCCESS, "Objective is sufficiently small")
            break  # quit

        if nf >= maxfun:
            model.save_point(xnew, rnew)  # save, just in case this point was an improvement
            exit_info = ExitInformation(EXIT_MAXFUN_WARNING, "Objective has been called MAXFUN times")
            break  # quit

        # Append xnew to model
        model.append_point(xnew, rnew)

    return exit_info, model, nf


def build_adaptive_model(model, delta, objfun, args, scaling_changes, nf, maxfun, params):
    # Adaptively grow the block size until some condition is met
    exit_info = None

    initial_p = model.p  # how many directions did we start with?

    max_block = min(model.n, params("adaptive.pmax"))
    # if nf >= 9*(model.n + 1) and np.random.uniform() <= 0.9:
    #     max_block = 0.5*model.n

    while initial_p <= model.p < max_block:  # add at least one point, continue until we get a full model
        # Find a new point orthogonal to existing directions
        model.factorise_system()  # now can get model.Q (None if no directions stored currently, which is ok for next line)
        d = random_orthog_directions_new_subspace_within_bounds(1, delta, model.xl - model.xopt(), model.xu - model.xopt(),
                                                                Q=model.Q, box_bound_thresh=params("geometry.direcion_box_bound_thresh"))[0,:]
        xnew = model.xopt() + d

        # Evaluate objective at xnew
        nf += 1
        rnew, fnew = eval_least_squares_objective(objfun, remove_scaling(xnew, scaling_changes), args=args,
                                                  eval_num=nf,
                                                  full_x_thresh=params("logging.n_to_print_whole_x_vector"),
                                                  check_for_overflow=params("general.check_objfun_for_overflow"))

        if fnew <= model.min_objective_value():
            model.save_point(xnew, rnew)  # save, since this point was an improvement
            exit_info = ExitInformation(EXIT_SUCCESS, "Objective is sufficiently small")
            break  # quit

        if nf >= maxfun:
            model.save_point(xnew, rnew)  # save, just in case this point was an improvement
            exit_info = ExitInformation(EXIT_MAXFUN_WARNING, "Objective has been called MAXFUN times")
            break  # quit

        # Append xnew to model
        model.append_point(xnew, rnew)

        # Calculate adaptive condition to see if we can finish
        fk = model.fopt()
        interp_ok, ck, Jk = model.interpolate_mini_models(jac_in_full_space=False)
        if not interp_ok:
            exit_info = ExitInformation(EXIT_LINALG_ERROR, "Failed to build interpolation model while building adaptive model")
            break  # quit
        gk = 2.0 * np.dot(Jk.T, ck)

        adaptive_method = params("adaptive.condition")
        if params("adaptive.adjust_alpha"):
            alpha = params("adaptive.alpha") * (max_block - model.p) / max_block
        else:
            alpha = params("adaptive.alpha")

        this_block_ok = None
        if adaptive_method == 'normg':
            this_block_ok = (np.linalg.norm(gk) >= alpha * delta)

        elif adaptive_method == 'modelred':
            Hk = Hessian(model.p, vals=2.0 * np.dot(Jk.T, Jk))
            sk_red, _, _ = trsbox(np.zeros((model.p,)), gk, Hk, -1e20 * np.ones((model.p,)), 1e20 * np.ones((model.p,)), delta)
            model_value = fk + np.dot(sk_red, gk + 0.5 * Hk.vec_mul(sk_red))
            this_block_ok = (model_value <= alpha * fk)

        elif adaptive_method == 'normgh':
            J_norm2 = np.linalg.norm(Jk, ord=2)
            H_norm2 = 2.0 * J_norm2**2
            g_norm = np.linalg.norm(gk)
            this_block_ok = (g_norm / max(1.0, H_norm2) >= alpha * delta)

        elif adaptive_method == 'delta':
            Hk = Hessian(model.p, vals=2.0 * np.dot(Jk.T, Jk))
            sk_red, _, _ = trsbox(np.zeros((model.p,)), gk, Hk, -1e20 * np.ones((model.p,)), 1e20 * np.ones((model.p,)), delta)
            pred_reduction = -np.dot(sk_red, gk + 0.5 * Hk.vec_mul(sk_red))
            this_block_ok = (pred_reduction >= alpha * delta)

        elif adaptive_method == 'deltasq':
            Hk = Hessian(model.p, vals=2.0 * np.dot(Jk.T, Jk))
            sk_red, _, _ = trsbox(np.zeros((model.p,)), gk, Hk, -1e20 * np.ones((model.p,)), 1e20 * np.ones((model.p,)), delta)
            pred_reduction = -np.dot(sk_red, gk + 0.5 * Hk.vec_mul(sk_red))
            this_block_ok = (pred_reduction >= alpha * delta**2)

        elif adaptive_method == 'deltasqcurv':
            Hk = Hessian(model.p, vals=2.0 * np.dot(Jk.T, Jk))
            sk_red, _, _ = trsbox(np.zeros((model.p,)), gk, Hk, -1e20 * np.ones((model.p,)), 1e20 * np.ones((model.p,)), delta)
            pred_reduction = -np.dot(sk_red, gk + 0.5 * Hk.vec_mul(sk_red))

            J_norm2 = np.linalg.norm(Jk, ord=2)
            H_norm2 = 2.0 * J_norm2 ** 2
            g_norm = np.linalg.norm(gk)

            this_block_ok = (pred_reduction >= alpha * delta ** 2) \
                            and (g_norm / max(1.0, H_norm2) >= alpha * delta)

        elif adaptive_method == 'armijo':
            Hk = Hessian(model.p, vals=2.0 * np.dot(Jk.T, Jk))
            sk_red, _, _ = trsbox(np.zeros((model.p,)), gk, Hk, -1e20 * np.ones((model.p,)), 1e20 * np.ones((model.p,)), delta)
            pred_reduction = -np.dot(sk_red, gk + 0.5 * Hk.vec_mul(sk_red))
            this_block_ok = (pred_reduction >= alpha * np.linalg.norm(sk_red) * abs(np.dot(gk, sk_red)))
        else:
            raise RuntimeError("Unknown param adaptive.condition: %s" % adaptive_method)

        if this_block_ok or np.random.uniform() <= params("adaptive.ignore_condition_prob"):
            break  # success - stop growing block and continue to TRS
        else:
            continue  # fail - increase block size
    
    return exit_info, model, nf


# def trs(g, H, delta, tol=1e-10, verbose=False):
#     # Solve min_x g.T*x + 0.5*x.T*H*x, s.t. ||x|| <= delta
#     # Tuple of nonlinear constraints (inequality: c(x) >= 0)
#     cons = ({'type':'ineq', 'fun': lambda x: delta**2 - np.dot(x,x), 'jac': lambda x: -2.0 * x},)  # delta^2 - ||x||^2 >= 0
#     fun = lambda x: np.dot(x, g + 0.5*np.dot(H, x))
#     jac = lambda x: g + np.dot(H, x)
#     x0 = np.zeros(g.shape)
#     res = sp_opt.minimize(fun, x0, method='SLSQP', jac=jac, constraints=cons, tol=tol)  # SLSQP doesn't use Hessian
#     if not res.success:
#         warnings.warn("Trust region solver didn't converge: %s" % res.message, RuntimeWarning)
#         # Replace result with Cauchy step
#         crv = np.dot(g, H.dot(g))
#         gnorm = np.linalg.norm(g)
#         alpha = min(delta / gnorm, gnorm ** 2 / crv) if crv > 0.0 else delta / gnorm
#         return -alpha * g
#     if verbose:
#         print("Success!" if res.success else "Fail!")
#         print(res.message)
#     return res.x


def solve_main(objfun, x0, args, xl, xu, rhobeg, rhoend, maxfun, nruns_so_far, nf_so_far,
               params, scaling_changes, fixed_block=None, true_objfun=None):

    # First, evaluate objfun at x0 - this gives us m
    nf = nf_so_far + 1
    r0, f0 = eval_least_squares_objective(objfun, remove_scaling(x0, scaling_changes), args=args, eval_num=nf,
                                          full_x_thresh=params("logging.n_to_print_whole_x_vector"),
                                          check_for_overflow=params("general.check_objfun_for_overflow"))
    f0 = sumsq(r0)  # used for termination tests later too
    exit_info = None

    if f0 <= params("model.abs_tol"):
        exit_info = ExitInformation(EXIT_SUCCESS, "Objective is sufficiently small")

    if exit_info is not None:
        return x0, r0, f0, None, nf, nruns_so_far + 1, exit_info, None

    # Initialize model  # TODO how best to set pinit?
    delta = rhobeg
    rho = rhobeg if params("tr_radius.use_rho") else 0.0
    model = InterpSet(fixed_block if fixed_block is not None else params("adaptive.p_init"), x0, r0, xl, xu,
                      abs_tol=params("model.abs_tol"), rel_tol=params("model.rel_tol"))
    # Evaluate initial points (requires pinit evaluations, so have done pinit+1 evals total after this step)
    exit_info, nf = model.initialise_interp_set(delta, objfun, args, scaling_changes, nf, maxfun,
                                                use_coord_directions=params("geometry.use_coord_directions"),
                                                box_bound_thresh=params("geometry.direcion_box_bound_thresh"),
                                                full_x_thresh=params("logging.n_to_print_whole_x_vector"),
                                                check_for_overflow=params("general.check_objfun_for_overflow"))

    if exit_info is not None:
        xopt, ropt, fopt = model.get_final_results()
        return xopt, ropt, fopt, None, nf, nruns_so_far + 1, exit_info, None

    if params("logging.save_diagnostic_info"):
        diagnostic_info = DiagnosticInfo(x0, r0, f0, delta, rho, nf, with_xk=params("logging.save_xk"),
                                         with_rk=params("logging.save_rk"), with_poisedness=params("logging.save_poisedness"),
                                         with_rho=params("tr_radius.use_rho"),
                                         with_J_comparisons=(true_objfun is not None))
    else:
        diagnostic_info = None

    # Things to track across iterations
    current_iter = 0
    last_fopts_for_slowterm = []
    num_consecutive_slow_iters = 0
    # Adaptive checking - how many safety/very successful steps have we had in a row so far?
    num_consecutive_safety_steps = 0
    # num_consecutive_very_successful_steps = 0  # not used
    num_consecutive_bad_steps = 0  # how many consecutive steps with small/negative ratio have we had?
    last_successful_iter = 0  # determines when we can reduce rho
    diffs = [0.0, 0.0, 0.0]

    while True:
        current_iter += 1
        if params("tr_radius.use_rho"):
            logging.debug("*** Iter %g (delta = %g, rho = %g) ***" % (current_iter, delta, rho))
        else:
            logging.debug("*** Iter %g (delta = %g) ***" % (current_iter, delta))
        # if current_iter > 4:  # TODO temporary!
        #     break  # quit
        model.point_ages += 1  # increment all ages for existing points

        # Don't bother sampling new points after a safety step - we just reduce delta and try again
        recycle_model = False  #(0 < num_consecutive_safety_steps <= params("geometry.safety_steps_before_redraw"))
        if not recycle_model:
            if fixed_block is None:
                # Add interpolation points based on adaptive criteria
                # (assuming we have already dropped old points at end of previous iteration)
                logging.debug("Starting adaptive model building from p=%g directions" % model.p)
                exit_info, model, nf = build_adaptive_model(model, delta, objfun, args, scaling_changes, nf, maxfun, params)
                if exit_info is not None:
                    break  # quit
                logging.debug("Finished adaptive model building with p=%g directions" % model.p)
            else:
                exit_info, model, nf = build_fixed_block_model(model, fixed_block, delta, objfun, args, scaling_changes, nf, maxfun, params)
                if exit_info is not None:
                    break  # quit
        else:
            logging.debug("Recycling model from previous iteration (safety step)")

        # Interpolate model (based at model.xopt)
        logging.debug("Using model with p=%g directions" % (model.p))

        xk = model.xopt()
        # logging.debug("Before interp, xopt = %s" % str(model.xopt()[:10]))
        # rk = model.ropt()
        fk = model.fopt()
        interp_ok, ck, Jk = model.interpolate_mini_models(jac_in_full_space=False)
        if not interp_ok:
            exit_info = ExitInformation(EXIT_LINALG_ERROR, "Failed to build interpolation model")
            break  # quit
        if true_objfun is not None:  # for diagnostic info
            true_rk, true_Jk = true_objfun(xk)
            # Now do finite differencing using step size delta
            fd_Jk = np.zeros((model.m, model.n))
            for i in range(model.n):
                step = np.zeros((model.n,))
                step[i] = delta
                r_perturb, _ = true_objfun(xk + step)
                fd_Jk[:, i] = (r_perturb - true_rk) / delta
            # Project into relevant subspace
            true_Jk = np.dot(true_Jk, model.Q)
            fd_Jk = np.dot(fd_Jk, model.Q)
        else:
            true_Jk = None
            fd_Jk = None
        # Q_used = model.Q.copy()
        # # True Jacobian for ARGLALE
        # J_true = -0.005 * np.ones((model.m, model.n))
        # for i in range(model.n):
        #     J_true[i,i] = 0.995
        # Jk = np.dot(J_true, Q_used)  # replace with actual Jacobian

        # (optionally) save poisedness of interpolation model
        poisedness = model.poisedness_in_reduced_space(delta) if params("logging.save_poisedness") else None
        # logging.info("Poisedness = %g" % model.poisedness_in_reduced_space(delta))

        # Build full model and calculate step/predicted reduction
        gk = 2.0 * Jk.T.dot(ck)
        Hk = Hessian(model.p, vals=2.0 * np.dot(Jk.T, Jk))
        # if True:
        sk_red, _, crvmin = trsbox(np.zeros((model.p,)), gk, Hk, -1e20 * np.ones((model.p,)), 1e20 * np.ones((model.p,)), delta)
        # else:
        # Scipy proxy for trust region subproblem solver
        # sk_red = trs(gk, 2.0*np.dot(Jk.T, Jk), delta)
        # crvmin = 0

        delta_used = delta  # for diagnostic info only
        rho_used = rho  # for diagnostic info only
        sk_full = model.project_to_full_space(sk_red)
        # logging.debug("Check = %g" % np.linalg.norm(sk_full - Q_used.dot(sk_red)))
        norm_sk = np.linalg.norm(sk_red)
        if params("tr_radius.use_rho"):
            logging.debug("||sk|| = %g, delta = %g, rho = %g" % (norm_sk, delta, rho))
        else:
            logging.debug("||sk|| = %g, delta = %g" % (norm_sk, delta))
        # if abs(norm_sk - np.linalg.norm(sk_full)) > 1e-10:
        #     logging.info("sk norm mismatch")
        #     exit()
        # if norm_sk > delta_used + 1e-10:
        #     logging.info("TRS violates ball constraint (||s|| = %g, delta = %g, diff = %g)" % (norm_sk, delta_used, norm_sk - delta_used))
        #     exit()
        # logging.debug("At step, xopt = %s" % str(model.xopt()[:10]))
        xnew = xk + sk_full
        pred_reduction = -np.dot(sk_red, gk + 0.5 * Hk.vec_mul(sk_red))

        if (params("general.use_safety_step") and norm_sk < params("general.safety_step_thresh") * (rho if params("tr_radius.use_rho") else delta)) \
                or pred_reduction < 2.0*np.finfo(float).eps:  # step too short or TRS gave model increase
            logging.debug("Safety step")
            iter_type = ITER_SAFETY
            num_consecutive_safety_steps += 1
            # num_consecutive_very_successful_steps = 0
            num_consecutive_bad_steps += 1  # safety is bad
            ratio = None  # used for diagnostic info only

            if params("tr_radius.use_rho"):
                delta = max(params("tr_radius.gamma_dec") * delta, rho)
                if not done_with_current_rho(rho, last_successful_iter, current_iter, diffs, crvmin, in_safety=True) or delta > rho:
                    # Delete a bad point (equivalent to fixing geometry)
                    try:
                        sigmas = model.poisedness_of_each_point(delta=delta)
                        if not params("geometry.drop_points_by_distance_only"):  # poisedness-aware removal
                            sqdists = np.square(model.distances_to_xopt(include_kopt=True))  # ||yt-xk||^2
                            vals = sigmas * np.maximum(sqdists ** 2 / delta ** 4, 1)  # BOBYQA point to remove criterion
                        else:
                            vals = sigmas
                        vals[model.kopt] = -1.0  # make sure kopt is never selected
                        k = np.argmax(vals)
                    except np.linalg.LinAlgError:
                        # If case poisedness calculation fails, revert to furthest point
                        sqdists = np.square(model.distances_to_xopt(include_kopt=True))  # ||yt-xk||^2
                        k = np.argmax(sqdists)
                    model.remove_point(k, check_not_kopt=True)

                    if params("general.use_safety_step") and norm_sk > rho:
                        last_successful_iter = current_iter
                else:
                    delta, rho = reduce_rho(delta, rho, rhoend, params)
                    last_successful_iter = current_iter

                if rho <= rhoend:
                    exit_info = ExitInformation(EXIT_SUCCESS, "rho has reached rhoend")
                    break  # quit
            else:
                delta = params("tr_radius.gamma_dec") * delta
                if delta <= rhoend:
                    exit_info = ExitInformation(EXIT_SUCCESS, "delta has reached rhoend")
                    break  # quit
            # (end of safety step)
        else:
            # (start of normal step)
            num_consecutive_safety_steps = 0

            # Evaluate objective at xnew
            nf += 1
            rnew, fnew = eval_least_squares_objective(objfun, remove_scaling(xnew, scaling_changes), args=args,
                                                eval_num=nf, full_x_thresh=params("logging.n_to_print_whole_x_vector"),
                                                check_for_overflow=params("general.check_objfun_for_overflow"))

            if fnew <= model.min_objective_value():
                model.save_point(xnew, rnew)  # save, since this point was an improvement
                exit_info = ExitInformation(EXIT_SUCCESS, "Objective is sufficiently small")
                break  # quit

            if nf >= maxfun:
                model.save_point(xnew, rnew)  # save, just in case this point was an improvement
                exit_info = ExitInformation(EXIT_MAXFUN_WARNING, "Objective has been called MAXFUN times")
                break  # quit

            # Slow termination checking
            if fnew < fk:
                if len(last_fopts_for_slowterm) <= params("slow.history_for_slow"):
                    last_fopts_for_slowterm.append(fk)
                    num_consecutive_slow_iters = 0
                else:
                    last_fopts_for_slowterm = last_fopts_for_slowterm[1:] + [fk]
                    this_iter_slow = (np.log(last_fopts_for_slowterm[0]) - np.log(fk)) / float(params("slow.history_for_slow")) < params("slow.thresh_for_slow")
                    if this_iter_slow:
                        num_consecutive_slow_iters += 1
                    else:
                        num_consecutive_slow_iters = 0

                # Do slow termination check
                if num_consecutive_slow_iters >= params("slow.max_slow_iters"):
                    model.save_point(xnew, rnew)  # save, since this point was an improvement
                    exit_info = ExitInformation(EXIT_SLOW_WARNING, "Maximum slow iterations reached")
                    break  # quit

            # Decide on type of step
            actual_reduction = fk - fnew
            ratio = actual_reduction / pred_reduction
            if min(norm_sk, delta) > rho:  # if ||sk|| >= rho, successful!
                last_successful_iter = current_iter
            diffs = [abs(actual_reduction - pred_reduction), diffs[0], diffs[1]]

            # logging.debug("Alt pred = %g" % (sumsq(ck) - sumsq(ck + Jk.dot(sk_red))))
            # logging.debug("Alt pred 2 = %g" % (sumsq(ck) - sumsq(ck + Jk.dot(np.dot(Q_used.T, sk_full)))))
            # logging.debug("fk = %g, ||ck||^2 = %g, ||ropt||^2 = %g" % (fk, sumsq(ck), sumsq(model.ropt())))
            # logging.debug("fnew = %g, ||rnew||^2 = %g, test = %g" % (fnew, sumsq(rnew), sumsq(objfun(xk + sk_full))))
            # logging.debug("Ratio = %g (actual = %g, predicted = %g)" % (ratio, actual_reduction, pred_reduction))
            # logging.debug("==========")
            # logging.debug(xk)
            # logging.debug(sk_full)
            # logging.debug(xk + sk_full)
            # logging.debug("==========")
            # True Jacobian for ARGLALE
            # J_true = -0.005 * np.ones((model.m, model.n))
            # for i in range(model.n):
            #     J_true[i,i] = 0.995
            # J_proj = np.dot(J_true, Q_used)
            # logging.debug("J error = %g" % np.linalg.norm(J_proj - Jk))
            # logging.debug("Poisedness = %g" % poisedness)

            # if False:
            #     logging.debug("Testing...")
            #     # Model is m(x) = c + J*project_to_reduced_space(x-xopt)
            #     for k in range(model.p + 1):
            #         if k == model.kopt:
            #             # error = np.linalg.norm(ck - model.rvec(k))
            #             error = np.linalg.norm(ck - objfun(model.xpt(k)))
            #             if error > 1e-12 or True:
            #                 logging.debug("Point %g is kopt, error = %g" % (k, error))
            #         else:
            #             # true_r = model.rvec(k)
            #             true_r = objfun(model.xpt(k))
            #             s_red = np.dot(model.Q.T, model.xpt(k) - model.xopt())
            #             model_r = ck + Jk.dot(s_red)
            #             error = np.linalg.norm(true_r - model_r)
            #             if error > 1e-12 or True:
            #                 logging.debug("Point %g, error = %g" % (k, error))
            #     error = np.linalg.norm(np.dot(model.Q.T, model.Q) - np.eye(model.p))
            #     # error2 = np.linalg.norm(np.dot(Q_used.T, Q_used) - np.eye(model.p))
            #     if error > 1e-12:
            #         logging.debug("Q not orthog error: %g" % error)
            #     # if error2 > 1e-12:
            #     #     logging.debug("Q not orthog error2: %g" % error2)
            #     logging.debug("End testing")

            # Update trust region radius
            J_norm2 = np.linalg.norm(Jk, ord=2)
            H_norm2 = 2.0 * J_norm2 ** 2
            norm_gk = np.linalg.norm(gk)
            norm_gk_Hk = norm_gk / max(1.0, H_norm2)
            delta, iter_type = update_tr(delta, rho, ratio, norm_sk, norm_gk, norm_gk_Hk, params)
            # if iter_type == ITER_VERY_SUCCESSFUL:
            #     num_consecutive_very_successful_steps += 1
            # else:
            #     num_consecutive_very_successful_steps = 0
            if iter_type in [ITER_ACCEPTABLE, ITER_UNSUCCESSFUL]:
                num_consecutive_bad_steps += 1
            else:
                num_consecutive_bad_steps = 0

            logging.debug("Ratio = %g (%s): actual = %g, predicted = %g" % (ratio, iter_type, actual_reduction, pred_reduction))
            if num_consecutive_bad_steps > 0:
                logging.debug("%g bad steps so far" % num_consecutive_bad_steps)

            # Add xnew to interpolation set
            if model.p < model.n:
                model.append_point(xnew, rnew)  # updates xopt
                xnew_appended = True
            else:
                # If the model is full, replace xnew with the point furthest from xk (from previous iteration)
                # if True:  # replace point based on geometry
                try:
                    sigmas = model.poisedness_of_each_point(d=sk_red)
                    if not params("geometry.drop_points_by_distance_only"):  # poisedness-aware removal
                        sqdists = np.square(model.distances_to_xopt(include_kopt=True))  # ||yt-xk||^2
                        vals = sigmas * np.maximum(sqdists ** 2 / delta ** 4, 1)  # BOBYQA point to remove criterion
                    else:
                        vals = sigmas
                    vals[model.kopt] = -1.0  # make sure kopt is never selected
                    knew = np.argmax(vals)
                except np.linalg.LinAlgError:
                    # If poisedness calculation fails, revert to dropping furthest points
                    sqdists = np.square(model.distances_to_xopt(include_kopt=True))  # ||yt-xk||^2
                    knew = np.argmax(sqdists)
                # else:  # replace point based on distance only
                #     knew = np.argmax(model.distances_to_xopt(include_kopt=True))  # include_kopt, otherwise index mismatch
                model.change_point(knew, xnew, rnew, check_not_kopt=True)  # updates xopt
                xnew_appended = False

            # Drop points no longer needed
            # Fixed block: remove at least 1 from xnew (if appended) and 1 to make space for a new direction in next iter
            # Adaptive: remove at least 1 from xnew (if appended) and another 2 (want to allow decreasing # dirns across iters, and adaptive always adds at least 1)
            min_npt_to_drop = (1 if xnew_appended else 0) + (2 if fixed_block is None else 1)
            alpha = params("adaptive.previous_J_save_frac")
            if params("adaptive.drop_more_on_unsuccessful") and iter_type in [ITER_ACCEPTABLE, ITER_UNSUCCESSFUL]:
                alpha = min(alpha, params("adaptive.save_frac_for_unsuccessful"))
            ndirs_to_keep = min(int(alpha * model.p), model.p - min_npt_to_drop)
            ndirs_to_keep = max(0, ndirs_to_keep)
            ndirs_to_drop = model.p - ndirs_to_keep
            # if False and num_consecutive_bad_steps > 2:
            #     ndirs_to_drop = min(model.p, int(num_consecutive_bad_steps * ndirs_to_drop))
            # if False and params("tr_radius.use_rho"):
            #     ndirs_to_drop = min(model.p, int((current_iter - last_successful_iter) * ndirs_to_drop))
            logging.debug("After adding xnew, model has %g directions, dropping %g" % (model.p, ndirs_to_drop))
            # Criteria of points to remove:
            try:
                sigmas = model.poisedness_of_each_point(delta=delta)
                if not params("geometry.drop_points_by_distance_only"):  # poisedness-aware removal
                    sqdists = np.square(model.distances_to_xopt(include_kopt=True))  # ||yt-xk||^2
                    vals = sigmas * np.maximum(sqdists**2 / delta**4, 1)  # BOBYQA point to remove criterion
                else:
                    vals = sigmas
                vals[model.kopt] = -1.0  # make sure kopt is never selected
            except np.linalg.LinAlgError:
                # If poisedness calculation fails, revert to dropping furthest points
                vals = np.square(model.distances_to_xopt(include_kopt=True))  # ||yt-xk||^2
                vals[model.kopt] = -1.0  # make sure kopt is never selected
            for i in range(ndirs_to_drop):
                # if False:  # delete points by distance to xopt
                #     k = np.argmax(model.distances_to_xopt(include_kopt=True))  # include_kopt, otherwise index mismatch
                # elif False:  # delete points by age
                #     pts_by_age = np.argsort(model.point_ages)  # point indices in increasing order of point age
                #     k = pts_by_age[-1] if pts_by_age[-1] != model.kopt else pts_by_age[-2]
                # else:  # delete points by geometry
                k = np.argmax(vals)
                vals = np.delete(vals, k)  # keep vals indices in line with indices of model.points
                model.remove_point(k, check_not_kopt=True)

            if params("geometry.impose_max_cond_num"):  # or iter_type in [ITER_ACCEPTABLE, ITER_UNSUCCESSFUL]:  # TODO only check geometry when things are going badly?
                # Note this removes points which are almost linearly dependent with others
                # Bound is on condition number of R, by removing small diagonal entries
                # i.e. Nothing done here to prevent large diagonal entries
                # (This is dealt with above by removing points far from xk)
                logging.debug("Checking and fixing geometry")
                model.factorise_system()
                Rdiag = np.abs(np.diag(model.R))
                thresh = np.max(Rdiag) / params("geometry.max_cond_num")  # smallest acceptable Rdiag value
                idx_to_drop = np.sort(np.where(Rdiag < thresh)[0])  # in ascending order
                idx_to_drop[idx_to_drop >= model.kopt] += 1  # adjust indices to include kopt
                for k in reversed(idx_to_drop):
                    # Go through indices in descending order, so can delete in order without having to relabel indices
                    model.remove_point(k, check_not_kopt=True)
                if len(idx_to_drop) > 0:
                    logging.debug("Geometry checked, %g points removed" % len(idx_to_drop))
                else:
                    logging.debug("Geometry checked, no points removed")

            if params("geometry.use_geometry_fixing_steps") and iter_type in [ITER_ACCEPTABLE, ITER_UNSUCCESSFUL] \
                and model.p > 1:  # no point fixing geometry if there is only 1 point left
                logging.debug("Geometry fixing step")
                distances = model.distances_to_xopt(include_kopt=True)
                kmax = np.argmax(distances)  # current point furthest from xk
                if distances[kmax] >= 2.0 * delta:  # the threshold from DFO-LS
                    logging.debug("Geometry needs fixing")
                    iter_type = ITER_ACCEPTABLE_GEOM if iter_type == ITER_ACCEPTABLE else ITER_UNSUCCESSFUL_GEOM
                    # Move point kmax to a better location in B(xk,adelt) [in DFO-LS, adelt != delta necessarily]
                    if params("tr_radius.use_rho"):
                        # Use the value from DFO-LS
                        adelt = max(min(0.1 * distances[kmax], delta), rho)
                    else:
                        other_distances = model.distances_to_xopt(include_kopt=False)
                        closest_dist = np.min(other_distances)
                        avg_dist = np.exp(np.mean(np.log(other_distances)))  # geometric mean
                        adelt = max(min(avg_dist, delta), closest_dist)  # want delta, but not if it's too large or small
                    # Maximise |c+g*s| for ||s|| <= adelt, which is at s = +/- (adelt/normg) * g
                    try:
                        c, g = model.lagrange_poly(kmax)
                        normg = np.linalg.norm(g)
                        if abs(c + adelt * normg) < abs(c - adelt * normg):
                            s = (adelt / normg) * g
                        else:
                            s = -(adelt / normg) * g
                        snew = model.project_to_full_space(s)
                    except np.linalg.LinAlgError:
                        # Then we had a singular R matrix, so this is clearly the point to remove
                        kmax_in_R = np.argmin(np.abs(np.diag(model.R)))  # minimum diagonal entry
                        kmax = kmax_in_R if kmax_in_R < model.kopt else kmax_in_R + 1  # adjust to correct for model.R not including kopt
                        # Move it to the orthogonal direction we are missing
                        snew = model.Q[:, kmax_in_R]

                    xnew = model.xopt() + snew

                    # Evaluate objective at xnew
                    nf += 1
                    rnew, fnew = eval_least_squares_objective(objfun, remove_scaling(xnew, scaling_changes), args=args,
                                                              eval_num=nf, full_x_thresh=params("logging.n_to_print_whole_x_vector"),
                                                              check_for_overflow=params("general.check_objfun_for_overflow"))

                    if fnew <= model.min_objective_value():
                        model.save_point(xnew, rnew)  # save, since this point was an improvement
                        exit_info = ExitInformation(EXIT_SUCCESS, "Objective is sufficiently small")
                        break  # quit

                    if nf >= maxfun:
                        model.save_point(xnew, rnew)  # save, just in case this point was an improvement
                        exit_info = ExitInformation(EXIT_MAXFUN_WARNING, "Objective has been called MAXFUN times")
                        break  # quit

                    model.change_point(kmax, xnew, rnew, check_not_kopt=True)  # updates xopt
                    logging.debug("Geometry fixed")
                else:
                    logging.debug("Geometry does not need fixing")

            logging.debug("Finish iteration with %g directions" % model.p)

            # Finally, decide if we need to reduce rho or not
            if params("tr_radius.use_rho"):
                if ratio < 0 and done_with_current_rho(rho, last_successful_iter, current_iter, diffs, crvmin, in_safety=False) and delta <= rho:
                    delta, rho = reduce_rho(delta, rho, rhoend, params)
                    last_successful_iter = current_iter

                if rho <= rhoend:
                    exit_info = ExitInformation(EXIT_SUCCESS, "rho has reached rhoend")
                    break  # quit
            else:
                if delta <= rhoend:
                    exit_info = ExitInformation(EXIT_SUCCESS, "delta has reached rhoend")
                    break  # quit
            # (end of normal step)

        # (in both safety and normal steps, update diagnostic info)
        if params("logging.save_diagnostic_info"):
            diagnostic_info.save_info(model, delta_used, rho_used, Jk, norm_sk, np.linalg.norm(gk),
                                      nruns_so_far + 1, nf, current_iter, iter_type, ratio, poisedness,
                                      trueJ=true_Jk, fdJ=fd_Jk)
        continue

    xopt, ropt, fopt = model.get_final_results()
    return xopt, ropt, fopt, None, nf, nruns_so_far + 1, exit_info, diagnostic_info  # TODO no Jacobian returned?


def solve(objfun, x0, args=(), fixed_block=None, bounds=None, rhobeg=None, rhoend=1e-8, maxfun=None,
          user_params=None, objfun_has_noise=False, scaling_within_bounds=False, true_objfun=None):
    n = len(x0)

    if fixed_block is not None:
        assert 1 <= fixed_block <= n, "fixed_block, if specified, must be in [1..n]"

    # Set missing inputs (if not specified) to some sensible defaults
    if bounds is None:
        xl = None
        xu = None
        scaling_within_bounds = False
    else:
        raise RuntimeError('DFBGN does not support bounds yet (needs update to trust region subproblem solver)')
        # assert len(bounds) == 2, "bounds must be a 2-tuple of (lower, upper), where both are arrays of size(x0)"
        # xl = bounds[0]
        # xu = bounds[1]

    if xl is None:
        xl = -1e20 * np.ones((n,))  # unconstrained
    if xu is None:
        xu = 1e20 * np.ones((n,))  # unconstrained

    if rhobeg is None:
        rhobeg = 0.1 if scaling_within_bounds else 0.1 * max(np.max(np.abs(x0)), 1.0)
    if maxfun is None:
        maxfun = min(100 * (n + 1), 1000)  # 100 gradients, capped at 1000

    # Set parameters
    params = ParameterList(int(n), int(maxfun), objfun_has_noise=objfun_has_noise)  # make sure int, not int
    if user_params is not None:
        for (key, val) in user_params.items():
            params(key, new_value=val)

    scaling_changes = None
    if scaling_within_bounds:
        shift = xl.copy()
        scale = xu - xl
        scaling_changes = (shift, scale)

    x0 = apply_scaling(x0, scaling_changes)
    xl = apply_scaling(xl, scaling_changes)
    xu = apply_scaling(xu, scaling_changes)

    exit_info = None
    # Input & parameter checks
    if exit_info is None and rhobeg < 0.0:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "rhobeg must be strictly positive")

    if exit_info is None and rhoend < 0.0:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "rhoend must be strictly positive")

    if exit_info is None and rhobeg <= rhoend:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "rhobeg must be > rhoend")

    if exit_info is None and maxfun <= 0:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "maxfun must be strictly positive")

    if exit_info is None and np.shape(x0) != (n,):
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "x0 must be a vector")

    if exit_info is None and np.shape(x0) != np.shape(xl):
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "lower bounds must have same shape as x0")

    if exit_info is None and np.shape(x0) != np.shape(xu):
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "upper bounds must have same shape as x0")

    if exit_info is None and np.min(xu - xl) < 2.0 * rhobeg:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "gap between lower and upper must be at least 2*rhobeg")

    if maxfun <= n:
        warnings.warn("maxfun <= n: Are you sure your budget is large enough?", RuntimeWarning)

    # Check invalid parameter values

    all_ok, bad_keys = params.check_all_params()
    if exit_info is None and not all_ok:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "Bad parameters: %s" % str(bad_keys))

    # If we had an input error, quit gracefully
    if exit_info is not None:
        exit_flag = exit_info.flag
        exit_msg = exit_info.message(with_stem=True)
        results = OptimResults(None, None, None, None, 0, 0, exit_flag, exit_msg)
        return results

    # Enforce lower & upper bounds on x0
    idx = (xl < x0) & (x0 <= xl + rhobeg)
    if np.any(idx):
        warnings.warn("x0 too close to lower bound, adjusting", RuntimeWarning)
    x0[idx] = xl[idx] + rhobeg

    idx = (x0 <= xl)
    if np.any(idx):
        warnings.warn("x0 below lower bound, adjusting", RuntimeWarning)
    x0[idx] = xl[idx]

    idx = (xu - rhobeg <= x0) & (x0 < xu)
    if np.any(idx):
        warnings.warn("x0 too close to upper bound, adjusting", RuntimeWarning)
    x0[idx] = xu[idx] - rhobeg

    idx = (x0 >= xu)
    if np.any(idx):
        warnings.warn("x0 above upper bound, adjusting", RuntimeWarning)
    x0[idx] = xu[idx]

    # Call main solver (first time)
    nruns = 0
    nf = 0
    xmin, rmin, fmin, jacmin, nf, nruns, exit_info, diagnostic_info = \
        solve_main(objfun, x0, args, xl, xu, rhobeg, rhoend, maxfun, nruns, nf, params, scaling_changes, fixed_block=fixed_block,
                   true_objfun=true_objfun)

    # Process final return values & package up
    exit_flag = exit_info.flag
    exit_msg = exit_info.message(with_stem=True)

    # Un-scale Jacobian
    if scaling_changes is not None and jacmin is not None:
        for i in range(n):
            jacmin[:, i] = jacmin[:, i] / scaling_changes[1][i]

    results = OptimResults(remove_scaling(xmin, scaling_changes), rmin, fmin, jacmin, nf, nruns, exit_flag, exit_msg)

    if params("logging.save_diagnostic_info") and diagnostic_info is not None:
        df = diagnostic_info.to_dataframe()
        results.diagnostic_info = df

    return results