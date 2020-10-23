"""
Diagnostic Info
====

A class containing diagnostic information (optionally) produced by the solver.


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
import pandas as pd


__all__ = ['DiagnosticInfo', 'ITER_VERY_SUCCESSFUL', 'ITER_SUCCESSFUL', 'ITER_ACCEPTABLE',
           'ITER_UNSUCCESSFUL', 'ITER_SAFETY', 'ITER_ACCEPTABLE_GEOM', 'ITER_UNSUCCESSFUL_GEOM']


ITER_VERY_SUCCESSFUL = "Very successful"   # ratio >= 0.7, no geometry update
ITER_SUCCESSFUL = "Successful"  # 0.1 <= ratio < 0.7, no geometry update
ITER_ACCEPTABLE = "Acceptable"  # 0 <= ratio < 0.1
ITER_ACCEPTABLE_GEOM = "Acceptable (geom fixed)"  # 0 <= ratio < 0.1, with geometry update
# ITER_ACCEPTABLE_NO_GEOM = "Acceptable (geom not fixed)"  # 0 <= ratio < 0.1, without geometry update
ITER_UNSUCCESSFUL = "Unsuccessful"  # ratio < 0
ITER_UNSUCCESSFUL_GEOM = "Unsuccessful (geom fixed)"  # ratio < 0, with geometry update
# ITER_UNSUCCESSFUL_NO_GEOM = "Unsuccessful (geom not fixed)"  # ratio < 0, without geometry update (possibly rho reduced)
ITER_SAFETY = "Safety"  # safety step taken (||s|| too small compared to rho)
ITER_INIT = "Initial setup"


def srank(A):  # stable rank of a matrix
    if np.linalg.norm(A, ord=2) < 1e-16:
        return 0.0
    else:
        return np.linalg.norm(A, ord='fro')**2 / np.linalg.norm(A, ord=2)**2


def cond(A, p=2):  # condition number for any matrix
    # if A.shape[0] == A.shape[1]:
    #     return np.linalg.cond(A, p=p)
    # else:
    try:
        return np.linalg.norm(A, ord=p) * np.linalg.norm(np.linalg.pinv(A), ord=p)
    except np.linalg.LinAlgError:
        return 1e15  # can't compute pseudoinverse, assume ill conditioned


class DiagnosticInfo:
    def __init__(self, x0, r0, f0, delta0, rho0, nf, with_xk=False, with_rk=False, with_poisedness=False, with_rho=False, with_J_comparisons=False):  # initialise with first info
        self.data = {}
        # Initialise everything we want to store
        self.save_xk = with_xk
        self.save_rk = with_rk
        self.save_poisedness = with_poisedness
        if self.save_xk:
            self.data["xk"] = [x0]
        if self.save_rk:
            self.data["rk"] = [r0]
        self.data["fk"] = [f0]

        self.data["delta"] = [delta0]
        self.save_rho = with_rho
        if self.save_rho:
            self.data["rho"] = [rho0]

        self.data["J_norm"] = [np.nan]
        self.data["J_norm2"] = [np.nan]
        self.data["J_cond"] = [np.nan]
        self.data["J_srank"] = [np.nan]
        self.data["J_rank"] = [np.nan]

        self.save_J_comparisons = with_J_comparisons
        if self.save_J_comparisons:
            self.data["true_J_norm"] = [np.nan]
            self.data["true_J_norm2"] = [np.nan]
            self.data["fd_J_norm"] = [np.nan]
            self.data["fd_J_norm2"] = [np.nan]
            self.data["model_v_true_J_norm"] = [np.nan]
            self.data["model_v_true_J_norm2"] = [np.nan]
            self.data["fd_v_true_J_norm"] = [np.nan]
            self.data["fd_v_true_J_norm2"] = [np.nan]
            self.data["model_v_fd_J_norm"] = [np.nan]
            self.data["model_v_fd_J_norm2"] = [np.nan]

        self.data["norm_gk"] = [np.nan]
        self.data["norm_sk"] = [np.nan]

        self.data["nruns"] = [1]
        self.data["nf"] = [nf]
        # self.data["nx"] = [nx]
        self.data["iter_this_run"] = [0]
        self.data["iters_total"] = [0]

        self.data["iter_type"] = [ITER_INIT]
        self.data["ratio"] = [np.nan]

        self.data["ndirs"] = [np.nan]  # number of directions used to construct model
        if self.save_poisedness:
            self.data["poisedness"] = [np.nan]  # poisedness of model in B(xk,delta) (reduced space) used to compute trust region step

        # And some things will be constant throughout
        self.data["n"] = [len(x0)]

        # Save a copy to look at changes between iterations
        # self.previous_J = J0.copy()
        # self.previous_s = None
        return

    def to_dataframe(self):
        data_to_save = {}
        for key in self.data:
            if (key == "xk" and not self.save_xk) or (key == "rk" and not self.save_rk):
                continue  # skip
            data_to_save[key] = self.data[key]
        return pd.DataFrame(data_to_save)

    def to_csv(self, filename):
        df = self.to_dataframe()
        df.to_csv(filename)

    def save_info(self, model, delta, rho, J, norm_sk, norm_gk, nruns, nf, iter_this_run, iter_type, ratio, model_poisedness, trueJ=None, fdJ=None):
        if self.save_xk:
            self.data["xk"].append(model.xopt())
        if self.save_rk:
            self.data["rk"].append(model.ropt())
        self.data["fk"].append(model.fopt())

        self.data["delta"].append(delta)
        if self.save_rho:
            self.data["rho"].append(rho)

        self.data["J_norm"].append(np.linalg.norm(J, ord='fro'))
        self.data["J_norm2"].append(np.linalg.norm(J, ord=2))
        self.data["J_cond"].append(cond(J, p=2))
        self.data["J_srank"].append(srank(J))
        self.data["J_rank"].append(np.linalg.matrix_rank(J))

        if self.save_J_comparisons:
            if trueJ is None:
                raise RuntimeError("Diagonistic info: trying to save J comparisons but true Jacobian not provided")
            if fdJ is None:
                raise RuntimeError("Diagonistic info: trying to save J comparisons but finite differencing Jacobian not provided")
            self.data["true_J_norm"].append(np.linalg.norm(trueJ, ord='fro'))
            self.data["true_J_norm2"].append(np.linalg.norm(trueJ, ord=2))
            self.data["fd_J_norm"].append(np.linalg.norm(fdJ, ord='fro'))
            self.data["fd_J_norm2"].append(np.linalg.norm(fdJ, ord=2))
            self.data["model_v_true_J_norm"].append(np.linalg.norm(J-trueJ, ord='fro'))
            self.data["model_v_true_J_norm2"].append(np.linalg.norm(J-trueJ, ord=2))
            self.data["fd_v_true_J_norm"].append(np.linalg.norm(fdJ-trueJ, ord='fro'))
            self.data["fd_v_true_J_norm2"].append(np.linalg.norm(fdJ-trueJ, ord=2))
            self.data["model_v_fd_J_norm"].append(np.linalg.norm(J-fdJ, ord='fro'))
            self.data["model_v_fd_J_norm2"].append(np.linalg.norm(J-fdJ, ord=2))

        # self.data["norm_gk"].append(2.0 * np.linalg.norm(np.dot(J.T, rk)))
        self.data["norm_gk"].append(norm_gk)
        self.data["norm_sk"].append(norm_sk)

        self.data["nruns"].append(nruns)
        self.data["nf"].append(nf)
        # self.data["nx"].append(nx)
        self.data["iter_this_run"].append(iter_this_run)
        self.data["iters_total"].append(len(self.data["iters_total"]))

        self.data["iter_type"].append(iter_type)
        self.data["ratio"].append(ratio)

        self.data["ndirs"].append(J.shape[1])
        if self.save_poisedness:
            self.data["poisedness"].append(model_poisedness)

        # Constant stuff - carry over previous value
        for key in ["n"]:
            self.data[key].append(self.data[key][-1])

        # Save a copy to look at changes between iterations
        # self.previous_J = J.copy()
        # self.previous_s = sk.copy()
        return
