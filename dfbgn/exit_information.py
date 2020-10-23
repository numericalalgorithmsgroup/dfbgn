"""
Wrapper for exit information

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


__all__ = ['EXIT_AUTO_DETECT_RESTART_WARNING', 'EXIT_FALSE_SUCCESS_WARNING', 'EXIT_SLOW_WARNING', 'EXIT_MAXFUN_WARNING',
           'EXIT_SUCCESS', 'EXIT_INPUT_ERROR', 'EXIT_TR_INCREASE_ERROR', 'EXIT_LINALG_ERROR', 'ExitInformation',
           'OptimResults']


EXIT_AUTO_DETECT_RESTART_WARNING = 4  # warning, auto-detected restart criteria
EXIT_FALSE_SUCCESS_WARNING = 3  # warning, maximum fake successful steps reached
EXIT_SLOW_WARNING = 2  # warning, maximum number of slow (successful) iterations reached
EXIT_MAXFUN_WARNING = 1  # warning, reached max function evals
EXIT_SUCCESS = 0  # successful finish (rho=rhoend, sufficient objective reduction, or everything in noise level)
EXIT_INPUT_ERROR = -1  # error, bad inputs
EXIT_TR_INCREASE_ERROR = -2  # error, trust region step increased model value
EXIT_LINALG_ERROR = -3  # error, linalg error (singular matrix encountered)
# EXIT_ALTMOV_MEMORY_ERROR = -4  # error, stpsav issue in ALTMOV


class ExitInformation:
    def __init__(self, flag, msg_details):
        self.flag = flag
        self.msg = msg_details

    def flag(self):
        return self.flag

    def message(self, with_stem=True):
        if not with_stem:
            return self.msg
        elif self.flag == EXIT_SUCCESS:
            return "Success: " + self.msg
        elif self.flag == EXIT_SLOW_WARNING:
            return "Warning (slow progress): " + self.msg
        elif self.flag == EXIT_MAXFUN_WARNING:
            return "Warning (max evals): " + self.msg
        elif self.flag == EXIT_INPUT_ERROR:
            return "Error (bad input): " + self.msg
        elif self.flag == EXIT_TR_INCREASE_ERROR:
            return "Error (trust region increase): " + self.msg
        elif self.flag == EXIT_LINALG_ERROR:
            return "Error (linear algebra): " + self.msg
        elif self.flag == EXIT_FALSE_SUCCESS_WARNING:
            return "Warning (max false good steps): " + self.msg
        # elif self.flag == EXIT_ALTMOV_MEMORY_ERROR:
        #     return "Error (geometry step): " + self.msg
        else:
            return "Unknown exit flag: " + self.msg

    def able_to_do_restart(self):
        if self.flag in [EXIT_TR_INCREASE_ERROR, EXIT_LINALG_ERROR, EXIT_SLOW_WARNING, EXIT_AUTO_DETECT_RESTART_WARNING]:
            return True
        elif self.flag in [EXIT_MAXFUN_WARNING, EXIT_INPUT_ERROR]:
            return False
        else:
            # Successful step (rho=rhoend, noise level termination, or value small)
            return "sufficiently small" not in self.msg  # restart for rho=rhoend and noise level termination


# A container for the results of the optimization routine
class OptimResults(object):
    def __init__(self, xmin, rmin, fmin, jacmin, nf, nruns, exit_flag, exit_msg):
        self.x = xmin
        self.resid = rmin
        self.f = fmin
        self.jacobian = jacmin
        self.nf = nf
        # self.nx = nx
        self.nruns = nruns
        self.flag = exit_flag
        self.msg = exit_msg
        self.diagnostic_info = None
        # Set standard names for exit flags
        self.EXIT_SLOW_WARNING = EXIT_SLOW_WARNING
        self.EXIT_MAXFUN_WARNING = EXIT_MAXFUN_WARNING
        self.EXIT_SUCCESS = EXIT_SUCCESS
        self.EXIT_INPUT_ERROR = EXIT_INPUT_ERROR
        self.EXIT_TR_INCREASE_ERROR = EXIT_TR_INCREASE_ERROR
        self.EXIT_LINALG_ERROR = EXIT_LINALG_ERROR
        self.EXIT_FALSE_SUCCESS_WARNING = EXIT_FALSE_SUCCESS_WARNING

    def __str__(self):
        # Result of calling print(soln)
        output = "****** DFBGN Results ******\n"
        if self.flag != self.EXIT_INPUT_ERROR:
            output += "Solution xmin = %s\n" % str(self.x)
            if len(self.resid) < 100:
                output += "Residual vector = %s\n" % str(self.resid)
            else:
                output += "Not showing residual vector because it is too long; check self.resid\n"
            output += "Objective value f(xmin) = %.10g\n" % self.f
            # output += "Needed %g objective evaluations (at %g points)\n" % (self.nf, self.nx)
            output += "Needed %g objective evaluations\n" % (self.nf)
            if self.nruns > 1:
                output += "Did a total of %g runs\n" % self.nruns
            if np.size(self.jacobian) < 200:
                output += "Approximate Jacobian = %s\n" % str(self.jacobian)
            else:
                output += "Not showing approximate Jacobian because it is too long; check self.jacobian\n"
            if self.diagnostic_info is not None:
                output += "Diagnostic information available; check self.diagnostic_info\n"
        output += "Exit flag = %g\n" % self.flag
        output += "%s\n" % self.msg
        output += "****************************\n"
        return output