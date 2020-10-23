"""
Parameters
====

A container class for all the solver parameter values.


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

__all__ = ['ParameterList']


class ParameterList(object):
    def __init__(self, n, maxfun, objfun_has_noise=False):
        self.n = n

        self.params = {}
        self.params["general.use_safety_step"] = False
        self.params["general.safety_step_thresh"] = 0.5  # safety step called if ||d|| <= thresh * rho
        self.params["general.check_objfun_for_overflow"] = True
        # Logging
        self.params["logging.n_to_print_whole_x_vector"] = 6
        self.params["logging.save_diagnostic_info"] = False
        self.params["logging.save_poisedness"] = False
        self.params["logging.save_xk"] = False
        self.params["logging.save_rk"] = False
        # Trust Region Radius management
        self.params["tr_radius.eta1"] = 0.1
        self.params["tr_radius.eta2"] = 0.7
        self.params["tr_radius.gamma_dec"] = 0.98 if objfun_has_noise else 0.5
        self.params["tr_radius.gamma_inc"] = 2.0
        self.params["tr_radius.gamma_inc_overline"] = 4.0
        self.params["tr_radius.delta_max"] = 1.0e10
        self.params["tr_radius.update_method"] = 'bobyqa'
        self.params["tr_radius.alpha1"] = 0.9 if objfun_has_noise else 0.1
        self.params["tr_radius.alpha2"] = 0.95 if objfun_has_noise else 0.5
        self.params["tr_radius.use_rho"] = False  # use delta/rho combination?
        # Least-Squares objective threshold
        self.params["model.abs_tol"] = 1e-12
        self.params["model.rel_tol"] = 1e-20
        # Slow progress thresholds
        self.params["slow.history_for_slow"] = 5
        self.params["slow.thresh_for_slow"] = 1e-4
        self.params["slow.max_slow_iters"] = 20 * n
        # Geometry Management
        # self.params["geometry.use_fin_diff"] = False  # sample with finite differencing?
        self.params["geometry.use_coord_directions"] = False  # make a block coordinate method?
        self.params["geometry.direcion_box_bound_thresh"] = 0.01  # how close to bounds before random sampling is changed
        self.params["geometry.safety_steps_before_redraw"] = 1  # how many consecutive safety steps before redrawing a model (1 = every time)
        # self.params["geometry.vsucc_steps_before_redraw"] = 1  # how many consecutive v. successful steps before redrawing a model (1 = every time)
        # self.params["geometry.adaptive_dimension"] = False  # if False, use ndirs() input, otherwise modify adaptively my way
        self.params["geometry.impose_max_cond_num"] = False  # do check for maximum condition number of interp system?
        self.params["geometry.max_cond_num"] = 1e8  # maximum condition number of interpolation system R matrix
        self.params["geometry.use_geometry_fixing_steps"] = False
        self.params["geometry.drop_points_by_distance_only"] = False  # use distance to xk as only dropping criterion (not using poisedness)
        # Adaptive block
        self.params["adaptive.p_init"] = 1  # baseline block size
        self.params["adaptive.pmax"] = n  # largest number of directions we are allowed to use
        self.params["adaptive.alpha"] = 0.2  # stop growing block when ||gk|| >= alpha*delta
        self.params["adaptive.condition"] = 'normgh'  # what adaptive measure to use
        self.params["adaptive.ignore_condition_prob"] = 0.0  # probability of ignoring adaptive measure (i.e. using smaller block than ideal)
        self.params["adaptive.previous_J_save_frac"] = 1.0  # proportion of old directions to preserve across iterations
        self.params["adaptive.adjust_alpha"] = True  # whether to decrease alpha as block size increases
        self.params["adaptive.drop_more_on_unsuccessful"] = False
        self.params["adaptive.save_frac_for_unsuccessful"] = 0.9

        self.params_changed = {}
        for p in self.params:
            self.params_changed[p] = False

    def __call__(self, key, new_value=None):  # self(key) or self(key, new_value)
        if key in self.params:
            if new_value is None:
                return self.params[key]
            else:
                if self.params_changed[key]:
                    raise ValueError("Trying to update parameter '%s' for a second time" % key)
                self.params[key] = new_value
                self.params_changed[key] = True
                return self.params[key]
        else:
            raise ValueError("Unknown parameter '%s'" % key)

    def param_type(self, key):
        # Use the check_* methods below, but switch based on key
        if key == "general.use_safety_step":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "general.safety_step_thresh":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, None
        elif key == "general.check_objfun_for_overflow":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "logging.n_to_print_whole_x_vector":
            type_str, nonetype_ok, lower, upper = 'int', False, 0, None
        elif key == "logging.save_diagnostic_info":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "logging.save_poisedness":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "logging.save_xk":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "logging.save_rk":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "tr_radius.eta1":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, 1.0
        elif key == "tr_radius.eta2":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, 1.0
        elif key == "tr_radius.gamma_dec":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, 1.0
        elif key == "tr_radius.gamma_inc":
            type_str, nonetype_ok, lower, upper = 'float', False, 1.0, None
        elif key == "tr_radius.gamma_inc_overline":
            type_str, nonetype_ok, lower, upper = 'float', False, 1.0, None
        elif key == "tr_radius.delta_max":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, None
        elif key == "tr_radius.update_method":
            type_str, nonetype_ok, lower, upper = 'str', False, None, None
        elif key == "tr_radius.alpha1":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, 1.0
        elif key == "tr_radius.alpha2":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, 1.0
        elif key == "tr_radius.use_rho":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "model.abs_tol":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, None
        elif key == "model.rel_tol":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, 1.0
        elif key == "slow.history_for_slow":
            type_str, nonetype_ok, lower, upper = 'int', False, 0, None
        elif key == "slow.thresh_for_slow":
            type_str, nonetype_ok, lower, upper = 'float', False, 0, None
        elif key == "slow.max_slow_iters":
            type_str, nonetype_ok, lower, upper = 'int', False, 0, None
        elif key == "geometry.use_fin_diff":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "geometry.use_coord_directions":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "geometry.direcion_box_bound_thresh":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, 1.0
        elif key == "geometry.safety_steps_before_redraw":
            type_str, nonetype_ok, lower, upper = 'int', False, 1, None
        elif key == "geometry.vsucc_steps_before_redraw":
            type_str, nonetype_ok, lower, upper = 'int', False, 1, None
        # elif key == "geometry.adaptive_dimension":
        #     type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "geometry.impose_max_cond_num":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "geometry.max_cond_num":
            type_str, nonetype_ok, lower, upper = 'float', False, 1.0, None
        elif key == "geometry.use_geometry_fixing_steps":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "geometry.drop_points_by_distance_only":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "adaptive.p_init":
            type_str, nonetype_ok, lower, upper = 'int', False, 1, self.n
        elif key == "adaptive.pmax":
            type_str, nonetype_ok, lower, upper = 'int', False, 1, self.n
        elif key == "adaptive.alpha":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, None
        elif key == "adaptive.condition":
            type_str, nonetype_ok, lower, upper = 'str', False, None, None
        elif key == "adaptive.ignore_condition_prob":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, 1.0
        elif key == "adaptive.previous_J_save_frac":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, 1.0
        elif key == "adaptive.adjust_alpha":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "adaptive.drop_more_on_unsuccessful":
            type_str, nonetype_ok, lower, upper = 'bool', False, None, None
        elif key == "adaptive.save_frac_for_unsuccessful":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, 1.0
        else:
            assert False, "ParameterList.param_type() has unknown key: %s" % key
        return type_str, nonetype_ok, lower, upper

    def check_param(self, key, value):
        type_str, nonetype_ok, lower, upper = self.param_type(key)
        if type_str == 'int':
            return check_integer(value, lower=lower, upper=upper, allow_nonetype=nonetype_ok)
        elif type_str == 'float':
            return check_float(value, lower=lower, upper=upper, allow_nonetype=nonetype_ok)
        elif type_str == 'bool':
            return check_bool(value, allow_nonetype=nonetype_ok)
        elif type_str == 'str':
            return check_str(value, allow_nonetype=nonetype_ok)
        else:
            assert False, "Unknown type_str '%s' for parameter '%s'" % (type_str, key)

    def check_all_params(self):
        bad_keys = []
        for key in self.params:
            if not self.check_param(key, self.params[key]):
                bad_keys.append(key)
        return len(bad_keys) == 0, bad_keys


def check_integer(val, lower=None, upper=None, allow_nonetype=False):
    # Check that val is an integer and (optionally) that lower <= val <= upper
    if val is None:
        return allow_nonetype
    elif not isinstance(val, int):
        return False
    else:  # is integer
        return (lower is None or val >= lower) and (upper is None or val <= upper)


def check_float(val, lower=None, upper=None, allow_nonetype=False):
    # Check that val is a float and (optionally) that lower <= val <= upper
    if val is None:
        return allow_nonetype
    elif not isinstance(val, float):
        return False
    else:  # is integer
        return (lower is None or val >= lower) and (upper is None or val <= upper)


def check_bool(val, allow_nonetype=False):
    if val is None:
        return allow_nonetype
    else:
        return isinstance(val, bool)


def check_str(val, allow_nonetype=False):
    if val is None:
        return allow_nonetype
    else:
        return isinstance(val,str) or isinstance(val, unicode)

