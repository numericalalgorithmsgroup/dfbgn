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

import unittest

from dfbgn.params import ParameterList


class TestAccess(unittest.TestCase):
    def runTest(self):
        n = 3
        maxfun = 50 * (n + 1)
        p = ParameterList(n, maxfun)
        self.assertTrue(p("general.check_objfun_for_overflow"), 'Bad check_overflow/access')
        p("general.check_objfun_for_overflow", False)  # set to False
        self.assertFalse(p("general.check_objfun_for_overflow"), 'Bad check_overflow/access')


class TestFail(unittest.TestCase):
    def runTest(self):
        n = 3
        maxfun = 50 * (n + 1)
        p = ParameterList(n, maxfun)
        p("general.check_objfun_for_overflow", False)  # set to False
        self.assertFalse(p("general.check_objfun_for_overflow"), 'Bad check_overflow/access')
        self.assertRaises(ValueError, lambda: p("general.check_objfun_for_overflow", False))  # should fail
        self.assertRaises(ValueError, lambda: p("fake_parameter_name"))  # should fail
        self.assertRaises(ValueError, lambda: p("fake_parameter_name", False))  # should fail

if __name__ == '__main__':
    unittest.main()