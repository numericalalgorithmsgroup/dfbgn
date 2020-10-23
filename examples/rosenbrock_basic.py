# DFBGN example: minimize the Rosenbrock function
from __future__ import print_function
import numpy as np
import dfbgn

# Define the objective function
def rosenbrock(x):
    return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])

# Define the starting point
x0 = np.array([-1.2, 1.0])

# For optional extra output details
# import logging
# logging.basicConfig(level=logging.INFO, format='%(message)s')

# DFBGN is a randomized algorithm - set random seed for reproducibility
np.random.seed(0)

# Call DFBGN
soln = dfbgn.solve(rosenbrock, x0, fixed_block=2)

# Display output
print(soln)

