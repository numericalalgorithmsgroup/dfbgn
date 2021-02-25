Using DFBGN
===========
This section describes the main interface to DFBGN and how to use it.

Nonlinear Least-Squares Minimization
------------------------------------
DFBGN is designed to solve the local optimization problem

.. math::

   \min_{x\in\mathbb{R}^n} \quad  f(x) := \sum_{i=1}^{m}r_{i}(x)^2

DFBGN iteratively constructs an interpolation-based model for the objective, and determines a step using a trust-region framework. For an in-depth technical description of the algorithm see the paper [CR2021]_.


How to use DFBGN
----------------
The main interface to DFBGN is via the function :code:`solve`

  .. code-block:: python
  
      soln = dfbgn.solve(objfun, x0, fixed_block=fixed_block)

The input :code:`objfun` is a Python function which takes an input :math:`x\in\mathbb{R}^n` and returns the vector of residuals :math:`[r_1(x)\: \cdots \: r_m(x)]\in\mathbb{R}^m`. Both the input and output of :code:`objfun` must be one-dimensional NumPy arrays (i.e. with :code:`x.shape == (n,)` and :code:`objfun(x).shape == (m,)`).

The input :code:`x0` is the starting point for the solver, and (where possible) should be set to be the best available estimate of the true solution :math:`x_{min}\in\mathbb{R}^n`. It should be specified as a one-dimensional NumPy array (i.e. with :code:`x0.shape == (n,)`).
As DFBGN is a local solver, providing different values for :code:`x0` may cause it to return different solutions, with possibly different objective values.

The input :code:`fixed_block` is the size of the exploration space. It should be an integer from 1 to :code:`len(x0)` inclusive, set based on how fast you want the internal linear algebra calculations to be (smaller values are faster).

The output of :code:`dfbgn.solve` is an object containing:

* :code:`soln.x` - an estimate of the solution, :math:`x_{min}\in\mathbb{R}^n`, a one-dimensional NumPy array.
* :code:`soln.resid` - the vector of residuals at the calculated solution, :math:`[r_1(x_{min})\:\cdots\: r_m(x_{min})]`, a one-dimensional NumPy array.
* :code:`soln.f` - the objective value at the calculated solution, :math:`f(x_{min})`, a Float.
* :code:`soln.nf` - the number of evaluations of :code:`objfun` that the algorithm needed, an Integer.
* :code:`soln.flag` - an exit flag, which can take one of several values (listed below), an Integer.
* :code:`soln.msg` - a description of why the algorithm finished, a String.
* :code:`soln.diagnostic_info` - a table of diagnostic information showing the progress of the solver, a Pandas DataFrame.

The possible values of :code:`soln.flag` are defined by the following variables:

* :code:`soln.EXIT_SUCCESS` - DFBGN terminated successfully (the objective value or trust region radius are sufficiently small).
* :code:`soln.EXIT_MAXFUN_WARNING` - maximum allowed objective evaluations reached. This is the most likely return value when using multiple restarts.
* :code:`soln.EXIT_SLOW_WARNING` - maximum number of slow iterations reached.
* :code:`soln.EXIT_FALSE_SUCCESS_WARNING` - DFBGN reached the maximum number of restarts which decreased the objective, but to a worse value than was found in a previous run.
* :code:`soln.EXIT_INPUT_ERROR` - error in the inputs.
* :code:`soln.EXIT_TR_INCREASE_ERROR` - error occurred when solving the trust region subproblem.
* :code:`soln.EXIT_LINALG_ERROR` - linear algebra error, e.g. the interpolation points produced a singular linear system.

These variables are defined in the :code:`soln` object, so can be accessed with, for example

  .. code-block:: python
  
      if soln.flag == soln.EXIT_SUCCESS:
          print("Success!")


A Simple Example
----------------
Suppose we wish to minimize the `Rosenbrock test function <https://en.wikipedia.org/wiki/Rosenbrock_function>`_:

.. math::

   \min_{(x_1,x_2)\in\mathbb{R}^2}  &\quad  100(x_2-x_1^2)^2 + (1-x_1)^2 \\

This function has exactly one local minimum :math:`f(x_{min})=0` at :math:`x_{min}=(1,1)`. We can write this as a least-squares problem as:

.. math::

   \min_{(x_1,x_2)\in\mathbb{R}^2}  &\quad  [10(x_2-x_1^2)]^2 + [1-x_1]^2 \\

A commonly-used starting point for testing purposes is :math:`x_0=(-1.2,1)`. The following script shows how to solve this problem using DFBGN:

  .. code-block:: python
  
      # DFBGN example: minimize the Rosenbrock function
      from __future__ import print_function
      import numpy as np
      import dfbgn

      # Define the objective function
      def rosenbrock(x):
          return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])
      
      # Define the starting point
      x0 = np.array([-1.2, 1.0])
      
      # DFBGN is a randomized algorithm - set random seed for reproducibility
      np.random.seed(0)

      # Call DFBGN
      soln = dfbgn.solve(rosenbrock, x0, fixed_block=2)
      
      # Display output
      print(soln)
      
Note that DFBGN is a randomized algorithm: the subspace it searches is randomly generated. The output of this script, showing that DFBGN finds the correct solution, is

  .. code-block:: none
  
      ****** DFBGN Results ******
      Solution xmin = [ 1.          0.99999998]
      Residual vector = [ -1.61462722e-07   0.00000000e+00]
      Objective value f(xmin) = 2.607021062e-14
      Needed 72 objective evaluations
      No approximate Jacobian available
      Exit flag = 0
      Success: Objective is sufficiently small
      ****************************

This and all following problems can be found in the `examples <https://github.com/numericalalgorithmsgroup/dfbgn/tree/master/examples>`_ directory on the DFBGN Github page.

More Output
-----------
We can get DFBGN to print out more detailed information about its progress using the `logging <https://docs.python.org/3/library/logging.html>`_ module. To do this, we need to add the following lines:

  .. code-block:: python
  
      import logging
      logging.basicConfig(level=logging.INFO, format='%(message)s')
      
      # ... (call dfbgn.solve)

And we can now see each evaluation of :code:`objfun`:

  .. code-block:: none
  
      Function eval 1 has f = 24.2 at x = [-1.2  1. ]
      Function eval 2 has f = 63.2346372977649 at x = [-1.30493146  0.94178154]
      Function eval 3 has f = 27.9653746738959 at x = [-1.25821846  1.10493146]
      Function eval 4 has f = 6.33451236346909 at x = [-1.08861669  1.04465151]
      ...
      Function eval 70 has f = 1.99643713755605e-12 at x = [ 1.          1.00000014]
      Function eval 71 has f = 110.765405382932 at x = [ 0.45748543 -0.84175933]
      Function eval 72 has f = 2.60702106219341e-14 at x = [ 1.          0.99999998]

If we wanted to save this output to a file, we could replace the above call to :code:`logging.basicConfig()` with

  .. code-block:: python
  
      logging.basicConfig(filename="myfile.log", level=logging.INFO, 
                          format='%(message)s', filemode='w')


Example: Noisy Objective Evaluation
-----------------------------------
As described in :doc:`info`, derivative-free algorithms such as DFBGN are particularly useful when :code:`objfun` has noise. Let's modify the previous example to include random noise in our objective evaluation, and compare it to a derivative-based solver:

  .. code-block:: python
  
      # DFBGN example: minimize the noisy Rosenbrock function
      from __future__ import print_function
      import numpy as np
      import dfbgn

      # Define the objective function
      def rosenbrock(x):
          return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])

      # Modified objective function: add 1% Gaussian noise
      def rosenbrock_noisy(x):
          return rosenbrock(x) * (1.0 + 1e-2 * np.random.normal(size=(2,)))

      # Define the starting point
      x0 = np.array([-1.2, 1.0])

      # Set random seed (for reproducibility)
      np.random.seed(0)

      print("Demonstrate noise in function evaluation:")
      for i in range(5):
          print("objfun(x0) = %s" % str(rosenbrock_noisy(x0)))
      print("")

      # Call DFBGN
      soln = dfbgn.solve(rosenbrock_noisy, x0, fixed_block=2)

      # Display output
      print(soln)

      # Compare with a derivative-based solver
      import scipy.optimize as opt
      soln = opt.least_squares(rosenbrock_noisy, x0)

      print("")
      print("** SciPy results **")
      print("Solution xmin = %s" % str(soln.x))
      print("Objective value f(xmin) = %.10g" % (2.0 * soln.cost))
      print("Needed %g objective evaluations" % soln.nfev)
      print("Exit flag = %g" % soln.status)
      print(soln.message)


The output of this is:

  .. code-block:: none
  
      Demonstrate noise in function evaluation:
      objfun(x0) = [-4.4776183   2.20880346]
      objfun(x0) = [-4.44306447  2.24929965]
      objfun(x0) = [-4.48217255  2.17849989]
      objfun(x0) = [-4.44180389  2.19667014]
      objfun(x0) = [-4.39545837  2.20903317]

      ****** DFBGN Results ******
      Solution xmin = [ 1.          0.99999994]
      Residual vector = [ -6.31017296e-07   5.73947373e-10]
      Objective value f(xmin) = 3.981831569e-13
      Needed 82 objective evaluations
      No approximate Jacobian available
      Exit flag = 0
      Success: Objective is sufficiently small
      ****************************


      ** SciPy results **
      Solution xmin = [-1.19999679  1.00000624]
      Objective value f(xmin) = 23.47462704
      Needed 8 objective evaluations
      Exit flag = 3
      `xtol` termination condition is satisfied.


DFBGN is able to find the solution with 10 more function evaluations as in the noise-free case. However SciPy's derivative-based solver, which has no trouble solving the noise-free problem, is unable to make any progress.

Example: Solving a Nonlinear System of Equations
------------------------------------------------
Lastly, we give an example of using DFBGN to solve a nonlinear system of equations (taken from `here <http://support.sas.com/documentation/cdl/en/imlug/66112/HTML/default/viewer.htm#imlug_genstatexpls_sect004.htm>`_). We wish to solve the following set of equations

.. math::

   x_1 + x_2 - x_1 x_2 + 2 &= 0, \\
   x_1 \exp(-x_2) - 1 &= 0.

The code for this is:

  .. code-block:: python
  
      # DFBGN example: Solving a nonlinear system of equations
      # Originally from:
      # http://support.sas.com/documentation/cdl/en/imlug/66112/HTML/default/viewer.htm#imlug_genstatexpls_sect004.htm

      from __future__ import print_function
      from math import exp
      import numpy as np
      import dfbgn

      # Want to solve:
      #   x1 + x2 - x1*x2 + 2 = 0
      #   x1 * exp(-x2) - 1   = 0
      def nonlinear_system(x):
          return np.array([x[0] + x[1] - x[0]*x[1] + 2,
                           x[0] * exp(-x[1]) - 1.0])

      # Warning: if there are multiple solutions, which one
      #          DFBGN returns will likely depend on x0!
      x0 = np.array([0.1, -2.0])

      # DFBGN is a randomized algorithm - set random seed for reproducibility
      np.random.seed(0)

      # Call DFBGN
      soln = dfbgn.solve(nonlinear_system, x0, fixed_block=2)

      # Display output
      print(soln)

The output of this is

  .. code-block:: none
  
      ****** DFBGN Results ******
      Solution xmin = [ 0.09777311 -2.32510592]
      Residual vector = [  2.38996951e-08   2.23316848e-07]
      Objective value f(xmin) = 5.044160988e-14
      Needed 18 objective evaluations
      No approximate Jacobian available
      Exit flag = 0
      Success: Objective is sufficiently small
      ****************************

Here, we see that both entries of the residual vector are very small, so both equations have been solved to high accuracy.

References
----------

.. [CR2021]   
   Coralia Cartis and Lindon Roberts, `Scalable Subspace Methods for Derivative-Free Nonlinear Least-Squares Optimization <https://arxiv.org/abs/2102.12016>`_, *arXiv preprint arXiv:2102.12016*, (2021).
