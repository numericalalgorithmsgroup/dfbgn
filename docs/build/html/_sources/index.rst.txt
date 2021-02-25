.. DFBGN documentation master file, created by
   sphinx-quickstart on Fri Oct 23 17:28:35 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DFBGN: Derivative-Free Block Gauss-Newton Optimizer for Least-Squares Minimization
==================================================================================

**Release:** |version|

**Date:** |today|

**Author:** `Lindon Roberts <lindon.roberts@anu.edu.au>`_

DFBGN is a package for finding local solutions to large-scale nonlinear least-squares minimization problems, without requiring any derivatives of the objective. DFBGN stands for Derivative-Free Block Gauss-Newton.

That is, DFBGN solves

.. math::

   \min_{x\in\mathbb{R}^n} \quad  f(x) := \sum_{i=1}^{m}r_{i}(x)^2

Full details of the DFBGN algorithm are given in our paper: Coralia Cartis and Lindon Roberts, `Scalable Subspace Methods for Derivative-Free Nonlinear Least-Squares Optimization <https://arxiv.org/abs/2102.12016>`_, *arXiv preprint arXiv:2102.12016*, (2021).

If you wish to solve small-scale least-squares problems, you may wish to try `DFO-LS <https://github.com/numericalalgorithmsgroup/dfols>`_. If you are interested in solving general optimization problems (without a least-squares structure), you may wish to try `Py-BOBYQA <https://github.com/numericalalgorithmsgroup/pybobyqa>`_.

DFBGN is released under the GNU General Public License. Please `contact NAG <http://www.nag.com/content/worldwide-contact-information>`_ for alternative licensing.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   userguide

Acknowledgements
----------------
This software was developed under the supervision of `Coralia Cartis <https://www.maths.ox.ac.uk/people/coralia.cartis>`_, and was supported by the EPSRC Centre For Doctoral Training in `Industrially Focused Mathematical Modelling <https://www.maths.ox.ac.uk/study-here/postgraduate-study/industrially-focused-mathematical-modelling-epsrc-cdt>`_ (EP/L015803/1) in collaboration with the `Numerical Algorithms Group <http://www.nag.com/>`_.

