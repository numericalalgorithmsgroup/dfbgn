=========================================
DFBGN: Derivative-Free Block Gauss-Newton
=========================================

.. image::  https://travis-ci.org/numericalalgorithmsgroup/dfbgn.svg?branch=master
   :target: https://travis-ci.org/numericalalgorithmsgroup/dfbgn
   :alt: Build Status

.. image::  https://img.shields.io/badge/License-GPL%20v3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0
   :alt: GNU GPL v3 License

.. image:: https://img.shields.io/pypi/v/dfbgn.svg
   :target: https://pypi.python.org/pypi/dfbgn
   :alt: Latest PyPI version


DFBGN is a Python package for  nonlinear least-squares minimization, where derivatives are not available.
It is particularly useful when evaluations of the objective are expensive and/or noisy, and the number of variables to be optimized is large.

DFBGN is based on `DFO-LS <https://github.com/numericalalgorithmsgroup/dfols>`_, but is better-suited when there are many variables to be optimized (so the linear algebra in DFO-LS is too slow).
Unlike DFO-LS, DFBGN does not currently support bound constraints on the variables.

If you are interested in solving general optimization problems (without a least-squares structure), you may wish to try `Py-BOBYQA <https://github.com/numericalalgorithmsgroup/pybobyqa>`_.

Requirements
------------
DFBGN requires the following software to be installed:

* Python 2.7 or Python 3 (http://www.python.org/)

Additionally, the following python packages should be installed (these will be installed automatically if using *pip*, see `Installation using pip`_):

* NumPy 1.11 or higher (http://www.numpy.org/)
* SciPy 0.18 or higher (http://www.scipy.org/)
* Pandas 0.17 or higher (http://pandas.pydata.org/)

Installation using pip
----------------------
For easy installation, use `pip <http://www.pip-installer.org/>`_ as root:

 .. code-block:: bash

    $ [sudo] pip install dfbgn

or alternatively *easy_install*:

 .. code-block:: bash

    $ [sudo] easy_install dfbgn

If you do not have root privileges or you want to install DFBGN for your private use, you can use:

 .. code-block:: bash

    $ pip install --user dfbgn

which will install DFBGN in your home directory.

Note that if an older install of DFBGN is present on your system you can use:

 .. code-block:: bash

    $ [sudo] pip install --upgrade dfbgn

to upgrade DFBGN to the latest version.

Manual installation
-------------------
Alternatively, you can download the source code from `Github <https://github.com/numericalalgorithmsgroup/dfbgn>`_ and unpack as follows:

 .. code-block:: bash

    $ git clone https://github.com/numericalalgorithmsgroup/dfbgn
    $ cd dfbgn

DFBGN is written in pure Python and requires no compilation. It can be installed using:

 .. code-block:: bash

    $ [sudo] pip install .

If you do not have root privileges or you want to install DFBGN for your private use, you can use:

 .. code-block:: bash

    $ pip install --user .

instead.

To upgrade DFBGN to the latest version, navigate to the top-level directory (i.e. the one containing :code:`setup.py`) and rerun the installation using :code:`pip`, as above:

 .. code-block:: bash

    $ git pull
    $ [sudo] pip install .  # with admin privileges

Testing
-------
If you installed DFBGN manually, you can test your installation by running:

 .. code-block:: bash

    $ python setup.py test

Alternatively, the HTML documentation provides some simple examples of how to run DFBGN.

Examples
--------
Examples of how to run DFBGN are given in the `documentation <https://numericalalgorithmsgroup.github.io/dfbgn/>`_, and the `examples <https://github.com/numericalalgorithmsgroup/dfbgn/tree/master/examples>`_ directory in Github.

Uninstallation
--------------
If DFBGN was installed using *pip* you can uninstall as follows:

 .. code-block:: bash

    $ [sudo] pip uninstall dfbgn

If DFBGN was installed manually you have to remove the installed files by hand (located in your python site-packages directory).

Bugs
----
Please report any bugs using GitHub's issue tracker.

License
-------
This algorithm is released under the GNU GPL license. Please `contact NAG <http://www.nag.com/content/worldwide-contact-information>`_ for alternative licensing.
