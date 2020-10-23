Installing DFBGN
================

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

Uninstallation
--------------
If DFBGN was installed using *pip* you can uninstall as follows:

 .. code-block:: bash

    $ [sudo] pip uninstall dfbgn

If DFBGN was installed manually you have to remove the installed files by hand (located in your python site-packages directory).
