#!/usr/bin/env python3

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

from setuptools import setup

# Get package version without "import dfbgn" (which requires dependencies to already be installed)
import os
version = {}
with open(os.path.join('dfbgn', 'version.py')) as fp:
    exec(fp.read(), version)
__version__ = version['__version__']

setup(
    name='dfbgn',
    version=__version__,
    description='A derivative-free solver for large-scale nonlinear least-squares minimization',
    long_description=open('README.rst').read(),
    author='Lindon Roberts',
    author_email='lindon.roberts@anu.edu.au',
    url='https://github.com/numericalalgorithmsgroup/dfbgn/',
    download_url='https://github.com/numericalalgorithmsgroup/dfbgn/archive/v0.1.tar.gz',
    packages=['dfbgn'],
    license='GNU GPL',
    keywords = 'mathematics derivative free optimization nonlinear least squares',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Framework :: IPython',
        'Framework :: Jupyter',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        ],
    install_requires = ['numpy >= 1.11', 'scipy >= 0.18', 'pandas >= 0.17'],
    test_suite='nose.collector',
    tests_require=['nose'],
    zip_safe = True,
    )

