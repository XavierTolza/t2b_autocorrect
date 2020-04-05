import os
import re
from multiprocessing import cpu_count
from os import listdir
from os.path import join

from setuptools import setup

name = 't2b'
DEBUG = os.environ.get("DEBUG")
DEBUG = DEBUG is not None and DEBUG == "1"

kwargs = {}
# Handle cython files
files = listdir(name)
r = re.compile(".+\.pyx")

try:
    from Cython.Build import cythonize
except ImportError:
    raise ImportError("You must first install cython to compile C extensions. pip install cython")

try:
    cython_files = [i for i in listdir(name) if r.fullmatch(i) is not None]
except AttributeError:
    cython_files = [i for i in listdir(name) if r.match(i) is not None]
if len(cython_files):
    from distutils.extension import Extension

    extra_compile_args = ['-fopenmp']
    if DEBUG:
        extra_compile_args.append("-O0")
    extensions = [Extension(join(name, i.split(".")[0]).replace("/", "."), [join(name, i)],
                            language="c++",
                            extra_compile_args=extra_compile_args,
                            extra_link_args=['-fopenmp'])
                  for i in cython_files]

    kwargs.update(dict(
        ext_modules=cythonize(extensions, annotate=True, gdb_debug=DEBUG, nthreads=cpu_count())
    ))

setup(
    name=name,
    version='1.0',
    description='Correction automatique du T2B',
    author='Xavier Tolza',
    author_email='tolza.xavier@gmail.com',
    packages=[name],  # same as name
    install_requires=[
        "bitstruct",
        "numpy",
        "pytools"
    ],
    dependency_links=['https://gitlab.com/api/v4/projects/7277028/repository/archive.tar.gz#egg=pytools-1.0'],
    **kwargs
)

# Pour compiler: python setup.py build_ext --inplace