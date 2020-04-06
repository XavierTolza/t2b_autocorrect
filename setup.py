from setuptools import setup
import os
import re
from multiprocessing import cpu_count
from os import listdir
from os.path import join
from distutils.extension import Extension

DEBUG = os.environ.get("DEBUG")
DEBUG = DEBUG is not None and DEBUG == "1"

kwargs = {}
name = 't2b'

try:
    from Cython.Build import cythonize
except ImportError:
    raise ImportError("Vous devez installer cython en premier: pip install cython")

# Handle cython files
files = listdir(name)
r = re.compile(".+\.pyx")

try:
    cython_files = [i for i in listdir(name) if r.fullmatch(i) is not None]
except AttributeError:
    cython_files = [i for i in listdir(name) if r.match(i) is not None]
if len(cython_files):
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
    install_requires="""pillow
numpy
opencv-python
matplotlib
scipy
click
imutils""".split('\n'),
    entry_points={
        'console_scripts': ['t2b=t2b.cli:cli'],
    },
    package_data={name: ['t2b/*.txt', 't2b/*.xz']},
    include_package_data=True,
    data_files=['t2b/correction.txt', 't2b/motifs.tar.xz'],
    **kwargs
)

# Pour compiler: python setup.py build_ext --inplace
