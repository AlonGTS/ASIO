from setuptools import setup
from Cython.Build import cythonize

setup(
    name="gts_tracker",
    ext_modules=cythonize(
        "gts_tracker.pyx",
        compiler_directives={"language_level": "3"},
    ),
)
