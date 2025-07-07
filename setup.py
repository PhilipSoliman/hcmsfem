import numpy
from setuptools import Extension, find_packages, setup

setup(
    packages=find_packages(where="."),
    package_dir={"": "."},
    ext_modules=[
        Extension(
            "hcmsfem.solvers.clib.custom_cg",
            ["hcmsfem/solvers/clib/custom_cg.cpp"],
            include_dirs=[numpy.get_include()],
            language="c++",
        )
    ],
)
