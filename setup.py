from setuptools import Extension, setup
setup(
    ext_modules=[
        Extension(
            name="_mutable_lattice",
            sources=["_mutable_lattice.c"],
            # To enable architecture-specific optimizations
            # extra_compile_args=['-march=native'],
            # To enable assertions:
            # undef_macros=['NDEBUG'],
        )
    ],
    packages=["mutable_lattice"],
)
