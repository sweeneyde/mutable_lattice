from setuptools import Extension, setup
setup(
    ext_modules=[
        Extension(
            name="_mutable_lattice",
            sources=["src/_mutable_lattice.c"],
            # extra_compile_args=['-march=native'],
            undef_macros=['NDEBUG'],
        )
    ],
    packages=["mutable_lattice"],
)
