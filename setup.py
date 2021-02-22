from setuptools import setup, find_packages
import sys


try:
    from Cython.Distutils import build_ext
except ImportError:
    def build_ext(*args, **kwargs):
        from Cython.Distutils import build_ext
        return build_ext(*args, **kwargs)


class lazy_extlist(list):
    def __init__(self, callback):
        self._list, self.callback = None, callback

    def c_list(self):
        if self._list is None:
            self._list = self.callback()
        return self._list

    def __iter__(self):
        for e in self.c_list():
            yield e

    def __getitem__(self, ii):
        return self.c_list()[ii]

    def __len__(self):
        return len(self.c_list())


def extensions():
    __builtins__.__NUMPY_SETUP__ = False
    from Cython.Distutils import Extension
    import numpy as np
    extra_compile_args = ["-O3"]
    extra_link_args = []
    if sys.platform == "darwin":
        extra_compile_args.append("-mmacosx-version-min=10.9")
        extra_compile_args.append('-stdlib=libc++')
        extra_link_args.append('-stdlib=libc++')
    return [Extension(
        'valens.dtw',
        ["valens/dtw.pyx"],
        cython_directives={'language_level': sys.version_info[0]},
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[np.get_include()],
        language="c++")
    ]


setup(
    name="valens",
    description='Computer vision module for valens',
    version="0.0.1",
    #long_description=open('README.rst').read(),
    packages=find_packages(),
    setup_requires=["numpy", 'cython'],
    ext_modules=lazy_extlist(extensions),
    cmdclass={'build_ext': build_ext},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 2",
    ]
)

