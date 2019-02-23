#!/usr/bin/env python

import pip._internal
from setuptools import find_packages
from setuptools import setup
from setuptools import Extension

import numpy as np
from Cython.Distutils import build_ext


reqs = pip._internal.req.parse_requirements(
    'requirements.txt',
    session=pip._internal.download.PipSession()
)
install_requires = [str(req.req) for req in reqs]
ext_modules = [
    Extension('_qiqc.preprocessing.modules.normalizers.rulebase',
              sources=['qiqc/preprocessing/modules/normalizers/rulebase.pyx']),
    Extension('_qiqc.preprocessing.modules.tokenizers.word',
              sources=['qiqc/preprocessing/modules/tokenizers/word.pyx']),
    Extension('_qiqc.utils',
              sources=['qiqc/utils.pyx']),
]

setup(
    version='0.0.0',
    packages=find_packages(exclude=['tests*']),
    install_requires=install_requires,
    scripts=[],
    test_suite='nose.collector',
    include_dirs=[np.get_include()],
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)
