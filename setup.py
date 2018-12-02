#!/usr/bin/env python

import pip._internal
from setuptools import find_packages
from setuptools import setup


reqs = pip._internal.req.parse_requirements(
    'requirements.txt',
    session=pip._internal.download.PipSession()
)

install_requires = [str(req.req) for req in reqs]

setup(
    version='0.0.0',
    packages=find_packages(exclude=['tests*']),
    install_requires=install_requires,
    scripts=[],
    test_suite='nose.collector',
)
