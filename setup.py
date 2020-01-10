#!/usr/bin/python

"""Copyright (c) 2019 Wyze Labs, Inc."""

import setuptools
with open('requirements.txt') as f:
    REQUIREMENTS = f.read().splitlines()

setuptools.setup(
    name ='audio_cnn',
    version = '0.1.0',
    install_requires = REQUIREMENTS,
    packages = setuptools.find_namespace_packages(),
    include_package_data = True,
    setup_requires = [],
    tests_requires= [],
    zip_safe = False,
    entry_points = '', 
)
