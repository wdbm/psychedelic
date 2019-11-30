#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import setuptools

def main():
    setuptools.setup(
        name                 = 'psychedelic',
        version              = '2019.11.30.1817',
        description          = 'utilities for machine learning etc.',
        long_description     = long_description(),
        url                  = 'https://github.com/wdbm/psychedelic',
        author               = 'Will Breaden Madden',
        author_email         = 'wbm@protonmail.ch',
        license              = 'GPLv3',
        packages             = setuptools.find_packages(),
        install_requires     = [
                               'keras-vis',
                               'graphviz',
                               'jupyter',
                               'keras',
                               'keras_tqdm',
                               'livelossplot>=0.4.1',
                               'matplotlib',
                               'numpy',
                               'pandas',
                               'pydot',
                               'seaborn',
                               'scikit-learn',
                               'scipy',
                               'seaborn',
                               'talos',
                               'tensorflow-gpu==1.12',
                               'tqdm',
                               'umap-learn'
                               ],
        include_package_data = True,
        zip_safe             = False
    )

def long_description(filename='README.md'):
    if os.path.isfile(os.path.expandvars(filename)):
      try:
          import pypandoc
          long_description = pypandoc.convert_file(filename, 'rst')
      except ImportError:
          long_description = open(filename).read()
    else:
        long_description = ''
    return long_description

if __name__ == '__main__':
    main()
