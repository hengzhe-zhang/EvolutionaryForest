#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst', encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst', encoding='utf-8') as history_file:
    history = history_file.read()

requirements = [
    'scipy',
    'hdfe',
    'numpy',
    'seaborn',
    'matplotlib',
    'deap',
    'sympy',
    'pandas',
    'scikit_learn',
    'dill',
    'lightgbm',
    'smt',
    'pytest',
    'joblib',
    'linear-tree',
    'mlxtend',
    'sklearn2pmml',
    'tpot',
    'gplearn',
    'scorch',
    'umap-learn',
    'category_encoders',
]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Hengzhe Zhang",
    author_email='zhenlingcn@foxmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="An open source python library for automated feature engineering based on Genetic Programming",
    install_requires=requirements,
    license="BSD license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='evolutionary_forest',
    name='evolutionary_forest',
    packages=find_packages(include=['evolutionary_forest', 'evolutionary_forest.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/zhenlingcn/evolutionary_forest',
    version='0.2.4',
    zip_safe=False,
)
