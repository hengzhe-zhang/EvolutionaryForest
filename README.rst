===================
Evolutionary Forest
===================


.. image:: https://img.shields.io/pypi/v/evolutionary_forest.svg
        :target: https://pypi.python.org/pypi/evolutionary_forest

.. image:: https://img.shields.io/travis/com/zhenlingcn/evolutionaryforest.svg
        :target: https://www.travis-ci.com/github/zhenlingcn/EvolutionaryForest

.. image:: https://readthedocs.org/projects/evolutionary-forest/badge/?version=latest
        :target: https://evolutionary-forest.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/zhenlingcn/evolutionary_forest/shield.svg
     :target: https://pyup.io/repos/github/zhenlingcn/evolutionary_forest/
     :alt: Updates



An open source python library for automated feature engineering based on Genetic Programming


* Free software: BSD license
* Documentation: https://evolutionary-forest.readthedocs.io.


Introduction
----------------

Feature engineering is a long-standing issue that has plagued machine learning practitioners for many years. Deep learning techniques have significantly reduced the need for manual feature engineering in recent years. However, a critical issue is that the features discovered by deep learning methods are difficult to interpret.

In the domain of interpretable machine learning, genetic programming has demonstrated to be a promising method for automated feature construction, as it can improve the performance of traditional machine learning systems while maintaining similar interpretability. Nonetheless, such a potent method is rarely mentioned by practitioners. We believe that the main reason for this phenomenon is that there is still a lack of a mature package that can automatically build features based on the genetic programming algorithm. As a result, we propose this package with the goal of providing a powerful feature construction tool for enhancing existing state-of-the-art machine learning algorithms, particularly decision-tree based algorithms.

Features
----------------

*   A powerful feature construction tool for generating interpretable machine learning features.
*   A reliable machine learning model has powerful performance on the small dataset.

Installation
--------------------------------
From PyPI:

.. code:: bash

    pip install -U evolutionary_forest

From GitHub (Latest Code):

.. code:: bash

    pip install git+https://github.com/hengzhe-zhang/EvolutionaryForest.git

Supported Algorithms
--------------------------------
* `Evolutionary Forest (TEVC 2021) <https://github.com/hengzhe-zhang/EvolutionaryForest/blob/master/experiment/methods/EF.py>`_
* `SR-Forest (TEVC 2023) <https://github.com/hengzhe-zhang/EvolutionaryForest/blob/master/experiment/methods/SRForest.py>`_


Example
----------------
An example of usage:

.. code:: Python

    X, y = load_diabetes(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    r = EvolutionaryForestRegressor(max_height=3, normalize=True, select='AutomaticLexicase',
                                    gene_num=10, boost_size=100, n_gen=20, n_pop=200, cross_pb=1,
                                    base_learner='Random-DT', verbose=True)
    r.fit(x_train, y_train)
    print(r2_score(y_test, r.predict(x_test)))

An example of improvements brought about by constructed features:

.. image:: https://raw.githubusercontent.com/zhenlingcn/EvolutionaryForest/master/docs/constructed_features.png

Tutorials
----------------
Here are some nodebook examples of using Evolutionary Forest:

* `Regression on Diabetes Dataset`_

.. _Regression on Diabetes Dataset: https://github.com/hengzhe-zhang/EvolutionaryForest/blob/master/tutorial/diabetes_regression.ipynb

Documentation
----------------
Tutorial: `English Version`_ | `中文版本`_

.. _English Version: https://github.com/zhenlingcn/EvolutionaryForest/blob/master/tutorial/diabetes_regression.ipynb
.. _中文版本: https://github.com/zhenlingcn/EvolutionaryForest/blob/master/tutorial/diabetes_regression-CN.md

Credits
---------------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

Citation
---------------

Please cite our paper if you find it helpful :)

.. code::

    @article{zhang2021evolutionary,
      title={An Evolutionary Forest for Regression},
      author={Zhang, Hengzhe and Zhou, Aimin and Zhang, Hu},
      journal={IEEE Transactions on Evolutionary Computation},
      volume={26},
      number={4},
      pages={735--749},
      year={2021},
      publisher={IEEE}
    }

    @article{zhang2023sr,
      title={SR-Forest: A Genetic Programming based Heterogeneous Ensemble Learning Method},
      author={Zhang, Hengzhe and Zhou, Aimin and Chen, Qi and Xue, Bing and Zhang, Mengjie},
      journal={IEEE Transactions on Evolutionary Computation},
      year={2023},
      publisher={IEEE}
    }
