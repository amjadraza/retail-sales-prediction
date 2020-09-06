=======================
retail-sales-prediction
=======================


.. image:: https://img.shields.io/pypi/v/retail_sales_prediction.svg
        :target: https://pypi.python.org/pypi/retail_sales_prediction

.. image:: https://travis-ci.com/amjadraza/retail-sales-prediction.svg?branch=master
    :target: https://travis-ci.com/amjadraza/retail-sales-prediction

.. image:: https://readthedocs.org/projects/retail-sales-prediction/badge/?version=latest
        :target: https://retail-sales-prediction.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


Python project to predict the sales of retail stores with machine learning. This project is
based on the data provided in Kaggle Competition.

Kaggle Link : https://www.kaggle.com/c/favorita-grocery-sales-forecasting/


* Free software: MIT license
* Documentation: https://retail-sales-prediction.readthedocs.io.

Project Environment
-------------------

We create the project environment using below command.

``conda env create -f environment.yml -p ./venv``

Update the existing conda environment

``conda env update -f environment.yml -p ./venv``

Activate the environment

``conda activate ./venv``


Features
--------
Machine learning pipeline to predict the sale forecasting. This project is the sand box
and needs a bit of work to complete it.

Currently it supports below features

* Running the Light GBM Model with fixed training, validation and test sets.
* Two variants of how unit_sales are filled NA. More can be added
* Notebooks with Exploratory data analysis
* Notebooks with Feature engineering and Model Training
* Documentation using Sphnix


TODO
----

Project Slides
==============

You can view the  project slides of my project at using this link_

.. _link: https://www.slideshare.net/AmjadRaza3/a-presentation-for-retail-sales-projects



Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
