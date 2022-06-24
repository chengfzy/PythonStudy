# PythonStudy
> Some study code using Python

[TOC]


## Contents
### util
1. heading: some useful code to format heading information

### system
1. rename_files: rename files in folder
1. elapsed_time: code to print elapsed time
1. tools: some system tools

### net
1. book_converter: Convert a book in website with html format to PDF

### others
1. stock: to monitor the stock price in terminal
1. stock_cost_calculate: calculate the cost and profile of stock


### math
1. sophus: python version using Sophus
1. line_fitting: 3D line fitting using SVD and optimization method
    - The opt method is not finished due to not familiar with the optimization using python, check matlab script for reference
1. plane_fitting: 3D plane fitting using SVD
1. opt_unconstrained_minimize  \
    Solve the Rosenbrock function of N variables. [Ref](https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#unconstrained-minimization-of-multivariate-scalar-functions-minimize)
1. opt_least_square \
    Solve least square function. [Ref](https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#least-squares-minimization-least-squares)
1. opt_bundle_adjustment    \
    Use SciPy to solve bundle adjustment. [Ref](https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html).
    Same problem using ceres can be found in [link](http://ceres-solver.org/nnls_tutorial.html#bundle-adjustment)
1. huber_loss   \
    Plot the huber loss function, the huber loss are define in [wiki]( https://en.wikipedia.org/wiki/Huber_loss)
1. compute_correlation \
    Calculate the time shift using correlation, [Ref](https://towardsdatascience.com/computing-cross-correlation-between-geophysical-time-series-488642be7bf0)
1. bspline_curve \
    Some function about B-spline.
    Ref: [scipy BSpline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html)
1. bspline
    B-Spline evaluation.
    Ref:
    1. [NURBS-Python Visualization](https://nurbs-python.readthedocs.io/en/latest/visualization.html).
    1. [NURBS-Python Example](https://github.com/orbingol/NURBS-Python_Examples/blob/master/visualization/mpl_curve2d_tangents.py).

### vision
1. misc: convert color, show image, etc
1. color: define the most used colors using OpenCV
1. show_image: show image in folder with some basic controller
1. resize_image: Read image in folder, resize it and save to file with .jpg format
1. generate_chessboard: generate chessboard and save into pdf file
1. generate_april_target: generate April target to pdf file


### ml
##### linear regression
1. linear_regression \
    Linear regression example.
    Ref: [scikit learn example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py)
1. non_negative_least_squares \
    Non negative least square regression
    Ref: [scikit learn example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_nnls.html#sphx-glr-auto-examples-linear-model-plot-nnls-py)



### dl
Some code about deep learning
1. coco2yolo: parse COCO dataset for selected categories and save to YOLO format, it's used for train.

##### torch_study
Some study notes using `PyTorch`
1. 01_network   \
    The neural networks.
    Ref: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
1. 02_train_classifier
1. 03_finetuning \
    Fine tuning torchvision models.
    Ref: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#


## Notes
1. If some dependencies are need for OpenCV library, and the `PATH` and `LD_LIBRARY_PATH` are set by terminal, we should open `PyCharm` in terminal and then open python project.