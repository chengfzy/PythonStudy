# PythonStudy
Some Useful Code about Python3


# Contents
## util
1. heading: some useful code to format heading information

## math
1. line_fitting: 3D line fitting using SVD and optimization method
    - The opt method is not finished due to not familiar with the optimization using python, check matlab script for reference
1. plane_fitting: 3D plane fitting using SVD
1. opt_unconstrained_minimize  \
    Solve the Rosenbrock function of N variables. [Ref](https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#unconstrained-minimization-of-multivariate-scalar-functions-minimize)
1. opt_least_square \
    Solve least square function. [Ref](https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#least-squares-minimization-least-squares)
1. opt_bundle_adjustment    \
    Use SciPy to solve bundle adjustment. [Ref](https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html).
    Same problem using ceres can be found in 
    [link](http://ceres-solver.org/nnls_tutorial.html#bundle-adjustment)
1. huber_loss   \
    Plot the huber loss function, the huber loss are define in [wiki]( https://en.wikipedia.org/wiki/Huber_loss)
1. bspline_curve \
    Some function about B-spline.
    Ref: [scipy BSpline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html)
1. bspline  \
    B-Spline evaluation. \
    Ref: \
    1. [NURBS-Python Visualization](https://nurbs-python.readthedocs.io/en/latest/visualization.html).   \
    1. [NURBS-Python Example](https://github.com/orbingol/NURBS-Python_Examples/blob/master/visualization/mpl_curve2d_tangents.py).

## vision
1. show_image: show image in folder with some basic controller
1. resize_image: Read image in folder, resize it and save to file with .jpg format
1. generate_chessboard: generate chessboard and save into pdf file

# system
1. rename_files: rename files in folder

# net
1. book_converter.py: Convert a book in website with html format to PDF


# Notes
1. If some dependencies are need for OpenCV library, and the `PATH` and `LD_LIBRARY_PATH` are set by terminal, we should open `PyCharm` in terminal and then open python project.