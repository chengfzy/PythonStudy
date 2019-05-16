# PythonStudy
Some Useful Code about Python3


# Contents
## common
1. debug_info: some useful code to format debug information

## math
1. line_fitting: 3D line fitting using SVD and optimization method
    - The opt method is not finished due to not familiar with the optimization using python, check matlab script for reference
1. plane_fitting: 3D plane fitting using SVD
1. opt_unconstrained_minimize  \
    Solve the Rosenbrock function of N variables. Ref: https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#unconstrained-minimization-of-multivariate-scalar-functions-minimize
1. opt_least_square \
    Solve least square function. Ref: https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#least-squares-minimization-least-squares
1. opt_bundle_adjustment    \
    Use SciPy to solve bundle adjustment. Ref: https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html.
    Save problem using ceres can be found in 
    http://ceres-solver.org/nnls_tutorial.html#bundle-adjustment
    
    

## vision
1. show_image: show image in folder with some basic controller
1. resize_image: Read image in folder, resize it and save to file with .jpg format

# Notes
1. If some dependencies are need for OpenCV library, and the `PATH` and `LD_LIBRARY_PATH` are set by terminal, we should open `PyCharm` in terminal and then open python project.