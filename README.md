# SemiSupervisedGraphClustering
Implementation of the semi supervised graph clustering algorithm, also known as SS-KERNEL-KMEANS or SSK-KMEANS.

## Project
Implementation of the graph clustering presented in [Semi-supervised Graph Clustering: A Kernel Approach](http://www.cs.utexas.edu/~inderjit/public_papers/kernel_icml.pdf)

Kulis, Brian, et al. "Semi-supervised graph clustering: a kernel approach." Machine learning 74.1 (2009): 1-22.

## Remarks
### Constraints
One important point is that we represent constraints as a n by n matrix of values between -1 and 1, where -1 is a must not link and +1 is a must link constraint. All values between is allowed in order to represent the uncertainty of the user.

### Initialization
In this process the initialization is crucial: no cannot link constraint can be broken otherwise the algorithm will returns an error. In the same way if there is no clustering verifying the constraint, the current initialization will return an error.

### Possible improvement
Integrate this code in an active learning framework in order to limit the number of constraints necessary for the convergence of the algorithm.

## Dependencies
Code tested with python 3.5 with numpy and scipy and our Kernel Constrained Kmeans.  
Sklearn and matplotlib necessary for the example.

## Example
A simple example is given in the notebook `Example.ipynb` which explore the constraint clustering with an rbf kernel on [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.
