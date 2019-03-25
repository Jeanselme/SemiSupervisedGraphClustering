"""
    Semi Supervised Graph Clustering
"""
import numpy as np
from KernelConstrainedKmeans.wkckmeans import weightedKernelConstrainedKmeans

def ssKmeans(affinity, assignation, objective, constraints, max_iteration = 100):
    """
        Semi supervised graph clustering

        Arguments:
            affinity {Array n*n} -- Affinity matrix
            assignation {Array n} -- Initial assignation 
            objective {str} -- Objective to compute ("ratio association", "ratio cut", "normalized cut")
            constraints {Array n*n} -- Constraints matrix with value between -1 and 1

        Keyword Arguments:
            max_iteration {int} -- Maximum iteration (default: {100})

        Returns:
            Assignation - Array n
    """
    # Number of points
    n = len(affinity)

    # Degrees affinity
    degrees = np.sum(affinity, axis = 0)

    if objective == "ratio association":
        weights = np.ones(n)
        affinity = affinity.copy() - degrees
    elif objective == "ratio cut":
        weights = np.ones(n)
    elif objective == "normalized cut":
        weights = degrees
    else:
        raise ValueError("Objective : {} unknown".format(objective))

    # Invert weights
    invWeights = np.diag(1. / weights)

    # Computes kernel
    kernel = np.matmul(invWeights, (affinity + constraints))
    sigma = - np.trace(kernel) / n # Value for diagonal shifting
    kernel = np.matmul(sigma + kernel, invWeights)

    # Oringial paper does not enforce constraints
    return weightedKernelConstrainedKmeans(kernel, assignation, None, weights, max_iteration)