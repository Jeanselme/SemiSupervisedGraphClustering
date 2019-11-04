"""
    Semi Supervised Graph Clustering
"""
import numpy as np
from numba import jit
from KernelConstrainedKmeans.initialization import Initialization
from KernelConstrainedKmeans.wkckmeans import weightedKernelConstrainedKmeans

@jit(nopython=True)
def fast_verification(row, col, data, assignation):
    respected, broken = 0, 0
    for i, j, val in zip(row, col, data):   
        if assignation[i] == assignation[j] and val > 0:
            respected += 1
        elif assignation[i] != assignation[j] and val < 0:
            respected += 1
        else:
            broken += 1
    return respected, broken

def verification_constraint(constraint_matrix, assignation):
    """
        Returns the number of constraint verified and broken
        
        Arguments:
            constraint_matrix {Array n*n} -- Constraint matrix
            assignation {Array n} -- Assignation

        Returns:
            number constraint respected, number constraint broken
    """
    return fast_verification(constraint_matrix.row, constraint_matrix.col, constraint_matrix.data, assignation)

def kta_score(constraint_matrix, assignation):
    """
        Returns the kta score
        
        Arguments:
            constraint_matrix {Array n*n} -- Constraint matrix
            assignation {Array n} -- Assignation

        Returns:
            KTA Score
    """
    respected, broken = verification_constraint(constraint_matrix, assignation)
    return respected / (respected + broken)

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

def crossValidationSskmeans(affinities, k, objective, constraints, max_iteration = 100):
    """
        Computes a cross validation on the different affinities matrices 
        Which maximizes the number of constraints respected

        Arguments:
            affinity {Array n*n} -- Affinity matrix
            k {Int} -- Number of cluster
            objective {str} -- Objective to compute ("ratio association", "ratio cut", "normalized cut")
            constraints {Array n*n} -- Constraints matrix with value between -1 and 1

        Keyword Arguments:
            max_iteration {int} -- Maximum iteration (default: {100})

        Returns:
            Assignation - Array n
    """
    initializer = Initialization(k, constraints)

    best_assignation, best_score = None, None
    for affinity in affinities:
        initialization =  initializer.farthest_initialization(affinity)
        assignation = ssKmeans(affinity, initialization, objective, constraints)
        score = kta_score(constraints, assignation)
        
        if best_score is None or best_score < score:
            best_score = score
            best_assignation = assignation
        
    return best_assignation

