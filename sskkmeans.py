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
        affinity = affinity.copy() - np.diag(degrees)
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
    sigma = 0.001 * np.eye(n) # Value for diagonal shifting
    kernel = np.matmul(sigma + kernel, invWeights)

    # Oringial paper does not enforce constraints : Set to None
    return weightedKernelConstrainedKmeans(kernel, assignation, None, weights, max_iteration)

def holdOutSskmeans(affinities, k, objective, constraints, evaluation = None, max_iteration = 100):
    """
        Computes the performances of the different affinities matrices 
        On a hold out set of constraints
        Which maximizes the number of constraints respected

        Arguments:
            affinity {Array n*n} -- Affinity matrix
            k {Int} -- Number of cluster
            objective {str} -- Objective to compute ("ratio association", "ratio cut", "normalized cut")
            constraints {Array n*n} -- Constraints matrix with value between -1 and 1
            evaluation {Array n*n} -- Constraints matrix with value between -1 and 1 (Superset of constraints)

        Keyword Arguments:
            max_iteration {int} -- Maximum iteration (default: {100})

        Returns:
            Assignation - Array n
    """
    initializer = Initialization(k, constraints)

    best_affinity, best_score = None, None
    for affinity in affinities:
        initialization =  initializer.farthest_initialization(affinity)
        assignation = ssKmeans(affinity, initialization, objective, constraints)
        if evaluation is None:
            score = kta_score(constraints, assignation)
        else:
            score = kta_score(evaluation, assignation)
        
        if best_score is None or best_score < score:
            best_score = score
            best_affinity = affinity
            
    initializer = Initialization(k, evaluation)
    initialization =  initializer.farthest_initialization(best_affinity)
    return ssKmeans(best_affinity, initialization, objective, evaluation)
