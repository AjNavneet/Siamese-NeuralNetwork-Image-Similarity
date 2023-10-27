import numpy as np

def get_squared_distance_matrix(x, diag_value=0.0):
    """
    Return the matrix of squared Euclidean distances

    :param x: array representing a set of N points in a vector space
    :param diag_value: value to put on the diagonal of the matrix
    :return: NxN matrix containing the squared Euclidean distance for each pair of points
    """
    # Expand the dimensions of the input array to prepare for distance calculation
    x1 = np.expand_dims(x, 1)
    x0 = np.expand_dims(x, 0)
    
    # Calculate the squared Euclidean distances between all pairs of points
    d2 = np.sum((x1 - x0) ** 2, 2)
    
    # Set the diagonal values of the distance matrix to the specified 'diag_value'
    d2[np.diag_indices_from(d2)] = diag_value
    
    # Return the resulting NxN distance matrix
    return d2
