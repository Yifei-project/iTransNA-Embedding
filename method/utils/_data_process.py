import numpy as np
# import warnings
# import math
from scipy.linalg import hankel
from scipy.spatial.distance import cdist
# from sklearn.neighbors import KDTree


def hankel_matrix(data, q, p=None):
    """
    Find the Hankel matrix dimensionwise for multiple multidimensional 
    time series
    
    Arguments
    data : [N, T, 1] or [N, T, D] ndarray
        A collection of N time series of length T and dimensionality D
    q : int
        The width of the matrix (the number of features)
    p : int
        The height of the matrix (the number of samples)
    
    """
    
    if len(data.shape) == 3:
        return np.stack([_hankel_matrix(item, q, p) for item in data])
    
    if len(data.shape) == 1:
        data = data[:, None]
    return _hankel_matrix(data, q, p)  
    

def _hankel_matrix(data, q, p=None):
    """
    Calculate the hankel matrix of a multivariate timeseries
    
    data : array
        T x D multidimensional time series
    """
    if len(data.shape) == 1:
        data = data[:, None]

    # Hankel parameters
    if not p:
        p = len(data) - q + 1
    all_hmats = list()
    for row in data.T:
        first, last = row[-(p + q -1) : -(p-1)], row[-p :]
        if (p==1):
            first, last = row[-(p + q -1) : ], row[-p :]
        out = hankel(first, last)
        all_hmats.append(out)
    out = np.dstack(all_hmats)
    return np.transpose(out, (1, 0, 2))#[:-1]


def standardize_ts(a, scale=1.0):
    """
    Standardize a T x D time series along its first dimension
    For dimensions with zero variance, divide by one instead of zero
    """
    stds = np.std(a, axis=0, keepdims=True)
    stds[stds==0] = 1
    return (a - np.mean(a, axis=0, keepdims=True))/(scale*stds)


def measurement_from_state(vec):
    """
    Generate corresponding measurements from (reconstucted) state vectors
    Usually, univariate time-delay state vector 'S_t = [s(t-m+1),...,s(t-1),s(t)]' represents the state at time
    t, so the corresponding measurement will be 's(t)'
    vec: (num_samples, embed_dim) [N,T] ndarray, return (num_samples,) ndarray 
    """
    # the last element of vector is the measurement
    m = vec[:,-1]
    return m


def rescale_attractor(vec, metric='minkowski', p=2):
    """
    vec: shape (num_samples, num_coordinates) a series of vectors (points) sampled in an attractor
    Rescale the attractor via standard deviation. By standard deviation we mean the estimate
    computed from the second moment of the distances from the mean of the reconstructed vectors.

    """
    
    std_ = cdist(vec, vec.mean(axis=0).reshape(-1, vec.shape[1]), metric=metric, p=p).std()
    assert std_ > 0
    return vec / std_

def filter_components(embed_vec, eig_val=None, threshold_strategy='hard', tol=2.3094, p=0.99):
    if threshold_strategy == 'hard' and eig_val is not None:
        # eig_val is a list whose elements are listed in descending order
        for ii in range(len(eig_val)):
            if eig_val[ii] < tol:
                break
        return embed_vec[:, :ii]
    L_var = []
    for i in range(embed_vec.shape[1]):
        L_var.append(np.std(embed_vec[:,i])**2)

    sum_var = sum(L_var)
    for i in range(embed_vec.shape[1]):
        L_var[i] = L_var[i] / sum_var

    indices = np.argsort(np.array(L_var))[::-1]
    # print(indices)
    total = L_var[indices[0]]
    coords = [embed_vec[:, indices[0]]]
    for i in range(1,len(indices)):
        coords.append(embed_vec[:, indices[i]])
        total += L_var[indices[i]]
        if total >= p:
            break
            
    coords = np.array(coords).T
    m = coords.shape[1] # the number of effective coordinates
    # if rescale:
    #     low = np.min(coords)
    #     high = np.max(coords)
    #     coords = (coords - low) / (high - low)
    return coords
