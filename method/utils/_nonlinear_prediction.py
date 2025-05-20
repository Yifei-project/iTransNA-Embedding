import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist

def simple_nonlinear_predict(past_embedding, past_value, reference_embedding, horizon, R=1, metric='minkowski', p=2):

    Tree = KDTree(past_embedding, metric=metric, p=p)

    inds = Tree.query(reference_embedding, k=R, return_distance=False)

    inds_ = (inds+horizon) % past_value.shape[0]
    preds = np.mean(past_value[inds_], axis=1)

    return preds