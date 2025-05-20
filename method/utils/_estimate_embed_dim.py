import numpy as np
from scipy.spatial.distance import cdist

# From https://github.com/llbxg/hundun/blob/main/hundun/exploration/
# False Nearest Neighbors - Algorithm


def embedding(u_seq, T, D):

    dim, length = u_seq.ndim, len(u_seq)
    if len(u_seq.shape) == 1:
        u_seq = u_seq.reshape(length, dim)

    idx = np.arange(0, D, 1)*T
    e_seq = np.array([u_seq[idx+i, :] for i in range(length-(D-1)*T)])

    if u_seq.shape[1] == 1:
        e_seq = e_seq.reshape(len(e_seq), D)
    return e_seq

def FalseNearestNeighbors(u_seq, T, D_max=10,
                        threshold_R=10, threshold_A=2, threshold_percent=1):

    def _dist(seq):
        return cdist(seq, seq, metric='euclidean')

    R_A = np.std(u_seq)

    e_seq_list = [embedding(u_seq, T, j) for j in range(1, D_max+2)]

    percentage_list = []
    for e_seq1, e_seq2 in zip(e_seq_list, e_seq_list[1:]):
        dist1 = _dist(e_seq1) + np.eye(len(e_seq1))*10000
        dist2 = _dist(e_seq2)
        l2 = len(dist2)
        idx_n_list = dist1[:l2, :l2].argmin(axis=0)

        dist1_min = dist1.min(axis=0)
        dist2_min = np.array([d[idx] for idx, d in zip(idx_n_list, dist2)])

        percentage = 0
        for R_Dk1, R_Dk2 in zip(dist1_min, dist2_min):
            d = np.sqrt((R_Dk2**2-R_Dk1**2)/(R_Dk1**2))
            criterion_1 = d > threshold_R
            criterion_2 = R_Dk2/R_A >= threshold_A
            if criterion_1 and criterion_2:
                percentage += 1
        percentage_list.append(percentage*(1/len(e_seq2)))

    percentages = np.array(percentage_list)*100
    dranges = np.arange(1, len(percentages)+1)
    dimension = dranges[percentages < threshold_percent][0]


    return dimension, percentages


def AveragedFalseNeighbors(u_seq, T, D_max=10, threshold_E1=0.9, threshold_E2=1):

    def _dist(seq):
        return cdist(seq, seq, metric='chebyshev')

    R_A = np.std(u_seq)

    e_seq_list = [embedding(u_seq, T, j) for j in range(1, D_max+3)]

    a_list, b_list = [], []
    for e_seq1, e_seq2 in zip(e_seq_list, e_seq_list[1:]):
        dist1 = _dist(e_seq1) + np.eye(len(e_seq1))*10000
        dist2 = _dist(e_seq2)
        l2 = len(dist2)
        idx_n_list = dist1[:l2, :l2].argmin(axis=0)

        dist1_min = dist1.min(axis=0)
        dist2_min = np.array([d[idx] for idx, d in zip(idx_n_list, dist2)])

        # E2の計算のためのEの計算
        b = [np.abs(e_line[-1] - e_seq2[idx][-1])/R_A
             for idx, e_line in zip(idx_n_list, e_seq2)]
        b_bar = np.average(b)
        b_list.append(b_bar)

        # E1の計算のためのEの計算
        a = [R_Dk2/R_Dk1 for R_Dk1, R_Dk2 in zip(dist1_min, dist2_min)]
        a_bar = np.average(a)
        a_list.append(a_bar)

    E_list = []
    for c in [a_list, b_list]:
        criterion = []
        for tau, tau_p1 in zip(c, c[1:]):
            criterion.append(tau_p1/tau)
        E_list.append(np.array(criterion))

    dranges = np.arange(1, len(E_list[0])+1)

    dimension_list = []
    for E, threshold in zip(E_list, [threshold_E1, threshold_E2]):
        rule = (E > threshold)
        if len(rule) > 0:
            dim = dranges[rule][0]
        else:
            dim = None
        dimension_list.append(dim)


    return dimension_list, E_list