import numpy as np
from itertools import combinations as _combinations
from typing import NamedTuple as _NamedTuple

# From https://github.com/llbxg/hundun/blob/main/hundun/exploration/_box3.py


class _DimensionResult(_NamedTuple):
    dimension: float  # 次元
    idx_dimension: int  # 最も相関係数が高かった時のインデックス

    min_correlation: float
    correlations: np.ndarray  # 相関係数
    slopes: np.ndarray  # 1次関数によるフィッテイング時の傾き
    intercepts: np.ndarray  # 1次関数によるフィッテイング時の切片

    log_frac_1_eps: np.ndarray
    log_Ns: np.ndarray
    dimensions: np.ndarray  # 定義に沿った容量次元


class CalcDimension(object):

    def __init__(self, u_seq, scale_down, depsilon, base, loop,
                 batch_ave, min_correlation):
        self.dim, self.u_seq = self.check_dim(u_seq)
        if scale_down:
            self.u_seq = _scale_down(self.u_seq)

        self.batch_ave = batch_ave
        self.min_correlation = min_correlation
        self.config_accuracy = (depsilon, base, loop)

        self.length = len(self.u_seq)

    def __call__(self, q=0):
        result = self.main()

        return result.dimension

    def main(self):
        accuracies, values = self.calc()

        log_accuracies = self.wrap_accuracies_in_main(accuracies)
        new_values = self.wrap_value_in_main(values)
        dimensions = new_values / (log_accuracies + 1e-8)

        correlations, slopes, intercepts = \
            self._get_correlations_and_slopes(new_values, log_accuracies)
        self.correlations = correlations
        self.slopes = slopes

        idx = self._decide_idx_ref_mode(correlations)

        dimension = self.decide_dimension(idx, dimensions)

        return _DimensionResult(float(dimension), idx, self.min_correlation,
                                correlations, slopes, intercepts,
                                log_accuracies, new_values, dimensions)

    def calc(self):
        '''
        accuracyごとに計算(func)を行う.
        '''
        accuracies = self.make_accuracies(*self.config_accuracy)
        value_list = [self.func(epsilon) for epsilon in accuracies]
        return accuracies, np.array(value_list)

    def func(self, epsilon):
        '''
        value = func(epsilon)
        '''

        x = self.u_seq[:, 0]
        xedges = self.make_edges(x, epsilon)

        if self.dim == 1:
            value = self.func_for_1dim(x, xedges)

        elif self.dim == 2:
            y = self.u_seq[:, 1]
            yedges = self.make_edges(y, epsilon)
            value = self.func_for_2dim(x, y, xedges, yedges)

        elif self.dim == 3:
            y, z = self.u_seq[:, 1], self.u_seq[:, 2]
            yedges = self.make_edges(y, epsilon)
            zedges = self.make_edges(z, epsilon)
            value_list = []
            for z_left in zedges:
                z_right = z_left + epsilon
                new_u_seq = self.u_seq[(z_left < z) & (z <= z_right)]
                new_x, new_y = new_u_seq[:, 0], new_u_seq[:, 1]
                value_list.append(self.func_for_2dim(new_x, new_y,
                                                     xedges, yedges))
            value = self.wrap_value_3dim(value_list)

        else:
            value = 0

        return self.wrap_value(value)

    def func_for_1dim(self, x, xedges):
        return 0

    def func_for_2dim(self, x, y, xedges, yedges):
        return 0

    def decide_dimension(self, idx, dimensions):
        return np.average(dimensions[idx:idx+self.batch_ave])

    def _decide_idx_ref_mode(self, correlations):
        correlations_over = np.where(
            correlations >= self.min_correlation, correlations, 0)

        idx = int(np.argmax(correlations_over))

        return idx

    def _get_correlations_and_slopes(self, h_seq, v_seq):
        batch_ave = self.batch_ave
        correlation_list, slope_list, intercept_list = [], [], []

        for i in range(len(h_seq)-batch_ave):
            h_seq_batch, v_seq_batch = (h_seq[i:i+batch_ave],
                                        v_seq[i:i+batch_ave])

            correlation = np.corrcoef(h_seq_batch, v_seq_batch)[0, 1]
            correlation_list.append(correlation)

            slope_now, intercept = np.polyfit(v_seq_batch, h_seq_batch, 1)
            slope_list.append(slope_now)
            intercept_list.append(intercept)

        correlations = np.array(correlation_list)
        slopes = np.array(slope_list)
        intercepts = np.array(intercept_list)

        return correlations, slopes, intercepts

    @staticmethod
    def wrap_value(value):
        return value

    @staticmethod
    def wrap_accuracies_in_main(accuracies):
        return np.log(1/accuracies + 1e-8)

    @staticmethod
    def wrap_value_3dim(value_list):
        return sum(value_list)

    @staticmethod
    def wrap_value_in_main(values):
        return np.log(values + 1e-8)

    @staticmethod
    def make_edges(a, ep):
        return np.arange(np.min(a)-ep, np.max(a)+2*ep, ep)

    @staticmethod
    def check_dim(u_seq):
        if len(u_seq.shape) == 1:
            u_seq = u_seq.reshape(len(u_seq), 1)
        return u_seq.shape[1], u_seq

    @staticmethod
    def make_accuracies(depsilon=0.02, base=7, loop=250):
        '''
        eベースでのaccuracyを作成する. 刻み幅はdepsilonとし,
        最小はe^(-base), 最大はe^((loop-1)*depsilon-base)となる.
        デフォルトではe^(-2)からe^(-7)のaccuracyのリストを返す.
        '''
        epsilon_list = [np.e**(i*depsilon-base) for i in range(loop+1)]
        return np.array(epsilon_list)[::-1]


class Capacity(CalcDimension):

    def func_for_1dim(self, x, xedges):
        H, _ = np.histogram(x, bins=xedges)
        return np.sum(H > 0)

    def func_for_2dim(self, x, y, xedges, yedges):
        H, _, _ = np.histogram2d(x, y, bins=(xedges, yedges))
        return np.sum(H > 0)


class Information(CalcDimension):

    def func_for_1dim(self, x, xedges):
        H, _ = np.histogram(x, bins=xedges)
        p = H/self.length
        return p[p > 0]

    def func_for_2dim(self, x, y, xedges, yedges):
        H, _, _ = np.histogram2d(x, y, bins=(xedges, yedges))
        p = H/self.length
        return p[p > 0]

    @staticmethod
    def wrap_value(p):
        return -1*np.sum(np.multiply(p, np.log2(p + 1e-8)))

    @staticmethod
    def wrap_value_3dim(value_list):
        return np.concatenate(value_list)

    @staticmethod
    def wrap_value_in_main(values):
        return values

    @staticmethod
    def wrap_accuracies_in_main(accuracies):
        return np.log2(1/accuracies + 1e-8)


class Correlation(CalcDimension):
    '''
    Grassberger-Procaccia Algorithm (グラスバーガー - プロカッチャ アルゴリズム)
    '''
    def __init__(self, *args, **kwargs):
        super(Correlation, self).__init__(*args, **kwargs)
        self.distance = np.array(
            [_dist(x_i, x_j) for x_i, x_j in _combinations(self.u_seq, 2)])

    def calc(self):
        '''
        accuracyごとに計算(func)を行う.
        '''
        accuracies = self.make_accuracies(*self.config_accuracy)
        crs = 2*np.array(
            [_correlation_integrals(r, self.distance, len(self.u_seq))
             for r in accuracies])

        return accuracies, np.array(crs)

    def decide_dimension(self, idx, dimensions):
        return np.average(self.slopes[
            self.min_correlation <= self.correlations])

    @staticmethod
    def wrap_value_in_main(values):
        return -np.log(values + 1e-8)


def _dist(a, b):
    return np.linalg.norm(a - b)


def _correlation_integrals(r, distance, N):
    return np.sum(r > distance)/(N**2)


def calc_dimension_capacity(u_seq, depsilon=0.02, base=7, loop=250,
                            min_correlation=0.999, scale_down=True,
                            batch_ave=10):

    capacity = Capacity(
        u_seq, scale_down=scale_down,
        depsilon=depsilon, base=base, loop=loop,
        batch_ave=batch_ave, min_correlation=min_correlation
        )

    return capacity()


def calc_dimension_information(u_seq, depsilon=0.02, base=7, loop=250,
                               min_correlation=0.999, scale_down=True,
                               batch_ave=10):

    infomation = Information(
        u_seq, scale_down=scale_down,
        depsilon=depsilon, base=base, loop=loop,
        batch_ave=batch_ave, min_correlation=min_correlation
        )

    return infomation(q=1)


def calc_dimension_correlation(u_seq, depsilon=0.02, base=7, loop=250,
                               min_correlation=0.999, scale_down=True,
                               batch_ave=10):

    correlation = Correlation(
        u_seq, scale_down=scale_down,
        depsilon=depsilon, base=base, loop=loop,
        batch_ave=batch_ave, min_correlation=min_correlation
        )

    return correlation(q=2)


def _scale_down(seq):
    v_max = seq.max(axis=0, keepdims=True)
    v_min = seq.min(axis=0, keepdims=True)
    return seq/np.max(v_max-v_min)