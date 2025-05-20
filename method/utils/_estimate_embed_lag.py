import numpy as np
from sklearn.metrics import mutual_info_score


def mutual_information(X, max_lag):
    '''
    Calculates the mutual information between the an unshifted time series
	and a shifted time series. Utilizes scikit-learn's implementation of
	the mutual information found in sklearn.metrics.
	Parameters
	----------
    X : 1-D time series, array-like
	max_lag : integer
	    maximum amount to shift the time series
	Returns
	-------
	m_score : 1-D array
	    mutual information at between the unshifted time series and the
	    shifted time series
	'''

	#number of bins - say ~ 20 pts / bin for joint distribution
	#and that at least 4 bins are required
    N = max(X.shape)
    num_bins = max(4.,np.floor(np.sqrt(N/20)))
    num_bins = int(num_bins)

    m_score = np.zeros((max_lag))

    for jj in range(max_lag):
        lag = jj+1

        ts = X[0:-lag]
        ts_shift = X[lag::]

        min_ts = np.min(X)
        max_ts = np.max(X)+.0001 #needed to bin them up

        bins = np.linspace(min_ts,max_ts,num_bins+1)

        bin_tracker = np.zeros_like(ts)
        bin_tracker_shift = np.zeros_like(ts_shift)

        for ii in range(num_bins):

            locs = np.logical_and( ts>=bins[ii], ts<bins[ii+1] )
            bin_tracker[locs] = ii

            locs_shift = np.logical_and( ts_shift>=bins[ii], ts_shift<bins[ii+1] )
            bin_tracker_shift[locs_shift]=ii


        m_score[jj] = mutual_info_score(bin_tracker,bin_tracker_shift)
    return m_score


