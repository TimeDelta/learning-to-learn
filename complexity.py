import numpy as np
import antropy
from astropy.stats import bayesian_blocks
from sklearn.neighbors import NearestNeighbors
from pyentrp import entropy

SERIES_STATS = [
    # lambda to map series into single value
    lambda series: len(series),
    lambda series: np.mean(hurst_exponent(series)),
    lambda series: lempel_ziv_complexity_continuous(series, quantize_signal_bayesian_block_feature_bins),
    lambda series: np.mean(optimized_multiscale_permutation_entropy(series)),
    lambda series: differential_entropy(series, quantize_signal_bayesian_block_feature_bins),
]

def differential_entropy(data, quantizer):
    discrete_signal = quantizer(data)
    hist, _ = np.histogram(discrete_signal, density=True)
    nonzero = hist > 0
    return -np.sum(hist[nonzero] * np.log(hist[nonzero])) / np.log(len(np.unique(discrete_signal)))

def lempel_ziv_complexity_continuous(data, quantizer):
    symbol_seq = quantizer(data)
    phrase_start = 0
    complexity = 0

    while phrase_start < len(symbol_seq):
        phrase_length = 1
        while True:
            # so that a substring of target phrase length sits entirely before phrase_start
            max_prefix_start = phrase_start - phrase_length + 1

            if max_prefix_start > 0:
                # all substrings of phrase_length in the prefix [0 : phrase_start]
                previous_substrings = {
                    tuple(symbol_seq[k : k + phrase_length])
                    for k in range(max_prefix_start)
                }
            else:
                previous_substrings = set()

            end_of_candidate = phrase_start + phrase_length

            # does it still perfectly match something in the prefix?
            if (
                end_of_candidate <= len(symbol_seq)
                and tuple(symbol_seq[phrase_start : end_of_candidate])
                in previous_substrings
            ):
                phrase_length += 1
                continue
            else:
                break
        complexity += 1
        phrase_start += phrase_length
    alphabet_size = len(np.unique(symbol_seq))
    max_complexity = len(symbol_seq) / np.emath.logn(alphabet_size, len(symbol_seq))
    return complexity / max_complexity

def _hurst_exponent_1d(data):
    """
    slope of log-log regression
    """
    RS = []
    # use logspace for mixed local / ranged correlation structure
    window_sizes = np.unique(np.floor(np.logspace(np.log10(10), np.log10(data.shape[0] // 2), num=20)).astype(int))
    window_sizes = window_sizes[window_sizes > 0]
    for window in window_sizes:
        n_segments = len(data) // window
        RS_vals = []
        for i in range(n_segments):
            segment = data[i * window:(i + 1) * window]
            mean_seg = np.mean(segment)
            Y = segment - mean_seg
            cumulative_Y = np.cumsum(Y)
            R = np.max(cumulative_Y) - np.min(cumulative_Y)
            S = np.std(segment)
            if S != 0:
                RS_vals.append(R / S)
        if RS_vals:
            RS.append(np.mean(RS_vals))
    if len(RS) == 0:
        raise ValueError("No valid RS values computed; check window sizes and data.")
    logs = np.log(window_sizes[:len(RS)])
    log_RS = np.log(RS)
    mask = np.isfinite(logs) & np.isfinite(log_RS)
    logs, log_RS = logs[mask], log_RS[mask]
    slope, _ = np.polyfit(logs, log_RS, 1)
    return slope

def hurst_exponent(data):
    if data.ndim == 1:
        return _hurst_exponent_1d(data)
    # compute separately for each feature (column)
    n_samples, n_features = data.shape
    hurst_vals = []
    for f_i in range(n_features):
        col = data[:, f_i]
        hurst_vals.append(_hurst_exponent_1d(col))
    return hurst_vals

def optimized_multiscale_permutation_entropy(time_series) -> float:
    """
    Compute the mean Multiscale Permutation Entropy (MPE) over:
      - orders m = 2 and 3 (averaged)
      - delays swept from min_delay to max_delay (averaged)
      - scale fixed to 3
    """
    scale = 3
    delays = list(range(1, time_series.shape[0]//20))
    def single_feature(feature_series):
        mpe_vals = []
        for order in [2, 3]: # Orders to average over (maintains N ≫ m! guideline)
            for delay in delays:
                mpe = entropy.multiscale_permutation_entropy(feature_series, order, delay, scale) / np.log2(math.factorial(order))
                mpe_vals.append(mpe.mean())
        return float(np.mean(mpe_vals))
    if time_series.ndim == 1:
        return single_feature(time_series)
    per_feature = []
    for f_i in range(time_series[0].shape[0]):
        per_feature.append(single_feature(time_series[:,f_i]))
    return per_feature

def quantize_signal_bayesian_block_feature_bins(data):
    if data.ndim == 1:
        edges = bayesian_blocks(data)
        quantized = np.digitize(data, edges) - 1
        return quantized.tolist()
    elif data.ndim == 2:
        n_samples, n_features = data.shape
        quantized_features = []
        bases = []

        # quantize each feature and record its number of bins
        for i in range(n_features):
            edges = bayesian_blocks(data[:, i])
            quantized_features.append(np.digitize(data[:, i], edges) - 1)
            bases.append(len(edges))

        quantized_features = np.stack(quantized_features, axis=1)
        bases = np.array(bases, dtype=int)

        # compute mixed‑radix weights: product of previous bases
        # weights[0] = 1, weights[i] = prod(bases[:i])
        weights = np.cumprod(np.concatenate(([1], bases[:-1])))
        composite_symbols = np.sum(quantized_features * weights, axis=1)
        return composite_symbols.tolist()
    else:
        raise ValueError("Data must be 1D or 2D.")
