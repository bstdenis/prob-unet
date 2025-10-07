import functools

import numpy as np
from scipy import fftpack, ndimage


def zigzag_indices(num_rows, num_columns=None):
    """Generate zigzag order indices for a 2d matrix (default square)."""
    if num_columns is None:
        num_columns = num_rows
    result = []
    for s in range(num_rows + num_columns - 1):
        if s % 2 == 0:  # even: go up
            r = min(s, num_rows - 1)
            c = s - r
            while r >= 0 and c < num_columns:
                result.append((r, c))
                r -= 1
                c += 1
        else:  # odd: go down
            c = min(s, num_columns - 1)
            r = s - c
            while c >= 0 and r < num_rows:
                result.append((r, c))
                r += 1
                c -= 1
    return result


def low_frequency_dct_coefficients(array, dct_type=2, norm='ortho', k=8, include_dc=True):
    """
    Take a 2D DCT and return the first k coefficients in zigzag order.
    If include_dc=False, skip the first (0,0) term and take the next k.
    """
    dct2_result = fftpack.dct(fftpack.dct(array, type=dct_type, axis=0, norm=norm), type=dct_type, axis=1, norm=norm)
    h, w = dct2_result.shape
    order = zigzag_indices(h, w)
    coefficients = []
    for (r, c) in order:
        if not include_dc and r == 0 and c == 0:
            continue
        coefficients.append(float(dct2_result[r, c]))
        if len(coefficients) >= k:
            break
    return coefficients


@functools.lru_cache()
def haar_filters():
    # Haar low/high-pass filters
    inverse_sqrt_2 = 1 / np.sqrt(2.0)
    h = np.array([inverse_sqrt_2, inverse_sqrt_2], dtype=np.float32)   # low-pass
    g = np.array([inverse_sqrt_2, -inverse_sqrt_2], dtype=np.float32)  # high-pass
    return h, g


def separable_convolve_downsample(img, f_row, f_col):
    """Convolve rows with f_row, columns with f_col, then downsample by 2 in each dim."""
    # Convolve along rows
    tmp = ndimage.convolve1d(img, f_row, axis=1, mode='reflect')
    # Convolve along columns
    tmp = ndimage.convolve1d(tmp, f_col, axis=0, mode='reflect')
    # Downsample 2x
    return tmp[::2, ::2]


def haar_decompose_level(img):
    """One-level 2D Haar: returns (LL, LH, HL, HH)."""
    h, g = haar_filters()
    ll = separable_convolve_downsample(img, h, h)
    lh = separable_convolve_downsample(img, h, g)  # low rows, high cols
    hl = separable_convolve_downsample(img, g, h)  # high rows, low cols
    hh = separable_convolve_downsample(img, g, g)
    return ll, lh, hl, hh


def wavelet_energies(img, levels=1):
    """
    Compute sum-of-squares energies of LH/HL/HH at each level.
    Returns list: [E_LH1, E_HL1, E_HH1, E_LH2, E_HL2, E_HH2, ...]
    """
    features = []
    local_img = img
    for _ in range(levels):
        ll, lh, hl, hh = haar_decompose_level(local_img)
        # Energy = mean square (scale-independent) or sum of squares. Use mean for size robustness.
        features.extend([float(np.mean(lh ** 2)), float(np.mean(hl**2)), float(np.mean(hh ** 2))])
        local_img = ll
    return features


def edge_density_and_orientation_histogram(img, bins=0, edge_threshold=None):
    """
    Sobel-based edge density and (optional) orientation histogram.
    If bins==0: only edge density is returned (list of length 1).
    If bins>0: returns [edge_density, hist_0 ... hist_{bins-1}] with L1-normalized orientation histogram.
    """
    gx = ndimage.sobel(img, axis=1, mode='reflect')  # x = columns
    gy = ndimage.sobel(img, axis=0, mode='reflect')  # y = rows
    mag = np.hypot(gx, gy)
    # Threshold for edge density
    if edge_threshold is None:
        # Robust-ish default: mean + 1*std
        tau = float(mag.mean() + mag.std())
    else:
        tau = float(edge_threshold)
    edge_density = float((mag > tau).mean())

    features = [edge_density]

    if bins and bins > 0:
        # Orientation in [0, pi)
        theta = np.arctan2(gy, gx)
        theta = np.mod(theta, np.pi)
        # Weight by magnitude to emphasize stronger edges
        local_histogram, _ = np.histogram(theta, bins=bins, range=(0.0, np.pi), weights=mag)
        # L1-normalize (avoid div by zero)
        s = local_histogram.sum()
        if s > 0:
            local_histogram = local_histogram / s
        features.extend(local_histogram.astype(np.float32).tolist())
    return features
