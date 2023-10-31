from filtrepicurienchanced import lee_enhanced_filter
from filtrepicuri import lee_filter
from filtrepicuri import kuan_filter
from filtrepicuri import frost_filter
from filtrepicuri import median_filter
from filtrepicuri import mean_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.filters import maximum_filter

from tqdm import tqdm
import numpy as np
import pandas as pd
import uniunedate as union_find


def _import_cv2():
    try:
        import cv2
        return cv2
    except:
        raise ImportError('cv2 must be installed manually. Try to: <pip install opencv-python>')



def scale(X, verbose=3):
    if verbose >= 3: print('>Scaling image between [0-255] and to uint8')
    try:
        X = X - X.min()
        X = X / X.max()
        X = X * 255
        X = np.uint8(X)
    except:
        if verbose >= 2: print('>WARNING: Scaling not possible.')
    return X


def togray(X, verbose=3):
    cv2 = _import_cv2()
    try:
        if verbose >= 3: print('>Conversion to gray image.')
        X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
    except:
        if verbose >= 2: print('>WARNING: Conversion to gray not possible.')
    return X



def resize(X, size=None, verbose=3):
    cv2 = _import_cv2()
    try:
        if size is not None:
            if verbose >= 3: print('>Resizing image to %s.' % (str(size)))
            X = cv2.resize(X, size)
    except:
        if verbose >= 2: print('>WARNING: Resizing not possible.')
    return X



def denoise(X, method='fastnl', window=9, cu=0.25, verbose=3):
    if window is None: window = 9
    if cu is None: cu = 0.25
    cv2 = _import_cv2()


    if verbose >= 3: print('>Minimizare zgomot cu [%s], fereastra: [%d].' % (method, window))
    if method == 'fastnl':
        if len(X.shape) == 2:
            X = cv2.fastNlMeansDenoising(X, h=window)
        if len(X.shape) == 3:
            if verbose >= 3: print('[findpeaks] >Denoising color image.')
            X = cv2.fastNlMeansDenoisingColored(X, h=window)
    elif method == 'bilateral':
        X = cv2.bilateralFilter(X, window, 75, 75)
    elif method == 'lee':
        X = lee_filter(X, win_size=window, cu=cu)
    elif method == 'lee_enhanced':
        X = lee_enhanced_filter(X, win_size=window, cu=cu, k=1, cmax=1.73)
    elif method == 'kuan':
        X = kuan_filter(X, win_size=window, cu=cu)
    elif method == 'frost':
        X = frost_filter(X, win_size=window, damping_factor=2)
    elif method == 'median':
        X = median_filter(X, win_size=window)
    elif method == 'mean':
        X = mean_filter(X, win_size=window)

    return X



def mask(X, limit=0, verbose=3):
    if limit is None: limit = 0

    if verbose >= 3: print('>Detect peaks using the mask method with limit=%s.' % (limit))

    neighborhood = generate_binary_structure(2, 2)

    local_max = maximum_filter(X, footprint=neighborhood) == X

    background = (X <= limit)

    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)


    Xdetect = local_max ^ eroded_background

    Xranked = Xdetect.astype(int)
    idxs = np.where(Xranked)
    for i, idx in enumerate(zip(idxs[0], idxs[1])):
        Xranked[idx] = i + 1


    result = {'Xdetect': Xdetect, 'Xranked': Xranked}


    return result



def topologie2d(X, limit=None, whitelist=['peak', 'valley'], verbose=3):
    result_peak = {'groups0': [], 'Xdetect': np.zeros_like(X).astype(float), 'Xranked': np.zeros_like(X).astype(float),
                   'peak': None, 'valley': None,
                   'persistence': pd.DataFrame(columns=['x', 'y', 'birth_level', 'death_level', 'score'])}
    result_valley = {'groups0': [], 'Xdetect': np.zeros_like(X).astype(float),
                     'Xranked': np.zeros_like(X).astype(float), 'peak': None, 'valley': None,
                     'persistence': pd.DataFrame(columns=['x', 'y', 'birth_level', 'death_level', 'score'])}


    if np.any(np.isin(whitelist, 'peak')):
        result_peak = topologie(X, limit=limit, reverse=True, verbose=verbose)
        result_peak['persistence']['peak'] = True
        result_peak['persistence']['valley'] = False


    if np.any(np.isin(whitelist, 'valley')):
        result_valley = topologie(X, limit=limit, reverse=False, verbose=verbose)
        result_valley['persistence']['peak'] = False
        result_valley['persistence']['valley'] = True
        result_valley['persistence']['death_level'] = result_valley['persistence']['death_level'] * -1
        result_valley['persistence']['birth_level'] = result_valley['persistence']['birth_level'] * -1


    persistence = pd.concat([result_peak['persistence'], result_valley['persistence']])
    persistence.sort_values(by='score', ascending=False, inplace=True)

    Xdetect = result_peak['Xdetect']
    idx = np.where(result_valley['Xdetect'])
    if len(idx[0]) > 0: Xdetect[idx] = result_valley['Xdetect'][idx] * -1
    Xranked = result_peak['Xranked']
    idx = np.where(result_valley['Xranked'])
    if len(idx[0]) > 0: Xranked[idx] = result_valley['Xranked'][idx] * -1
    groups0 = result_peak['groups0'] + result_valley['groups0']
    result = {'persistence': persistence, 'Xdetect': Xdetect, 'Xranked': Xranked, 'groups0': groups0}
    return result



def topologie(X, limit=None, reverse=True, verbose=3):
    if verbose >= 3: print('>Detectie picuri prin topologie cu limita la %s.' % (limit))

    results = {'groups0': [], 'Xdetect': np.zeros_like(X).astype(float), 'Xranked': np.zeros_like(X).astype(float),
               'peak': None, 'valley': None,
               'persistence': pd.DataFrame(columns=['x', 'y', 'birth_level', 'death_level', 'score'])}
    h, w = X.shape
    max_peaks, min_peaks = None, None
    groups0 = {}


    indices = [(i, j) for i in range(h) for j in range(w)]
    indices.sort(key=lambda p: _get_indices(X, p), reverse=reverse)


    uf = union_find.UnionFind()

    def _get_comp_birth(p):
        return _get_indices(X, uf[p])


    for i, p in tqdm(enumerate(indices), disable=disable_tqdm(verbose)):
        v = _get_indices(X, p)
        ni = [uf[q] for q in _iter_neighbors(p, w, h) if q in uf]
        nc = sorted([(_get_comp_birth(q), q) for q in set(ni)], reverse=True)

        if i == 0: groups0[p] = (v, v, None)
        uf.add(p, -i)

        if len(nc) > 0:
            oldp = nc[0][1]
            uf.union(oldp, p)

            for bl, q in nc[1:]:
                if uf[q] not in groups0:
                    groups0[uf[q]] = (float(bl), float(bl) - float(v), p)
                uf.union(oldp, q)

    groups0 = [(k, groups0[k][0], groups0[k][1], groups0[k][2]) for k in groups0]
    groups0.sort(key=lambda g: g[2], reverse=True)


    if (limit is not None):
        Ikeep = np.array(list(map(lambda x: x[2], groups0))) > limit
        groups0 = np.array(groups0, dtype='object')
        groups0 = groups0[Ikeep].tolist()

    if len(groups0) > 0:

        max_peaks = np.array(list(map(lambda x: [x[0][0], x[1]], groups0)))
        idxsort = np.argsort(max_peaks[:, 0])
        max_peaks = max_peaks[idxsort, :]

        min_peaks = np.array(list(map(lambda x: [(x[3][0] if x[3] is not None else 0), x[2]], groups0)))
        idxsort = np.argsort(min_peaks[:, 0])
        min_peaks = min_peaks[idxsort, :]

        Xdetect = np.zeros_like(X).astype(float)
        Xranked = np.zeros_like(X).astype(int)
        for i, homclass in enumerate(groups0):
            p_birth, bl, pers, p_death = homclass
            y, x = p_birth
            Xdetect[y, x] = pers
            Xranked[y, x] = i + 1


        if (X.shape[1] == 2) and (np.all(Xdetect[:, 1] == 0)):
            Xdetect = Xdetect[:, 0]
            Xranked = Xranked[:, 0]


        df_persistence = pd.DataFrame()
        df_persistence['x'] = np.array(list(map(lambda x: x[0][1], groups0)))
        df_persistence['y'] = np.array(list(map(lambda x: x[0][0], groups0)))
        df_persistence['birth_level'] = np.array(list(map(lambda x: float(x[1]), groups0)))
        df_persistence['death_level'] = np.array(list(map(lambda x: float(x[1]) - float(x[2]), groups0)))
        df_persistence['score'] = np.array(list(map(lambda x: float(x[2]), groups0)))

        results = {}
        results['groups0'] = groups0
        results['Xdetect'] = Xdetect
        results['Xranked'] = Xranked
        results['peak'] = max_peaks
        results['valley'] = min_peaks
        results['persistence'] = df_persistence


    return results


def _get_indices(im, p):
    return im[p[0]][p[1]]


def _iter_neighbors(p, w, h):
    y, x = p


    neigh = [(y + j, x + i) for i in [-1, 0, 1] for j in [-1, 0, 1]]


    for j, i in neigh:
        if j < 0 or j >= h:
            continue
        if i < 0 or i >= w:
            continue
        if j == y and i == x:
            continue
        yield j, i


def _post_processing(X, Xraw, min_peaks, max_peaks, interpolate, lookahead, labxRaw=None, verbose=3):
    if lookahead < 1: raise Exception('lookhead parameter should be at least 1.')
    labx_s = np.zeros((len(X))) * np.nan
    results = {}
    results['min_peaks_s'] = None
    results['max_peaks_s'] = None
    results['xs'] = np.arange(0, len(Xraw))
    results['labx_s'] = labx_s
    results['labx'] = np.zeros((len(Xraw))) * np.nan
    results['min_peaks'] = None
    results['max_peaks'] = None

    if len(min_peaks) > 0 and len(max_peaks) > 0 and (max_peaks[0][0] is not None):

        idx_peaks, _ = zip(*max_peaks)
        idx_peaks = np.array(list(idx_peaks)).astype(int)
        idx_valleys, _ = zip(*min_peaks)
        idx_valleys = np.append(np.array(list(idx_valleys)), len(X) - 1).astype(int)
        idx_valleys = np.append(0, idx_valleys)


        count = 1
        for i in range(0, len(idx_valleys) - 1):
            if idx_valleys[i] != idx_valleys[i + 1]:
                labx_s[idx_valleys[i]:idx_valleys[i + 1] + 1] = count
                count = count + 1


        if interpolate is not None:
            min_peaks = np.minimum(np.ceil(((idx_valleys / len(X)) * len(Xraw))).astype(int), len(Xraw) - 1)
            max_peaks = np.minimum(np.ceil(((idx_peaks / len(X)) * len(Xraw))).astype(int), len(Xraw) - 1)
            
            max_peaks_corr = []
            for max_peak in max_peaks:
                getrange = np.arange(np.maximum(max_peak - lookahead, 0), np.minimum(max_peak + lookahead, len(Xraw)))
                max_peaks_corr.append(getrange[np.argmax(Xraw[getrange])])
            
            min_peaks_corr = []
            for min_peak in min_peaks:
                getrange = np.arange(np.maximum(min_peak - lookahead, 0), np.minimum(min_peak + lookahead, len(Xraw)))
                min_peaks_corr.append(getrange[np.argmin(Xraw[getrange])])
            
            count = 1
            labx = np.zeros((len(Xraw))) * np.nan
            for i in range(0, len(min_peaks) - 1):
                if min_peaks[i] != min_peaks[i + 1]:
                    labx[min_peaks[i]:min_peaks[i + 1] + 1] = count
                    count = count + 1


            results['labx'] = labx
            results['min_peaks'] = np.c_[min_peaks_corr, Xraw[min_peaks_corr]]
            results['max_peaks'] = np.c_[max_peaks_corr, Xraw[max_peaks_corr]]

        results['min_peaks_s'] = np.c_[idx_valleys, X[idx_valleys]]
        results['max_peaks_s'] = np.c_[idx_peaks, X[idx_peaks]]
        if labxRaw is None:
            results['labx_s'] = labx_s
        else:
            results['labx_s'] = labxRaw


    return results



def disable_tqdm(verbose):
    return (True if ((verbose < 4 or verbose is None) or verbose > 5) else False)