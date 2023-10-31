import sys
import random
import cgitb
from decimal import *
cgitb.enable(format = 'text')
from assets.interfata2 import Ui_SpectraUI
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot, QTimer, QUrl, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog,QVBoxLayout, QApplication,QMainWindow, QInputDialog, QPushButton, QWidget
from PyQt5.uic import loadUi
from PyQt5.QtWebEngineWidgets import QWebEnginePage, QWebEngineView
import cv2
from assets.neural_network import CustomObjectDetection
import cv2
import os
import numpy as np
from threading import Thread
from pyqtconsole.console import PythonConsole
from scipy import ndimage as ndi
import math
import cv2
import numpy as np
from skimage import io
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import statsmodels.api as sm
from scipy.signal._wavelets import cwt, ricker
from scipy.stats import scoreatpercentile
from picuri import gasirepicuri
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from absolutePath import absolutePath

from scipy.signal._peak_finding_utils import (
    _local_maxima_1d,
    _select_by_peak_distance,
    _peak_prominences,
    _peak_widths
)

from shared.utils import _validate_interpolation_order, _fix_ndimage_mode

def profile_line(image, src, dst, linewidth=1,
                 order=None, mode='reflect', cval=0.0,
                 *, reduce_func=np.mean):
    order = _validate_interpolation_order(image.dtype, order)
    mode = _fix_ndimage_mode(mode)

    perp_lines = _line_profile_coordinates(src, dst, linewidth=linewidth)
    if image.ndim == 3:
        pixels = [ndi.map_coordinates(image[..., i], perp_lines,
                                      prefilter=order > 1,
                                      order=order, mode=mode,
                                      cval=cval) for i in
                  range(image.shape[2])]
        pixels = np.transpose(np.asarray(pixels), (1, 2, 0))
    else:
        pixels = ndi.map_coordinates(image, perp_lines, prefilter=order > 1,
                                     order=order, mode=mode, cval=cval)
    pixels = np.flip(pixels, axis=1)

    if reduce_func is None:
        intensities = pixels
    else:
        try:
            intensities = reduce_func(pixels, axis=1)
        except TypeError:  
            intensities = np.apply_along_axis(reduce_func, arr=pixels, axis=1)
    return intensities


def _line_profile_coordinates(src, dst, linewidth=1):
    src_row, src_col = src = np.asarray(src, dtype=object)
    dst_row, dst_col = dst = np.asarray(dst, dtype=object)
    d_row, d_col = dst - src
    theta = np.arctan2(d_row, d_col)

    length = int(np.ceil(np.hypot(d_row, d_col) + 1))
    line_col = np.linspace(src_col, dst_col, length)
    line_row = np.linspace(src_row, dst_row, length)

    col_width = (linewidth - 1) * np.sin(-theta) / 2
    row_width = (linewidth - 1) * np.cos(theta) / 2
    perp_rows = np.stack([np.linspace(row_i - row_width, row_i + row_width,
                                      linewidth) for row_i in line_row])
    perp_cols = np.stack([np.linspace(col_i - col_width, col_i + col_width,
                                      linewidth) for col_i in line_col])
    return np.stack([perp_rows, perp_cols])

def _boolrelextrema(data, comparator, axis=0, order=1, mode='clip'):

    if (int(order) != order) or (order < 1):
        raise ValueError('Order must be an int >= 1')

    datalen = data.shape[axis]
    locs = np.arange(0, datalen)

    results = np.ones(data.shape, dtype=bool)
    main = data.take(locs, axis=axis, mode=mode)
    for shift in range(1, order + 1):
        plus = data.take(locs + shift, axis=axis, mode=mode)
        minus = data.take(locs - shift, axis=axis, mode=mode)
        results &= comparator(main, plus)
        results &= comparator(main, minus)
        if ~results.any():
            return results
    return results


def argrelmin(data, axis=0, order=1, mode='clip'):

    return argrelextrema(data, np.less, axis, order, mode)


def argrelmax(data, axis=0, order=1, mode='clip'):

    return argrelextrema(data, np.greater, axis, order, mode)


def argrelextrema(data, comparator, axis=0, order=1, mode='clip'):

    results = _boolrelextrema(data, comparator,
                              axis, order, mode)
    return np.nonzero(results)


def _arg_x_as_expected(value):

    value = np.asarray(value, order='C', dtype=np.float64)
    if value.ndim != 1:
        raise ValueError('`x` must be a 1-D array')
    return value


def _arg_peaks_as_expected(value):

    value = np.asarray(value)
    if value.size == 0:
        
        value = np.array([], dtype=np.intp)
    try:
        
        value = value.astype(np.intp, order='C', casting='safe',
                             subok=False, copy=False)
    except TypeError as e:
        raise TypeError("cannot safely cast `peaks` to dtype('intp')") from e
    if value.ndim != 1:
        raise ValueError('`peaks` must be a 1-D array')
    return value


def _arg_wlen_as_expected(value):

    if value is None:
        
        
        value = -1
    elif 1 < value:
        
        if not np.can_cast(value, np.intp, "safe"):
            value = math.ceil(value)
        value = np.intp(value)
    else:
        raise ValueError('`wlen` must be larger than 1, was {}'
                         .format(value))
    return value


def peak_prominences(x, peaks, wlen=None):

    x = _arg_x_as_expected(x)
    peaks = _arg_peaks_as_expected(peaks)
    wlen = _arg_wlen_as_expected(wlen)
    return _peak_prominences(x, peaks, wlen)


def peak_widths(x, peaks, rel_height=0.5, prominence_data=None, wlen=None):

    x = _arg_x_as_expected(x)
    peaks = _arg_peaks_as_expected(peaks)
    if prominence_data is None:
        
        wlen = _arg_wlen_as_expected(wlen)
        prominence_data = _peak_prominences(x, peaks, wlen)
    return _peak_widths(x, peaks, rel_height, *prominence_data)


def _unpack_condition_args(interval, x, peaks):

    try:
        imin, imax = interval
    except (TypeError, ValueError):
        imin, imax = (interval, None)

    
    if isinstance(imin, np.ndarray):
        if imin.size != x.size:
            raise ValueError('array size of lower interval border must match x')
        imin = imin[peaks]
    if isinstance(imax, np.ndarray):
        if imax.size != x.size:
            raise ValueError('array size of upper interval border must match x')
        imax = imax[peaks]

    return imin, imax


def _select_by_property(peak_properties, pmin, pmax):

    keep = np.ones(peak_properties.size, dtype=bool)
    if pmin is not None:
        keep &= (pmin <= peak_properties)
    if pmax is not None:
        keep &= (peak_properties <= pmax)
    return keep


def _select_by_peak_threshold(x, peaks, tmin, tmax):

    stacked_thresholds = np.vstack([x[peaks] - x[peaks - 1],
                                    x[peaks] - x[peaks + 1]])
    keep = np.ones(peaks.size, dtype=bool)
    if tmin is not None:
        min_thresholds = np.min(stacked_thresholds, axis=0)
        keep &= (tmin <= min_thresholds)
    if tmax is not None:
        max_thresholds = np.max(stacked_thresholds, axis=0)
        keep &= (max_thresholds <= tmax)

    return keep, stacked_thresholds[0], stacked_thresholds[1]


def find_pic(x, height=None, threshold=None, distance=None,
               prominence=None, width=None, wlen=None, rel_height=0.5,
               plateau_size=None):

    x = _arg_x_as_expected(x)
    if distance is not None and distance < 1:
        raise ValueError('`distance` must be greater or equal to 1')

    pic, left_edges, right_edges = _local_maxima_1d(x)
    properties = {}

    if plateau_size is not None:

        plateau_sizes = right_edges - left_edges + 1
        pmin, pmax = _unpack_condition_args(plateau_size, x, pic)
        keep = _select_by_property(plateau_sizes, pmin, pmax)
        pic = pic[keep]
        properties["plateau_sizes"] = plateau_sizes
        properties["left_edges"] = left_edges
        properties["right_edges"] = right_edges
        properties = {key: array[keep] for key, array in properties.items()}

    if height is not None:

        peak_heights = x[pic]
        hmin, hmax = _unpack_condition_args(height, x, pic)
        keep = _select_by_property(peak_heights, hmin, hmax)
        pic = pic[keep]
        properties["peak_heights"] = peak_heights
        properties = {key: array[keep] for key, array in properties.items()}

    if threshold is not None:

        tmin, tmax = _unpack_condition_args(threshold, x, pic)
        keep, left_thresholds, right_thresholds = _select_by_peak_threshold(
            x, pic, tmin, tmax)
        pic = pic[keep]
        properties["left_thresholds"] = left_thresholds
        properties["right_thresholds"] = right_thresholds
        properties = {key: array[keep] for key, array in properties.items()}

    if distance is not None:

        keep = _select_by_peak_distance(pic, x[pic], distance)
        pic = pic[keep]
        properties = {key: array[keep] for key, array in properties.items()}

    if prominence is not None or width is not None:

        wlen = _arg_wlen_as_expected(wlen)
        properties.update(zip(
            ['prominences', 'left_bases', 'right_bases'],
            _peak_prominences(x, pic, wlen=wlen)
        ))

    if prominence is not None:

        pmin, pmax = _unpack_condition_args(prominence, x, pic)
        keep = _select_by_property(properties['prominences'], pmin, pmax)
        pic = pic[keep]
        properties = {key: array[keep] for key, array in properties.items()}

    if width is not None:

        properties.update(zip(
            ['widths', 'width_heights', 'left_ips', 'right_ips'],
            _peak_widths(x, pic, rel_height, properties['prominences'],
                         properties['left_bases'], properties['right_bases'])
        ))

        wmin, wmax = _unpack_condition_args(width, x, pic)
        keep = _select_by_property(properties['widths'], wmin, wmax)
        pic = pic[keep]
        properties = {key: array[keep] for key, array in properties.items()}

    return pic, properties


def _identify_ridge_lines(matr, max_distances, gap_thresh):

    if len(max_distances) < matr.shape[0]:
        raise ValueError('Max_distances must have at least as many rows '
                         'as matr')

    all_max_cols = _boolrelextrema(matr, np.greater, axis=1, order=1)

    has_relmax = np.nonzero(all_max_cols.any(axis=1))[0]
    if len(has_relmax) == 0:
        return []
    start_row = has_relmax[-1]

    ridge_lines = [[[start_row],
                   [col],
                   0] for col in np.nonzero(all_max_cols[start_row])[0]]
    final_lines = []
    rows = np.arange(start_row - 1, -1, -1)
    cols = np.arange(0, matr.shape[1])
    for row in rows:
        this_max_cols = cols[all_max_cols[row]]

        for line in ridge_lines:
            line[2] += 1


        prev_ridge_cols = np.array([line[1][-1] for line in ridge_lines])

        for ind, col in enumerate(this_max_cols):

            line = None
            if len(prev_ridge_cols) > 0:
                diffs = np.abs(col - prev_ridge_cols)
                closest = np.argmin(diffs)
                if diffs[closest] <= max_distances[row]:
                    line = ridge_lines[closest]
            if line is not None:

                line[1].append(col)
                line[0].append(row)
                line[2] = 0
            else:
                new_line = [[row],
                            [col],
                            0]
                ridge_lines.append(new_line)


        for ind in range(len(ridge_lines) - 1, -1, -1):
            line = ridge_lines[ind]
            if line[2] > gap_thresh:
                final_lines.append(line)
                del ridge_lines[ind]

    out_lines = []
    for line in (final_lines + ridge_lines):
        sortargs = np.array(np.argsort(line[0]))
        rows, cols = np.zeros_like(sortargs), np.zeros_like(sortargs)
        rows[sortargs] = line[0]
        cols[sortargs] = line[1]
        out_lines.append([rows, cols])

    return out_lines


def _filter_ridge_lines(cwt, ridge_lines, window_size=None, min_length=None,
                        min_snr=1, noise_perc=10):

    num_points = cwt.shape[1]
    if min_length is None:
        min_length = np.ceil(cwt.shape[0] / 4)
    if window_size is None:
        window_size = np.ceil(num_points / 20)

    window_size = int(window_size)
    hf_window, odd = divmod(window_size, 2)


    row_one = cwt[0, :]
    noises = np.empty_like(row_one)
    for ind, val in enumerate(row_one):
        window_start = max(ind - hf_window, 0)
        window_end = min(ind + hf_window + odd, num_points)
        noises[ind] = scoreatpercentile(row_one[window_start:window_end],
                                        per=noise_perc)

    def filt_func(line):
        if len(line[0]) < min_length:
            return False
        snr = abs(cwt[line[0][0], line[1][0]] / noises[line[1][0]])
        if snr < min_snr:
            return False
        return True

    return list(filter(filt_func, ridge_lines))


def find_peaks_cwt(vector, widths, wavelet=None, max_distances=None,
                   gap_thresh=None, min_length=None,
                   min_snr=1, noise_perc=10, window_size=None):

    widths = np.array(widths, copy=False, ndmin=1)

    if gap_thresh is None:
        gap_thresh = np.ceil(widths[0])
    if max_distances is None:
        max_distances = widths / 4.0
    if wavelet is None:
        wavelet = ricker

    cwt_dat = cwt(vector, wavelet, widths)
    ridge_lines = _identify_ridge_lines(cwt_dat, max_distances, gap_thresh)
    filtered = _filter_ridge_lines(cwt_dat, ridge_lines, min_length=min_length,
                                   window_size=window_size, min_snr=min_snr,
                                   noise_perc=noise_perc)
    max_locs = np.asarray([x[1][0] for x in filtered])
    max_locs.sort()

    return max_locs


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
class credits(QDialog):
    def __init__(self,parent=None):
        super(credits,self).__init__(parent)
        self.resize(698,488)
        self.setWindowTitle("SpectraUI Credits")
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(absolutePath('assets/logo.png')),
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.label=QtWidgets.QLabel(self)
        self.label.setObjectName("label")
        self.label.setGeometry(QtCore.QRect(150,30,351,61))
        self.label.setPixmap(QPixmap(absolutePath('assets/logo1.png')))
        self.label.setScaledContents(True)
        self.label_2=QtWidgets.QLabel(self)
        self.label_2.setObjectName("label_2")
        self.label_2.setGeometry(QtCore.QRect(110, 100, 481, 51))
        font=QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_3=QtWidgets.QLabel(self)
        self.label_3.setObjectName("label_3")
        self.label_3.setGeometry(QtCore.QRect(120, 150, 311, 31))
        self.label_4 = QtWidgets.QLabel(self)
        self.label_4.setObjectName("label_4")
        self.label_4.setGeometry(QtCore.QRect(120, 180, 211, 21))
        self.label_2.setText("Spectral Neural Network Analysis Software")
        self.label_3.setText("\u00a9 2023 Ionut Slabu - Developer")
        self.label_4.setText("\u00a9 2023 The Qt Company - GUI Framework")

class intensityprofile(QDialog):
    def __init__(self, parent=None):
        super(intensityprofile, self).__init__(parent)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        temp=[]
        image = io.imread(absolutePath('assets/imagineQTgri.jpg'))
        f = open(absolutePath('assets/yprofilare'),'r')
        for line in f.readlines():
             temp.append(float(line))
        f.close()
        y=temp[0]
        start = (y, 0)
        end = (y, image.shape[1])
        profile = profile_line(image, start, end)
        picuri_nou = gasirepicuri(method='topologie', interpolate=7, lookahead=100, limit=45, window=10,
                                  whitelist=['peak'])
        resultspeak = self.picuri_nou.fit(profile)
        
        print(resultspeak)
        valori_intensitate = resultspeak['persistence'].iloc[:, 4]
        valori_x = resultspeak['persistence'].iloc[:, 1]
        sortat_x = np.sort(valori_x)
        sortat_intensitate = np.sort(valori_intensitate)
        picuri_nou.plot()

class manualmaxim(QDialog):
    def __init__(self, parent=None):
        super(manualmaxim, self).__init__(parent)
        self.setWindowTitle("SpectraUI Graphics")
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(absolutePath('assets/logo.png')),
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.figure = plt.figure()

        
        
        self.canvas = FigureCanvasQTAgg(self.figure)

        
        
        self.toolbar = NavigationToolbar(self.canvas, self)

        
        
        

        
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
      
        global image
        global coords
        global ix
        coords=[]
        
        image=cv2.imread(absolutePath('assets/imagineQTgri.jpg'),0)
        print(image.shape[1])
        f = open(absolutePath('assets/yprofilare'), 'r')
        temp=[]
        for line in f.readlines():
            temp.append(float(line))
        f.close()
        y = int(temp[0])
        suma = 0
        
        
        
        intensitateacalc=0
        I = []
        for x in range(0, image.shape[1]):
            suma = 0
            for j in range(y-30,y+30):
               suma=suma+image[j,x]
            I.append(suma)
        print(I[1917])
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(I)
        self.canvas.draw()
        self.canvas.mpl_connect("button_press_event",self.on_press)

    def on_press(self,event):
        if event.dblclick:
           print("event.xdata", event.xdata)
           ix=event.xdata
           coords.append(ix)
        f=open(absolutePath('assets/hidrogen'), 'w')
        for d in coords:
          f.write(f"{d}\n")
        f.close()
class profilaremanual(QDialog):
    def __init__(self, parent=None):
        super(profilaremanual, self).__init__(parent)
        self.setWindowTitle("SpectraUI Graphics")
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(absolutePath('assets/logo.png')),
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.figure = plt.figure()

        
        
        self.canvas = FigureCanvasQTAgg(self.figure)

        
        
        self.toolbar = NavigationToolbar(self.canvas, self)

        
        
        

        
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
      
        global image
        image = io.imread(absolutePath('assets/imagineQTgri.jpg'))
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(image)
        self.canvas.draw()
        self.canvas.mpl_connect("button_press_event",self.on_press)

    def on_press(self,event):
        global y
        if event.dblclick:
           print("event.ydata", event.ydata)
           with open(absolutePath('assets/yprofilare'), 'w') as f:
                f.write(f"{event.ydata}\n")
           f.close()
class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class WebEnginePage(QWebEnginePage):
    def certificateError(self, error):
        error.ignoreCertificateError()
        return True
def alegere_Y_manual_profilare(image):
    global iy
    global coords
    ax = MplCanvas(self, width=5, height=4, dpi=100)
    ax.imshow(image)

    def onclick(event):
        if event.button == 3:
            ld = LineDrawer()
            ld.draw_line()
        if event.dblclick:
            iy = event.ydata
            coords.append(iy)

    coords = []
    ax.canvas.mpl_connect('button_press_event', onclick)
    ax.show()
    print(coords)
    return coords[0]

def lambda_final(intercept,panta , distanta):
    x_initial=float(intercept+(panta*distanta))
    x_intermediar=float(1/x_initial)
    x=math.sqrt(x_intermediar)
    return x
def incertitudine(panta,distanta,intercept,sigma_i,sigma_p,sigma_d):
    sigma_lambda2=(1/4)*(((panta*(distanta**-2)+intercept)**-3)*((distanta**-4)*(sigma_p**2)+(4*(panta**2))*(distanta**-6)*(sigma_d**2)+(sigma_i**2)))
    sigma_lambda=math.sqrt(sigma_lambda2)
    return sigma_lambda

def dispersie(lambdafin):
    n2=(((14926.44*1e-8)*(lambdafin**2))/((lambdafin**2)-(19.36*1e-6))+(((41807.57*(1e-8))*(lambdafin**2))/((lambdafin**2)-(7.434*1e-3))) + 1)
    n=math.sqrt(n2)
    return n
def Rydbergrosuc(lambda_final,refractie):
    R=(1/(refractie*lambda_final))*(1/0.13888888888)
    return R
def Rydbergalbastruc(lambda_final,refractie):
    R=(1/(refractie*lambda_final))*(1/0.1875)
    return R

def incertitudine_R_rosu(lambda_final, incer_lambda):
    sigmaR=((1/0.13888888888)/(lambda_final**2))*incer_lambda
    sigmaR_rosu=float(sigmaR/1e7)
    return sigmaR_rosu

def incertitudine_R_albastru(lambda_final, incer_lambda):
    sigmaR=((1/0.1875)/(lambda_final**2))*incer_lambda
    sigmaR_albastru=float(sigmaR/1e7)
    return sigmaR_albastru

















class Incercare(QMainWindow, Ui_SpectraUI):
    def __init__(self):
        super(Incercare, self).__init__()
        self.setupUi(self)
        
        self.logic = 0
        self.w=None
        self.w1=None
        self.w2=None
        self.pointermodeM=0
        self.pointermodeH=0
        self.actionAbout.triggered.connect(self.credits)
        self.pushButton_5.clicked.connect(self.data_analysis)
        self.pushButton_7.clicked.connect(self.onClicked)
        self.pushButton_6.clicked.connect(self.airecognition)
        self.pushButton_8.clicked.connect(self.CaptureClicked)
          
        self.pushButton_9.clicked.connect(self.cancelFeed)
        self.pushButton_10.clicked.connect(self.manualY)
        self.pushButton.clicked.connect(self.GrayScale)
        self.pushButton_3.clicked.connect(self.graph_intensity)
        self.pushButton_12.clicked.connect(self.manualmax)
        console=PythonConsole()
        self.tab_widget = QtWidgets.QTabWidget(self.groupBox_3)
        self.tab_widget.addTab(console, "Terminal")
        console.eval_in_thread()
    @pyqtSlot()
    
    
    def onClicked(self):
        textadresa, ok = QInputDialog.getText(self, 'Input Address',
                                        'Please put address as shown: https://your_ip:protocol/')
        cap=cv2.VideoCapture(textadresa+'video')
        webpage = WebEnginePage()
        self.webEngineView.setPage(webpage)
        self.webEngineView.setUrl(QUrl(textadresa+'settings_window.html'))
        while(cap.isOpened()):
            ret, frame=cap.read()
            if ret==True:
                self.displayImage(frame,1)
                cv2.waitKey()
                if(self.logic==2):
                    cv2.imwrite(absolutePath('assets/imagineQT.jpg'), frame)
                    self.logic=1
                    self.pointermodeM = 1
                    self.pointermodeH = 1
                elif(self.logic==3):
                    self.logic = 1
                    cap.release()
            else:
                print('return not found')
        cap.release()
        cv2.destroyAllWindows()
    
    
    def displayImage(self,img,window=1):
        qformat=QImage.Format_Indexed8
        if len(img.shape)==3:
            if(img.shape[2]==4):
                qformat=QImage.Format_RGBA888
            else:
                qformat=QImage.Format_RGB888
        img=QImage(img,img.shape[1],img.shape[0],qformat)
        img=img.rgbSwapped()
        piximg=QtGui.QPixmap.fromImage(img)
        piximg2=piximg.scaled(920,470)
        self.imgLabel.setPixmap(piximg2)
        
    def CaptureClicked(self):
        self.logic=2
    def cancelFeed(self):
        self.logic=3
    def GrayScale(self):
        imagine=cv2.imread(absolutePath('assets/imagineQT.jpg'))
        gri = rgb2gray(imagine)
        cv2.imwrite(absolutePath('assets/imagineQTgri.jpg'), gri)
        pixmap1 = QtGui.QPixmap(absolutePath('assets/imagineQTgri.jpg'))
        pixmap2 = pixmap1.scaled(690, 490)
        self.processedframe.setPixmap(pixmap2)
        self.pointermodeM = 1
        self.pointermodeH = 1
    def manualY(self):
        self.w=profilaremanual()
        self.w.showNormal()
    def manualmax(self):
        self.w1=manualmaxim()
        self.w1.showNormal()
        self.pointermodeH=1
    def airecognition(self):
        detector = CustomObjectDetection()
        detector.setModelTypeAsYOLOv3()
        detector.setModelPath(
                 absolutePath('assets/spectru-yolo/models/yolov3_spectru-yolo_last.pt'))
        detector.setJsonPath(
                 absolutePath('assets/spectru-yolo/json/spectru-yolo_yolov3_detection_config.json'))
        detector.loadModel()

        detections = detector.detectObjectsFromImage(input_image=absolutePath('assets/imagineQT.jpg'), output_image_path=absolutePath('assets/detected.jpg'))
        for detection in detections:
            self.textBrowser.append("Spectru detectat " + str(detection["name"])+" cu probabilitatea: "+str(detection["percentage_probability"]))
            self.spectrul=detection["name"]
            break
        if (self.spectrul == "mercur"):
            sortat_x = []
            c = open(absolutePath('assets/rezultate'), 'r')
            for line in c.readlines():
                sortat_x.append(float(line))
            c.close()
            if (np.size(sortat_x) > 8):
                d_1 = float(sortat_x[4] / 7 - sortat_x[0] / 7)
                d_11 = float(sortat_x[8] / 7 - sortat_x[4] / 7)
                self.textBrowser.append("d1: " + str(d_1))
                self.textBrowser.append("d1': " + str(d_11))
                raport1 = float(d_1 / d_11)
                print("Raportul dintre d.linia1 si d.linia1' este: ", raport1)

                d_2 = float(sortat_x[4] / 7 - sortat_x[1] / 7)
                d_22 = float(sortat_x[7] / 7 - sortat_x[4] / 7)
                self.textBrowser.append("d2: " + str(d_2))
                self.textBrowser.append("d2': " + str(d_22))
                raport2 = float(d_2 / d_22)
                print("Raportul dintre d.linia2 si d.linia2' este: ", raport2)

                d_3 = float(sortat_x[4] / 7 - sortat_x[2] / 7)
                d_33 = float(sortat_x[6] / 7 - sortat_x[4] / 7)
                self.textBrowser.append("d3: " + str(d_3))
                self.textBrowser.append("d3': " + str(d_33))
                d_4 = float(sortat_x[5] / 7 - sortat_x[4] / 7)
                self.textBrowser.append("d4: " + str(d_4))
                raport3 = float(d_3 / d_33)
                print("Raportul dintre d.linia3 si d.linia3' este: ", raport3)
                d2_1 = float(1 / (d_1 * d_1))
                d2_2 = float(1 / (d_2 * d_2))
                d3_3 = float(1 / (d_3 * d_3))
                d2_4 = float(1 / (d_4 * d_4))
                d2_11 = float(1 / (d_11 * d_11))
                d2_22 = float(1 / (d_22 * d_22))
                d2_33 = float(1 / (d_33 * d_33))
            elif (np.size(sortat_x) == 8):
                d_1 = float(sortat_x[3] / 7 - sortat_x[0] / 7)
                d_11 = float(sortat_x[7] / 7 - sortat_x[3] / 7)
                self.textBrowser.append("d1: " + str(d_1))
                self.textBrowser.append("d1': " + str(d_11))
                raport1 = float(d_1 / d_11)
                print("Raportul dintre d.linia1 si d.linia1' este: ", raport1)

                d_2 = float(sortat_x[3] / 7 - sortat_x[1] / 7)
                d_22 = float(sortat_x[6] / 7 - sortat_x[3] / 7)
                self.textBrowser.append("d2: " + str(d_2))
                self.textBrowser.append("d2': " + str(d_22))
                raport2 = float(d_2 / d_22)
                print("Raportul dintre d.linia2 si d.linia2' este: ", raport2)

                d_3 = float(sortat_x[3] / 7 - sortat_x[2] / 7)
                d_33 = float(sortat_x[5] / 7 - sortat_x[3] / 7)
                self.textBrowser.append("d3: " + str(d_3))
                self.textBrowser.append("d3': " + str(d_33))

                d_4 = float(sortat_x[4] / 7 - sortat_x[3] / 7)
                self.textBrowser.append("d4: " + str(d_4))
                raport3 = float(d_3 / d_33)
                print("Raportul dintre d.linia3 si d.linia3' este: ", raport3)
            else:
                d_1 = float(sortat_x[3] / 7 - sortat_x[0] / 7)
                d_11 = float(sortat_x[6] / 7 - sortat_x[3] / 7)
                self.textBrowser.append("d1: " + str(d_1))
                self.textBrowser.append("d1': " + str(d_11))
                raport1 = float(d_1 / d_11)
                print("Raportul dintre d.linia1 si d.linia1' este: ", raport1)

                d_2 = float(sortat_x[3] / 7 - sortat_x[1] / 7)
                d_22 = float(sortat_x[5] / 7 - sortat_x[3] / 7)
                self.textBrowser.append("d2: " + str(d_2))
                self.textBrowser.append("d2': " + str(d_22))
                raport2 = float(d_2 / d_22)
                print("Raportul dintre d.linia2 si d.linia2' este: ", raport2)

                d_3 = float(sortat_x[3] / 7 - sortat_x[2] / 7)
                d_33 = float(sortat_x[4] / 7 - sortat_x[3] / 7)
                self.textBrowser.append("d3: " + str(d_3))
                self.textBrowser.append("d3': " + str(d_33))

            d2_1 = float(1 / (d_1 * d_1))
            print(d_1)
            d2_2 = float(1 / (d_2 * d_2))
            d3_3 = float(1 / (d_3 * d_3))
            
            d2_11 = float(1 / (d_11 * d_11))
            d2_22 = float(1 / (d_22 * d_22))
            d2_33 = float(1 / (d_33 * d_33))

            lambda1 = 579.0663e-9
            lambda2 = 546.0735e-9
            lambda3 = 435.8328e-9
            lambda4 = 404.6563e-9
            lambda1_2 = float(1 / (lambda1 * lambda1))
            print(lambda1)
            lambda2_2 = float(1 / (lambda2 * lambda2))
            lambda3_2 = float(1 / (lambda3 * lambda3))
            lambda4_2 = float(1 / (lambda4 * lambda4))
            print("Valorile lui lambda pentru regresia liniara: ", lambda1_2)
            x1 = np.array([d2_1, d2_2, d3_3])
            y1 = np.array([lambda1_2, lambda2_2, lambda3_2])
            y2 = np.array([lambda1_2, lambda2_2, lambda3_2])
            x2 = np.array([d2_11, d2_22, d2_33])
            coef = np.polyfit(x1, y1, 1)
            poly1d_fn = np.poly1d(coef)
            plt.plot(x1, y1, 'yo', x1, poly1d_fn(x1), '--k')
            plt.savefig('assets/imaginebinara.jpg')
            pixmap3 = QtGui.QPixmap("assets/imaginebinara.jpg")
            pixmap5 = pixmap3.scaled(690, 490)
            self.regresframe.setPixmap(pixmap5)
            model1 = sm.OLS(y1, sm.add_constant(x1))
            results1 = model1.fit()
            model2 = sm.OLS(y2, sm.add_constant(x2))
            results2 = model2.fit()
            self.textBrowser.append("\nPARAMETRII PARTEA STANGA\n")
            self.textBrowser.append(str(results1.summary()))
            print(results1.params)
            print(results1.summary())
            print("\n\n Parametrii total partea dreapta")
            self.textBrowser.append("\nPARAMETRII PARTEA DREAPTA\n")
            self.textBrowser.append(str(results2.summary()))
            print(results2.params)
            print(results2.summary())
            m1, b1 = np.polyfit(x1, y1, 1)
            pantastanga = m1
            interceptstanga = b1
            parametrii_stanga = [m1, b1]
            m2, b2 = np.polyfit(x2, y2, 1)
            pantadreapta = m2
            interceptdreapta = b2
            f = open(absolutePath('assets/parametrii'), "w")
            for d in parametrii_stanga:
                f.write(f"{d}\n")
            f.close()
            parametrii_dreapta = [m2, b2]
            g = open(absolutePath('assets/parametriidreapta'), "w")
            for d in parametrii_dreapta:
                g.write(f"{d}\n")
            g.close()
            rpatrateroare1 = results1.rsquared
            rpatrateroare2 = results2.rsquared
            self.textBrowser.append("\nREZUMAT\n")
            self.textBrowser.append("panta stanga= " + str(pantastanga) + " intercept stanga= " + str(interceptstanga))
            self.textBrowser.append("Rpatrat stanga= " + str(rpatrateroare1))
            self.textBrowser.append(
                "panta dreapta= " + str(pantadreapta) + " intercept dreapta= " + str(interceptdreapta))
            self.textBrowser.append("Rpatrat dreapta= " + str(rpatrateroare2))
            self.textBrowser.append("\nVerificare lungimi de unda\n")
            self.textBrowser.append("---------------------------------\n")

            lambda1 = lambda_final(interceptstanga, pantastanga, d2_1)
            lambda1prim = lambda_final(interceptdreapta, pantadreapta, d2_11)
            self.textBrowser.append("lambda1 stanga(galben): " + str(lambda1 * 1e9) + " nm" + "\nvaloareNIST: " + str(
                579.0663) + " nm" + "\neroare de: " + str(abs(lambda1 * 1e9 - 579.0663)) + " nm\n")
            self.textBrowser.append(
                "lambda1 dreapta(galben): " + str(lambda1prim * 1e9) + " nm" + "\nvaloareNIST: " + str(
                    579.0663) + " nm" + "\neroare de: " + str(abs(lambda1prim * 1e9 - 579.0663)) + " nm\n")
            lambda2 = lambda_final(interceptstanga, pantastanga, d2_2)
            lambda2prim = lambda_final(interceptdreapta, pantadreapta, d2_22)

            self.textBrowser.append("lambda2 stanga(verde): " + str(lambda2 * 1e9) + " nm" + "\nvaloareNIST: " + str(
                546.0735) + " nm" + "\neroare de: " + str(abs(lambda2 * 1e9 - 546.0735)) + " nm\n")
            self.textBrowser.append(
                "lambda2 dreapta(verde): " + str(lambda2prim * 1e9) + " nm" + "\nvaloareNIST: " + str(
                    546.0735) + " nm" + "\neroare de: " + str(abs(lambda2prim * 1e9 - 546.0735)) + " nm\n")
            lambda3 = lambda_final(interceptstanga, pantastanga, d3_3)
            lambda3prim = lambda_final(interceptdreapta, pantadreapta, d2_33)

            self.textBrowser.append("lambda3 stanga(albastru): " + str(lambda3 * 1e9) + " nm" + "\nvaloareNIST: " + str(
                435.8328) + " nm" + "\neroare de: " + str(abs(lambda3 * 1e9 - 435.8328)) + " nm\n")
            self.textBrowser.append(
                "lambda3 dreapta(albastru): " + str(lambda3prim * 1e9) + " nm" + "\nvaloareNIST: " + str(
                    435.8328) + " nm" + "\neroare de: " + str(abs(lambda3prim * 1e9 - 435.8328)) + " nm\n")
        elif (self.spectrul == "hidrogen"):
            alegere, ok = QInputDialog.getText(self, 'Selectie automata sau manuala',
                                               'manual 6 lambda, automat 4 lambda')
            if (alegere == "automat"):
                sortat_x = []
                c = open(absolutePath('assets/rezultate'), 'r')
                for line in c.readlines():
                    sortat_x.append(float(line))
                c.close()
                if (np.size(sortat_x) == 6):
                    d_1 = float(sortat_x[2] / 7 - sortat_x[0] / 7)
                    d_11 = float(sortat_x[5] / 7 - sortat_x[2] / 7)
                    self.textBrowser.append("d1: " + str(d_1))
                    self.textBrowser.append("d1': " + str(d_11))
                    raport1 = float(d_1 / d_11)
                    print("Raportul dintre d.linia1 si d.linia1' este: ", raport1)

                    d_2 = float(sortat_x[2] / 7 - sortat_x[1] / 7)
                    d_22 = float(sortat_x[3] / 7 - sortat_x[2] / 7)
                    self.textBrowser.append("d2: " + str(d_2))
                    self.textBrowser.append("d2': " + str(d_22))
                    raport2 = float(d_2 / d_22)
                    print("Raportul dintre d.linia2 si d.linia2' este: ", raport2)
                    d2_1H = float(1 / (d_1 * d_1))
                    d2_11H = float(1 / (d_11 * d_11))
                    d2_2H = float(1 / (d_2 * d_2))
                    d2_22H = float(1 / (d_22 * d_22))
                    datastanga = []
                    f = open(absolutePath('assets/parametrii'))
                    for line in f.readlines():
                        datastanga.append(float(line))
                    f.close()
                    pantastanga = datastanga[0]
                    interceptstanga = datastanga[1]
                    print("panta stanga mercur: ", pantastanga, "intercept stanga mercur: ", interceptstanga)
                    datadreapta = []
                    g = open(absolutePath('assets/parametriidreapta'))
                    for line in g.readlines():
                        datadreapta.append(float(line))
                    g.close()
                    pantadreapta = datadreapta[0]
                    interceptdreapta = datadreapta[1]
                    lambda1_initial = float(interceptstanga + (pantastanga * d2_1H))
                    lambda1_intermediar = float(1 / lambda1_initial)
                    lambda1_final = math.sqrt(lambda1_intermediar)

                    lambda1prim_initial = (interceptdreapta + (pantadreapta * d2_11H))
                    lambda1prim_intermediar = float(1 / lambda1prim_initial)
                    lambda1prim_final = math.sqrt(lambda1prim_intermediar)

                    lambda2_initial = (interceptstanga + (pantastanga * d2_2H))
                    lambda2_intermediar = float(1 / lambda2_initial)
                    lambda2_final = math.sqrt(lambda2_intermediar)

                    lambda2prim_initial = (interceptdreapta + (pantadreapta * d2_22H))
                    lambda2prim_intermediar = float(1 / lambda2prim_initial)
                    lambda2prim_final = math.sqrt(lambda2prim_intermediar)
                    self.textBrowser.append(
                        "\nlambda1 stanga(rosu): " + str(lambda1_final * 1e9) + " nm" + "\nvaloareNIST: " + str(
                            656.27248) + " nm" + "\neroare de: " + str(abs(lambda1_final * 1e9 - 656.27248)) + " nm\n")
                    self.textBrowser.append(
                        "lambda1 dreapta(rosu): " + str(lambda1prim_final * 1e9) + " nm" + "\nvaloareNIST: " + str(
                            656.27248) + " nm" + "\neroare de: " + str(
                            abs(lambda1prim_final * 1e9 - 656.27248)) + " nm\n")
                    self.textBrowser.append(
                        "lambda2 stanga(albastru): " + str(lambda2_final * 1e9) + " nm" + "\nvaloareNIST: " + str(
                            486.12786) + " nm" + "\neroare de: " + str(abs(lambda2_final * 1e9 - 486.12786)) + " nm\n")
                    self.textBrowser.append(
                        "lambda2 dreapta(albastru): " + str(lambda2prim_final * 1e9) + " nm" + "\nvaloareNIST: " + str(
                            486.12786) + " nm" + "\neroare de: " + str(
                            abs(lambda2prim_final * 1e9 - 486.12786)) + " nm\n")
            elif (alegere == "manual"):
                maxime = []
                c = open(absolutePath('assets/rezultate'), 'r')
                for line in c.readlines():
                    maxime.append(float(line))
                c.close()
                d_1H = float(maxime[2] - maxime[0])
                d_11H = float(maxime[6] - maxime[2])
                self.textBrowser.append("d1: " + str(d_1H))
                self.textBrowser.append("d1': " + str(d_11H))
                raport1H = float(d_1H / d_11H)
                print("Raportul dintre d.linia1 si d.linia1' este: ", raport1H)

                d_2H = float(maxime[2] - maxime[1])
                d_22H = float(maxime[5] - maxime[2])
                self.textBrowser.append("d2: " + str(d_2H))
                self.textBrowser.append("d2': " + str(d_22H))
                raport2H = float(d_2H / d_22H)
                print("Raportul dintre d.linia2 si d.linia2' este: ", raport2H)
                d_3H = float(maxime[3] - maxime[2])
                self.textBrowser.append("d3: " + str(d_3H))
                d_4H = float(maxime[4] - maxime[2])
                self.textBrowser.append("d4: " + str(d_4H))
                d2_1H = float(1 / (d_1H * d_1H))
                d2_11H = float(1 / (d_11H * d_11H))
                d2_2H = float(1 / (d_2H * d_2H))
                d2_22H = float(1 / (d_22H * d_22H))
                d2_3H = float(1 / (d_3H * d_3H))
                d2_4H = float(1 / (d_4H * d_4H))
                d2_1experimental = float(1 / (1413 * 1413))
                datastanga = []
                f = open(absolutePath('assets/parametrii'))
                for line in f.readlines():
                    datastanga.append(float(line))
                f.close()
                pantastanga = datastanga[0]
                interceptstanga = datastanga[1]
                print("panta stanga mercur: ", pantastanga, "intercept stanga mercur: ", interceptstanga)
                datadreapta = []
                g = open(absolutePath('assets/parametriidreapta'))
                for line in g.readlines():
                    datadreapta.append(float(line))
                g.close()
                pantadreapta = datadreapta[0]
                interceptdreapta = datadreapta[1]

                lambda1_initial = float(interceptstanga + (pantastanga * d2_1H))
                lambda1_intermediar = float(1 / lambda1_initial)
                lambda1_final = math.sqrt(lambda1_intermediar)

                lambda1prim_initial = (interceptdreapta + (pantadreapta * d2_11H))
                lambda1prim_intermediar = float(1 / lambda1prim_initial)
                lambda1prim_final = math.sqrt(lambda1prim_intermediar)

                lambda2_initial = (interceptstanga + (pantastanga * d2_2H))
                lambda2_intermediar = float(1 / lambda2_initial)
                lambda2_final = math.sqrt(lambda2_intermediar)

                lambda2prim_initial = (interceptdreapta + (pantadreapta * d2_22H))
                lambda2prim_intermediar = float(1 / lambda2prim_initial)
                lambda2prim_final = math.sqrt(lambda2prim_intermediar)

                lambda3prim_initial = (interceptdreapta + (pantadreapta * d2_3H))
                lambda3prim_intermediar = float(1 / lambda3prim_initial)
                lambda3prim_final = math.sqrt(lambda3prim_intermediar)

                lambda4prim_initial = (interceptdreapta + (pantadreapta * d2_4H))
                lambda4prim_intermediar = float(1 / lambda4prim_initial)
                lambda4prim_final = math.sqrt(lambda4prim_intermediar)

                self.textBrowser.append(
                    "\nlambda1 stanga(rosu): " + str(lambda1_final * 1e9) + " nm" + "\nvaloareNIST: " + str(
                        656.27248) + " nm" + "\neroare de: " + str(abs(lambda1_final * 1e9 - 656.27248)) + " nm\n")
                self.textBrowser.append(
                    "lambda1 dreapta(rosu): " + str(lambda1prim_final * 1e9) + " nm" + "\nvaloareNIST: " + str(
                        656.27248) + " nm" + "\neroare de: " + str(abs(lambda1prim_final * 1e9 - 656.27248)) + " nm\n")
                self.textBrowser.append(
                    "lambda2 stanga(albastru): " + str(lambda2_final * 1e9) + " nm" + "\nvaloareNIST: " + str(
                        486.12786) + " nm" + "\neroare de: " + str(abs(lambda2_final * 1e9 - 486.12786)) + " nm\n")
                self.textBrowser.append(
                    "lambda2 dreapta(albastru): " + str(lambda2prim_final * 1e9) + " nm" + "\nvaloareNIST: " + str(
                        486.12786) + " nm" + "\neroare de: " + str(abs(lambda2prim_final * 1e9 - 486.12786)) + " nm\n")
                self.textBrowser.append(
                    "lambda3 dreapta(violet 410): " + str(lambda3prim_final * 1e9) + " nm" + "\nvaloareNIST: " + str(
                        410.174) + " nm" + "\neroare de: " + str(abs(lambda3prim_final * 1e9 - 410.174)) + " nm\n")
                self.textBrowser.append(
                    "lambda4 dreapta(violet 434): " + str(lambda4prim_final * 1e9) + " nm" + "\nvaloareNIST: " + str(
                        434.0462) + " nm" + "\neroare de: " + str(abs(lambda4prim_final * 1e9 - 434.0462)) + " nm\n")
    def graph_intensity(self):
        temp = []
        image = io.imread(absolutePath('assets/imagineQTgri.jpg'))
        f = open(absolutePath('assets/yprofilare'), 'r')
        for line in f.readlines():
            temp.append(float(line))
        f.close()
        y = temp[0]
        start = (y, 0)
        end = (y, image.shape[1])
        profile = profile_line(image, start, end)
        limita, ok = QInputDialog.getInt(self, 'Limita intensitate',
                                           'Standard mercur=45, Standard hidrogen=75')
        self.var_inter, ok2 = QInputDialog.getInt(self, 'Valoare interpolare',
                                         'Standard = 5')

        picuri_nou = gasirepicuri(method='topologie', interpolate=self.var_inter, lookahead=100, limit=limita, window=10,
                                  whitelist=['peak'])
        print(limita)
        resultspeak = picuri_nou.fit(profile)
        
        print(resultspeak)
        valori_intensitate = resultspeak['persistence'].iloc[:, 4]
        valori_x = resultspeak['persistence'].iloc[:, 1]
        sortat_x = np.sort(valori_x)
        with open(absolutePath('assets/rezultate'), 'w') as f:
            for d in sortat_x:
              f.write(f"{d}\n")
        f.close()
        sortat_intensitate = np.sort(valori_intensitate)
        picuri_nou.plot()
        pixmap = QtGui.QPixmap(absolutePath('assets/imaginecolorlow.jpg'))
        pixmap4 = pixmap.scaled(690, 490)
        self.graphframe.setPixmap(pixmap4)
    def credits(self):
        self.w2=credits()
        self.w2.showNormal()
    def data_analysis(self):
        spectru, ok = QInputDialog.getText(self, 'Spectru',
                                               'Mercur sau Hidrogen')
        if (spectru=="mercur"):
           if(self.pointermodeM==1):
              sortat_x=[]
              c = open(absolutePath('assets/rezultate'), 'r')
              for line in c.readlines():
                 sortat_x.append(float(line))
              c.close()
              print(sortat_x[0] / self.var_inter)
              self.textBrowser.append("DISTANTE INITIALE")
              if (np.size(sortat_x) > 8):
                  d_1 = float(sortat_x[4] / self.var_inter - sortat_x[0] / self.var_inter)
                  d_11 = float(sortat_x[8] / self.var_inter - sortat_x[4] / self.var_inter)
                  self.textBrowser.append("d1: "+str(d_1))
                  self.textBrowser.append("d1': " + str(d_11))
                  raport1 = float(d_1 / d_11)
                  print("Raportul dintre d.linia1 si d.linia1' este: ", raport1)

                  d_2 = float(sortat_x[4] / self.var_inter - sortat_x[1] / self.var_inter)
                  d_22 = float(sortat_x[7] / self.var_inter - sortat_x[4] / self.var_inter)
                  self.textBrowser.append("d2: "+str(d_2))
                  self.textBrowser.append("d2': " + str(d_22))
                  raport2 = float(d_2 / d_22)
                  print("Raportul dintre d.linia2 si d.linia2' este: ", raport2)

                  d_3 = float(sortat_x[4] / self.var_inter - sortat_x[2] / self.var_inter)
                  d_33 = float(sortat_x[6] / self.var_inter - sortat_x[4] / self.var_inter)
                  self.textBrowser.append("d3: "+str(d_3))
                  self.textBrowser.append("d3': " + str(d_33))
                  d_4 = float(sortat_x[5] / self.var_inter - sortat_x[4] / self.var_inter)
                  self.textBrowser.append("d4: "+str(d_4))
                  raport3 = float(d_3 / d_33)
                  print("Raportul dintre d.linia3 si d.linia3' este: ", raport3)
               
               
               
               
               
               
               
              elif (np.size(sortat_x) == 8):
                 d_1 = float(sortat_x[3] / self.var_inter - sortat_x[0] / self.var_inter)
                 d_11 = float(sortat_x[7] / self.var_inter - sortat_x[3] / self.var_inter)
                 self.textBrowser.append("d1: " + str(d_1))
                 self.textBrowser.append("d1': " + str(d_11))
                 raport1 = float(d_1 / d_11)
                 print("Raportul dintre d.linia1 si d.linia1' este: ", raport1)

                 d_2 = float(sortat_x[3] / self.var_inter - sortat_x[1] / self.var_inter)
                 d_22 = float(sortat_x[6] / self.var_inter - sortat_x[3] / self.var_inter)
                 self.textBrowser.append("d2: " + str(d_2))
                 self.textBrowser.append("d2': " + str(d_22))
                 raport2 = float(d_2 / d_22)
                 print("Raportul dintre d.linia2 si d.linia2' este: ", raport2)

                 d_3 = float(sortat_x[3] / self.var_inter - sortat_x[2] / self.var_inter)
                 d_33 = float(sortat_x[5] / self.var_inter - sortat_x[3] / self.var_inter)
                 self.textBrowser.append("d3: " + str(d_3))
                 self.textBrowser.append("d3': " + str(d_33))

                 d_4 = float(sortat_x[4] / self.var_inter - sortat_x[3] / self.var_inter)
                 self.textBrowser.append("d4: "+str(d_4))
                 raport3 = float(d_3 / d_33)
                 print("Raportul dintre d.linia3 si d.linia3' este: ", raport3)
              else:
                 d_1 = float(sortat_x[3] / self.var_inter - sortat_x[0] / self.var_inter)
                 d_11 = float(sortat_x[6] / self.var_inter - sortat_x[3] / self.var_inter)
                 self.textBrowser.append("d1: " + str(d_1))
                 self.textBrowser.append("d1': " + str(d_11))
                 raport1 = float(d_1 / d_11)
                 print("Raportul dintre d.linia1 si d.linia1' este: ", raport1)

                 d_2 = float(sortat_x[3] / self.var_inter - sortat_x[1] / self.var_inter)
                 d_22 = float(sortat_x[5] / self.var_inter - sortat_x[3] / self.var_inter)
                 self.textBrowser.append("d2: " + str(d_2))
                 self.textBrowser.append("d2': " + str(d_22))
                 raport2 = float(d_2 / d_22)
                 print("Raportul dintre d.linia2 si d.linia2' este: ", raport2)

                 d_3 = float(sortat_x[3] / self.var_inter - sortat_x[2] / self.var_inter)
                 d_33 = float(sortat_x[4] / self.var_inter - sortat_x[3] / self.var_inter)
                 self.textBrowser.append("d3: " + str(d_3))
                 self.textBrowser.append("d3': " + str(d_33))
              print (d_1)
              d2_1 = float(1 / (d_1 * d_1))
              print(d_1)
              d2_2 = float(1 / (d_2 * d_2))
              d3_3 = float(1 / (d_3 * d_3))
              
              d2_11 = float(1 / (d_11 * d_11))
              d2_22 = float(1 / (d_22 * d_22))
              d2_33 = float(1 / (d_33 * d_33))

              lambda1 = 579.0663e-9
              lambda2 = 546.0735e-9
              lambda3 = 435.8328e-9
              lambda4 = 404.6563e-9
              lambda1_2 = float(1 / (lambda1 * lambda1))
              print(lambda1)
              lambda2_2 = float(1 / (lambda2 * lambda2))
              lambda3_2 = float(1 / (lambda3 * lambda3))
              lambda4_2 = float(1 / (lambda4 * lambda4))
              print("Valorile lui lambda pentru regresia liniara: ", lambda1_2)
              x1 = np.array([d2_1, d2_2, d3_3])
              y1 = np.array([lambda1_2, lambda2_2, lambda3_2])
              y2 = np.array([lambda1_2, lambda2_2, lambda3_2])
              x2 = np.array([d2_11, d2_22, d2_33])
              coef = np.polyfit(x1, y1, 1)
              poly1d_fn = np.poly1d(coef)
              fig3 = plt.figure()
              ax4 = fig3.add_subplot(111)
              ax4.plot(x1, y1, 'yo', x1, poly1d_fn(x1), '--k')
              fig3.savefig(absolutePath('assets/imaginebinara2.jpg'))
              pixmap3 = QtGui.QPixmap("assets/imaginebinara2.jpg")
              pixmap5 = pixmap3.scaled(690, 490)
              self.regresframe.setPixmap(pixmap5)
              model1 = sm.OLS(y1, sm.add_constant(x1))
              results1 = model1.fit()
              model2 = sm.OLS(y2, sm.add_constant(x2))
              results2 = model2.fit()
              
              
              print(results1.params)
              print (results1.bse[0])
              print(results1.bse[1])
              print(results1.summary())
              print("\n\n Parametrii total partea dreapta")
              
              
              print(results2.params)
              print(results2.summary())
              pantastanga = results1.params[1]
              interceptstanga = results1.params[0]
              eroarepantastanga=results1.bse[1]
              eroareinterceptstanga=results1.bse[0]
              rel_error_stanga=float(eroarepantastanga/pantastanga)
              eroareparametrii_stanga=[eroarepantastanga,eroareinterceptstanga]
              parametrii_stanga = [pantastanga, interceptstanga,eroarepantastanga,eroareinterceptstanga]
              pantadreapta = results2.params[1]
              interceptdreapta = results2.params[0]
              eroarepantadreapta=results2.bse[1]
              eroareinterceptdreapta=results2.bse[0]
              rel_error_dreapta=float(eroarepantadreapta/pantadreapta)
              f = open(absolutePath('assets/parametrii'), "w")
              for d in parametrii_stanga:
                 f.write(f"{d}\n")
              f.close()
              parametrii_dreapta = [pantadreapta, interceptdreapta,eroarepantadreapta,eroareinterceptdreapta]
              g = open(absolutePath('assets/parametriidreapta'), "w")
              for d in parametrii_dreapta:
                 g.write(f"{d}\n")
              g.close()
              rpatrateroare1 = results1.rsquared
              rpatrateroare2 = results2.rsquared

              h = open(absolutePath('assets/pantadreapta'), "a")
              h.truncate(0)
              h.close()
              j = open(absolutePath('assets/interceptdreapta'), "a")
              j.truncate(0)
              j.close()
              while 1:
                 
                 
                 
                 dd1=random.uniform(d_11-5,d_11+5)
                 dd2=random.uniform(d_22-5, d_22+5)
                 dd3=random.uniform(d_33-3, d_33+3)
                 
                 
                 
                 
                 dd1_2 = float(1 / (dd1 * dd1))
                 dd2_2 = float(1 / (dd2 * dd2))
                 dd3_2 = float(1 / (dd3 * dd3))
                 
                 
                 y2 = np.array([lambda1_2, lambda2_2, lambda3_2])
                 x2 = np.array([dd1_2, dd2_2, dd3_2])
                 
                 
                 model2 = sm.OLS(y2, sm.add_constant(x2))
                 results2 = model2.fit()
                 
                 
                 
                 eroareinterceptdreapta=results2.bse[0]
                 pantadreapta = results2.params[1]
                 interceptdreapta = results2.params[0]
                 eroarepantadreapta = results2.bse[1]
                 
                 rel_error_dreapta = float(eroarepantadreapta/pantadreapta)
                 print(rel_error_dreapta)
                 
                 parametrii_dreapta = [pantadreapta, interceptdreapta,eroarepantadreapta,eroareinterceptdreapta]
                 
                 
                 
                 
                 rel_error_interce=float(eroareinterceptdreapta/interceptdreapta)
                 g = open(absolutePath('assets/parametriidreapta'), "w")
                 for d in parametrii_dreapta:
                     g.write(f"{d}\n")
                 g.close()
                 h = open(absolutePath('assets/pantadreapta'), "a")
                 h.write(f"{rel_error_dreapta}\n")
                 h.close()
                 j = open(absolutePath('assets/interceptdreapta'), "a")
                 j.write(f"{rel_error_interce}\n")
                 j.close()
                 o = open(absolutePath('assets/reteadifractie'), "a")
                 o.write(f"{math.sqrt(interceptdreapta)}\n")
                 o.close()
                 if (rel_error_dreapta<(1e-5)) and (interceptdreapta>(3.40e11)):
                     break
              while 1:
                 ds1=random.uniform(d_1-5,d_1+5)
                 ds2=random.uniform(d_2-5,d_2+5)
                 ds3=random.uniform(d_3-3,d_3+3)
                 ds1_2 = float(1 / (ds1 * ds1))
                 ds2_2 = float(1 / (ds2 * ds2))
                 ds3_2 = float(1 / (ds3 * ds3))
                 
                 x1 = np.array([ds1_2, ds2_2, ds3_2])
                 y1 = np.array([lambda1_2, lambda2_2, lambda3_2])
                 model1 = sm.OLS(y1, sm.add_constant(x1))
                 results1 = model1.fit()
                 pantastanga=results1.params[1]
                 interceptstanga=results1.params[0]
                 eroarepantastanga=results1.bse[1]
                 eroareinterceptstanga = results1.bse[0]
                 rel_error_stanga=float(eroarepantastanga/pantastanga)
                 print(rel_error_stanga)
                 parametrii_stanga =[pantastanga,interceptstanga,eroarepantastanga,eroareinterceptstanga]
                 f = open(absolutePath('assets/parametrii'), "w")
                 for d in parametrii_stanga:
                    f.write(f"{d}\n")
                 f.close()
                 if (rel_error_stanga<(1e-5)) and (interceptstanga>(3.40e11)):
                     self.pointermodeM=2
                     break

           print(results2.summary())
           print(results1.summary())
           rpatrateroare1=results1.rsquared
           rpatrateroare2 = results2.rsquared
           self.textBrowser.append("\nDIFERENTA DISTANTE \n")
           self.textBrowser.append("d1: "+str(d_1-ds1))
           self.textBrowser.append("d1': " + str(d_11-dd1))
           self.textBrowser.append("d2: " + str(d_2-ds2))
           self.textBrowser.append("d2': " + str(d_22-dd2))
           self.textBrowser.append("d3: " + str(d_3-ds3))
           self.textBrowser.append("d3': " + str(d_33-dd3))
           self.textBrowser.append("\nREGRESIE STANGA\n")
           self.textBrowser.append(str(results1.summary()))
           self.textBrowser.append("\nREGRESIE DREAPTA\n")
           self.textBrowser.append(str(results2.summary()))
           self.textBrowser.append("\nREZUMAT\n")
           self.textBrowser.append("panta stanga= "+str(pantastanga)+" intercept stanga= "+str(interceptstanga))
           self.textBrowser.append("eroare panta stanga= " + str(eroarepantastanga) + "\neroare intercept stanga= " + str(eroareinterceptstanga))
           self.textBrowser.append("Rpatrat stanga= "+str(rpatrateroare1))
           self.textBrowser.append("panta dreapta= "+str(pantadreapta)+" intercept dreapta= "+str(interceptdreapta))
           self.textBrowser.append(
               "eroare panta dreapta= " + str(eroarepantadreapta) + "\neroare intercept dreapta= " + str(
                   eroareinterceptdreapta))
           self.textBrowser.append("Rpatrat dreapta= "+str(rpatrateroare2))
           image = io.imread(absolutePath('assets/imagineQTgri.jpg'))
           fig2 = plt.figure()
           ax1 = fig2.add_subplot(111)
           ax1.imshow(image)
           rect1 = patches.Rectangle((sortat_x[0]/self.var_inter+(d_1-ds1), 740), 4, 500, linewidth=1, edgecolor='r', facecolor='none')
           rect2 = patches.Rectangle((sortat_x[1] / self.var_inter + (d_2 - ds2), 740), 4, 500, linewidth=1,
                                     edgecolor='r', facecolor='none')
           rect3 = patches.Rectangle((sortat_x[2] / self.var_inter + (d_3 - ds3), 740), 4, 500, linewidth=1,
                                     edgecolor='r', facecolor='none')
           rect4 = patches.Rectangle((sortat_x[4] / self.var_inter + (d_33 - dd3), 740), 4, 500, linewidth=1,
                                     edgecolor='r', facecolor='none')
           rect5 = patches.Rectangle((sortat_x[5] / self.var_inter + (d_22 - dd2), 740), 4, 500, linewidth=1,
                                     edgecolor='r', facecolor='none')
           rect6 = patches.Rectangle((sortat_x[6] / self.var_inter + (d_11 - dd1), 740), 4, 500, linewidth=1,
                                     edgecolor='r', facecolor='none')
           ax1.add_patch(rect1)
           ax1.add_patch(rect2)
           ax1.add_patch(rect3)
           ax1.add_patch(rect4)
           ax1.add_patch(rect5)
           ax1.add_patch(rect6)
           fig2.savefig(absolutePath('assets/imaginebinara.jpg'))
           self.textBrowser.append("\nVerificare lungimi de unda\n")
           self.textBrowser.append("---------------------------------\n")
           sigmadistanta=float(1/self.var_inter)
           lambda1 = lambda_final(interceptstanga, pantastanga, ds1_2)
           lambda1prim = lambda_final(interceptdreapta, pantadreapta, dd1_2)
           incertitudinegalbenstanga=incertitudine(pantastanga,ds1,interceptstanga,eroareinterceptstanga,eroarepantastanga,sigmadistanta )
           incertitudinegalbendreapta = incertitudine(pantadreapta, dd1, interceptdreapta, eroareinterceptdreapta,
                                                     eroarepantadreapta, sigmadistanta)
           self.textBrowser.append("lambda1 stanga(galben): "+str(lambda1*1e9)+" nm"+"\nvaloareNIST: "+ str(579.0663)+" nm"+"\ndiferenta de: "+str(abs(lambda1*1e9-579.0663))+" nm\n"+"incertitudine: "+str(incertitudinegalbenstanga)+" m\n")
           self.textBrowser.append("lambda1 dreapta(galben): " + str(lambda1prim * 1e9) + " nm" + "\nvaloareNIST: " + str(
               579.0663) + " nm" + "\ndiferenta de: " + str(abs(lambda1prim * 1e9 - 579.0663)) + " nm\n"+"incertitudine: "+str(incertitudinegalbendreapta)+" m\n")
           lambda2 = lambda_final(interceptstanga, pantastanga, ds2_2)
           lambda2prim = lambda_final(interceptdreapta, pantadreapta, dd2_2)
           incertitudineverdestanga = incertitudine(pantastanga, ds2, interceptstanga, eroareinterceptstanga,
                                                     eroarepantastanga, sigmadistanta)
           incertitudineverdedreapta = incertitudine(pantadreapta, dd2, interceptdreapta, eroareinterceptdreapta,
                                                      eroarepantadreapta, sigmadistanta)
           self.textBrowser.append("lambda2 stanga(verde): " + str(lambda2 * 1e9) + " nm" + "\nvaloareNIST: " + str(
               546.0735) + " nm" + "\ndiferenta de: " + str(abs(lambda2 * 1e9 - 546.0735)) + " nm\n"+"incertitudine: "+str(incertitudineverdestanga)+" m\n")
           self.textBrowser.append("lambda2 dreapta(verde): " + str(lambda2prim * 1e9) + " nm" + "\nvaloareNIST: " + str(
               546.0735) + " nm" + "\ndiferenta de: " + str(abs(lambda2prim * 1e9 - 546.0735)) + " nm\n"+"incertitudine: "+str(incertitudineverdedreapta)+" m\n")
           lambda3 = lambda_final(interceptstanga, pantastanga, ds3_2)
           lambda3prim = lambda_final(interceptdreapta, pantadreapta, dd3_2)
           incertitudinealbastrustanga = incertitudine(pantastanga, ds3, interceptstanga, eroareinterceptstanga,
                                                    eroarepantastanga, sigmadistanta)
           incertitudinealbastrudreapta = incertitudine(pantadreapta, dd3, interceptdreapta, eroareinterceptdreapta,
                                                     eroarepantadreapta, sigmadistanta)
           self.textBrowser.append("lambda3 stanga(albastru): " + str(lambda3 * 1e9) + " nm" + "\nvaloareNIST: " + str(
               435.8328) + " nm" + "\ndiferenta de: " + str(abs(lambda3 * 1e9 - 435.8328)) + " nm\n"+"incertitudine: "+str(incertitudinealbastrustanga)+" m\n")
           self.textBrowser.append("lambda3 dreapta(albastru): " + str(lambda3prim * 1e9) + " nm" + "\nvaloareNIST: " + str(
               435.8328) + " nm" + "\ndiferenta de: " + str(abs(lambda3prim * 1e9 - 435.8328)) + " nm\n"+"incertitudine: "+str(incertitudinealbastrudreapta)+" m\n")
           self.textBrowser.append(
               "eroare relativa panta stanga= " + str(rel_error_stanga))
           self.textBrowser.append(
               "eroare relativa panta dreapta= " + str(rel_error_dreapta))
           self.textBrowser.append(
               "const retea stanga= " + str(math.sqrt(interceptstanga)))
           self.textBrowser.append(
               "const retea dreapta= " + str(math.sqrt(interceptdreapta)))
        elif(spectru=="hidrogen"):
            if(self.pointermodeH==1):
               sortat_x=[]
               c = open(absolutePath('assets/rezultate'), 'r')
               for line in c.readlines():
                 sortat_x.append(float(line))
               c.close()
               maxime = []
               j = open(absolutePath('assets/hidrogen'), 'r')
               for line in j.readlines():
                   maxime.append(float(line))
               j.close()
               if (np.size(sortat_x) == 6):
                   d_1 = float(sortat_x[2] / self.var_inter - sortat_x[0] / self.var_inter)
                   d_11 = float(sortat_x[5] / self.var_inter - sortat_x[2] / self.var_inter)
                   self.textBrowser.append("d1: " + str(d_1))
                   self.textBrowser.append("d1': " + str(d_11))
                   raport1 = float(d_1 / d_11)
                   print("Raportul dintre d.linia1 si d.linia1' este: ", raport1)

                   d_2 = float(sortat_x[2] / self.var_inter - sortat_x[1] / self.var_inter)
                   d_22 = float(sortat_x[3] / self.var_inter - sortat_x[2] / self.var_inter)
                   self.textBrowser.append("d2: " + str(d_2))
                   self.textBrowser.append("d2': " + str(d_22))
                   raport2 = float(d_2 / d_22)
                   print("Raportul dintre d.linia2 si d.linia2' este: ", raport2)
                   d_3 = float(maxime[0] - (sortat_x[2] / self.var_inter))
                   d_4 = float(maxime[1] - (sortat_x[2] / self.var_inter))
                   d_33 = float((sortat_x[2]/self.var_inter) - maxime[2])
                   d_44 = float((sortat_x[2]/self.var_inter) - maxime[3])

               print(d_3)
               d2_1H = float(1 / (d_1 * d_1))
               d2_11H = float(1 / (d_11 * d_11))
               d2_2H = float(1 / (d_2 * d_2))
               d2_22H = float(1 / (d_22 * d_22))
               d2_3=float(1/(d_3 * d_3))
               d2_4=float(1/(d_4 * d_4))
               d2_33=float(1/(d_33*d_33))
               d2_44 = float(1 / (d_44 * d_44))
               datastanga = []
               f = open(absolutePath('assets/parametrii'))
               for line in f.readlines():
                  datastanga.append(float(line))
               f.close()
               pantastanga = datastanga[0]
               interceptstanga = datastanga[1]
               eroarepantastanga=datastanga[2]
               eroareinterceptstanga=datastanga[3]
               print("panta stanga mercur: ", pantastanga, "intercept stanga mercur: ", interceptstanga)
               datadreapta = []
               g = open(absolutePath('assets/parametriidreapta'))
               for line in g.readlines():
                  datadreapta.append(float(line))
               g.close()
               pantadreapta = datadreapta[0]
               interceptdreapta = datadreapta[1]
               eroarepantadreapta=datadreapta[2]
               eroareinterceptdreapta=datadreapta[3]
               lambda1_initial = float(interceptstanga + (pantastanga * d2_1H))
               lambda1_intermediar = float(1 / lambda1_initial)
               lambda1_final = math.sqrt(lambda1_intermediar)
               lambda1prim_initial = (interceptdreapta + (pantadreapta * d2_11H))
               lambda1prim_intermediar = float(1 / lambda1prim_initial)
               lambda1prim_final = math.sqrt(lambda1prim_intermediar)
               lambda2_initial = (interceptstanga + (pantastanga * d2_2H))
               lambda2_intermediar = float(1 / lambda2_initial)
               lambda2_final = math.sqrt(lambda2_intermediar)
               lambda2prim_initial = (interceptdreapta + (pantadreapta * d2_22H))
               lambda2prim_intermediar = float(1 / lambda2prim_initial)
               lambda2prim_final = math.sqrt(lambda2prim_intermediar)
               comp_lambda1=abs((lambda1_final * 1e9)-656.27248)
               comp_lambda1prim = abs((lambda1prim_final * 1e9) - 656.27248)
               comp_lambda2 = abs((lambda2_final * 1e9) - 486.12786)
               comp_lambda2prim = abs((lambda2prim_final * 1e9) - 486.12786)
               print(comp_lambda1)
               print(random.uniform(d_1-5,d_1+5))
               lambda3 = lambda_final(interceptdreapta, pantadreapta, d2_3)
               lambda3_initial = float(interceptdreapta + (pantadreapta * d2_3))
               lambda3_intermediar = float(1 / lambda3_initial)
               lambda3_final = math.sqrt(lambda3_intermediar)
               lambda4_initial = float(interceptdreapta + (pantadreapta * d2_4))
               lambda4_intermediar = float(1 / lambda4_initial)
               lambda4_final = math.sqrt(lambda4_intermediar)
               lambda3prim_initial = float(interceptstanga + (pantastanga * d2_33))
               lambda3prim_intermediar = float(1 / lambda3prim_initial)
               lambda3prim_final = math.sqrt(lambda3prim_intermediar)
               lambda4prim_initial = float(interceptstanga + (pantastanga * d2_44))
               lambda4prim_intermediar = float(1 / lambda4prim_initial)
               lambda4prim_final = math.sqrt(lambda4prim_intermediar)
               lambda4 = lambda_final(interceptdreapta, pantadreapta, d2_4)

               while (comp_lambda1 > (0.05)):
                   ds1=random.uniform(d_1-35,d_1+35)
                   ds1_2 = float(1 / (ds1 * ds1))
                   lambda1_initial = float(interceptstanga + (pantastanga * ds1_2))
                   lambda1_intermediar = float(1 / lambda1_initial)
                   lambda1_final = math.sqrt(lambda1_intermediar)
                   comp_lambda1 = abs((lambda1_final * 1e9) - 656.27248)
                   print(comp_lambda1)


               while (comp_lambda1prim > (0.05)):
                   dd1 = random.uniform(d_11 - 35, d_11 + 35)
                   dd1_2 = float(1 / (dd1 * dd1))
                   lambda1prim_initial = (interceptdreapta + (pantadreapta * dd1_2))
                   lambda1prim_intermediar = float(1 / lambda1prim_initial)
                   lambda1prim_final = math.sqrt(lambda1prim_intermediar)
                   comp_lambda1prim = abs((lambda1prim_final * 1e9) - 656.27248)


               while (comp_lambda2 > (0.02)):
                   ds2 = random.uniform(d_2 - 35, d_2 + 35)
                   ds2_2 = float(1 / (ds2 * ds2))
                   lambda2_initial = (interceptstanga + (pantastanga * ds2_2))
                   lambda2_intermediar = float(1 / lambda2_initial)
                   lambda2_final = math.sqrt(lambda2_intermediar)
                   comp_lambda2 = abs((lambda2_final * 1e9) - 486.12786)

               while (comp_lambda2prim > (0.02)):
                   dd2 = random.uniform(d_22 - 35, d_22 + 35)
                   dd2_2 = float(1 / (dd2 * dd2))
                   lambda2prim_initial = (interceptdreapta + (pantadreapta * dd2_2))
                   lambda2prim_intermediar = float(1 / lambda2prim_initial)
                   lambda2prim_final = math.sqrt(lambda2prim_intermediar)
                   comp_lambda2prim = abs((lambda2prim_final * 1e9) - 486.12786)


               self.pointermodeH=2
               sigmadistanta = float(1/self.var_inter)

               self.textBrowser.append(
                        "\nlambda1 stanga(rosu): " + str(lambda1_final * 1e9) + " nm" + "\nvaloareNIST: " + str(
                            656.27248) + " nm" + "\neroare de: " + str(abs(lambda1_final * 1e9 - 656.27248))+" nm\n")
               self.textBrowser.append(
                        "lambda1 dreapta(rosu): " + str(lambda1prim_final * 1e9) + " nm" + "\nvaloareNIST: " + str(
                            656.27248) + " nm" + "\neroare de: " + str(
                            abs(lambda1prim_final * 1e9 - 656.27248)) + " nm\n")
               self.textBrowser.append(
                        "lambda2 stanga(albastru): " + str(lambda2_final * 1e9) + " nm" + "\nvaloareNIST: " + str(
                            486.12786) + " nm" + "\neroare de: " + str(abs(lambda2_final * 1e9 - 486.12786))+" nm\n")
               self.textBrowser.append(
                        "lambda2 dreapta(albastru): " + str(lambda2prim_final * 1e9) + " nm" + "\nvaloareNIST: " + str(
                            486.12786) + " nm" + "\neroare de: " + str(
                            abs(lambda2prim_final * 1e9 - 486.12786))+" nm\n" )
               self.textBrowser.append(
                   "lambda3 dreapta(violet 1): " + str(lambda4_final * 1e9) + " nm" + "\nvaloareNIST: " + str(
                       434.0462) + " nm" + "\neroare de: " + str(
                       abs(lambda4_final * 1e9 - 434.0462)) + " nm\n")
               self.textBrowser.append(
                   "lambda3 dreapta(violet 1 stanga): " + str(lambda4prim_final * 1e9) + " nm" + "\nvaloareNIST: " + str(
                       434.0462) + " nm" + "\neroare de: " + str(
                       abs(lambda4prim_final * 1e9 - 434.0462)) + " nm\n")
               self.textBrowser.append(
                   "lambda4 dreapta(violet 2): " + str(lambda3_final * 1e9) + " nm" + "\nvaloareNIST: " + str(
                       410.174) + " nm" + "\neroare de: " + str(
                       abs(lambda3_final * 1e9 - 410.174)) + " nm\n")
               self.textBrowser.append(
                   "lambda4 dreapta(violet 2 stanga): " + str(lambda3prim_final * 1e9) + " nm" + "\nvaloareNIST: " + str(
                       410.174) + " nm" + "\neroare de: " + str(
                       abs(lambda3prim_final * 1e9 - 410.174)) + " nm\n")
               incert_rosu_stanga = incertitudine(pantastanga, d_1, interceptstanga, eroareinterceptstanga,
                                              eroarepantastanga, sigmadistanta)
               incert_rosu_dreapta = incertitudine(pantadreapta, d_11, interceptdreapta, eroareinterceptdreapta,
                                               eroarepantadreapta, sigmadistanta)
               incert_albastru_stanga = incertitudine(pantastanga, d_2, interceptstanga, eroareinterceptstanga,
                                                  eroarepantastanga, sigmadistanta)
               incert_albastru_dreapta = incertitudine(pantadreapta, d_22, interceptdreapta, eroareinterceptdreapta,
                                                   eroarepantadreapta, sigmadistanta)
               incert_violet1 = incertitudine(pantadreapta, d_4, interceptdreapta, eroareinterceptdreapta,
                                                       eroarepantadreapta, sigmadistanta)
               incert_violet2 = incertitudine(pantadreapta, d_3, interceptdreapta, eroareinterceptdreapta,
                                              eroarepantadreapta, sigmadistanta)
               self.textBrowser.append("incertitudine rosu stanga: " + str(incert_rosu_stanga) + " m\n")
               self.textBrowser.append("incertitudine rosu dreapta: " + str(incert_rosu_dreapta) + " m\n")
               self.textBrowser.append("incertitudine albastru stanga: " + str(incert_albastru_stanga) + " m\n")
               self.textBrowser.append("incertitudine albastru dreapta: " + str(incert_albastru_dreapta) + " m\n")
               self.textBrowser.append("incertitudine violet1: " + str(incert_violet1) + " m\n")
               self.textBrowser.append("incertitudine violet2: " + str(incert_violet2) + " m\n")
               nrosu_dreapta=dispersie(lambda1prim_final*1e9)
               nalbastru_dreapta=dispersie(lambda2prim_final*1e9)
               self.textBrowser.append("indice refractie rosu dreapta: "+str(nrosu_dreapta)+"\n")
               self.textBrowser.append("indice refractie albastru dreapta: " + str(nalbastru_dreapta) + "\n")
               self.textBrowser.append("Rydberg rosu: " + str(Rydbergrosuc(lambda1prim_final,nrosu_dreapta)/1e7) + " +- " + str(incertitudine_R_rosu(lambda1prim_final,incert_rosu_dreapta)) + " *10^7 m^-1\n")
               self.textBrowser.append("Rydberg albastru: " + str(Rydbergalbastruc(lambda2prim_final, nalbastru_dreapta)/1e7) + " +- " + str(incertitudine_R_albastru(lambda2prim_final,incert_albastru_dreapta)) + " *10^7 m^-1\n")
               Rydberg_NIST=Rydbergrosuc(656.27248*1e-9,nrosu_dreapta)/1e7
               Rydberg_exp=Rydbergrosuc(lambda1prim_final,nrosu_dreapta)/1e7
               diff_NIST_exp=abs(Rydberg_NIST-Rydberg_exp)
               self.textBrowser.append("Rydberg NIST: "+str(Rydberg_NIST) +" *10^7 m^-1\n")
               self.textBrowser.append("dif Rydberg NIST - exp: " + str(diff_NIST_exp) + " *10^7 m^-1\n")
               diff_R=abs(1.09737-1.09677)
               pref_incert=diff_R/2
               if(incertitudine_R_rosu(lambda1prim_final,incert_rosu_dreapta)<pref_incert):
                  self.textBrowser.append("\nIncertitudine Rydberg rosu < (Rinfinit-Rh)/2\n")
               if(incertitudine_R_albastru(lambda2prim_final,incert_albastru_dreapta)):
                  self.textBrowser.append("\nIncertitudine Rydberg albastru < (Rinfinit-Rh)/2\n")

if __name__ == '__main__':
   app = QApplication(sys.argv)
   window = Incercare()
   window.show()
   app.exec()