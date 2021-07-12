from __future__ import division
from PyQt4 import QtCore, QtGui
from scipy.signal import hamming
from scipy.fftpack import fft, fftshift, dct
from scipy.io.wavfile import read
from pydub import AudioSegment
from pydub.playback import play
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import pyaudio
import wave
import MySQLdb
import datetime
import time
import sched, time
import pyqtgraph
import SWHear
from PyQt4.Qt import QFrame
from pyqtgraph import PlotWidget
import suara
import _pickle as cPickle
from _datetime import date




#Koneksi Database
con = None
db = MySQLdb.connect("127.0.0.1", "root", "", "pengenalsuara")
kursor = db.cursor()


# Mengambil nilai id terbesar
def id():
    nilai = kursor.execute("Select * from pegawai")
    ambil = kursor.fetchall()
    id = len(ambil)
    return (id)

# Metode MFCC
def hertz_to_mel(freq):
    return 1125 * np.log(1 + freq / 700)

def mel_to_hertz(m):
    return 700 * (np.exp(m / 1125) - 1)

# calculate mel frequency filter bank
def mel_filterbank(nfft, nfiltbank, fs):
    # set limits of mel scale from 300Hz to 8000Hz
    lower_mel = hertz_to_mel(20)
    upper_mel = hertz_to_mel(44000)
    mel = np.linspace(lower_mel, upper_mel, nfiltbank + 2)
    hertz = [mel_to_hertz(m) for m in mel]
    fbins = [int(hz * (nfft / 2 + 1) / fs) for hz in hertz]
    fbank = np.empty((nfft // 2 + 1, nfiltbank))
    for i in range(1, nfiltbank + 1):
        for k in range(int(nfft / 2 + 1)):
            if k < fbins[i - 1]:
                fbank[k, i - 1] = 0
            elif k >= fbins[i - 1] and k < fbins[i]:
                fbank[k, i - 1] = (k - fbins[i - 1]) / (fbins[i] - fbins[i - 1])
            elif k >= fbins[i] and k <= fbins[i + 1]:
                fbank[k, i - 1] = (fbins[i + 1] - k) / (fbins[i + 1] - fbins[i])
            else:
                fbank[k, i - 1] = 0
    return fbank

def mfcc(s, fs, nfiltbank):
    # divide into segments of 25 ms with overlap of 10ms
    nSamples = np.int32(0.025 * fs)
    overlap = np.int32(0.01 * fs)
    nFrames = np.int32(np.ceil(len(s) / (nSamples - overlap)))
    # zero padding to make signal length long enough to have nFrames
    padding = ((nSamples - overlap) * nFrames) - len(s)
    if padding > 0:
        signal = np.append(s, np.zeros(padding))
    else:
        signal = s
    segment = np.empty((nSamples, nFrames))
    start = 0
    for i in range(nFrames):
        segment[:, i] = signal[start:start + nSamples]
        start = (nSamples - overlap) * i
    # compute periodogram
    nfft = 512
    periodogram = np.empty((nFrames, nfft // 2 + 1))
    for i in range(nFrames):
        x = segment[:, i] * hamming(nSamples)
        spectrum = fftshift(fft(x, nfft))
        periodogram[i, :] = abs(spectrum[nfft // 2 - 1:]) / nSamples
    # calculating mfccs
    fbank = mel_filterbank(nfft, nfiltbank, fs)
    # nfiltbank MFCCs for each frame
    mel_coeff = np.empty((nfiltbank, nFrames))
    for i in range(nfiltbank):
        for k in range(nFrames):
            mel_coeff[i, k] = np.sum(periodogram[k, :] * fbank[:, i])
    mel_coeff = np.log10(mel_coeff)
    mel_coeff = dct(mel_coeff)
    # exclude 0th order coefficient (much larger than others)
    mel_coeff[0, :] = np.zeros(nFrames)
    return mel_coeff

# Metode Euglidience Distance
def EUDistance(d, c):
    # np.shape(d)[0] = np.shape(c)[0] 
    n = np.shape(d)[1] # mencari jumlah kolom array d
    p = np.shape(c)[1] # mencari jumlah kolom array c
    distance = np.empty((n, p))
    if n < p:
        for i in range(n):
            copies = np.transpose(np.tile(d[:, i], (p, 1))) #np.tile menggandakan element 
            distance[i, :] = np.sum((copies - c) ** 2, 0) # np.sum menjumlahkan semua elemen array
    else:
        for i in range(p):
            copies = np.transpose(np.tile(c[:, i], (n, 1)))
            distance[:, i] = np.transpose(np.sum((d - copies) ** 2, 0))
    distance = np.sqrt(distance)
    return distance

# Algoritma LBG
def lbg(features, M):
    eps = 0.01
    codebook = np.mean(features, 1)
    distortion = 1
    nCentroid = 1
    while nCentroid < M:
        # double the size of codebook
        new_codebook = np.empty((len(codebook), nCentroid * 2))
        if nCentroid == 1:
            new_codebook[:, 0] = codebook * (1 + eps)
            new_codebook[:, 1] = codebook * (1 - eps)
        else:
            for i in range(nCentroid):
                new_codebook[:, 2 * i] = codebook[:, i] * (1 + eps)
                new_codebook[:, 2 * i + 1] = codebook[:, i] * (1 - eps)
        codebook = new_codebook
        nCentroid = np.shape(codebook)[1]
        D = EUDistance(features, codebook)
        while np.abs(distortion) > eps:
            # nearest neighbour search
            prev_distance = np.mean(D)
            nearest_codebook = np.argmin(D, axis=1)
            # cluster vectors and find new centroid
            for i in range(nCentroid):
                codebook[:, i] = np.mean(features[:, np.where(nearest_codebook == i)], 2).T  # add along 3rd dimension
            # replace all NaN values with 0
            codebook = np.nan_to_num(codebook)
            D = EUDistance(features, codebook)
            distortion = (prev_distance - np.mean(D)) / prev_distance
    return codebook

# Tahapan Training
def training(nfiltbank):
    nSpeaker = sum(1 for f in os.listdir('C:\\Users\\asus\\workspace\\bismillah\\train') if os.path.isfile(os.path.join('C:\\Users\\asus\\workspace\\bismillah\\train', f)) and f[0] != '.')
    nCentroid = 16
    codebooks_mfcc = np.empty((nSpeaker, nfiltbank, nCentroid))
    directory_train = os.getcwd() + '/train';
    fname = str()
    for i in range(nSpeaker):
        fname = '/' + str(i + 1) + '.wav'
        (fs, s) = read(directory_train + fname)
        mel_coeff = mfcc(s, fs, nfiltbank)
        codebooks_mfcc[i, :, :] = lbg(mel_coeff, nCentroid)
    codebooks = np.empty((2, nfiltbank, nCentroid))
    mel_coeff = np.empty((2, nfiltbank, 68))
    for i in range(2):
        fname = '/' + str(i+2) + '.wav'
        (fs,s) = read(directory_train + fname)
        mel_coeff[i,:,:] = mfcc(s, fs, nfiltbank)[:,0:68]
        codebooks_mfcc[i,:,:] = lbg(mel_coeff[i,:,:], nCentroid)
    codebooks = np.empty((2, nfiltbank, nCentroid))
    mel_coeff = np.empty((2, nfiltbank, 68))
    simpan("codebooks.xml", codebooks_mfcc)
    return codebooks_mfcc

# Tahapan Test
def minDistance(features, codebooks):
    speaker = 0
    distmin = np.inf
    #print (distmin)
    for k in range(np.shape(codebooks)[0]):
        D = EUDistance(features, codebooks[k, :, :])
        dist = np.sum(np.min(D, axis=1)) / (np.shape(D)[0])
        #print(dist)
        if dist < distmin: #distmin = tak hingga
            distmin = dist
            if dist <15:
                speaker = k+1
                #print(speaker)
                #print(dist)
            else:
                speaker = 0    
    return speaker

def simpan(filename, model):
    output = open(filename, 'wb')
    cPickle.dump(model, output)
    output.close()

    
def baca(filename):
    pkl_file = open(filename, 'rb')
    res = cPickle.load(pkl_file)
    pkl_file.close()
    return res 
          

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class awal(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(awal, self).__init__()
        self.setupUi(MainWindow)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(900, 585)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 150, 811, 51))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("High Tower Text"))
        font.setPointSize(30)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        font.setStrikeOut(False)
        self.label.setFont(font)
        self.label.setObjectName(_fromUtf8("label"))
        self.labelwaktu = QtGui.QLabel(self.centralwidget)
        self.labelwaktu.setGeometry(QtCore.QRect(740, 440, 111, 21))
        font = QtGui.QFont()
        font.setPointSize(17)
        font.setBold(False)
        font.setWeight(50)
        self.labelwaktu.setFont(font)
        self.labelwaktu.setText(_fromUtf8(""))
        self.labelwaktu.setAlignment(QtCore.Qt.AlignCenter)
        self.labelwaktu.setObjectName(_fromUtf8("labelwaktu"))
        self.labeltgl = QtGui.QLabel(self.centralwidget)
        self.labeltgl.setGeometry(QtCore.QRect(740, 470, 111, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.labeltgl.setFont(font)
        self.labeltgl.setText(_fromUtf8(""))
        self.labeltgl.setAlignment(QtCore.Qt.AlignCenter)
        self.labeltgl.setObjectName(_fromUtf8("labeltgl"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 900, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuHome = QtGui.QMenu(self.menubar)
        self.menuHome.setObjectName(_fromUtf8("menuHome"))
        self.menuDatabase = QtGui.QMenu(self.menubar)
        self.menuDatabase.setObjectName(_fromUtf8("menuDatabase"))
        self.menuRcognizer = QtGui.QMenu(self.menubar)
        self.menuRcognizer.setObjectName(_fromUtf8("menuRcognizer"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionDosen = QtGui.QAction(MainWindow)
        self.actionDosen.setObjectName(_fromUtf8("actionDosen"))
        self.actionPenelpon = QtGui.QAction(MainWindow)
        self.actionPenelpon.setObjectName(_fromUtf8("actionPenelpon"))
        self.actionHome = QtGui.QAction(MainWindow)
        self.actionHome.setObjectName(_fromUtf8("actionHome"))
        self.actionHotline = QtGui.QAction(MainWindow)
        self.actionHotline.setObjectName(_fromUtf8("actionHotline"))
        self.menuHome.addAction(self.actionHome)
        self.menuDatabase.addAction(self.actionDosen)
        self.menuDatabase.addAction(self.actionPenelpon)
        self.menuRcognizer.addAction(self.actionHotline)
        self.menubar.addAction(self.menuHome.menuAction())
        self.menubar.addAction(self.menuDatabase.menuAction())
        self.menubar.addAction(self.menuRcognizer.menuAction())
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        #self.timer = QtCore.QTimer(self)
        #self.timer.timeout.connect(self.waktu)
        #self.timer.start()

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.label.setText(_translate("MainWindow", "Aplikasi Pengenalan Penutur Hotline UPNVYK", None))
        self.menuHome.setTitle(_translate("MainWindow", "Home", None))
        self.menuDatabase.setTitle(_translate("MainWindow", "Database", None))
        self.menuRcognizer.setTitle(_translate("MainWindow", "Recognizer", None))
        self.actionDosen.setText(_translate("MainWindow", "Dosen", None))
        self.actionPenelpon.setText(_translate("MainWindow", "Penelpon", None))
        self.actionHome.setText(_translate("MainWindow", "Home", None))
        self.actionHotline.setText(_translate("MainWindow", "Hotline", None))
        
        self.actionDosen.triggered.connect(self.tampil_database_dosen)
        self.actionPenelpon.triggered.connect(self.tampil_database_penelpon)
        self.actionHotline.triggered.connect(self.pengenal_penutur)
        
    def tampil_database_dosen(self):
        #self.timer.stop()
        self.mainwindow24 = database_dosen(self)
        self.mainwindow24.setupUi(MainWindow)  
        
    def tampil_database_penelpon(self):
        self.timer.stop()
        self.mainwindow25 = database_penelpon(self)
        self.mainwindow25.setupUi(MainWindow)  

    def pengenal_penutur(self):
        #self.timer.stop()
        self.mainwindow26 = pengenal_penutur(self)
        self.mainwindow26.setupUi(MainWindow)
        
    def waktu(self):
        self.waktu = datetime.datetime.strftime(datetime.datetime.now(), "%H:%M:%S")
        self.isitanggal = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d")
        self.tanggal = datetime.datetime.strptime(self.isitanggal, "%Y-%m-%d")
        self.tgl = self.tanggal.strftime("%d %B %Y")
        self.labelwaktu.setText(self.waktu)
        self.labeltgl.setText(self.tgl)
        
        
class pengenal_penutur(QtGui.QMainWindow):
    def __init__(self, parent=None):
        pyqtgraph.setConfigOption('background', 'w') #before loading widget
        super(pengenal_penutur, self).__init__(parent)
        self.setupUi(MainWindow)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(900, 585)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        MainWindow.setFont(font)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.labelJudulHome = QtGui.QLabel(self.centralwidget)
        self.labelJudulHome.setGeometry(QtCore.QRect(350, 10, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.labelJudulHome.setFont(font)
        self.labelJudulHome.setObjectName(_fromUtf8("labelJudulHome"))
        self.labelFotoHome = QtGui.QLabel(self.centralwidget)
        self.labelFotoHome.setGeometry(QtCore.QRect(90, 90, 301, 251))
        self.labelFotoHome.setFrameShape(QtGui.QFrame.Box)
        self.labelFotoHome.setText(_fromUtf8(""))
        self.labelFotoHome.setObjectName(_fromUtf8("labelFotoHome"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(460, 90, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName(_fromUtf8("label"))
        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(460, 140, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.label_3 = QtGui.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(460, 240, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.label_4 = QtGui.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(460, 290, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.label_5 = QtGui.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(460, 340, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.labelNamaHome = QtGui.QLabel(self.centralwidget)
        self.labelNamaHome.setGeometry(QtCore.QRect(560, 90, 231, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.labelNamaHome.setFont(font)
        self.labelNamaHome.setText(_fromUtf8(""))
        self.labelNamaHome.setObjectName(_fromUtf8("labelNamaHome"))
        self.labelNIKHome = QtGui.QLabel(self.centralwidget)
        self.labelNIKHome.setGeometry(QtCore.QRect(560, 140, 231, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.labelNIKHome.setFont(font)
        self.labelNIKHome.setText(_fromUtf8(""))
        self.labelNIKHome.setObjectName(_fromUtf8("labelNIKHome"))
        self.labelJabatanHome = QtGui.QLabel(self.centralwidget)
        self.labelJabatanHome.setGeometry(QtCore.QRect(560, 240, 231, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.labelJabatanHome.setFont(font)
        self.labelJabatanHome.setText(_fromUtf8(""))
        self.labelJabatanHome.setObjectName(_fromUtf8("labelJabatanHome"))
        self.labelJurusanHome = QtGui.QLabel(self.centralwidget)
        self.labelJurusanHome.setGeometry(QtCore.QRect(560, 290, 231, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.labelJurusanHome.setFont(font)
        self.labelJurusanHome.setText(_fromUtf8(""))
        self.labelJurusanHome.setObjectName(_fromUtf8("labelJurusanHome"))
        self.labelFakultasHome = QtGui.QLabel(self.centralwidget)
        self.labelFakultasHome.setGeometry(QtCore.QRect(560, 340, 231, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.labelFakultasHome.setFont(font)
        self.labelFakultasHome.setText(_fromUtf8(""))
        self.labelFakultasHome.setObjectName(_fromUtf8("labelFakultasHome"))
        self.labelTeleponHome = QtGui.QLabel(self.centralwidget)
        self.labelTeleponHome.setGeometry(QtCore.QRect(560, 190, 231, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.labelTeleponHome.setFont(font)
        self.labelTeleponHome.setText(_fromUtf8(""))
        self.labelTeleponHome.setObjectName(_fromUtf8("labelTeleponHome"))
        self.label_8 = QtGui.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(460, 190, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.buttonmulai = QtGui.QPushButton(self.centralwidget)
        self.buttonmulai.setGeometry(QtCore.QRect(490, 440, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonmulai.setFont(font)
        self.buttonmulai.setObjectName(_fromUtf8("buttonmulai"))
        self.buttonberhenti = QtGui.QPushButton(self.centralwidget)
        self.buttonberhenti.setGeometry(QtCore.QRect(670, 440, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonberhenti.setFont(font)
        self.buttonberhenti.setObjectName(_fromUtf8("buttonberhenti"))
        self.labelWaktuHome = QtGui.QLabel(self.centralwidget)
        self.labelWaktuHome.setGeometry(QtCore.QRect(80, 420, 141, 21))
        font = QtGui.QFont()
        font.setPointSize(17)
        font.setBold(False)
        font.setWeight(50)
        self.labelWaktuHome.setFont(font)
        self.labelWaktuHome.setText(_fromUtf8(""))
        self.labelWaktuHome.setAlignment(QtCore.Qt.AlignCenter)
        self.labelWaktuHome.setObjectName(_fromUtf8("labelWaktuHome"))
        self.labelTanggalHome = QtGui.QLabel(self.centralwidget)
        self.labelTanggalHome.setGeometry(QtCore.QRect(80, 450, 141, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.labelTanggalHome.setFont(font)
        self.labelTanggalHome.setText(_fromUtf8(""))
        self.labelTanggalHome.setAlignment(QtCore.Qt.AlignCenter)
        self.labelTanggalHome.setObjectName(_fromUtf8("labelTanggalHome"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 900, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuHome = QtGui.QMenu(self.menubar)
        self.menuHome.setObjectName(_fromUtf8("menuHome"))
        self.menuDatabase = QtGui.QMenu(self.menubar)
        self.menuDatabase.setObjectName(_fromUtf8("menuDatabase"))
        self.menuRecognizer = QtGui.QMenu(self.menubar)
        self.menuRecognizer.setObjectName(_fromUtf8("menuRecognizer"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionDatabaseDosen = QtGui.QAction(MainWindow)
        self.actionDatabaseDosen.setObjectName(_fromUtf8("actionDatabaseDosen"))
        self.actionDatabasePenelpon = QtGui.QAction(MainWindow)
        self.actionDatabasePenelpon.setObjectName(_fromUtf8("actionDatabasePenelpon"))
        self.actionHome = QtGui.QAction(MainWindow)
        self.actionHome.setObjectName(_fromUtf8("actionHome"))
        self.actionHotline = QtGui.QAction(MainWindow)
        self.actionHotline.setObjectName(_fromUtf8("actionHotline"))
        self.menuHome.addAction(self.actionHome)
        self.menuDatabase.addAction(self.actionDatabaseDosen)
        self.menuDatabase.addAction(self.actionDatabasePenelpon)
        self.menuRecognizer.addAction(self.actionHotline)
        self.menubar.addAction(self.menuHome.menuAction())
        self.menubar.addAction(self.menuDatabase.menuAction())
        self.menubar.addAction(self.menuRecognizer.menuAction())
       
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        #self.timer = QtCore.QTimer(self)
        #self.timer.timeout.connect(self.waktu)
        #self.timer.start()
        nfiltbank = 12                
        training(nfiltbank)      
        

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.labelJudulHome.setText(_translate("MainWindow", "Pengenalan Penutur", None))
        self.label.setText(_translate("MainWindow", "Nama          :", None))
        self.label_2.setText(_translate("MainWindow", "NIK/NIP      :", None))
        self.label_3.setText(_translate("MainWindow", "Jabatan      :", None))
        self.label_4.setText(_translate("MainWindow", "Jurusan      :", None))
        self.label_5.setText(_translate("MainWindow", "Fakultas      :", None))
        self.label_8.setText(_translate("MainWindow", "No Telepon  :", None))
        self.buttonmulai.setText(_translate("MainWindow", "Mulai", None))
        self.buttonberhenti.setText(_translate("MainWindow", "Berhenti", None))
        self.menuHome.setTitle(_translate("MainWindow", "Home", None))
        self.menuDatabase.setTitle(_translate("MainWindow", "Database", None))
        self.menuRecognizer.setTitle(_translate("MainWindow", "Recognizer", None))
        self.actionDatabaseDosen.setText(_translate("MainWindow", "Dosen", None))
        self.actionDatabasePenelpon.setText(_translate("MainWindow", "Penelpon", None))
        self.actionHome.setText(_translate("MainWindow", "Home", None))
        self.actionHotline.setText(_translate("MainWindow", "Hotline", None))

        self.buttonmulai.clicked.connect(self.proses)
        self.buttonberhenti.clicked.connect(self.berhenti)
        self.actionDatabaseDosen.triggered.connect(self.tampil_database_dosen)
        self.actionDatabasePenelpon.triggered.connect(self.tampil_database_penelpon)
        self.actionHome.triggered.connect(self.awal)
        self.actionHotline.triggered.connect(self.pengenal_penutur)

    def awal(self):
        self.mainwindow27 = awal(self)
        self.mainwindow27.setupUi(MainWindow)

    def tampil_database_dosen(self):
        self.mainwindow2 = database_dosen(self)
        self.mainwindow2.setupUi(MainWindow)

    def tampil_database_penelpon(self):
        self.mainwindow3 = database_penelpon(self)
        self.mainwindow3.setupUi(MainWindow)

    def pengenal_penutur(self):
        self.mainwindow28 = pengenal_penutur(self)
        self.mainwindow28.setupUi(MainWindow)

    def mulai(self):
        #self.timer = QtCore.QTimer(self)
        #self.timer.timeout.connect(self.proses)
        #self.timer.setInterval(500)
        #self.timer.start()
        for i in range(5):
            self.proses() 
        
    def pengenal(self, codebooks_mfcc):
        codebooks_mfcc = codebooks_mfcc 
      
    def tampil_penelpon(self, id_penelpon, id_dosen):
        kursor = db.cursor()
        kursor.execute("select * from masuk where id_penelpon = %s"%(id_penelpon))
        masuk = kursor.fetchall()
        #print(masuk)
        x = len(masuk)
        if x == 0:
            self.labelNamaHome.setText("")
            self.labelNIKHome.setText("")
            self.labelTeleponHome.setText("")
            self.labelJabatanHome.setText("")
            self.labelJurusanHome.setText("")
            self.labelFakultasHome.setText("")
        if x >= 1:    
            if masuk[0][2] == 'Dikenal':
                kursor = db.cursor()
                kursor.execute("select p.nama, p.nik, p.no_telp, p.jabatan, p.jurusan, p.fakultas, f.path from pegawai p, foto f where p.id_dosen ='%s'"%id_dosen+"and p.id_dosen = f.id_dosen")
                orang = kursor.fetchall()
                for orang in orang:
                    self.labelNamaHome.setText(orang[0])
                    self.labelNIKHome.setText(orang[1])
                    self.labelTeleponHome.setText(orang[2])
                    self.labelJabatanHome.setText(orang[3])
                    self.labelJurusanHome.setText(orang[4])
                    self.labelFakultasHome.setText((orang[5]))
                    pixmap = QtGui.QPixmap(orang[6])
                    pixmap = pixmap.scaled(301, 251, QtCore.Qt.KeepAspectRatio)
                    self.labelFotoHome.setPixmap(pixmap)
                    self.labelFotoHome.show()
                    #self.labelWaktuHome.setText(datetime.datetime.now().time())
            elif masuk[0][2] == 'Tidak Dikenal':
                self.labelNamaHome.setText("Tidak Dikenal")
                self.labelNIKHome.setText("Tidak Dikenal")
                self.labelTeleponHome.setText("Tidak Dikenal")
                self.labelJabatanHome.setText("Tidak Dikenal")
                self.labelJurusanHome.setText("Tidak Dikenal")
                self.labelFakultasHome.setText("Tidak Dikenal")
                pixmap = QtGui.QPixmap('C:\\Users\\asus\\Pictures\\Blank.png')
                pixmap = pixmap.scaled(301, 251, QtCore.Qt.KeepAspectRatio)
                self.labelFotoHome.setPixmap(pixmap)
                self.labelFotoHome.show()
                #self.labelWaktuHome.setText(datetime.datetime.now().time())
                #self.labelFotoHome.setPixmap(None)
   
    def proses(self):
        nfiltbank = 12
        #print(codebooks_mfcc)
        codebooks_mfcc = baca("codebooks.xml")
        #directory = ('C:/Users/asus/workspace/bismillah/test/')
        nSpeaker = sum(1 for f in os.listdir('C:\\Users\\asus\\workspace\\bismillah\\pengujian') if os.path.isfile(os.path.join('C:\\Users\\asus\\workspace\\bismillah\\pengujian', f)) and f[0] != '.')
        for i in range(nSpeaker):
            kursor = db.cursor()
            kursor.execute("SELECT * From pengujian")
            objek = kursor.fetchall()
            id = len(objek)
            if id >= 1:
                directory = os.getcwd() + '/pengujian'
                nCorrect_MFCC = 0
                fname = '/' + str(i+1) + '.wav'
                #print(fname)
                (fs, s) = read(directory + fname)
                mel_coefs = mfcc(s, fs, nfiltbank)
                print (fs)
                sp_mfcc = minDistance(mel_coefs, codebooks_mfcc)
                kursor.execute("select id_dosen from suara where id_file = %s"%(sp_mfcc))
                dosen = kursor.fetchall()
                jml = len(dosen)
                if jml >= 1:
                    id_dosen = dosen[0]
                    kursor = db.cursor()
                    kursor.execute("select * from pegawai where id_dosen = %s" % (id_dosen))
                    orang = kursor.fetchall()
                    for orang in orang:
                        kursor.execute("INSERT INTO pengujian (id_penelepon,no_telp,status,id_dosen,waktu,tanggal) VALUES (%s,%s,%s,%s,%s,%s)",
                            [id+1, '08xxxxx', 'Dikenal', id_dosen, datetime.datetime.now().time(), datetime.datetime.now().date()])
                        db.commit()
                        self.tampil_penelpon((id+1), id_dosen)
                elif jml == 0:
                    nilai = "Tidak Dikenal"
                    kursor.execute("INSERT INTO pengujian (id_penelepon,no_telp,status,id_dosen,waktu,tanggal) VALUES (%s,%s,%s,%s,%s,%s)",
                        [id+1, '08xxxxx', nilai, 0, datetime.datetime.now().time(), datetime.datetime.now().date()])
                    db.commit()
                    self.tampil_penelpon((id+1), 0)
            else:
                nfiltbank = 12
                codebooks_mfcc = training(nfiltbank)
                directory = os.getcwd() + '/pengujian';
                nCorrect_MFCC = 0
                fname = '/1.wav'
                #print(fname)
                (fs, s) = read(directory + fname)
                mel_coefs = mfcc(s, fs, nfiltbank)
                sp_mfcc = minDistance(mel_coefs, codebooks_mfcc)
                kursor.execute("select id_dosen from suara where id_file = %s"%(sp_mfcc))
                dosen = kursor.fetchall()
                jml = len(dosen)
                #print (sp_mfcc)
                if  jml >= 1:
                    id_dosen = dosen[0]
                    kursor = db.cursor()
                    kursor.execute("select * from pegawai where id_dosen = %s" % (id_dosen))
                    orang = kursor.fetchall()
                    for orang in orang:
                        kursor.execute("INSERT INTO pengujian (id_penelepon,no_telp,status,id_dosen,waktu,tanggal) VALUES (%s,%s,%s,%s,%s,%s)",
                            [str(id+1), '08xxxxx', 'Dikenal', id_dosen, datetime.datetime.now().time(), datetime.datetime.now().date()])
                        db.commit()
                        self.tampil_penelpon((id+1), id_dosen)
                elif jml == 0:
                    nilai = "Tidak Dikenal"
                    kursor.execute("INSERT INTO pengujian (id_penelepon,no_telp,status,id_dosen,waktu,tanggal) VALUES (%s,%s,%s,%s,%s,%s)",
                        [str(id+1), '08xxxxx', nilai, nilai, datetime.datetime.now().time(), datetime.datetime.now().date()])
                    db.commit()
                    self.tampil_penelpon((id+1), 0)

    def berhenti(self):
        self.timer.stop()
        
    def waktu(self):
        self.waktu = datetime.datetime.strftime(datetime.datetime.now(), "%H:%M:%S")
        self.isitanggal = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d")
        self.tanggal = datetime.datetime.strptime(self.isitanggal, "%Y-%m-%d")
        self.tgl = self.tanggal.strftime("%d %B %Y")
        self.labelWaktuHome.setText(self.waktu)
        self.labelTanggalHome.setText(self.tgl)
                

class database_dosen(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(database_dosen, self).__init__()
        self.setupUi(MainWindow)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(900, 585)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.labelJudulDatabaseDosen = QtGui.QLabel(self.centralwidget)
        self.labelJudulDatabaseDosen.setGeometry(QtCore.QRect(360, 10, 151, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.labelJudulDatabaseDosen.setFont(font)
        self.labelJudulDatabaseDosen.setObjectName(_fromUtf8("labelJudulDatabaseDosen"))
        self.tableDatabaseDosen = QtGui.QTableWidget(self.centralwidget)
        self.tableDatabaseDosen.setGeometry(QtCore.QRect(20, 80, 711, 192))
        self.tableDatabaseDosen.setObjectName(_fromUtf8("tableDatabaseDosen"))
        self.tableDatabaseDosen.setColumnCount(7)
        self.tableDatabaseDosen.setRowCount(0)
        item = QtGui.QTableWidgetItem()
        self.tableDatabaseDosen.setHorizontalHeaderItem(0, item)
        item = QtGui.QTableWidgetItem()
        self.tableDatabaseDosen.setHorizontalHeaderItem(1, item)
        item = QtGui.QTableWidgetItem()
        self.tableDatabaseDosen.setHorizontalHeaderItem(2, item)
        item = QtGui.QTableWidgetItem()
        self.tableDatabaseDosen.setHorizontalHeaderItem(3, item)
        item = QtGui.QTableWidgetItem()
        self.tableDatabaseDosen.setHorizontalHeaderItem(4, item)
        item = QtGui.QTableWidgetItem()
        self.tableDatabaseDosen.setHorizontalHeaderItem(5, item)
        item = QtGui.QTableWidgetItem()
        self.tableDatabaseDosen.setHorizontalHeaderItem(6, item)
        self.buttonUbahDatabaseDosen = QtGui.QPushButton(self.centralwidget)
        self.buttonUbahDatabaseDosen.setGeometry(QtCore.QRect(750, 190, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonUbahDatabaseDosen.setFont(font)
        self.buttonUbahDatabaseDosen.setObjectName(_fromUtf8("buttonUbahDatabaseDosen"))
        self.buttonTambahDatabaseDosen = QtGui.QPushButton(self.centralwidget)
        self.buttonTambahDatabaseDosen.setGeometry(QtCore.QRect(750, 120, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonTambahDatabaseDosen.setFont(font)
        self.buttonTambahDatabaseDosen.setObjectName(_fromUtf8("buttonTambahDatabaseDosen"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 900, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuHome = QtGui.QMenu(self.menubar)
        self.menuHome.setObjectName(_fromUtf8("menuHome"))
        self.menuDatabase = QtGui.QMenu(self.menubar)
        self.menuDatabase.setObjectName(_fromUtf8("menuDatabase"))
        self.menuRecognizer = QtGui.QMenu(self.menubar)
        self.menuRecognizer.setObjectName(_fromUtf8("menuRecognizer"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionDosen = QtGui.QAction(MainWindow)
        self.actionDosen.setObjectName(_fromUtf8("actionDosen"))
        self.actionPenelpon = QtGui.QAction(MainWindow)
        self.actionPenelpon.setObjectName(_fromUtf8("actionPenelpon"))
        self.actionHome = QtGui.QAction(MainWindow)
        self.actionHome.setObjectName(_fromUtf8("actionHome"))
        self.actionHotline = QtGui.QAction(MainWindow)
        self.actionHotline.setObjectName(_fromUtf8("actionHotline"))
        self.menuHome.addAction(self.actionHome)
        self.menuDatabase.addAction(self.actionDosen)
        self.menuDatabase.addAction(self.actionPenelpon)
        self.menuRecognizer.addAction(self.actionHotline)
        self.menubar.addAction(self.menuHome.menuAction())
        self.menubar.addAction(self.menuDatabase.menuAction())
        self.menubar.addAction(self.menuRecognizer.menuAction())

        self.retranslateUi(MainWindow)
        self.data_dosen(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        


    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.labelJudulDatabaseDosen.setText(_translate("MainWindow", "DATA PEGAWAI", None))
        item = self.tableDatabaseDosen.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "ID Pegawai", None))
        item = self.tableDatabaseDosen.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Nama", None))
        item = self.tableDatabaseDosen.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "NIK", None))
        item = self.tableDatabaseDosen.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Jabatan", None))
        item = self.tableDatabaseDosen.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow", "Jurusan", None))
        item = self.tableDatabaseDosen.horizontalHeaderItem(5)
        item.setText(_translate("MainWindow", "Fakultas", None))
        item = self.tableDatabaseDosen.horizontalHeaderItem(6)
        item.setText(_translate("MainWindow", "No Telepon", None))
        self.buttonUbahDatabaseDosen.setText(_translate("MainWindow", "UBAH", None))
        self.buttonTambahDatabaseDosen.setText(_translate("MainWindow", "TAMBAH", None))
        self.menuHome.setTitle(_translate("MainWindow", "Home", None))
        self.menuDatabase.setTitle(_translate("MainWindow", "Database", None))
        self.menuRecognizer.setTitle(_translate("MainWindow", "Recognizer", None))
        self.actionDosen.setText(_translate("MainWindow", "Dosen", None))
        self.actionPenelpon.setText(_translate("MainWindow", "Penelpon", None))
        self.actionHome.setText(_translate("MainWindow", "Home", None))
        self.actionHotline.setText(_translate("MainWindow", "Hotline", None))

        self.buttonTambahDatabaseDosen.clicked.connect(self.tampil_tambah)
        self.actionHotline.triggered.connect(self.pengenal_penutur)
        self.actionPenelpon.triggered.connect(self.tampil_database_penelpon)
        self.buttonUbahDatabaseDosen.clicked.connect(self.ubah)
        self.actionHome.triggered.connect(self.awal)
 
    def awal(self):
        self.mainwindow29 = awal(self)
        self.mainwindow29.setupUi(MainWindow)       
    
    def tampil_tambah(self):
        self.mainwindow4 = tambah_database_dosen(self)
        self.mainwindow4.setupUi(MainWindow)
            
    def pengenal_penutur(self):
        self.mainwindow5 = pengenal_penutur(self)
        self.mainwindow5.setupUi(MainWindow)
        
    def tampil_database_penelpon(self):
        self.mainwindow6 = database_penelpon(self)
        self.mainwindow6.setupUi(MainWindow)

    def data_dosen(self, MainWindow):
        kursor = db.cursor()
        kursor.execute("SELECT id_dosen, nama, nik, jabatan, jurusan, fakultas, no_telp FROM pegawai")
        data = kursor.fetchall()
        rowCount = len(data)
        colCount = 7
        self.tableDatabaseDosen.setRowCount(rowCount)
        self.tableDatabaseDosen.setColumnCount(colCount)
        for s in range(colCount):
            for i, row in enumerate(data):
                for j, col in enumerate(row):
                    item = QtGui.QTableWidgetItem('%s' % (col))
                    self.tableDatabaseDosen.setItem(i, j, item)
        
    def hapus(self):
        file = self.tableDatabaseDosen.selectedItems()
        nameItem = file[0]
        data = nameItem.data(0).toString()
        ambil = self.tableDatabaseDosen.currentRow()
        self.tableDatabaseDosen.removeRow(ambil) 
        try:
            cursor = db.cursor()
            cursor.execute("DELETE FROM pegawai WHERE no_dosen = %s" % data)
            db.commit()
            a = 1
            if a == 1:
                cursor.execute("DELETE FROM suara WHERE id_dosen = %s" % data)
                db.commit()
                b = 1
                if b == 1:
                    cursor.execute("DELETE FROM foto WHERE id_dosen = %s" % data)
                    db.commit()
                    a = 0
                    b = 0
        except:
            db.rollback() 
    
    def ubah(self):
        file = self.tableDatabaseDosen.selectedItems()
        nameItem = file[0]
        pilih = nameItem.data(0)
        self.mainwindow6 = ubah_database_dosen(self)
        self.mainwindow6.tampildosen(pilih)


class tambah_database_dosen(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(tambah_database_dosen, self).__init__()
        self.setupUi(MainWindow)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(900, 585)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(470, 90, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName(_fromUtf8("label"))
        self.labelFotoTambahDatabaseDosen = QtGui.QLabel(self.centralwidget)
        self.labelFotoTambahDatabaseDosen.setGeometry(QtCore.QRect(60, 80, 371, 291))
        self.labelFotoTambahDatabaseDosen.setFrameShape(QtGui.QFrame.Box)
        self.labelFotoTambahDatabaseDosen.setText(_fromUtf8(""))
        self.labelFotoTambahDatabaseDosen.setObjectName(_fromUtf8("labelFotoTambahDatabaseDosen"))
        self.label_3 = QtGui.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(470, 190, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(470, 140, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.label_4 = QtGui.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(470, 240, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.label_5 = QtGui.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(470, 290, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.labelJudulHome = QtGui.QLabel(self.centralwidget)
        self.labelJudulHome.setGeometry(QtCore.QRect(360, 10, 211, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.labelJudulHome.setFont(font)
        self.labelJudulHome.setObjectName(_fromUtf8("labelJudulHome"))
        self.buttonCariTambahDatabaseDosen = QtGui.QPushButton(self.centralwidget)
        self.buttonCariTambahDatabaseDosen.setGeometry(QtCore.QRect(60, 390, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonCariTambahDatabaseDosen.setFont(font)
        self.buttonCariTambahDatabaseDosen.setObjectName(_fromUtf8("buttonCariTambahDatabaseDosen"))
        self.buttonRecordTambahDatabaseDosen = QtGui.QPushButton(self.centralwidget)
        self.buttonRecordTambahDatabaseDosen.setGeometry(QtCore.QRect(60, 450, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonRecordTambahDatabaseDosen.setFont(font)
        self.buttonRecordTambahDatabaseDosen.setObjectName(_fromUtf8("buttonRecordTambahDatabaseDosen"))
        self.lineCariTambahDatabaseDosen = QtGui.QLineEdit(self.centralwidget)
        self.lineCariTambahDatabaseDosen.setGeometry(QtCore.QRect(170, 400, 271, 20))
        self.lineCariTambahDatabaseDosen.setObjectName(_fromUtf8("lineCariTambahDatabaseDosen"))
        self.lineNamaTambahDatabaseDosen = QtGui.QLineEdit(self.centralwidget)
        self.lineNamaTambahDatabaseDosen.setGeometry(QtCore.QRect(570, 90, 271, 20))
        self.lineNamaTambahDatabaseDosen.setObjectName(_fromUtf8("lineNamaTambahDatabaseDosen"))
        self.lineFakultasTambahDatabaseDosen = QtGui.QLineEdit(self.centralwidget)
        self.lineFakultasTambahDatabaseDosen.setGeometry(QtCore.QRect(570, 290, 271, 20))
        self.lineFakultasTambahDatabaseDosen.setObjectName(_fromUtf8("lineFakultasTambahDatabaseDosen"))
        self.lineJurusanTambahDatabaseDosen = QtGui.QLineEdit(self.centralwidget)
        self.lineJurusanTambahDatabaseDosen.setGeometry(QtCore.QRect(570, 240, 271, 20))
        self.lineJurusanTambahDatabaseDosen.setObjectName(_fromUtf8("lineJurusanTambahDatabaseDosen"))
        self.lineJabatanTambahDatabaseDosen = QtGui.QLineEdit(self.centralwidget)
        self.lineJabatanTambahDatabaseDosen.setGeometry(QtCore.QRect(570, 190, 271, 20))
        self.lineJabatanTambahDatabaseDosen.setObjectName(_fromUtf8("lineJabatanTambahDatabaseDosen"))
        self.lineNIKTambahDatabaseDosen = QtGui.QLineEdit(self.centralwidget)
        self.lineNIKTambahDatabaseDosen.setGeometry(QtCore.QRect(570, 140, 271, 20))
        self.lineNIKTambahDatabaseDosen.setObjectName(_fromUtf8("lineNIKTambahDatabaseDosen"))
        self.lineRecordTambahDatabaseDosen = QtGui.QLineEdit(self.centralwidget)
        self.lineRecordTambahDatabaseDosen.setGeometry(QtCore.QRect(170, 460, 271, 20))
        self.lineRecordTambahDatabaseDosen.setObjectName(_fromUtf8("lineRecordTambahDatabaseDosen"))
        self.lineNomorTeleponTambahDatabaseDosen_2 = QtGui.QLineEdit(self.centralwidget)
        self.lineNomorTeleponTambahDatabaseDosen_2.setGeometry(QtCore.QRect(570, 340, 271, 20))
        self.lineNomorTeleponTambahDatabaseDosen_2.setObjectName(_fromUtf8("lineNomorTeleponTambahDatabaseDosen_2"))
        self.label_6 = QtGui.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(450, 340, 111, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.buttonSimpanTambahDatabaseDosen = QtGui.QPushButton(self.centralwidget)
        self.buttonSimpanTambahDatabaseDosen.setGeometry(QtCore.QRect(620, 430, 111, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonSimpanTambahDatabaseDosen.setFont(font)
        self.buttonSimpanTambahDatabaseDosen.setObjectName(_fromUtf8("buttonSimpanTambahDatabaseDosen"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 900, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuHome = QtGui.QMenu(self.menubar)
        self.menuHome.setObjectName(_fromUtf8("menuHome"))
        self.menuDatabase = QtGui.QMenu(self.menubar)
        self.menuDatabase.setObjectName(_fromUtf8("menuDatabase"))
        self.menuRecognizer = QtGui.QMenu(self.menubar)
        self.menuRecognizer.setObjectName(_fromUtf8("menuRecognizer"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionDosen = QtGui.QAction(MainWindow)
        self.actionDosen.setObjectName(_fromUtf8("actionDosen"))
        self.actionPenelpon = QtGui.QAction(MainWindow)
        self.actionPenelpon.setObjectName(_fromUtf8("actionPenelpon"))
        self.actionHome = QtGui.QAction(MainWindow)
        self.actionHome.setObjectName(_fromUtf8("actionHome"))
        self.actionHotline = QtGui.QAction(MainWindow)
        self.actionHotline.setObjectName(_fromUtf8("actionHotline"))
        self.menuHome.addAction(self.actionHome)
        self.menuDatabase.addAction(self.actionDosen)
        self.menuDatabase.addAction(self.actionPenelpon)
        self.menuRecognizer.addAction(self.actionHotline)
        self.menubar.addAction(self.menuHome.menuAction())
        self.menubar.addAction(self.menuDatabase.menuAction())
        self.menubar.addAction(self.menuRecognizer.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.label.setText(_translate("MainWindow", "Nama          :", None))
        self.label_3.setText(_translate("MainWindow", "Jabatan      :", None))
        self.label_2.setText(_translate("MainWindow", "NIK/NIP       :", None))
        self.label_4.setText(_translate("MainWindow", "Jurusan      :", None))
        self.label_5.setText(_translate("MainWindow", "Fakultas      :", None))
        self.labelJudulHome.setText(_translate("MainWindow", "TAMBAH DATA DOSEN", None))
        self.buttonCariTambahDatabaseDosen.setText(_translate("MainWindow", "Cari", None))
        self.buttonRecordTambahDatabaseDosen.setText(_translate("MainWindow", "Record", None))
        self.label_6.setText(_translate("MainWindow", "Nomor Telepon :", None))
        self.buttonSimpanTambahDatabaseDosen.setText(_translate("MainWindow", "Simpan", None))
        self.menuHome.setTitle(_translate("MainWindow", "Home", None))
        self.menuDatabase.setTitle(_translate("MainWindow", "Database", None))
        self.menuRecognizer.setTitle(_translate("MainWindow", "Recognizer", None))
        self.actionDosen.setText(_translate("MainWindow", "Dosen", None))
        self.actionPenelpon.setText(_translate("MainWindow", "Penelpon", None))
        self.actionHome.setText(_translate("MainWindow", "Home", None))
        self.actionHotline.setText(_translate("MainWindow", "Hotline", None))

        self.buttonCariTambahDatabaseDosen.clicked.connect(self.cari)
        self.buttonRecordTambahDatabaseDosen.clicked.connect(self.rekam)
        self.buttonSimpanTambahDatabaseDosen.clicked.connect(self.simpan)
        self.actionDosen.triggered.connect(self.tampil_database_dosen)
        self.actionPenelpon.triggered.connect(self.tampil_database_penelpon)
        self.actionHome.triggered.connect(self.awal)
        self.actionHotline.triggered.connect(self.pengenal_penutur)

    def awal(self):
        self.mainwindow30 = awal(self)
        self.mainwindow30.setupUi(MainWindow)
    
    def tampil_database_dosen(self):
        self.mainwindow7 = database_dosen(self)
        self.mainwindow7.setupUi(MainWindow)  
        
    def tampil_database_penelpon(self):
        self.mainwindow8 = database_penelpon(self)
        self.mainwindow8.setupUi(MainWindow)  
        
    def pengenal_penutur(self):
        self.mainwindow9 = pengenal_penutur(self)
        self.mainwindow9.setupUi(MainWindow)
    
    def ambil_suara(self):
        nilai = kursor.execute("Select * from suara")
        ambil = kursor.fetchall()
        id = len(ambil)
        return (id)
    
    def ambil_foto(self):
        nilai = kursor.execute("Select * from foto")
        ambil = kursor.fetchall()
        id = len(ambil)
        return (id)

    def rekam(self):
        directory = ('C:/Users/asus/workspace/bismillah/train/')
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        CHUNK = 1024
        RECORD_SECONDS = 5
        WAVE_OUTPUT_FILENAME = directory + str(self.ambil_suara()+1) + '.wav'
        audio = pyaudio.PyAudio()
        # start Recording
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)
        print('recording...')
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print('finished recording')
        # stop Recording
        stream.stop_stream()
        stream.close()
        audio.terminate()
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        self.lineRecordTambahDatabaseDosen.setText(WAVE_OUTPUT_FILENAME)
        print (id())
        if id() == 0:
            kursor.execute ("INSERT INTO pegawai (id_dosen) VALUES (%s)",
                            [int(id()+1)])
            db.commit() 
        else:
            kursor.execute("Select * from pegawai where id_dosen = (select max(id_dosen) from pegawai)")
            pegawai = kursor.fetchall()
            for pegawai in pegawai:
                print(pegawai[1])
                if pegawai[1] != None:
                    kursor.execute ("INSERT INTO pegawai (id_dosen) VALUES (%s)",
                                    [int(id()+1)])
                    db.commit() 
                    if self.lineRecordTambahDatabaseDosen.text() != '':
                        kursor.execute ("INSERT INTO suara (id_file,id_dosen,path) VALUES (%s,%s,%s)",
                                            [int(self.ambil_suara()+1), int(id()), self.lineRecordTambahDatabaseDosen.text()])
                        alertPopup = QtGui.QMessageBox()
                        alertPopup.setText("Data suara berhasil disimpan")
                        alertPopup.setIcon(alertPopup.Information)
                        alertPopup.exec_()
                        db.commit()  
                else:
                    if self.lineRecordTambahDatabaseDosen.text() != '':
                        kursor.execute ("INSERT INTO suara (id_file,id_dosen,path) VALUES (%s,%s,%s)",
                                            [int(self.ambil_suara()+1), int(id()), self.lineRecordTambahDatabaseDosen.text()])
                        alertPopup = QtGui.QMessageBox()
                        alertPopup.setText("Data suara berhasil disimpan")
                        alertPopup.setIcon(alertPopup.Information)
                        alertPopup.exec_()
                        db.commit()   
            
    def tambah(self):
        self.lineRecordTambahDatabaseDosen.setText("")
    
    def cari(self):
        file = QtGui.QFileDialog.getOpenFileName(self, "Pilih Gambar")
        self.lineCariTambahDatabaseDosen.setText(file)
        pixmap = QtGui.QPixmap(file)
        pixmap = pixmap.scaled(301, 251, QtCore.Qt.KeepAspectRatio)
        self.labelFotoTambahDatabaseDosen.setPixmap(pixmap)
        self.labelFotoTambahDatabaseDosen.show()
        return file

    def simpan(self):
        if self.lineNamaTambahDatabaseDosen.text() != '' and self.lineNIKTambahDatabaseDosen.text() != '' and self.lineJabatanTambahDatabaseDosen.text() != '' and self.lineJurusanTambahDatabaseDosen.text() != '' and  self.lineFakultasTambahDatabaseDosen.text() != '':
            kursor.execute ("UPDATE pegawai SET nama='%s'" % self.lineNamaTambahDatabaseDosen.text() + ",nik='%s'" % self.lineNIKTambahDatabaseDosen.text() + ",jabatan='%s'" % self.lineJabatanTambahDatabaseDosen.text() + ",jurusan='%s'" % self.lineJurusanTambahDatabaseDosen.text() + ",fakultas='%s'" % self.lineFakultasTambahDatabaseDosen.text() + ",no_telp='%s'" % self.lineNomorTeleponTambahDatabaseDosen_2.text() + " WHERE id_dosen=%s" % id())
            db.commit()
            alertPopup = QtGui.QMessageBox()
            alertPopup.setText("Data berhasil disimpan")
            alertPopup.setIcon(alertPopup.Information)
            alertPopup.exec_()
            if self.lineCariTambahDatabaseDosen.text() != '':
                kursor.execute ("INSERT INTO foto (id_foto,id_dosen,path) VALUES (%s,%s,%s)",
                                [int(self.ambil_foto()+1), int(id()), self.lineCariTambahDatabaseDosen.text()])
                db.commit()  
            self.labelFotoTambahDatabaseDosen.setText("")
            self.lineNamaTambahDatabaseDosen.setText("")
            self.lineNIKTambahDatabaseDosen.setText("")
            self.lineJabatanTambahDatabaseDosen.setText("")
            self.lineJurusanTambahDatabaseDosen.setText("")
            self.lineFakultasTambahDatabaseDosen.setText("")
            self.lineNomorTeleponTambahDatabaseDosen_2.setText("")
            self.lineCariTambahDatabaseDosen.setText("")
            self.lineRecordTambahDatabaseDosen.setText("")
        else:
            alertPopup = QtGui.QMessageBox()
            alertPopup.setText("Data belum terisi lengkap")
            alertPopup.setIcon(alertPopup.Critical)
            alertPopup.exec_()



class ubah_database_dosen(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(ubah_database_dosen, self).__init__()
        self.setupUi(MainWindow)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(900, 585)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.labelJudulHome = QtGui.QLabel(self.centralwidget)
        self.labelJudulHome.setGeometry(QtCore.QRect(380, 20, 181, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.labelJudulHome.setFont(font)
        self.labelJudulHome.setObjectName(_fromUtf8("labelJudulHome"))
        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(460, 140, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.lineJurusanUbahDatabaseDosen = QtGui.QLineEdit(self.centralwidget)
        self.lineJurusanUbahDatabaseDosen.setGeometry(QtCore.QRect(560, 240, 271, 20))
        self.lineJurusanUbahDatabaseDosen.setObjectName(_fromUtf8("lineJurusanUbahDatabaseDosen"))
        self.label_5 = QtGui.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(460, 290, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.lineNamaUbahDatabaseDosen = QtGui.QLineEdit(self.centralwidget)
        self.lineNamaUbahDatabaseDosen.setGeometry(QtCore.QRect(560, 90, 271, 20))
        self.lineNamaUbahDatabaseDosen.setObjectName(_fromUtf8("lineNamaUbahDatabaseDosen"))
        self.lineFakultasUbahDatabaseDosen = QtGui.QLineEdit(self.centralwidget)
        self.lineFakultasUbahDatabaseDosen.setGeometry(QtCore.QRect(560, 290, 271, 20))
        self.lineFakultasUbahDatabaseDosen.setObjectName(_fromUtf8("lineFakultasUbahDatabaseDosen"))
        self.label_3 = QtGui.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(460, 190, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.lineJabatanUbahDatabaseDosen = QtGui.QLineEdit(self.centralwidget)
        self.lineJabatanUbahDatabaseDosen.setGeometry(QtCore.QRect(560, 190, 271, 20))
        self.lineJabatanUbahDatabaseDosen.setObjectName(_fromUtf8("lineJabatanUbahDatabaseDosen"))
        self.lineNomorTeleponUbahDatabaseDosen_2 = QtGui.QLineEdit(self.centralwidget)
        self.lineNomorTeleponUbahDatabaseDosen_2.setGeometry(QtCore.QRect(560, 340, 271, 20))
        self.lineNomorTeleponUbahDatabaseDosen_2.setObjectName(_fromUtf8("lineNomorTeleponUbahDatabaseDosen_2"))
        self.lineNIKUbahDatabaseDosen = QtGui.QLineEdit(self.centralwidget)
        self.lineNIKUbahDatabaseDosen.setGeometry(QtCore.QRect(560, 140, 271, 20))
        self.lineNIKUbahDatabaseDosen.setObjectName(_fromUtf8("lineNIKUbahDatabaseDosen"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(460, 90, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName(_fromUtf8("label"))
        self.label_6 = QtGui.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(440, 340, 111, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.label_4 = QtGui.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(460, 240, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.buttonCariUbahDatabaseDosen = QtGui.QPushButton(self.centralwidget)
        self.buttonCariUbahDatabaseDosen.setGeometry(QtCore.QRect(40, 400, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonCariUbahDatabaseDosen.setFont(font)
        self.buttonCariUbahDatabaseDosen.setObjectName(_fromUtf8("buttonCariUbahDatabaseDosen"))
        self.labelFotoTambahDatabaseDosen = QtGui.QLabel(self.centralwidget)
        self.labelFotoTambahDatabaseDosen.setGeometry(QtCore.QRect(40, 90, 381, 291))
        self.labelFotoTambahDatabaseDosen.setFrameShape(QtGui.QFrame.Box)
        self.labelFotoTambahDatabaseDosen.setText(_fromUtf8(""))
        self.labelFotoTambahDatabaseDosen.setObjectName(_fromUtf8("labelFotoTambahDatabaseDosen"))
        self.lineCariUbahDatabaseDosen = QtGui.QLineEdit(self.centralwidget)
        self.lineCariUbahDatabaseDosen.setGeometry(QtCore.QRect(150, 410, 271, 20))
        self.lineCariUbahDatabaseDosen.setObjectName(_fromUtf8("lineCariUbahDatabaseDosen"))
        self.buttonDataSuaraUbahDatabaseDosen_2 = QtGui.QPushButton(self.centralwidget)
        self.buttonDataSuaraUbahDatabaseDosen_2.setGeometry(QtCore.QRect(160, 450, 131, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonDataSuaraUbahDatabaseDosen_2.setFont(font)
        self.buttonDataSuaraUbahDatabaseDosen_2.setObjectName(_fromUtf8("buttonDataSuaraUbahDatabaseDosen_2"))
        self.buttonSimpanUbahDatabaseDosen_3 = QtGui.QPushButton(self.centralwidget)
        self.buttonSimpanUbahDatabaseDosen_3.setGeometry(QtCore.QRect(580, 400, 131, 61))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonSimpanUbahDatabaseDosen_3.setFont(font)
        self.buttonSimpanUbahDatabaseDosen_3.setObjectName(_fromUtf8("buttonSimpanUbahDatabaseDosen_3"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 900, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuHome = QtGui.QMenu(self.menubar)
        self.menuHome.setObjectName(_fromUtf8("menuHome"))
        self.menuDatabase = QtGui.QMenu(self.menubar)
        self.menuDatabase.setObjectName(_fromUtf8("menuDatabase"))
        self.menuRecognizer = QtGui.QMenu(self.menubar)
        self.menuRecognizer.setObjectName(_fromUtf8("menuRecognizer"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionDosen = QtGui.QAction(MainWindow)
        self.actionDosen.setObjectName(_fromUtf8("actionDosen"))
        self.actionPenelpon = QtGui.QAction(MainWindow)
        self.actionPenelpon.setObjectName(_fromUtf8("actionPenelpon"))
        self.actionHome = QtGui.QAction(MainWindow)
        self.actionHome.setObjectName(_fromUtf8("actionHome"))
        self.actionHotline = QtGui.QAction(MainWindow)
        self.actionHotline.setObjectName(_fromUtf8("actionHotline"))
        self.menuHome.addAction(self.actionHome)
        self.menuDatabase.addAction(self.actionDosen)
        self.menuDatabase.addAction(self.actionPenelpon)
        self.menuRecognizer.addAction(self.actionHotline)
        self.menubar.addAction(self.menuHome.menuAction())
        self.menubar.addAction(self.menuDatabase.menuAction())
        self.menubar.addAction(self.menuRecognizer.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.labelJudulHome.setText(_translate("MainWindow", "UBAH DATA DOSEN", None))
        self.label_2.setText(_translate("MainWindow", "NIK/NIP       :", None))
        self.label_5.setText(_translate("MainWindow", "Fakultas      :", None))
        self.label_3.setText(_translate("MainWindow", "Jabatan      :", None))
        self.label.setText(_translate("MainWindow", "Nama          :", None))
        self.label_6.setText(_translate("MainWindow", "Nomor Telepon :", None))
        self.label_4.setText(_translate("MainWindow", "Jurusan      :", None))
        self.buttonCariUbahDatabaseDosen.setText(_translate("MainWindow", "Cari", None))
        self.buttonDataSuaraUbahDatabaseDosen_2.setText(_translate("MainWindow", "Tampil Data Suara", None))
        self.buttonSimpanUbahDatabaseDosen_3.setText(_translate("MainWindow", "Simpan", None))
        self.menuHome.setTitle(_translate("MainWindow", "Home", None))
        self.menuDatabase.setTitle(_translate("MainWindow", "Database", None))
        self.menuRecognizer.setTitle(_translate("MainWindow", "Recognizer", None))
        self.actionDosen.setText(_translate("MainWindow", "Dosen", None))
        self.actionPenelpon.setText(_translate("MainWindow", "Penelpon", None))
        self.actionHome.setText(_translate("MainWindow", "Home", None))
        self.actionHotline.setText(_translate("MainWindow", "Hotline", None))
        
        self.buttonCariUbahDatabaseDosen.clicked.connect(self.cari)
        self.actionDosen.triggered.connect(self.awal)
        self.actionPenelpon.triggered.connect(self.tampil_database_penelpon)
        self.actionHome.triggered.connect(self.pengenal_penutur)
        self.actionHotline.triggered.connect(self.pengenal_penutur)
        self.buttonDataSuaraUbahDatabaseDosen_2.clicked.connect(self.datasuara)

    def awal(self):
        self.mainwindow31 = awal(self)
        self.mainwindow31.setupUi(MainWindow)
    
    def tampil_database_dosen(self):
        self.mainwindow10 = database_dosen(self)
        self.mainwindow10.setupUi(MainWindow)  
        
    def tampil_database_penelpon(self):
        self.mainwindow11 = database_penelpon(self)
        self.mainwindow11.setupUi(MainWindow)  
        
    def pengenal_penutur(self):
        self.mainwindow12 = pengenal_penutur(self)
        self.mainwindow12.setupUi(MainWindow)
        
    def datasuara(self):
        kursor.execute("Select id_dosen from pegawai where nik=%s"%self.lineNIKUbahDatabaseDosen.text())
        pilih = kursor.fetchone()
        db.commit()
        print(pilih[0])
        self.mainwindow19 = datasuara_penelpon(self)
        self.mainwindow19.tampil(pilih[0])
    
    def tampildosen(self, pilih):
        kursor.execute("Select * from pegawai where id_dosen = %s" %pilih)
        pegawai = kursor.fetchall()
        db.commit()
        counter = 0
        for pegawai in pegawai:
            counter = 1
            self.lineNamaUbahDatabaseDosen.setText(pegawai[1])
            self.lineNIKUbahDatabaseDosen.setText(pegawai[2])
            self.lineJabatanUbahDatabaseDosen.setText(pegawai[3])
            self.lineJurusanUbahDatabaseDosen.setText(pegawai[4])
            self.lineFakultasUbahDatabaseDosen.setText(pegawai[5])
            self.lineNomorTeleponUbahDatabaseDosen_2.setText(pegawai[6])
            kursor.execute("Select * from foto where id_dosen = %s" %pilih)
            foto = kursor.fetchall()
            db.commit()
            for foto in foto:
                self.lineCariUbahDatabaseDosen.setText(foto[2])
                pixmap = QtGui.QPixmap(foto[2])
                pixmap = pixmap.scaled(301, 251, QtCore.Qt.KeepAspectRatio)
                self.labelFotoTambahDatabaseDosen.setPixmap(pixmap)
                self.labelFotoTambahDatabaseDosen.show()
                
    def cari(self):
        file = QtGui.QFileDialog.getOpenFileName(self, "Pilih Gambar")
        self.lineCariUbahDatabaseDosen.setText(file)
        pixmap = QtGui.QPixmap(file)
        pixmap = pixmap.scaled(301, 251, QtCore.Qt.KeepAspectRatio)
        self.labelFotoTambahDatabaseDosen.setPixmap(pixmap)
        self.labelFotoTambahDatabaseDosen.show()
        return file

    
    def simpan(self):
        if self.lineNamaTambahDatabaseDosen.text() != '' and self.lineNIKTambahDatabaseDosen.text() != '' and self.lineJabatanTambahDatabaseDosen.text() != '' and self.lineJurusanTambahDatabaseDosen.text() != '' and  self.lineFakultasTambahDatabaseDosen.text() != '':
            kursor.execute ("UPDATE pegawai SET nama='%s'" % self.lineNamaTambahDatabaseDosen.text() + ",nik='%s'" % self.lineNIKTambahDatabaseDosen.text() + ",jabatan='%s'" % self.lineJabatanTambahDatabaseDosen.text() + ",jurusan='%s'" % self.lineJurusanTambahDatabaseDosen.text() + ",fakultas='%s'" % self.lineFakultasTambahDatabaseDosen.text() + ",no_telp='%s'" % self.lineNomorTeleponTambahDatabaseDosen_2.text() + " WHERE id_dosen=%s" % id())
            db.commit()
            alertPopup = QtGui.QMessageBox()
            alertPopup.setText("Data berhasil disimpan")
            alertPopup.setIcon(alertPopup.Information)
            alertPopup.exec_()
            if self.lineCariTambahDatabaseDosen.text() != '':
                kursor.execute ("UPDATE foto SET path='%s'" % self.lineCariUbahDatabaseDosen.text() + " WHERE id_foto=%s" % id())
                db.commit()  
            self.labelFotoTambahDatabaseDosen.setText("")
            self.lineNamaTambahDatabaseDosen.setText("")
            self.lineNIKTambahDatabaseDosen.setText("")
            self.lineJabatanTambahDatabaseDosen.setText("")
            self.lineJurusanTambahDatabaseDosen.setText("")
            self.lineFakultasTambahDatabaseDosen.setText("")
            self.lineNomorTeleponTambahDatabaseDosen_2.setText("")
            self.lineCariTambahDatabaseDosen.setText("")
            #self.lineRecordTambahDatabaseDosen.setText("")
        else:
            alertPopup = QtGui.QMessageBox()
            alertPopup.setText("Data belum terisi lengkap")
            alertPopup.setIcon(alertPopup.Critical)
            alertPopup.exec_()
    
    

class database_penelpon(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(database_penelpon, self).__init__()
        self.setupUi(MainWindow)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(900, 585)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.tableDatabasePenelpon = QtGui.QTableWidget(self.centralwidget)
        self.tableDatabasePenelpon.setGeometry(QtCore.QRect(20, 80, 861, 301))
        self.tableDatabasePenelpon.setObjectName(_fromUtf8("tableDatabasePenelpon"))
        self.tableDatabasePenelpon.setColumnCount(8)
        self.tableDatabasePenelpon.setRowCount(0)
        item = QtGui.QTableWidgetItem()
        self.tableDatabasePenelpon.setHorizontalHeaderItem(0, item)
        item = QtGui.QTableWidgetItem()
        self.tableDatabasePenelpon.setHorizontalHeaderItem(1, item)
        item = QtGui.QTableWidgetItem()
        self.tableDatabasePenelpon.setHorizontalHeaderItem(2, item)
        item = QtGui.QTableWidgetItem()
        self.tableDatabasePenelpon.setHorizontalHeaderItem(3, item)
        item = QtGui.QTableWidgetItem()
        self.tableDatabasePenelpon.setHorizontalHeaderItem(4, item)
        item = QtGui.QTableWidgetItem()
        self.tableDatabasePenelpon.setHorizontalHeaderItem(5, item)
        item = QtGui.QTableWidgetItem()
        self.tableDatabasePenelpon.setHorizontalHeaderItem(6, item)
        item = QtGui.QTableWidgetItem()
        self.tableDatabasePenelpon.setHorizontalHeaderItem(7, item)
        self.labelJudulHome = QtGui.QLabel(self.centralwidget)
        self.labelJudulHome.setGeometry(QtCore.QRect(350, 10, 241, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.labelJudulHome.setFont(font)
        self.labelJudulHome.setObjectName(_fromUtf8("labelJudulHome"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 40, 161, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName(_fromUtf8("label"))
        self.labeljumlah = QtGui.QLabel(self.centralwidget)
        self.labeljumlah.setGeometry(QtCore.QRect(180, 40, 161, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.labeljumlah.setFont(font)
        self.labeljumlah.setText(_fromUtf8(""))
        self.labeljumlah.setObjectName(_fromUtf8("labeljumlah"))
        self.buttontidakdikenal = QtGui.QPushButton(self.centralwidget)
        self.buttontidakdikenal.setGeometry(QtCore.QRect(770, 40, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttontidakdikenal.setFont(font)
        self.buttontidakdikenal.setObjectName(_fromUtf8("buttontidakdikenal"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 900, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuHome = QtGui.QMenu(self.menubar)
        self.menuHome.setObjectName(_fromUtf8("menuHome"))
        self.menuDatabase = QtGui.QMenu(self.menubar)
        self.menuDatabase.setObjectName(_fromUtf8("menuDatabase"))
        self.menuRecognizer = QtGui.QMenu(self.menubar)
        self.menuRecognizer.setObjectName(_fromUtf8("menuRecognizer"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionDosen = QtGui.QAction(MainWindow)
        self.actionDosen.setObjectName(_fromUtf8("actionDosen"))
        self.actionPenelpon = QtGui.QAction(MainWindow)
        self.actionPenelpon.setObjectName(_fromUtf8("actionPenelpon"))
        self.actionHome = QtGui.QAction(MainWindow)
        self.actionHome.setObjectName(_fromUtf8("actionHome"))
        self.actionHotline = QtGui.QAction(MainWindow)
        self.actionHotline.setObjectName(_fromUtf8("actionHotline"))
        self.menuHome.addAction(self.actionHome)
        self.menuDatabase.addAction(self.actionDosen)
        self.menuDatabase.addAction(self.actionPenelpon)
        self.menuRecognizer.addAction(self.actionHotline)
        self.menubar.addAction(self.menuHome.menuAction())
        self.menubar.addAction(self.menuDatabase.menuAction())
        self.menubar.addAction(self.menuRecognizer.menuAction())

        self.retranslateUi(MainWindow)
        self.tampil(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        item = self.tableDatabasePenelpon.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "ID", None))
        item = self.tableDatabasePenelpon.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Status", None))
        item = self.tableDatabasePenelpon.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Nama", None))
        item = self.tableDatabasePenelpon.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Jurusan", None))
        item = self.tableDatabasePenelpon.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow", "Jabatan", None))
        item = self.tableDatabasePenelpon.horizontalHeaderItem(5)
        item.setText(_translate("MainWindow", "No telepon", None))
        item = self.tableDatabasePenelpon.horizontalHeaderItem(6)
        item.setText(_translate("MainWindow", "Waktu", None))
        item = self.tableDatabasePenelpon.horizontalHeaderItem(7)
        item.setText(_translate("MainWindow", "Tanggal", None))
        self.labelJudulHome.setText(_translate("MainWindow", "DATA PENELPON DIKENAL", None))
        self.label.setText(_translate("MainWindow", "Jumlah penelepon dikenal :", None))
        self.buttontidakdikenal.setText(_translate("MainWindow", "Tidak Dikenal", None))
        self.menuHome.setTitle(_translate("MainWindow", "Home", None))
        self.menuDatabase.setTitle(_translate("MainWindow", "Database", None))
        self.menuRecognizer.setTitle(_translate("MainWindow", "Recognizer", None))
        self.actionDosen.setText(_translate("MainWindow", "Dosen", None))
        self.actionPenelpon.setText(_translate("MainWindow", "Penelpon", None))
        self.actionHome.setText(_translate("MainWindow", "Home", None))
        self.actionHotline.setText(_translate("MainWindow", "Hotline", None))
        
        self.buttontidakdikenal.clicked.connect(self.tidakdikenal)
        self.actionDosen.triggered.connect(self.tampil_database_dosen)
        self.actionPenelpon.triggered.connect(self.tampil_database_penelpon)
        self.actionHotline.triggered.connect(self.pengenal_penutur)
        self.actionHome.triggered.connect(self.awal)

    def awal(self):
        self.mainwindow32 = awal(self)
        self.mainwindow32.setupUi(MainWindow)
    
    def tampil_database_dosen(self):
        self.mainwindow13 = database_dosen(self)
        self.mainwindow13.setupUi(MainWindow)  
        
    def tampil_database_penelpon(self):
        self.mainwindow14 = database_penelpon(self)
        self.mainwindow14.setupUi(MainWindow)  
        
    def pengenal_penutur(self):
        self.mainwindow15 = pengenal_penutur(self)
        self.mainwindow15.setupUi(MainWindow)
        
    def tidakdikenal(self):
        self.mainwindow23 = database_penelpon_tidakdikenal(self)
        self.mainwindow23.setupUi(MainWindow)
    
    def tampil(self, MainWindow):
        kursor = db.cursor()
        kursor.execute("select m.id_penelpon, m.status, p.nama, p.jurusan, p.jabatan, p.no_telp, m.waktu, m.tanggal from masuk m, pegawai p where m.id_dosen=p.id_dosen")
        orang = kursor.fetchall()
        rowCount = len(orang)
        self.labeljumlah.setText(str(rowCount))
        colCount = 8
        print (orang)
        self.tableDatabasePenelpon.setRowCount(rowCount)
        self.tableDatabasePenelpon.setColumnCount(colCount)
        for s in range(colCount):
            for i, row in enumerate(orang):
                for j, col in enumerate(row):
                    item = QtGui.QTableWidgetItem('%s' % (col))
                    self.tableDatabasePenelpon.setItem(i, j, item)

class database_penelpon_tidakdikenal(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(database_penelpon_tidakdikenal, self).__init__()
        self.setupUi(MainWindow)
        
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(900, 585)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.tableDatabasePenelpon = QtGui.QTableWidget(self.centralwidget)
        self.tableDatabasePenelpon.setGeometry(QtCore.QRect(70, 90, 351, 301))
        self.tableDatabasePenelpon.setObjectName(_fromUtf8("tableDatabasePenelpon"))
        self.tableDatabasePenelpon.setColumnCount(3)
        self.tableDatabasePenelpon.setRowCount(0)
        item = QtGui.QTableWidgetItem()
        self.tableDatabasePenelpon.setHorizontalHeaderItem(0, item)
        item = QtGui.QTableWidgetItem()
        self.tableDatabasePenelpon.setHorizontalHeaderItem(1, item)
        item = QtGui.QTableWidgetItem()
        self.tableDatabasePenelpon.setHorizontalHeaderItem(2, item)
        self.labelJudulHome = QtGui.QLabel(self.centralwidget)
        self.labelJudulHome.setGeometry(QtCore.QRect(350, 10, 311, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.labelJudulHome.setFont(font)
        self.labelJudulHome.setObjectName(_fromUtf8("labelJudulHome"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(490, 100, 191, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName(_fromUtf8("label"))
        self.labeljumlah = QtGui.QLabel(self.centralwidget)
        self.labeljumlah.setGeometry(QtCore.QRect(680, 100, 161, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.labeljumlah.setFont(font)
        self.labeljumlah.setText(_fromUtf8(""))
        self.labeljumlah.setObjectName(_fromUtf8("labeljumlah"))
        self.buttonkembali = QtGui.QPushButton(self.centralwidget)
        self.buttonkembali.setGeometry(QtCore.QRect(610, 310, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonkembali.setFont(font)
        self.buttonkembali.setObjectName(_fromUtf8("buttonkembali"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 900, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuHome = QtGui.QMenu(self.menubar)
        self.menuHome.setObjectName(_fromUtf8("menuHome"))
        self.menuDatabase = QtGui.QMenu(self.menubar)
        self.menuDatabase.setObjectName(_fromUtf8("menuDatabase"))
        self.menuRecognizer = QtGui.QMenu(self.menubar)
        self.menuRecognizer.setObjectName(_fromUtf8("menuRecognizer"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionDosen = QtGui.QAction(MainWindow)
        self.actionDosen.setObjectName(_fromUtf8("actionDosen"))
        self.actionPenelpon = QtGui.QAction(MainWindow)
        self.actionPenelpon.setObjectName(_fromUtf8("actionPenelpon"))
        self.actionHome = QtGui.QAction(MainWindow)
        self.actionHome.setObjectName(_fromUtf8("actionHome"))
        self.actionHotline = QtGui.QAction(MainWindow)
        self.actionHotline.setObjectName(_fromUtf8("actionHotline"))
        self.menuHome.addAction(self.actionHome)
        self.menuDatabase.addAction(self.actionDosen)
        self.menuDatabase.addAction(self.actionPenelpon)
        self.menuRecognizer.addAction(self.actionHotline)
        self.menubar.addAction(self.menuHome.menuAction())
        self.menubar.addAction(self.menuDatabase.menuAction())
        self.menubar.addAction(self.menuRecognizer.menuAction())

        self.retranslateUi(MainWindow)
        self.tampil(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        item = self.tableDatabasePenelpon.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "ID", None))
        item = self.tableDatabasePenelpon.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Waktu", None))
        item = self.tableDatabasePenelpon.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Tanggal", None))
        self.labelJudulHome.setText(_translate("MainWindow", "DATA PENELPON TIDAK DIKENAL", None))
        self.label.setText(_translate("MainWindow", "Jumlah penelepon tidak dikenal :", None))
        self.buttonkembali.setText(_translate("MainWindow", "Kembali", None))
        self.menuHome.setTitle(_translate("MainWindow", "Home", None))
        self.menuDatabase.setTitle(_translate("MainWindow", "Database", None))
        self.menuRecognizer.setTitle(_translate("MainWindow", "Recognizer", None))
        self.actionDosen.setText(_translate("MainWindow", "Dosen", None))
        self.actionPenelpon.setText(_translate("MainWindow", "Penelpon", None))
        self.actionHome.setText(_translate("MainWindow", "Home", None))
        self.actionHotline.setText(_translate("MainWindow", "Hotline", None))
        
        self.actionDosen.triggered.connect(self.tampil_database_dosen)
        self.actionPenelpon.triggered.connect(self.tampil_database_penelpon)
        self.buttonkembali.clicked.connect(self.kembali)
        self.actionHotline.triggered.connect(self.pengenal_penutur)
        self.actionHome.triggered.connect(self.awal)

    def awal(self):
        self.mainwindow33 = awal(self)
        self.mainwindow33.setupUi(MainWindow)
        
    def tampil_database_dosen(self):
        self.mainwindow19 = database_dosen(self)
        self.mainwindow19.setupUi(MainWindow)  
        
    def tampil_database_penelpon(self):
        self.mainwindow20 = database_penelpon(self)
        self.mainwindow20.setupUi(MainWindow)  
        
    def pengenal_penutur(self):
        self.mainwindow21 = pengenal_penutur(self)
        self.mainwindow21.setupUi(MainWindow)
        
    def kembali(self):
        self.mainwindow22 = database_penelpon(self)
        self.mainwindow22.setupUi(MainWindow)
        
    def tampil(self, MainWindow):
        kursor = db.cursor()
        kursor.execute("select id_penelpon, waktu, tanggal from masuk where status = 'Tidak Dikenal'")
        orang = kursor.fetchall()
        rowCount = len(orang)
        self.labeljumlah.setText(str(rowCount))
        colCount = 3
        self.tableDatabasePenelpon.setRowCount(rowCount)
        self.tableDatabasePenelpon.setColumnCount(colCount)
        for s in range(colCount):
            for i, row in enumerate(orang):
                for j, col in enumerate(row):
                    item = QtGui.QTableWidgetItem('%s' % (col))
                    self.tableDatabasePenelpon.setItem(i, j, item)

class datasuara_penelpon(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(datasuara_penelpon, self).__init__()
        self.setupUi(MainWindow)
    
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(900, 585)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.labelJudulDatabaseDosen = QtGui.QLabel(self.centralwidget)
        self.labelJudulDatabaseDosen.setGeometry(QtCore.QRect(350, 30, 211, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.labelJudulDatabaseDosen.setFont(font)
        self.labelJudulDatabaseDosen.setObjectName(_fromUtf8("labelJudulDatabaseDosen"))
        self.tableDatabaseDosen = QtGui.QTableWidget(self.centralwidget)
        self.tableDatabaseDosen.setGeometry(QtCore.QRect(50, 110, 611, 321))
        self.tableDatabaseDosen.setObjectName(_fromUtf8("tableDatabaseDosen"))
        self.tableDatabaseDosen.setColumnCount(2)
        self.tableDatabaseDosen.setRowCount(0)
        item = QtGui.QTableWidgetItem()
        self.tableDatabaseDosen.setHorizontalHeaderItem(0, item)
        item = QtGui.QTableWidgetItem()
        self.tableDatabaseDosen.setHorizontalHeaderItem(1, item)
        self.buttonPlay = QtGui.QPushButton(self.centralwidget)
        self.buttonPlay.setGeometry(QtCore.QRect(750, 140, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonPlay.setFont(font)
        self.buttonPlay.setObjectName(_fromUtf8("buttonPlay"))
        self.buttonTambah = QtGui.QPushButton(self.centralwidget)
        self.buttonTambah.setGeometry(QtCore.QRect(750, 220, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonTambah.setFont(font)
        self.buttonTambah.setObjectName(_fromUtf8("buttonTambah"))
        self.buttonUbah = QtGui.QPushButton(self.centralwidget)
        self.buttonUbah.setGeometry(QtCore.QRect(750, 300, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonUbah.setFont(font)
        self.buttonUbah.setObjectName(_fromUtf8("buttonUbah"))
        self.buttonKembali = QtGui.QPushButton(self.centralwidget)
        self.buttonKembali.setGeometry(QtCore.QRect(750, 370, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonKembali.setFont(font)
        self.buttonKembali.setObjectName(_fromUtf8("buttonKembali"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(810, 40, 46, 13))
        self.label.setText(_fromUtf8(""))
        self.label.setObjectName(_fromUtf8("label"))
        self.labelid = QtGui.QLabel(self.centralwidget)
        self.labelid.setGeometry(QtCore.QRect(70, 80, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.labelid.setFont(font)
        self.labelid.setObjectName(_fromUtf8("labelid"))
        self.labeliddosen = QtGui.QLabel(self.centralwidget)
        self.labeliddosen.setGeometry(QtCore.QRect(100, 80, 20, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.labeliddosen.setFont(font)
        self.labeliddosen.setText(_fromUtf8(""))
        self.labeliddosen.setObjectName(_fromUtf8("labeliddosen"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 900, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuHome = QtGui.QMenu(self.menubar)
        self.menuHome.setObjectName(_fromUtf8("menuHome"))
        self.menuDatabase = QtGui.QMenu(self.menubar)
        self.menuDatabase.setObjectName(_fromUtf8("menuDatabase"))
        self.menuRecognizer = QtGui.QMenu(self.menubar)
        self.menuRecognizer.setObjectName(_fromUtf8("menuRecognizer"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionDosen = QtGui.QAction(MainWindow)
        self.actionDosen.setObjectName(_fromUtf8("actionDosen"))
        self.actionPenelpon = QtGui.QAction(MainWindow)
        self.actionPenelpon.setObjectName(_fromUtf8("actionPenelpon"))
        self.actionHome = QtGui.QAction(MainWindow)
        self.actionHome.setObjectName(_fromUtf8("actionHome"))
        self.actionHotline = QtGui.QAction(MainWindow)
        self.actionHotline.setObjectName(_fromUtf8("actionHotline"))
        self.menuHome.addAction(self.actionHome)
        self.menuDatabase.addAction(self.actionDosen)
        self.menuDatabase.addAction(self.actionPenelpon)
        self.menuRecognizer.addAction(self.actionHotline)
        self.menubar.addAction(self.menuHome.menuAction())
        self.menubar.addAction(self.menuDatabase.menuAction())
        self.menubar.addAction(self.menuRecognizer.menuAction())


        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.timer = QtCore.QTimer(self)
        self.timer.stop()

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.labelJudulDatabaseDosen.setText(_translate("MainWindow", "DATA REKAMAN SUARA", None))
        item = self.tableDatabaseDosen.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "ID File", None))
        item = self.tableDatabaseDosen.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Path", None))
        self.buttonPlay.setText(_translate("MainWindow", "PLAY", None))
        self.buttonTambah.setText(_translate("MainWindow", "TAMBAH", None))
        self.buttonUbah.setText(_translate("MainWindow", "UBAH", None))
        self.buttonKembali.setText(_translate("MainWindow", "KEMBALI", None))
        self.labelid.setText(_translate("MainWindow", "Id : ", None))
        self.menuHome.setTitle(_translate("MainWindow", "Home", None))
        self.menuDatabase.setTitle(_translate("MainWindow", "Database", None))
        self.menuRecognizer.setTitle(_translate("MainWindow", "Recognizer", None))
        self.actionDosen.setText(_translate("MainWindow", "Dosen", None))
        self.actionPenelpon.setText(_translate("MainWindow", "Penelpon", None))
        self.actionHome.setText(_translate("MainWindow", "Home", None))
        self.actionHotline.setText(_translate("MainWindow", "Hotline", None))
        
        self.actionDosen.triggered.connect(self.tampil_database_dosen)
        self.actionPenelpon.triggered.connect(self.tampil_database_penelpon)
        self.buttonPlay.clicked.connect(self.play)
        self.buttonTambah.clicked.connect(self.rekam)
        self.buttonKembali.clicked.connect(self.kembali)
        self.buttonUbah.clicked.connect(self.ubah)
        self.actionHotline.triggered.connect(self.pengenal_penutur)
        self.actionHome.triggered.connect(self.awal)

    def awal(self):
        self.mainwindow34 = awal(self)
        self.mainwindow34.setupUi(MainWindow)
        
    def kembali(self):
        self.pilih = int(self.labeliddosen.text())
        self.mainwindow35 = ubah_database_dosen(self)
        self.mainwindow35.tampidosen(self.pilih)

    def tampil_database_dosen(self):
        self.mainwindow16 = database_dosen(self)
        self.mainwindow16.setupUi(MainWindow)  
        
    def tampil_database_penelpon(self):
        self.mainwindow17 = database_penelpon(self)
        self.mainwindow17.setupUi(MainWindow)  
        
    def pengenal_penutur(self):
        self.mainwindow18 = pengenal_penutur(self)
        self.mainwindow18.setupUi(MainWindow)
        
    def ambil_suara(self):
        nilai = kursor.execute("Select * from suara")
        ambil = kursor.fetchall()
        db.commit()
        id = len(ambil)
        return (id)
    
    def refresh(self):
        kursor = db.cursor()
        kursor.execute("SELECT id_file, path FROM suara where id_dosen=%s"%self.labelJudulDatabaseDosen.text())
        data = kursor.fetchall()
        db.commit()
        rowCount = len(data)
        colCount = 2
        self.tableDatabaseDosen.setRowCount(rowCount)
        self.tableDatabaseDosen.setColumnCount(colCount)
        for s in range(colCount):
            for i, row in enumerate(data):
                for j, col in enumerate(row):
                    item = QtGui.QTableWidgetItem('%s' % (col))
                    self.tableDatabaseDosen.setItem(i, j, item) 
    
    def tampil(self, pilih):
        kursor = db.cursor()
        kursor.execute("SELECT id_file, path FROM suara where id_dosen=%s"%pilih)
        data = kursor.fetchall()
        db.commit()
        self.labeliddosen.setText(str(pilih))
        rowCount = len(data)
        colCount = 2
        self.tableDatabaseDosen.setRowCount(rowCount)
        self.tableDatabaseDosen.setColumnCount(colCount)
        for s in range(colCount):
            for i, row in enumerate(data):
                for j, col in enumerate(row):
                    item = QtGui.QTableWidgetItem('%s' % (col))
                    self.tableDatabaseDosen.setItem(i, j, item)   
                    
    def rekam(self):
        directory = ('C:/Users/asus/workspace/bismillah/train/')
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        CHUNK = 1024
        RECORD_SECONDS = 5
        WAVE_OUTPUT_FILENAME = directory + str(self.ambil_suara()+1) + '.wav'
        audio = pyaudio.PyAudio()
        # start Recording
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)
        print('recording...')
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print('finished recording')
        # stop Recording
        stream.stop_stream()
        stream.close()
        audio.terminate()
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        kursor.execute ("INSERT INTO suara (id_file,id_dosen,path) VALUES (%s,%s,%s)",
                        [int(self.ambil_suara()+1), int(self.labeliddosen.text()), WAVE_OUTPUT_FILENAME])
        alertPopup = QtGui.QMessageBox()
        alertPopup.setText("Data suara berhasil disimpan")
        alertPopup.setIcon(alertPopup.Information)
        alertPopup.exec_()
        db.commit()    
        self.refresh()
    
    def ubah(self):
        file = self.tableDatabaseDosen.selectedItems()
        nameItem = file[0]
        pilih = nameItem.data(0)
        directory = ('C:/Users/asus/workspace/bismillah/train/')
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        CHUNK = 1024
        RECORD_SECONDS = 4
        WAVE_OUTPUT_FILENAME = directory + str(pilih) + '.wav'
        audio = pyaudio.PyAudio()
        # start Recording
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)
        print('recording...')
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print('finished recording')
        # stop Recording
        stream.stop_stream()
        stream.close()
        audio.terminate()
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        kursor.execute ("UPDATE suara SET path='%s'" % WAVE_OUTPUT_FILENAME + " WHERE id_file=%s" % pilih)
        db.commit()  
        alertPopup = QtGui.QMessageBox()
        alertPopup.setText("Data suara berhasil disimpan")
        alertPopup.setIcon(alertPopup.Information)
        alertPopup.exec_()

    def play(self):
        file = self.tableDatabaseDosen.selectedItems()
        nameItem = file[0]
        pilih = nameItem.data(0)
        data_suara = AudioSegment.from_wav('C:/Users/asus/workspace/bismillah/train/'+str(pilih)+'.wav')
        play(data_suara)

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = awal()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
    suara.form.show()
    suara.form.update()
