from __future__ import division
from PyQt4 import QtCore, QtGui
from scipy.signal import hamming
from scipy.fftpack import fft, fftshift, dct
from scipy.io.wavfile import read
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import pyaudio
import wave
import MySQLdb
import datetime
import time


#Koneksi Database
con = None
db = MySQLdb.connect("127.0.0.1", "root", "", "pengenalsuara")
kursor = db.cursor()

# Mengambil nilai id terbesar
nilai = kursor.execute("Select max(id_dosen) from datadosen")
ambil = kursor.fetchall()
id = ambil[0][0]


# Metode MFCC
def hertz_to_mel(freq):
    return 1125 * np.log(1 + freq / 700)


def mel_to_hertz(m):
    return 700 * (np.exp(m / 1125) - 1)


# calculate mel frequency filter bank
def mel_filterbank(nfft, nfiltbank, fs):
    # set limits of mel scale from 300Hz to 8000Hz
    lower_mel = hertz_to_mel(300)
    upper_mel = hertz_to_mel(8000)
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
    n = np.shape(d)[1]
    p = np.shape(c)[1]
    distance = np.empty((n, p))

    if n < p:
        for i in range(n):
            copies = np.transpose(np.tile(d[:, i], (p, 1)))
            distance[i, :] = np.sum((copies - c) ** 2, 0)
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
    nSpeaker = id
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
        fname = '/' + str(i + 2) + '.wav'
        (fs, s) = read(directory_train + fname)
        mel_coeff[i, :, :] = mfcc(s, fs, nfiltbank)[:, 0:68]
        codebooks[i, :, :] = lbg(mel_coeff[i, :, :], nCentroid)
    return (codebooks_mfcc)

# Tahapan Test
def minDistance(features, codebooks):
    speaker = 0
    distmin = np.inf
    for k in range(np.shape(codebooks)[0]):
        D = EUDistance(features, codebooks[k, :, :])
        dist = np.sum(np.min(D, axis=1)) / (np.shape(D)[0])
        if dist < distmin:
            distmin = dist
            speaker = k
        else:
            speaker = 0

    return speaker

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

class login(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(login, self).__init__()
        self.setupUi(MainWindow)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(330, 219)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.buttonLogin = QtGui.QPushButton(self.centralwidget)
        self.buttonLogin.setGeometry(QtCore.QRect(120, 150, 75, 23))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonLogin.setFont(font)
        self.buttonLogin.setObjectName(_fromUtf8("buttonLogin"))
        self.label_5 = QtGui.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(130, 10, 71, 16))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.label_4 = QtGui.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(20, 50, 91, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.lineUsernameLogin = QtGui.QLineEdit(self.centralwidget)
        self.lineUsernameLogin.setGeometry(QtCore.QRect(120, 50, 191, 20))
        self.lineUsernameLogin.setObjectName(_fromUtf8("lineUsernameLogin"))
        self.linePasswordLogin = QtGui.QLineEdit(self.centralwidget)
        self.linePasswordLogin.setGeometry(QtCore.QRect(120, 100, 191, 20))
        self.linePasswordLogin.setObjectName(_fromUtf8("linePasswordLogin"))
        self.label_6 = QtGui.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(20, 100, 91, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 330, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.buttonLogin.setText(_translate("MainWindow", "Login", None))
        self.label_5.setText(_translate("MainWindow", "LOGIN", None))
        self.label_4.setText(_translate("MainWindow", "Username     :", None))
        self.label_6.setText(_translate("MainWindow", "Password     :", None))

class home(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(home, self).__init__()
        self.setupUi(MainWindow)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(795, 508)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.labelJudulHome = QtGui.QLabel(self.centralwidget)
        self.labelJudulHome.setGeometry(QtCore.QRect(90, 10, 591, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.labelJudulHome.setFont(font)
        self.labelJudulHome.setObjectName(_fromUtf8("labelJudulHome"))
        self.labelFotoHome = QtGui.QLabel(self.centralwidget)
        self.labelFotoHome.setGeometry(QtCore.QRect(70, 80, 301, 251))
        self.labelFotoHome.setFrameShape(QtGui.QFrame.Box)
        self.labelFotoHome.setText(_fromUtf8(""))
        self.labelFotoHome.setObjectName(_fromUtf8("labelFotoHome"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(410, 90, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName(_fromUtf8("label"))
        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(410, 140, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.label_3 = QtGui.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(410, 240, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.label_4 = QtGui.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(410, 290, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.label_5 = QtGui.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(410, 340, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.label_6 = QtGui.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(90, 360, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.label_7 = QtGui.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(90, 410, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.labelNamaHome = QtGui.QLabel(self.centralwidget)
        self.labelNamaHome.setGeometry(QtCore.QRect(510, 90, 231, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.labelNamaHome.setFont(font)
        self.labelNamaHome.setText(_fromUtf8(""))
        self.labelNamaHome.setObjectName(_fromUtf8("labelNamaHome"))
        self.labelNIKHome = QtGui.QLabel(self.centralwidget)
        self.labelNIKHome.setGeometry(QtCore.QRect(510, 140, 231, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.labelNIKHome.setFont(font)
        self.labelNIKHome.setText(_fromUtf8(""))
        self.labelNIKHome.setObjectName(_fromUtf8("labelNIKHome"))
        self.labelJabatanHome = QtGui.QLabel(self.centralwidget)
        self.labelJabatanHome.setGeometry(QtCore.QRect(510, 240, 231, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.labelJabatanHome.setFont(font)
        self.labelJabatanHome.setText(_fromUtf8(""))
        self.labelJabatanHome.setObjectName(_fromUtf8("labelJabatanHome"))
        self.labelJurusanHome = QtGui.QLabel(self.centralwidget)
        self.labelJurusanHome.setGeometry(QtCore.QRect(510, 290, 231, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.labelJurusanHome.setFont(font)
        self.labelJurusanHome.setText(_fromUtf8(""))
        self.labelJurusanHome.setObjectName(_fromUtf8("labelJurusanHome"))
        self.labelFakultasHome = QtGui.QLabel(self.centralwidget)
        self.labelFakultasHome.setGeometry(QtCore.QRect(510, 340, 231, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.labelFakultasHome.setFont(font)
        self.labelFakultasHome.setText(_fromUtf8(""))
        self.labelFakultasHome.setObjectName(_fromUtf8("labelFakultasHome"))
        self.buttonGrafikHome = QtGui.QPushButton(self.centralwidget)
        self.buttonGrafikHome.setGeometry(QtCore.QRect(660, 390, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonGrafikHome.setFont(font)
        self.buttonGrafikHome.setObjectName(_fromUtf8("buttonGrafikHome"))
        self.labelWaktuHome = QtGui.QLabel(self.centralwidget)
        self.labelWaktuHome.setGeometry(QtCore.QRect(180, 360, 151, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.labelWaktuHome.setFont(font)
        self.labelWaktuHome.setText(_fromUtf8(""))
        self.labelWaktuHome.setObjectName(_fromUtf8("labelWaktuHome"))
        self.labelTanggalHome = QtGui.QLabel(self.centralwidget)
        self.labelTanggalHome.setGeometry(QtCore.QRect(180, 410, 231, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.labelTanggalHome.setFont(font)
        self.labelTanggalHome.setText(_fromUtf8(""))
        self.labelTanggalHome.setObjectName(_fromUtf8("labelTanggalHome"))
        self.buttonLogoutHome = QtGui.QPushButton(self.centralwidget)
        self.buttonLogoutHome.setGeometry(QtCore.QRect(720, 0, 75, 23))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonLogoutHome.setFont(font)
        self.buttonLogoutHome.setObjectName(_fromUtf8("buttonLogoutHome"))
        self.buttonMulaiHome = QtGui.QPushButton(self.centralwidget)
        self.buttonMulaiHome.setGeometry(QtCore.QRect(420, 390, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonMulaiHome.setFont(font)
        self.buttonMulaiHome.setObjectName(_fromUtf8("buttonMulaiHome"))
        self.buttonBerhentiHome = QtGui.QPushButton(self.centralwidget)
        self.buttonBerhentiHome.setGeometry(QtCore.QRect(540, 390, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonBerhentiHome.setFont(font)
        self.buttonBerhentiHome.setObjectName(_fromUtf8("buttonBerhentiHome"))
        self.labelTeleponHome = QtGui.QLabel(self.centralwidget)
        self.labelTeleponHome.setGeometry(QtCore.QRect(510, 190, 231, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.labelTeleponHome.setFont(font)
        self.labelTeleponHome.setText(_fromUtf8(""))
        self.labelTeleponHome.setObjectName(_fromUtf8("labelTeleponHome"))
        self.label_8 = QtGui.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(410, 190, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setObjectName(_fromUtf8("label_8"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 795, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuHome = QtGui.QMenu(self.menubar)
        self.menuHome.setObjectName(_fromUtf8("menuHome"))
        self.menuDatabase = QtGui.QMenu(self.menubar)
        self.menuDatabase.setObjectName(_fromUtf8("menuDatabase"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionDatabaseDosen = QtGui.QAction(MainWindow)
        self.actionDatabaseDosen.setObjectName(_fromUtf8("actionDatabaseDosen"))
        self.actionDatabasePenelpon = QtGui.QAction(MainWindow)
        self.actionDatabasePenelpon.setObjectName(_fromUtf8("actionDatabasePenelpon"))
        self.menuDatabase.addAction(self.actionDatabaseDosen)
        self.menuDatabase.addAction(self.actionDatabasePenelpon)
        self.menubar.addAction(self.menuHome.menuAction())
        self.menubar.addAction(self.menuDatabase.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.labelJudulHome.setText(_translate("MainWindow", "HOTLINE UNIVERSITAS PEMBANGUNAN \"VETERAN\" YOGYAKARTA", None))
        self.label.setText(_translate("MainWindow", "Nama          :", None))
        self.label_2.setText(_translate("MainWindow", "NIK              :", None))
        self.label_3.setText(_translate("MainWindow", "Jabatan      :", None))
        self.label_4.setText(_translate("MainWindow", "Jurusan      :", None))
        self.label_5.setText(_translate("MainWindow", "Fakultas      :", None))
        self.label_6.setText(_translate("MainWindow", "WAKTU      :", None))
        self.label_7.setText(_translate("MainWindow", "TANGGAL   :", None))
        self.buttonGrafikHome.setText(_translate("MainWindow", "Tampil Grafik", None))
        self.buttonLogoutHome.setText(_translate("MainWindow", "Logout", None))
        self.buttonMulaiHome.setText(_translate("MainWindow", "Mulai", None))
        self.buttonBerhentiHome.setText(_translate("MainWindow", "Berhenti", None))
        self.label_8.setText(_translate("MainWindow", "No Telepon  :", None))
        self.menuHome.setTitle(_translate("MainWindow", "Home", None))
        self.menuDatabase.setTitle(_translate("MainWindow", "Database", None))
        self.actionDatabaseDosen.setText(_translate("MainWindow", "Dosen", None))
        self.actionDatabasePenelpon.setText(_translate("MainWindow", "Penelpon", None))

        self.buttonMulaiHome.clicked.connect(self.mulai)
        self.buttonBerhentiHome.clicked.connect(self.berhenti)
        self.actionDatabaseDosen.triggered.connect(self.tampil_database_dosen)
        self.actionDatabasePenelpon.triggered.connect(self.tampil_database_penelpon)

    def tampil_database_dosen(self):
        self.mainwindow2 = database_dosen(self)
        self.mainwindow2.setupUi(MainWindow)

    def tampil_database_penelpon(self):
        self.mainwindow3 = database_penelpon(self)
        self.mainwindow3.setupUi(MainWindow)

    def berhenti(self):
        kursor.execute("Select * from penelpon")
        data=kursor.fetchall()
        #id=np.array(data[len(data)][0])
        print (data)


    def mulai(self, nCorrect_MFCC):
        for i in range(5):
            FORMAT = pyaudio.paInt16
            CHANNELS = 2
            RATE = 44100
            CHUNK = 1024
            RECORD_SECONDS = 3
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
            kursor = db.cursor()
            kursor.execute("SELECT * From penelpon")
            objek = kursor.fetchall()
            id = len(objek)
            if id >= 1:
                WAVE_OUTPUT_FILENAME = 'C:/Users/asus/PycharmProjects/SpeakerRecognitionCoba1/test/' + str(id) + '.wav'
                waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                waveFile.setnchannels(CHANNELS)
                waveFile.setsampwidth(audio.get_sample_size(FORMAT))
                waveFile.setframerate(RATE)
                waveFile.writeframes(b''.join(frames))
                waveFile.close()
                nfiltbank = 12
                codebooks_mfcc = training(nfiltbank)
                directory = os.getcwd() + '/test';
                nCorrect_MFCC = 0
                fname = '/' + str(id) + '.wav'
                print(fname)
                (fs, s) = read(directory + fname)
                mel_coefs = mfcc(s, fs, nfiltbank)
                sp_mfcc = minDistance(mel_coefs, codebooks_mfcc)
                kursor = db.cursor()
                kursor.execute("select * from datadosen where id_dosen = %s" % (sp_mfcc + 1))
                orang = kursor.fetchall()
                counter = 0
                if self.labelNamaHome.text() == '':
                    for orang in orang:
                        counter = counter + 1
                        self.labelNamaHome.setText(orang[1])
                        self.labelNIKHome.setText(orang[2])
                        self.labelTeleponHome.setText(orang[3])
                        self.labelJabatanHome.setText(orang[4])
                        self.labelJurusanHome.setText(orang[5])
                        self.labelFakultasHome.setText((orang[6]))
                        pixmap = QtGui.QPixmap(orang[8])
                        pixmap = pixmap.scaled(301, 251, QtCore.Qt.KeepAspectRatio)
                        self.labelFotoHome.setPixmap(pixmap)
                        self.labelFotoHome.show()
                        kursor.execute("INSERT INTO penelpon (nama,nik,no_telp,waktu,tanggal) VALUES (%s,%s,%s,%s,%s)",
                            [orang[1], orang[2], datetime.datetime.now().time(), datetime.datetime.now().date()])
                        db.commit()
            else:
                WAVE_OUTPUT_FILENAME = 'C:/Users/asus/PycharmProjects/SpeakerRecognitionCoba1/test/0.wav'
                waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                waveFile.setnchannels(CHANNELS)
                waveFile.setsampwidth(audio.get_sample_size(FORMAT))
                waveFile.setframerate(RATE)
                waveFile.writeframes(b''.join(frames))
                waveFile.close()
                nfiltbank = 12
                codebooks_mfcc = training(nfiltbank)
                directory = os.getcwd() + '/test';
                nCorrect_MFCC = 0
                fname = '/0.wav'
                print(fname)
                (fs, s) = read(directory + fname)
                mel_coefs = mfcc(s, fs, nfiltbank)
                sp_mfcc = minDistance(mel_coefs, codebooks_mfcc)
                kursor = db.cursor()
                kursor.execute("select * from datadosen where id_dosen = %s" % (sp_mfcc + 1))
                orang = kursor.fetchall()
                counter = 0
                if self.labelNamaHome.text() == '':
                    for orang in orang:
                        counter = counter + 1
                        self.labelNamaHome.setText(orang[1])
                        self.labelNIKHome.setText(orang[2])
                        self.labelTeleponHome.setText(orang[3])
                        self.labelJabatanHome.setText(orang[4])
                        self.labelJurusanHome.setText(orang[5])
                        self.labelFakultasHome.setText((orang[6]))
                        pixmap = QtGui.QPixmap(orang[8])
                        pixmap = pixmap.scaled(301, 251, QtCore.Qt.KeepAspectRatio)
                        self.labelFotoHome.setPixmap(pixmap)
                        self.labelFotoHome.show()
                        kursor.execute("INSERT INTO penelpon (nama, nik, no_telp ,waktu, tanggal) VALUES (%s,%s,%s,%s,%s)",
                            [orang[1], orang[2], orang[3], datetime.datetime.now().time(), datetime.datetime.now().date()])
                        db.commit()



class database_dosen(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(database_dosen, self).__init__()
        self.setupUi(MainWindow)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(796, 507)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.labelJudulDatabaseDosen = QtGui.QLabel(self.centralwidget)
        self.labelJudulDatabaseDosen.setGeometry(QtCore.QRect(320, 10, 131, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.labelJudulDatabaseDosen.setFont(font)
        self.labelJudulDatabaseDosen.setObjectName(_fromUtf8("labelJudulDatabaseDosen"))
        self.tableDatabaseDosen = QtGui.QTableWidget(self.centralwidget)
        self.tableDatabaseDosen.setGeometry(QtCore.QRect(30, 80, 511, 192))
        self.tableDatabaseDosen.setObjectName(_fromUtf8("tableDatabaseDosen"))
        self.tableDatabaseDosen.setColumnCount(5)
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
        self.buttonPlayDatabaseDosen = QtGui.QPushButton(self.centralwidget)
        self.buttonPlayDatabaseDosen.setGeometry(QtCore.QRect(620, 110, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonPlayDatabaseDosen.setFont(font)
        self.buttonPlayDatabaseDosen.setObjectName(_fromUtf8("buttonPlayDatabaseDosen"))
        self.buttonUbahDatabaseDosen = QtGui.QPushButton(self.centralwidget)
        self.buttonUbahDatabaseDosen.setGeometry(QtCore.QRect(620, 250, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonUbahDatabaseDosen.setFont(font)
        self.buttonUbahDatabaseDosen.setObjectName(_fromUtf8("buttonUbahDatabaseDosen"))
        self.buttonTambahDatabaseDosen = QtGui.QPushButton(self.centralwidget)
        self.buttonTambahDatabaseDosen.setGeometry(QtCore.QRect(620, 180, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonTambahDatabaseDosen.setFont(font)
        self.buttonTambahDatabaseDosen.setObjectName(_fromUtf8("buttonTambahDatabaseDosen"))
        self.buttonHapusDatabaseDosen = QtGui.QPushButton(self.centralwidget)
        self.buttonHapusDatabaseDosen.setGeometry(QtCore.QRect(620, 320, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonHapusDatabaseDosen.setFont(font)
        self.buttonHapusDatabaseDosen.setObjectName(_fromUtf8("buttonHapusDatabaseDosen"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 796, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuHome = QtGui.QMenu(self.menubar)
        self.menuHome.setObjectName(_fromUtf8("menuHome"))
        self.menuDatabase = QtGui.QMenu(self.menubar)
        self.menuDatabase.setObjectName(_fromUtf8("menuDatabase"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionDosen = QtGui.QAction(MainWindow)
        self.actionDosen.setObjectName(_fromUtf8("actionDosen"))
        self.actionPenelpon = QtGui.QAction(MainWindow)
        self.actionPenelpon.setObjectName(_fromUtf8("actionPenelpon"))
        self.menuDatabase.addAction(self.actionDosen)
        self.menuDatabase.addAction(self.actionPenelpon)
        self.menubar.addAction(self.menuHome.menuAction())
        self.menubar.addAction(self.menuDatabase.menuAction())

        self.retranslateUi(MainWindow)
        self.data_dosen(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.labelJudulDatabaseDosen.setText(_translate("MainWindow", "DATA DOSEN", None))
        item = self.tableDatabaseDosen.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Nama", None))
        item = self.tableDatabaseDosen.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "NIK", None))
        item = self.tableDatabaseDosen.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Jabatan", None))
        item = self.tableDatabaseDosen.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Jurusan", None))
        item = self.tableDatabaseDosen.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow", "Fakultas", None))
        self.buttonPlayDatabaseDosen.setText(_translate("MainWindow", "PLAY", None))
        self.buttonUbahDatabaseDosen.setText(_translate("MainWindow", "UBAH", None))
        self.buttonTambahDatabaseDosen.setText(_translate("MainWindow", "TAMBAH", None))
        self.buttonHapusDatabaseDosen.setText(_translate("MainWindow", "HAPUS", None))
        self.menuHome.setTitle(_translate("MainWindow", "Home", None))
        self.menuDatabase.setTitle(_translate("MainWindow", "Database", None))
        self.actionDosen.setText(_translate("MainWindow", "Dosen", None))
        self.actionPenelpon.setText(_translate("MainWindow", "Penelpon", None))

        self.buttonTambahDatabaseDosen.clicked.connect(self.tampil_tambah)

    def data_dosen(self, MainWindow):
        kursor = db.cursor()
        kursor.execute("SELECT nama, nik, jurusan, fakultas FROM datadosen")
        data = kursor.fetchall()
        rowCount = len(data)
        colCount = 4
        self.tableDatabaseDosen.setRowCount(rowCount)
        self.tableDatabaseDosen.setColumnCount(colCount)
        for s in range(colCount):
            for i, row in enumerate(data):
                for j, col in enumerate(row):
                    item = QtGui.QTableWidgetItem('%s' % (col))
                    self.tableDatabaseDosen.setItem(i, j, item)

    def tampil_tambah(self):
        self.mainwindow4 = tambah_database_dosen(self)
        self.mainwindow4.setupUi(MainWindow)

class tambah_database_dosen(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(tambah_database_dosen, self).__init__()
        self.setupUi(MainWindow)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(795, 506)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(400, 90, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName(_fromUtf8("label"))
        self.labelFotoTambahDatabaseDosen = QtGui.QLabel(self.centralwidget)
        self.labelFotoTambahDatabaseDosen.setGeometry(QtCore.QRect(60, 80, 301, 251))
        self.labelFotoTambahDatabaseDosen.setFrameShape(QtGui.QFrame.Box)
        self.labelFotoTambahDatabaseDosen.setText(_fromUtf8(""))
        self.labelFotoTambahDatabaseDosen.setObjectName(_fromUtf8("labelFotoTambahDatabaseDosen"))
        self.label_3 = QtGui.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(400, 190, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(400, 140, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.label_4 = QtGui.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(400, 240, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.label_5 = QtGui.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(400, 290, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.labelJudulHome = QtGui.QLabel(self.centralwidget)
        self.labelJudulHome.setGeometry(QtCore.QRect(300, 10, 201, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.labelJudulHome.setFont(font)
        self.labelJudulHome.setObjectName(_fromUtf8("labelJudulHome"))
        self.buttonCariTambahDatabaseDosen = QtGui.QPushButton(self.centralwidget)
        self.buttonCariTambahDatabaseDosen.setGeometry(QtCore.QRect(60, 350, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonCariTambahDatabaseDosen.setFont(font)
        self.buttonCariTambahDatabaseDosen.setObjectName(_fromUtf8("buttonCariTambahDatabaseDosen"))
        self.buttonTambahDatabaseDosen = QtGui.QPushButton(self.centralwidget)
        self.buttonTambahDatabaseDosen.setGeometry(QtCore.QRect(570, 410, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonTambahDatabaseDosen.setFont(font)
        self.buttonTambahDatabaseDosen.setObjectName(_fromUtf8("buttonTambahDatabaseDosen"))
        self.buttonRecordTambahDatabaseDosen = QtGui.QPushButton(self.centralwidget)
        self.buttonRecordTambahDatabaseDosen.setGeometry(QtCore.QRect(390, 330, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonRecordTambahDatabaseDosen.setFont(font)
        self.buttonRecordTambahDatabaseDosen.setObjectName(_fromUtf8("buttonRecordTambahDatabaseDosen"))
        self.lineCariTambahDatabaseDosen = QtGui.QLineEdit(self.centralwidget)
        self.lineCariTambahDatabaseDosen.setGeometry(QtCore.QRect(170, 360, 191, 20))
        self.lineCariTambahDatabaseDosen.setObjectName(_fromUtf8("lineCariTambahDatabaseDosen"))
        self.lineNamaTambahDatabaseDosen = QtGui.QLineEdit(self.centralwidget)
        self.lineNamaTambahDatabaseDosen.setGeometry(QtCore.QRect(500, 90, 271, 20))
        self.lineNamaTambahDatabaseDosen.setObjectName(_fromUtf8("lineNamaTambahDatabaseDosen"))
        self.lineFakultasTambahDatabaseDosen = QtGui.QLineEdit(self.centralwidget)
        self.lineFakultasTambahDatabaseDosen.setGeometry(QtCore.QRect(500, 290, 271, 20))
        self.lineFakultasTambahDatabaseDosen.setObjectName(_fromUtf8("lineFakultasTambahDatabaseDosen"))
        self.lineJurusanTambahDatabaseDosen = QtGui.QLineEdit(self.centralwidget)
        self.lineJurusanTambahDatabaseDosen.setGeometry(QtCore.QRect(500, 240, 271, 20))
        self.lineJurusanTambahDatabaseDosen.setObjectName(_fromUtf8("lineJurusanTambahDatabaseDosen"))
        self.lineJabatanTambahDatabaseDosen = QtGui.QLineEdit(self.centralwidget)
        self.lineJabatanTambahDatabaseDosen.setGeometry(QtCore.QRect(500, 190, 271, 20))
        self.lineJabatanTambahDatabaseDosen.setObjectName(_fromUtf8("lineJabatanTambahDatabaseDosen"))
        self.lineNIKTambahDatabaseDosen = QtGui.QLineEdit(self.centralwidget)
        self.lineNIKTambahDatabaseDosen.setGeometry(QtCore.QRect(500, 140, 271, 20))
        self.lineNIKTambahDatabaseDosen.setObjectName(_fromUtf8("lineNIKTambahDatabaseDosen"))
        self.lineRecordTambahDatabaseDosen = QtGui.QLineEdit(self.centralwidget)
        self.lineRecordTambahDatabaseDosen.setGeometry(QtCore.QRect(500, 340, 271, 20))
        self.lineRecordTambahDatabaseDosen.setObjectName(_fromUtf8("lineRecordTambahDatabaseDosen"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 795, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuHome = QtGui.QMenu(self.menubar)
        self.menuHome.setObjectName(_fromUtf8("menuHome"))
        self.menuDatabase = QtGui.QMenu(self.menubar)
        self.menuDatabase.setObjectName(_fromUtf8("menuDatabase"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionDosen = QtGui.QAction(MainWindow)
        self.actionDosen.setObjectName(_fromUtf8("actionDosen"))
        self.actionPenelpon = QtGui.QAction(MainWindow)
        self.actionPenelpon.setObjectName(_fromUtf8("actionPenelpon"))
        self.menuDatabase.addAction(self.actionDosen)
        self.menuDatabase.addAction(self.actionPenelpon)
        self.menubar.addAction(self.menuHome.menuAction())
        self.menubar.addAction(self.menuDatabase.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.label.setText(_translate("MainWindow", "Nama          :", None))
        self.label_3.setText(_translate("MainWindow", "Jabatan      :", None))
        self.label_2.setText(_translate("MainWindow", "NIK              :", None))
        self.label_4.setText(_translate("MainWindow", "Jurusan      :", None))
        self.label_5.setText(_translate("MainWindow", "Fakultas      :", None))
        self.labelJudulHome.setText(_translate("MainWindow", "TAMBAH DATA DOSEN", None))
        self.buttonCariTambahDatabaseDosen.setText(_translate("MainWindow", "Cari", None))
        self.buttonRecordTambahDatabaseDosen.setText(_translate("MainWindow", "Record", None))
        self.buttonTambahDatabaseDosen.setText(_translate("MainWindow", "Tambah", None))
        self.menuHome.setTitle(_translate("MainWindow", "Home", None))
        self.menuDatabase.setTitle(_translate("MainWindow", "Database", None))
        self.actionDosen.setText(_translate("MainWindow", "Dosen", None))
        self.actionPenelpon.setText(_translate("MainWindow", "Penelpon", None))

        self.buttonCariTambahDatabaseDosen.clicked.connect(self.cari)
        self.buttonRecordTambahDatabaseDosen.clicked.connect(self.rekam)
        self.buttonTambahDatabaseDosen.clicked.connect(self.tambah)

    def rekam(self):
        directory = ('C:/Users/asus/PycharmProjects/SpeakerRecognitionCoba1/train/')
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        CHUNK = 1024
        RECORD_SECONDS = 4
        WAVE_OUTPUT_FILENAME = directory + str(id+1) + '.wav'
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

    def cari(self):
        file = QtGui.QFileDialog.getOpenFileName(self, "Pilih Gambar")
        self.lineCariTambahDatabaseDosen.setText(file)
        pixmap = QtGui.QPixmap(file)
        pixmap = pixmap.scaled(301, 251, QtCore.Qt.KeepAspectRatio)
        self.labelFotoTambahDatabaseDosen.setPixmap(pixmap)
        self.labelFotoTambahDatabaseDosen.show()
        return file
    def tes(self):
        print ("nilai = %d" %(id+1))

    def tambah(self):
        if self.lineNamaTambahDatabaseDosen.text() != '' and self.lineNIKTambahDatabaseDosen.text() != '' and self.lineJabatanTambahDatabaseDosen.text() != '' and self.lineJurusanTambahDatabaseDosen.text() != '' and  self.lineFakultasTambahDatabaseDosen.text() != '' and self.lineCariTambahDatabaseDosen.text() != '' and  self.lineRecordTambahDatabaseDosen.text() != '':
                kursor.execute ("INSERT INTO datadosen (id_dosen,nama,nik,jabatan,jurusan,fakultas,rekaman_suara,foto) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                                  [int(id+1), self.lineNamaTambahDatabaseDosen.text(), self.lineNIKTambahDatabaseDosen.text(), self.lineJabatanTambahDatabaseDosen.text(),
                                    self.lineJurusanTambahDatabaseDosen.text(), self.lineFakultasTambahDatabaseDosen.text(), self.lineRecordTambahDatabaseDosen.text(), self.lineCariTambahDatabaseDosen.text()])
                alertPopup = QtGui.QMessageBox()
                alertPopup.setText("Data berhasil disimpan")
                alertPopup.setIcon(alertPopup.Information)
                alertPopup.exec_()
                db.commit()
                self.labelFotoTambahDatabaseDosen.setText("")
                self.lineNamaTambahDatabaseDosen.setText("")
                self.lineNIKTambahDatabaseDosen.setText("")
                self.lineJabatanTambahDatabaseDosen.setText("")
                self.lineJurusanTambahDatabaseDosen.setText("")
                self.lineFakultasTambahDatabaseDosen.setText("")
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
        MainWindow.resize(793, 507)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.label_5 = QtGui.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(390, 300, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.buttonCariUbahDatabaseDosen = QtGui.QPushButton(self.centralwidget)
        self.buttonCariUbahDatabaseDosen.setGeometry(QtCore.QRect(50, 360, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonCariUbahDatabaseDosen.setFont(font)
        self.buttonCariUbahDatabaseDosen.setObjectName(_fromUtf8("buttonCariUbahDatabaseDosen"))
        self.labelJudulHome = QtGui.QLabel(self.centralwidget)
        self.labelJudulHome.setGeometry(QtCore.QRect(320, 20, 181, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.labelJudulHome.setFont(font)
        self.labelJudulHome.setObjectName(_fromUtf8("labelJudulHome"))
        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(390, 150, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.label_4 = QtGui.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(390, 250, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.label_3 = QtGui.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(390, 200, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.lineCariUbahDatabaseDosen = QtGui.QLineEdit(self.centralwidget)
        self.lineCariUbahDatabaseDosen.setGeometry(QtCore.QRect(160, 370, 191, 20))
        self.lineCariUbahDatabaseDosen.setObjectName(_fromUtf8("lineCariUbahDatabaseDosen"))
        self.lineJabatanUbahDatabaseDosen = QtGui.QLineEdit(self.centralwidget)
        self.lineJabatanUbahDatabaseDosen.setGeometry(QtCore.QRect(490, 200, 271, 20))
        self.lineJabatanUbahDatabaseDosen.setObjectName(_fromUtf8("lineJabatanUbahDatabaseDosen"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(390, 100, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName(_fromUtf8("label"))
        self.lineFakultasUbahDatabaseDosen = QtGui.QLineEdit(self.centralwidget)
        self.lineFakultasUbahDatabaseDosen.setGeometry(QtCore.QRect(490, 300, 271, 20))
        self.lineFakultasUbahDatabaseDosen.setObjectName(_fromUtf8("lineFakultasUbahDatabaseDosen"))
        self.lineNIKUbahDatabaseDosen = QtGui.QLineEdit(self.centralwidget)
        self.lineNIKUbahDatabaseDosen.setGeometry(QtCore.QRect(490, 150, 271, 20))
        self.lineNIKUbahDatabaseDosen.setObjectName(_fromUtf8("lineNIKUbahDatabaseDosen"))
        self.buttonRecordUbahDatabaseDosen = QtGui.QPushButton(self.centralwidget)
        self.buttonRecordUbahDatabaseDosen.setGeometry(QtCore.QRect(380, 340, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonRecordUbahDatabaseDosen.setFont(font)
        self.buttonRecordUbahDatabaseDosen.setObjectName(_fromUtf8("buttonRecordUbahDatabaseDosen"))
        self.lineRecordUbahDatabaseDosen = QtGui.QLineEdit(self.centralwidget)
        self.lineRecordUbahDatabaseDosen.setGeometry(QtCore.QRect(490, 350, 271, 20))
        self.lineRecordUbahDatabaseDosen.setObjectName(_fromUtf8("lineRecordUbahDatabaseDosen"))
        self.lineNamaUbahDatabaseDosen = QtGui.QLineEdit(self.centralwidget)
        self.lineNamaUbahDatabaseDosen.setGeometry(QtCore.QRect(490, 100, 271, 20))
        self.lineNamaUbahDatabaseDosen.setObjectName(_fromUtf8("lineNamaUbahDatabaseDosen"))
        self.lineJurusanUbahDatabaseDosen = QtGui.QLineEdit(self.centralwidget)
        self.lineJurusanUbahDatabaseDosen.setGeometry(QtCore.QRect(490, 250, 271, 20))
        self.lineJurusanUbahDatabaseDosen.setObjectName(_fromUtf8("lineJurusanUbahDatabaseDosen"))
        self.labelFotoUbahDatabaseDosen = QtGui.QLabel(self.centralwidget)
        self.labelFotoUbahDatabaseDosen.setGeometry(QtCore.QRect(50, 90, 301, 251))
        self.labelFotoUbahDatabaseDosen.setFrameShape(QtGui.QFrame.Box)
        self.labelFotoUbahDatabaseDosen.setText(_fromUtf8(""))
        self.labelFotoUbahDatabaseDosen.setObjectName(_fromUtf8("labelFotoUbahDatabaseDosen"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 793, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuHome = QtGui.QMenu(self.menubar)
        self.menuHome.setObjectName(_fromUtf8("menuHome"))
        self.menuDatabase = QtGui.QMenu(self.menubar)
        self.menuDatabase.setObjectName(_fromUtf8("menuDatabase"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionDosen = QtGui.QAction(MainWindow)
        self.actionDosen.setObjectName(_fromUtf8("actionDosen"))
        self.actionPenelpon = QtGui.QAction(MainWindow)
        self.actionPenelpon.setObjectName(_fromUtf8("actionPenelpon"))
        self.menuDatabase.addAction(self.actionDosen)
        self.menuDatabase.addAction(self.actionPenelpon)
        self.menubar.addAction(self.menuHome.menuAction())
        self.menubar.addAction(self.menuDatabase.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.label_5.setText(_translate("MainWindow", "Fakultas      :", None))
        self.buttonCariUbahDatabaseDosen.setText(_translate("MainWindow", "Cari", None))
        self.labelJudulHome.setText(_translate("MainWindow", "UBAH DATA DOSEN", None))
        self.label_2.setText(_translate("MainWindow", "NIK              :", None))
        self.label_4.setText(_translate("MainWindow", "Jurusan      :", None))
        self.label_3.setText(_translate("MainWindow", "Jabatan      :", None))
        self.label.setText(_translate("MainWindow", "Nama          :", None))
        self.buttonRecordUbahDatabaseDosen.setText(_translate("MainWindow", "Record", None))
        self.menuHome.setTitle(_translate("MainWindow", "Home", None))
        self.menuDatabase.setTitle(_translate("MainWindow", "Database", None))
        self.actionDosen.setText(_translate("MainWindow", "Dosen", None))
        self.actionPenelpon.setText(_translate("MainWindow", "Penelpon", None))

class database_penelpon(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(database_penelpon, self).__init__()
        self.setupUi(MainWindow)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(835, 425)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.tableDatabasePenelpon = QtGui.QTableWidget(self.centralwidget)
        self.tableDatabasePenelpon.setGeometry(QtCore.QRect(20, 80, 711, 301))
        self.tableDatabasePenelpon.setObjectName(_fromUtf8("tableDatabasePenelpon"))
        self.tableDatabasePenelpon.setColumnCount(7)
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
        self.labelJudulHome = QtGui.QLabel(self.centralwidget)
        self.labelJudulHome.setGeometry(QtCore.QRect(330, 20, 151, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.labelJudulHome.setFont(font)
        self.labelJudulHome.setObjectName(_fromUtf8("labelJudulHome"))
        self.buttonMulaiPenelpon = QtGui.QPushButton(self.centralwidget)
        self.buttonMulaiPenelpon.setGeometry(QtCore.QRect(750, 140, 75, 23))
        self.buttonMulaiPenelpon.setObjectName(_fromUtf8("buttonMulaiPenelpon"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 835, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuHome = QtGui.QMenu(self.menubar)
        self.menuHome.setObjectName(_fromUtf8("menuHome"))
        self.menuDatabase = QtGui.QMenu(self.menubar)
        self.menuDatabase.setObjectName(_fromUtf8("menuDatabase"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionDosen = QtGui.QAction(MainWindow)
        self.actionDosen.setObjectName(_fromUtf8("actionDosen"))
        self.actionPenelpon = QtGui.QAction(MainWindow)
        self.actionPenelpon.setObjectName(_fromUtf8("actionPenelpon"))
        self.menuDatabase.addAction(self.actionDosen)
        self.menuDatabase.addAction(self.actionPenelpon)
        self.menubar.addAction(self.menuHome.menuAction())
        self.menubar.addAction(self.menuDatabase.menuAction())

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
        item.setText(_translate("MainWindow", "No telepon", None))
        item = self.tableDatabasePenelpon.horizontalHeaderItem(5)
        item.setText(_translate("MainWindow", "Waktu", None))
        item = self.tableDatabasePenelpon.horizontalHeaderItem(6)
        item.setText(_translate("MainWindow", "Tanggal", None))
        self.labelJudulHome.setText(_translate("MainWindow", "DATA PENELPON", None))
        self.buttonMulaiPenelpon.setText(_translate("MainWindow", "Mulai", None))
        self.menuHome.setTitle(_translate("MainWindow", "Home", None))
        self.menuDatabase.setTitle(_translate("MainWindow", "Database", None))
        self.actionDosen.setText(_translate("MainWindow", "Dosen", None))
        self.actionPenelpon.setText(_translate("MainWindow", "Penelpon", None))

        self.buttonMulaiPenelpon.clicked.connect(self.mulai)

    def tampil(self, MainWindow):
        kursor = db.cursor()
        kursor.execute("select * from penelpon")
        orang = kursor.fetchall()
        rowCount = len(orang)
        colCount = 7
        self.tableDatabasePenelpon.setRowCount(rowCount)
        self.tableDatabasePenelpon.setColumnCount(colCount)
        for s in range(colCount):
            for i, row in enumerate(orang):
                for j, col in enumerate(row):
                    item = QtGui.QTableWidgetItem('%s' % (col))
                    self.tableDatabasePenelpon.setItem(i, j, item)

    def mulai(self, nCorrect_MFCC):
        for i in range(5):
            FORMAT = pyaudio.paInt16
            CHANNELS = 2
            RATE = 44100
            CHUNK = 1024
            RECORD_SECONDS = 3
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
            kursor = db.cursor()
            kursor.execute("SELECT * From penelpon")
            objek = kursor.fetchall()
            id = len(objek)
            if id >= 1:
                WAVE_OUTPUT_FILENAME = 'C:/Users/asus/PycharmProjects/SpeakerRecognitionCoba1/test/' + str(id) + '.wav'
                waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                waveFile.setnchannels(CHANNELS)
                waveFile.setsampwidth(audio.get_sample_size(FORMAT))
                waveFile.setframerate(RATE)
                waveFile.writeframes(b''.join(frames))
                waveFile.close()
                nfiltbank = 12
                codebooks_mfcc = training(nfiltbank)
                directory = os.getcwd() + '/test';
                nCorrect_MFCC = 0
                fname = '/' + str(id) + '.wav'
                print(fname)
                (fs, s) = read(directory + fname)
                mel_coefs = mfcc(s, fs, nfiltbank)
                sp_mfcc = minDistance(mel_coefs, codebooks_mfcc)
                if sp_mfcc >= 1:
                    kursor = db.cursor()
                    kursor.execute("select * from datadosen where id_dosen = %s" % (sp_mfcc))
                    orang = kursor.fetchall()
                    for orang in orang:
                        kursor.execute("INSERT INTO penelpon (nama,nik,no_telp,waktu,tanggal) VALUES (%s,%s,%s,%s,%s)",
                        [orang[1], orang[2], orang[3], datetime.datetime.now().time(), datetime.datetime.now().date()])
                        db.commit()
                elif sp_mfcc == 0:
                    nilai = "Tidak Dikenal"
                    kursor.execute("INSERT INTO penelpon (nama,nik,no_telp,waktu,tanggal) VALUES (%s,%s,%s,%s,%s)",
                                   [nilai, nilai, nilai, datetime.datetime.now().time(), datetime.datetime.now().date()])
                    db.commit()
            else:
                WAVE_OUTPUT_FILENAME = 'C:/Users/asus/PycharmProjects/SpeakerRecognitionCoba1/test/0.wav'
                waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                waveFile.setnchannels(CHANNELS)
                waveFile.setsampwidth(audio.get_sample_size(FORMAT))
                waveFile.setframerate(RATE)
                waveFile.writeframes(b''.join(frames))
                waveFile.close()
                nfiltbank = 12
                codebooks_mfcc = training(nfiltbank)
                directory = os.getcwd() + '/test';
                nCorrect_MFCC = 0
                fname = '/0.wav'
                print(fname)
                (fs, s) = read(directory + fname)
                mel_coefs = mfcc(s, fs, nfiltbank)
                sp_mfcc = minDistance(mel_coefs, codebooks_mfcc)
                if sp_mfcc >= 1:
                    kursor = db.cursor()
                    kursor.execute("select * from datadosen where id_dosen = %s" % (sp_mfcc))
                    orang = kursor.fetchall()
                    for orang in orang:
                        kursor.execute("INSERT INTO penelpon (nama,nik,no_telp,waktu,tanggal) VALUES (%s,%s,%s,%s,%s)",
                        [orang[1], orang[2], orang[3], datetime.datetime.now().time(), datetime.datetime.now().date()])
                        db.commit()
                elif sp_mfcc == 0:
                    nilai = "Tidak Dikenal"
                    kursor.execute("INSERT INTO penelpon (nama,nik,no_telp,waktu,tanggal) VALUES (%s,%s,%s,%s,%s)",
                                   [nilai, nilai, nilai, datetime.datetime.now().time(), datetime.datetime.now().date()])
                    db.commit()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = database_penelpon()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())