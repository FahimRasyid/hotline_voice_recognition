from __future__ import division
from scipy.signal import hamming
from scipy.fftpack import fft, fftshift, dct
from scipy.io.wavfile import read
import sys
import numpy as np
import os
import pyaudio
import wave
import MySQLdb
import datetime

#Koneksi Database
con = None
db = MySQLdb.connect("127.0.0.1", "root", "", "pengenalsuara")
kursor = db.cursor()

# Mengambil nilai id terbesar
nilai = kursor.execute("Select * from pegawai")
ambil = kursor.fetchall()
id = len(ambil)

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
def training(nfiltbank, id_pegawai):
    nSpeaker = 5 #id_pegawai
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
    codebooks = np.empty((2, nfiltbank, nCentroid))
    mel_coeff = np.empty((2, nfiltbank, 68))
    for i in range(2):
        fname = '/' + str(i + 2) + '.wav'
        (fs, s) = read(directory_train + fname)
        mel_coeff[i, :, :] = mfcc(s, fs, nfiltbank)[:, 0:68]
        codebooks[i, :, :] = lbg(mel_coeff[i, :, :], nCentroid)
    return codebooks_mfcc

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
    print (speaker)
    return speaker

def record(id_masuk):
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 3
    audio = pyaudio.PyAudio()
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, frames_per_buffer=CHUNK)
    print('recording...')
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
    WAVE_OUTPUT_FILENAME = 'C:/Users/asus/Documents/NetBeansProjects/bismillah/src/test/' + str(id_masuk+1) + '.wav'
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
        
def test(id_masuk, id_pegawai):
    nfiltbank = 12
    codebooks_mfcc = training(nfiltbank, id_pegawai)
    directory = os.getcwd() + '/test';
    nCorrect_MFCC = 0
    fname = '/' + str(id_masuk+1) + '.wav'
    print(fname)
    (fs, s) = read(directory + fname)
    mel_coefs = mfcc(s, fs, nfiltbank)
    sp_mfcc = minDistance(mel_coefs, codebooks_mfcc)
    print (sp_mfcc)
    if sp_mfcc >= 1:
        kursor = db.cursor()
        kursor.execute("select * from pegawai where id_dosen = %s" % (sp_mfcc))
        orang = kursor.fetchall()
        nilai = orang
        for orang in orang:
            kursor.execute("INSERT INTO masuk (id_penelpon,status,nik,nama,waktu,tanggal) VALUES (%s,%s,%s,%s,%s,%s)",
            [str(id_masuk+1), "Dikenal", orang[2], orang[1], datetime.datetime.now().time(), datetime.datetime.now().date()])
            db.commit()
    elif sp_mfcc == 0:
        nilai = "Tidak Dikenal"
        kursor = db.cursor()
        kursor.execute("INSERT INTO masuk (id_penelpon,status,nik,nama,waktu,tanggal) VALUES (%s,%s,%s,%s,%s,%s)",
            [str(id_masuk+1), nilai, nilai, nilai, datetime.datetime.now().time(), datetime.datetime.now().date()])
        db.commit()
    return sp_mfcc

#if __name__ == '__main__':
 #   app.run()