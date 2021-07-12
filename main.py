from __future__ import division
from scipy.io.wavfile import read
from flask import Flask, render_template
from method import record, test
import os
import pyaudio
import wave
import MySQLdb
import datetime

app = Flask(__name__)

#Koneksi Database
con = None
db = MySQLdb.connect("127.0.0.1", "root", "", "pengenalsuara")
kursor = db.cursor()

# Mengambil nilai id pegawai terbesar
def ambil_id_pegawai():
    nilai = kursor.execute("Select * from pegawai")
    ambil = kursor.fetchall()
    id_pegawai = len(ambil)
    return id_pegawai

def ambil_id_masuk():
    kursor = db.cursor()
    kursor.execute("SELECT * From masuk")
    objek = kursor.fetchall() 
    id_masuk = len(objek)
    return id_masuk

    
app = Flask(__name__, static_url_path='')
@app.route('/')
    
def tampil():
    id_pegawai = ambil_id_pegawai()
    id_masuk = ambil_id_masuk()
    record(id_masuk)
    test(id_masuk, id_pegawai)
    #return render_template('utama.html', orang=orang)

if __name__ == "__main__":
    tampil()
