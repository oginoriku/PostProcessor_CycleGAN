import torch
import os
import pickle
import numpy as np
import random
from pesq import pesq
from pystoi import stoi
import scipy.signal as sp
from scipy.io import wavfile
import logging
import oct2py

logging.basicConfig(level=logging.ERROR)
oc = oct2py.Oct2Py(logger=logging.getLogger())

COMPOSITE = "composite.m"

def my_makedir(SAMPLE_DIR):
    if not os.path.exists(SAMPLE_DIR):
        os.makedirs(SAMPLE_DIR)

def loadPickle(fileName):
    with open(fileName, mode="rb") as f:
        return pickle.load(f)

def dumpPickle(fileName, obj):
    with open(fileName, mode="wb") as f:
        pickle.dump(obj, f)

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

def str2bool(v):
    return v.lower() in ('true')

def cal_PESQ(clean_stft_data, stft_data):
    pesq_mos = 0
    if len(clean_stft_data)!=len(stft_data):
        raise Exception
    for i in range(len(stft_data)):
        #逆フーリエ変換
        _, clean_data=sp.istft(clean_stft_data[i].cpu().numpy(), fs=16000, window="hann", nperseg=512, noverlap=256)
        _, data=sp.istft(stft_data[i].cpu().numpy(), fs=16000, window="hann", nperseg=512, noverlap=256)
        pesq_mos += pesq(16000, clean_data, data, 'wb')
    PESQ_average = pesq_mos/len(stft_data)
    return PESQ_average

def cal_PESQ_fake(clean_stft_data, ase_stft_data, fake_stft_data, device):
    pesq_mos = 0
    for i in range(len(clean_stft_data)):
        #入力信号の振幅を取得
        amp=torch.abs(ase_stft_data[i])
        #入力信号の位相を取得
        phase=ase_stft_data[i]/torch.maximum(amp,torch.full(ase_stft_data[i].size(),fill_value=1.e-20).to(device))
        #出力信号の振幅に入力信号の位相をかける
        y_fake = fake_stft_data[i] * phase
        #逆フーリエ変換
        _, clean_data=sp.istft(clean_stft_data[i].cpu().numpy(), fs=16000, window="hann", nperseg=512, noverlap=256)
        _, fake_data=sp.istft(y_fake.detach().cpu().numpy(), fs=16000, window="hann", nperseg=512, noverlap=256)
        pesq_mos += pesq(16000, clean_data, fake_data, 'wb')

    PESQ_average = pesq_mos/len(clean_stft_data)

    return PESQ_average

def cal_pesq(clean: str, enhanced: str):
    sr1, clean_wav = wavfile.read(clean)
    sr2, enhanced_wav = wavfile.read(enhanced)
    assert sr1 == sr2
    mode = "nb" if sr1 < 16000 else "wb"
    return pesq(sr1, clean_wav, enhanced_wav, mode)

def composite(clean: str, enhanced: str):
    pesq_score = cal_pesq(clean, enhanced)
    csig, cbak, covl, ssnr = oc.feval(COMPOSITE, clean, enhanced, nout=4)
    csig += 0.603 * pesq_score
    cbak += 0.478 * pesq_score
    covl += 0.805 * pesq_score
    return pesq_score, csig, cbak, covl, ssnr

def cal_STOI_PESQ_CSIG_CBAK_COVL_SSNR(clean_stft_data, ase_stft_data, fake_stft_data, device):
    STOI_ase = 0
    PESQ_ase = 0
    CSIG_ase = 0
    CBAK_ase = 0
    COVL_ase = 0
    SSNR_ase = 0
    STOI_fake = 0
    PESQ_fake = 0
    CSIG_fake = 0
    CBAK_fake = 0
    COVL_fake = 0
    SSNR_fake = 0
    for i in range(len(clean_stft_data)):
        #入力信号の振幅を取得
        amp=torch.abs(ase_stft_data[i])
        #入力信号の位相を取得
        phase=ase_stft_data[i]/torch.maximum(amp,torch.full(ase_stft_data[i].size(),fill_value=1.e-20).to(device))
        #出力信号の振幅に入力信号の位相をかける
        y_fake = fake_stft_data[i] * phase
        #逆フーリエ変換
        _, clean_data=sp.istft(clean_stft_data[i].cpu().numpy(), fs=16000, window="hann", nperseg=512, noverlap=256)
        _, ase_data=sp.istft(ase_stft_data[i].cpu().numpy(), fs=16000, window="hann", nperseg=512, noverlap=256)
        _, fake_data=sp.istft(y_fake.detach().cpu().numpy(), fs=16000, window="hann", nperseg=512, noverlap=256)
        STOI_ase += stoi(clean_data, ase_data, 16000, extended=False)
        STOI_fake += stoi(clean_data, fake_data, 16000, extended=False)
        wavfile.write("clean.wav", 16000, clean_data)
        wavfile.write("ase.wav", 16000, ase_data)
        wavfile.write("fake.wav", 16000, fake_data)
        eval_index_ase = composite("clean.wav", "ase.wav")
        eval_index_fake = composite("clean.wav", "fake.wav")
        PESQ_ase += eval_index_ase[0]
        CSIG_ase += eval_index_ase[1]
        CBAK_ase += eval_index_ase[2]
        COVL_ase += eval_index_ase[3]
        SSNR_ase += eval_index_ase[4]
        PESQ_fake += eval_index_fake[0]
        CSIG_fake += eval_index_fake[1]
        CBAK_fake += eval_index_fake[2]
        COVL_fake += eval_index_fake[3]
        SSNR_fake += eval_index_fake[4]

    STOI_ase = STOI_ase/len(clean_stft_data)
    PESQ_ase = PESQ_ase/len(clean_stft_data)
    CSIG_ase = CSIG_ase/len(clean_stft_data)
    CBAK_ase = CBAK_ase/len(clean_stft_data)
    COVL_ase = COVL_ase/len(clean_stft_data)
    SSNR_ase = SSNR_ase/len(clean_stft_data)
    STOI_fake = STOI_fake/len(clean_stft_data)
    PESQ_fake = PESQ_fake/len(clean_stft_data)
    CSIG_fake = CSIG_fake/len(clean_stft_data)
    CBAK_fake = CBAK_fake/len(clean_stft_data)
    COVL_fake = COVL_fake/len(clean_stft_data)
    SSNR_fake = SSNR_fake/len(clean_stft_data)

    return STOI_ase, PESQ_ase, CSIG_ase, CBAK_ase, COVL_ase, SSNR_ase, STOI_fake, PESQ_fake, CSIG_fake, CBAK_fake, COVL_fake, SSNR_fake