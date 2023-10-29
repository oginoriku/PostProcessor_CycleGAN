import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
import os
import glob
import random
from tqdm import tqdm
import scipy.signal as sp
import pickle

def my_makedir(SAMPLE_DIR):
    if not os.path.exists(SAMPLE_DIR):
        os.makedirs(SAMPLE_DIR)

def dumpPickle(fileName, obj):
    with open(fileName, mode="wb") as f:
        pickle.dump(obj, f)

#音の長さを変更
def length_adjust(src,length):
    while len(src)<length:
        src = np.concatenate([src,np.zeros((length-np.shape(src)[0]))],axis=0)
    src = src[:length]
    return src

def simulate_src(audio, room_dim, fs, absorption, max_order, mic_locs, src_loc):
    #部屋の宣言
    room = pra.ShoeBox(
            room_dim, fs=fs, absorption=absorption, max_order=max_order
            )

    # マイクの追加
    room.add_microphone_array(mic_locs)
    #音源の追加
    room.add_source(src_loc, signal=audio)
    room.simulate()
    return room.mic_array.signals

#与えられたSNRに調整
def adjust_SNR(speech,noise,snr):
    speech_power = np.sum(np.square(speech))
    noise_power = np.sum(np.square(noise))
    alpha = np.sqrt((noise_power/speech_power)*(10**(snr / 10)))
    speech = alpha*speech
    return speech,alpha

def norm16bit(mix, target, noise, clean):
    if (np.max(mix) > 32767):
        mix_max = 32767/np.max(mix)
        mix = mix * mix_max
        target = target * mix_max
        noise = noise * mix_max
        clean = clean * mix_max
    return mix, target, noise, clean

#エリア収音
def area_sound_enhansment(mix1, mix2, nperseg, noverlap):
    data1_l = mix1[0]
    data1_r = mix1[1]
    data2_l = mix2[0]
    data2_r = mix2[1]
    #短時間フーリエ変換
    f1_l, t1_l, stft_data1_l = sp.stft(data1_l, fs=16000, window="hann", nperseg=nperseg, noverlap=noverlap)
    f1_r, t1_r, stft_data1_r = sp.stft(data1_r, fs=16000, window="hann", nperseg=nperseg, noverlap=noverlap)
    f2_l, t2_l, stft_data2_l = sp.stft(data2_l, fs=16000, window="hann", nperseg=nperseg, noverlap=noverlap)
    f2_r, t2_r, stft_data2_r = sp.stft(data2_r, fs=16000, window="hann", nperseg=nperseg, noverlap=noverlap)
    #NBF
    nbf1 = np.abs(stft_data1_l - stft_data1_r)
    nbf2 = np.abs(stft_data2_l - stft_data2_r)
    #入力信号の振幅を取得
    amp=np.abs(stft_data1_l)
    #入力信号の振幅の1%を下回らないようにする
    eps=0.01*amp
    #係数aを作成, 高周波数帯に向かうにつれて小さい値となる
    a = np.array([])
    for i in range(len(stft_data1_l)):
        a_in = np.array([])
        for j in range(len(stft_data1_l[0])):
            #a_in = np.append(a_in, 0.1*len(stft_data1_l)/(i+1))
            a_in = np.append(a_in, 0.45*len(stft_data1_l)/(i+1))
        if i == 0:
            a = np.append(a, a_in)
        else:
            a = np.vstack([a, a_in])
    b = a
    x = np.maximum(amp-(a*nbf1)-(b*nbf2), eps)
    #入力信号の位相を取得
    phase=stft_data1_l/np.maximum(amp,1.e-20)
    #出力信号の振幅に入力信号の位相をかける
    ase_data = x * phase
    return ase_data

def make_sound(target_PATH, noise_PATH, snr, fs, fn, absorption, max_order, room_dim, mic_locs1, mic_locs2, target_loc, noise_loc, rt):
    #ファイルを読み込む
    fs1, target = wavfile.read(target_PATH)#目的音
    fs2, noise = wavfile.read(noise_PATH)#雑音
    #音源の長さを変える
    target = length_adjust(target, fn)
    noise = length_adjust(noise, fn)
    #シミュレート
    target_1 = simulate_src(target, room_dim, fs, absorption, max_order, mic_locs1, target_loc)
    noise_1 = simulate_src(noise, room_dim, fs, absorption, max_order, mic_locs1, noise_loc)
    target_2 = simulate_src(target, room_dim, fs, absorption, max_order, mic_locs2, target_loc)
    noise_2 = simulate_src(noise, room_dim, fs, absorption, max_order, mic_locs2, noise_loc)
    clean = simulate_src(target, room_dim, fs, 0.0, 0, mic_locs1, target_loc)
    #音源の長さを変える
    target_1 = target_1[0:, :fn]
    noise_1 = noise_1[0:, :fn]
    target_2 = target_2[0:, :fn]
    noise_2 = noise_2[0:, :fn]
    clean = clean[0:, :fn]

    #マイク1_lのSNRを調整し, それに合わせてスケールを調整
    _, alpha = adjust_SNR(target_1[0], noise_1[0], snr)
    target_1 = alpha*target_1
    target_2 = alpha*target_2
    clean = alpha*clean

    #足し合わせて混合音にする
    mix_1 = target_1 + noise_1
    mix_2 = target_2 + noise_2
    #最大値が32767を超えると音割れが生じるため, 調整
    mix_1, target_1, noise_1, clean = norm16bit(mix_1, target_1, noise_1, clean)
    mix_2, target_2, noise_2, clean = norm16bit(mix_2, target_2, noise_2, clean)

    return mix_1, mix_2, target_1, clean

def main(number, data_number, timit_dir, save_root_dir, mode):
    ###################
    #各設定を宣言しておく#
    ##################
    d_mic = 0.03 #2chマイクのマイク同士の距離[m]
    d_micarray = 0.4 #２つの2chマイクの距離[m]
    angle_micarray = 25 #2chマイクの角度[°]
    x_delta = (d_mic/2)*np.cos(np.radians(angle_micarray))
    y_delta = (d_mic/2)*np.sin(np.radians(angle_micarray))
    x_room = 7
    y_room = 7
    z_room = 3
    room_dim = [x_room, y_room+(y_delta*2), z_room] #部屋の大きさ[m, m, ms]
    fs = 16000 #サンプリング周波数[Hz]
    fn = 48000 #音の長さ[フレーム数]
    nperseg=512
    noverlap=256
    rt60 = [0.2, 0.5] #残響時間[s]
    d_noise = 0.8 #妨害音の距離[m]
    angle_noise_list = [80, 90, 100] #妨害音の角度[°]
    snr_list = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5] #SNR[dB]

    mic1_x = ((x_room-d_micarray)/2)   #マイクアレイ1の中心位置[x]: 3.4
    mic2_x = ((x_room+d_micarray)/2)   #マイクアレイ2の中心位置[x]: 3.6
    # マイクロホンアレイの位置
    mic_locs1 = np.c_[
        [mic1_x-x_delta, (y_room/2)-y_delta, z_room/2], [mic1_x+x_delta, (y_room/2)+y_delta, z_room/2]# mic 1, 2
    ]
    mic_locs2 = np.c_[
        [mic2_x-x_delta, (y_room/2)+y_delta, z_room/2], [mic2_x+x_delta, (y_room/2)-y_delta, z_room/2]# mic 1, 2
    ]
    #音源の位置
    target_loc = [ x_room/2, (y_room/2)-((d_micarray/2)/np.tan(np.radians(angle_micarray))), z_room/2] #2つのマイクの指向性が交差する位置
    ##################
    ##################

    if mode == "train":
        save_dir = os.path.join(save_root_dir, "train_data")
        corpus_dir_PATH = os.path.join(timit_dir, "train")
        corpus_dir_list = [filename for filename in os.listdir(corpus_dir_PATH) if not filename.startswith('.')]
        corpus_data_PATH_list = []
        for f in corpus_dir_list[:231]:
            corpus_data_PATH = glob.glob(os.path.join(corpus_dir_PATH, f, "*.wav"))
            corpus_data_PATH_list.extend(corpus_data_PATH)
    if mode == "clean":
        save_dir = os.path.join(save_root_dir, "clean_data")
        corpus_dir_PATH = os.path.join(timit_dir, "train")
        corpus_dir_list = [filename for filename in os.listdir(corpus_dir_PATH) if not filename.startswith('.')]
        corpus_data_PATH_list = []
        for f in corpus_dir_list[231:]:
            corpus_data_PATH = glob.glob(os.path.join(corpus_dir_PATH, f, "*.wav"))
            corpus_data_PATH_list.extend(corpus_data_PATH)
    elif mode == "test":
        save_dir = os.path.join(save_root_dir, "test_data")
        corpus_dir_PATH = os.path.join(timit_dir, "test")
        corpus_dir_list = [filename for filename in os.listdir(corpus_dir_PATH) if not filename.startswith('.')]
        corpus_data_PATH_list = []
        print(len(corpus_dir_list))
        for f in corpus_dir_list[:100]:
            corpus_data_PATH = glob.glob(os.path.join(corpus_dir_PATH, f, "*.wav"))
            corpus_data_PATH_list.extend(corpus_data_PATH)

    my_makedir(save_dir)
    my_makedir(os.path.join(save_dir, "time_domain"))
    print(len(corpus_data_PATH_list))
    snr_check_list = []
    for i, target_data_PATH in enumerate(tqdm(corpus_data_PATH_list)):
        random.seed(i+data_number)
        snr = random.choice(snr_list)
        rt = random.choice(rt60)
        absorption, max_order = pra.inverse_sabine(rt, room_dim)
        angle_noise = random.choice(angle_noise_list)
        noise_loc = [x_room/2+(d_noise*np.cos(np.radians(angle_noise))), (y_room/2)-(d_noise*np.sin(np.radians(angle_noise))), z_room/2]
        noise_data_PATH_cands = random.sample(corpus_data_PATH_list, 11)
        snr_check_list.append(snr)

        for noise_data_PATH_cand in noise_data_PATH_cands:
            target_data_PATH_split = target_data_PATH.split('/')
            noise_data_PATH_cand_split = noise_data_PATH_cand.split('/')
            if target_data_PATH_split[-2]!=noise_data_PATH_cand_split[-2] and target_data_PATH_split[-1]!=noise_data_PATH_cand_split[-1]:
                noise_data_PATH = noise_data_PATH_cand
                break
        try:
            mix1, mix2, target1, clean = make_sound(target_data_PATH, noise_data_PATH, snr, fs, fn, absorption, max_order, room_dim, mic_locs1, mic_locs2, target_loc, noise_loc, rt)
        except Exception:
            print("エラー")
            continue

        ase_stft_data = area_sound_enhansment(mix1, mix2, nperseg, noverlap)
        t, ase_data = sp.istft(ase_stft_data, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)

        cleanR_data = target1[0][:fn]
        clean_data = clean[0][:fn]
        mix_data = mix1[0][:fn]
        ase_data = ase_data[:fn]

        _, _, clean_stft_data = sp.stft(clean_data, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
        d = {'ase_src': ase_stft_data, 'clean_src': clean_stft_data}
        td = {'clean_data': clean_data, 'mix_data': mix_data, 'ase_data': ase_data, 'cleanR_data':cleanR_data}
        dumpPickle(os.path.join(save_dir, f"data{data_number}.pickle"), d)
        dumpPickle(os.path.join(save_dir, f"time_domain/data{data_number}.pickle"), td)
        data_number += 1
    return data_number

########################
#main
########################
save_root_dir = "save root directory path (ex. /root/dataset/MUBASE)"
timit_dir ="TIMIT corpus directory path (ex. /DB/TIMIT/by_speaker16000)"

my_makedir(save_root_dir)

# train data
mode = "train"
data_number = 0
for number in [0, 1, 2, 3, 4]:
    print(f"{number}週目")
    data_number = main(number, data_number, timit_dir, save_root_dir, mode)

# clean data
mode = "clean"
data_number = 0
for number in [0, 1, 2, 3, 4]:
    print(f"{number}週目")
    data_number = main(number, data_number, timit_dir, save_root_dir, mode)

# test data
mode = "test"
data_number = 0
for number in [0]:
    print(f"{number}週目")
    data_number = main(number, data_number, timit_dir, save_root_dir, mode)