import numpy as np
import os
from tqdm import tqdm
from Generator import U_Net
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from utils import my_makedir, loadPickle, dumpPickle, cal_STOI_PESQ_CSIG_CBAK_COVL_SSNR


n_cuda = 0
NUMBER_Test = 1000
batch_size = 1
test_data_path = "test_data directory path (ex. /root/dataset/MUBASE/test_data)"
model_G_path = "trained model path (ex. /root/results/sample/G_modelA2B_245.pth)"
save_dir_path = "save directory path (ex. /root/results/CycleGAN)"

my_makedir(os.path.join(save_dir_path, "log"))
log_path = os.path.join(save_dir_path, "log")

cudnn.deterministic = True

#データセットをまとめる, バッチに分ける
print("=====================================データセット読み込み開始============================================")
testset = []
for i in tqdm(range(NUMBER_Test)):
    testdata = loadPickle(os.path.join(test_data_path, f"data{i}.pickle"))
    testset.append(testdata)
testset = np.array(testset)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

print("trainset.shape: ", testset.shape)
print("バッチサイズ: ", batch_size)
print("イテレーション数: ", len(testloader))
print("=====================================データセット読み込み終了============================================")

#モデルをGPUに転送
device = torch.device(f'cuda:{n_cuda}' if torch.cuda.is_available() else 'cpu')
print(f'using: {device}')

#main
#テストを行う
STOI_fake_data = []
PESQ_fake_data = []
CSIG_fake_data = []
CBAK_fake_data = []
COVL_fake_data = []
SSNR_fake_data = []

netG = U_Net().to(device)
netG.eval()

print("=====================================start test============================================")
model_path = model_G_path
netG.load_state_dict(torch.load(model_path))

PESQ_ase = 0
STOI_ase = 0
CSIG_ase = 0
CBAK_ase = 0
COVL_ase = 0
SSNR_ase = 0
PESQ_fake = 0
STOI_fake = 0
CSIG_fake = 0
CBAK_fake = 0
COVL_fake = 0
SSNR_fake = 0

bar = tqdm(total = len(testloader))

for count, d in enumerate(testloader, 1):
    bar.update(1)
    clean_src = d["clean_src"].to(device, non_blocking=True)
    ase_src = d["ase_src"].to(device, non_blocking=True)
    #clean_src = torch.abs(d["clean_src"]).to(device, dtype=torch.float, non_blocking=True)

    fake_src = netG(torch.abs(ase_src).to(device, dtype=torch.float, non_blocking=True))
    eval_index = cal_STOI_PESQ_CSIG_CBAK_COVL_SSNR(clean_src, ase_src, fake_src, device)
    STOI_ase += eval_index[0]
    PESQ_ase += eval_index[1]
    CSIG_ase += eval_index[2]
    CBAK_ase += eval_index[3]
    COVL_ase += eval_index[4]
    SSNR_ase += eval_index[5]
    STOI_fake += eval_index[6]
    PESQ_fake += eval_index[7]
    CSIG_fake += eval_index[8]
    CBAK_fake += eval_index[9]
    COVL_fake += eval_index[10]
    SSNR_fake += eval_index[11]

STOI_average_ase = STOI_ase/len(testloader)
PESQ_average_ase = PESQ_ase/len(testloader)
CSIG_average_ase = CSIG_ase/len(testloader)
CBAK_average_ase = CBAK_ase/len(testloader)
COVL_average_ase = COVL_ase/len(testloader)
SSNR_average_ase = SSNR_ase/len(testloader)

STOI_average_fake = STOI_fake/len(testloader)
PESQ_average_fake = PESQ_fake/len(testloader)
CSIG_average_fake = CSIG_fake/len(testloader)
CBAK_average_fake = CBAK_fake/len(testloader)
COVL_average_fake = COVL_fake/len(testloader)
SSNR_average_fake = SSNR_fake/len(testloader)

print("-----------------STOI----------------")
print("STOI_ase: ", STOI_average_ase)
print("STOI_fake: ", STOI_average_fake)
print("-----------------PESQ----------------")
print("PESQ_ase: ", PESQ_average_ase)
print("PESQ_fake: ", PESQ_average_fake)
print("-----------------CSIG----------------")
print("CSIG_ase: ", CSIG_average_ase)
print("CSIG_fake: ", CSIG_average_fake)
print("-----------------CBAK----------------")
print("CBAK_ase: ", CBAK_average_ase)
print("CBAK_fake: ", CBAK_average_fake)
print("-----------------COVL----------------")
print("COVL_ase: ", COVL_average_ase)
print("COVL_fake: ", COVL_average_fake)
print("-----------------SSNR----------------")
print("SSNR_ase: ", SSNR_average_ase)
print("SSNR_fake: ", SSNR_average_fake)

STOI_fake_data.append(STOI_average_fake)
PESQ_fake_data.append(PESQ_average_fake)
CSIG_fake_data.append(CSIG_average_fake)
CBAK_fake_data.append(CBAK_average_fake)
COVL_fake_data.append(COVL_average_fake)
SSNR_fake_data.append(SSNR_average_fake)

dumpPickle(os.path.join(log_path, "STOI_fake.pickle"), STOI_fake_data)
dumpPickle(os.path.join(log_path, "PESQ_fake.pickle"), PESQ_fake_data)
dumpPickle(os.path.join(log_path, "CSIG_fake.pickle"), CSIG_fake_data)
dumpPickle(os.path.join(log_path, "CBAK_fake.pickle"), CBAK_fake_data)
dumpPickle(os.path.join(log_path, "COVL_fake.pickle"), COVL_fake_data)
dumpPickle(os.path.join(log_path, "SSNR_fake.pickle"), SSNR_fake_data)
