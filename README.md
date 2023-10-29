# PostProcessor CycleGAN

## Data Generation
以下のコマンドを実行してください。
```
python make_dataset.py 
```
**`make_dataset.py` 内の各パラメータについて**  
```
save_root_dir：データセットを保存するディレクトリのパス  
timit_dir：TIMIT corpus（16kHz）のディレクトリパス (ex. /DB/TIMIT/by_speaker16000)
```

データは、学習時のデータロード時間を短縮するために、pickle形式で保存されます。  

## Training
ポストプロセッサーの学習。

以下コマンドを実行してください。
```
bash main.sh
```
**`main.sh` 内の各パラメータについて**  
```
data_path：データセットのパス  
save_path：学習結果を保存するディレクトリのパス  
cuda_num：推論に使用するGPUの番号  
clean_data_num：学習に使うクリーンデータの数  
train_data_num：学習に使うエリア収音データの数  
val_data_num：学習に使う検証データの数  
batch_size：バッチサイズ  
total_epochs：学習エポック数  
```

## Inference
学習済みモデルを用いた推論。

以下コマンドを実行してください。
```
python test.py 
```

**`test.py` 内の各パラメータについて**  
```
n_cuda：推論に使用するGPUの番号  
NUMBER_Test：推論に使うデータ数  
batch_size：バッチサイズ  
test_data_path：データセットのパス  
model_G_path：学習済みモデルのパス  
save_dir_path：保存するディレクトリのパス　 
```

## Ciation
[1] "Design of Discriminators in GAN-Based Unsupervised Learning of Neural Post-Processors for Suppressing Localized Spectral Distortion", Riku Ogino, Kohei Saijo, Tetsuji Ogawa, APSIPA2022   
[https://ieeexplore.ieee.org/document/9979833](https://ieeexplore.ieee.org/document/9979833)