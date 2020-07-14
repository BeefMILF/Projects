This repository is a short summary of my final solution in a kaggle competition. The model achieved a (MAP@3) of 0.91377 on private and 0.94352 on public. 

# Freesound General-Purpose Audio Tagging Challenge

---

During this challenge, we have to build general-purpose automatic audio tagging system using dataset of audio files of 41 classes of different environment sounds(like guitar or baby's laugh).

[Kaggle](https://www.kaggle.com/c/freesound-audio-tagging/overview) - competition 

## Requirements 

```
# see requirements.txt 
librosa=0.7.2=pypi_0

pytorch=1.7.0.dev20200714=py3.8_cuda10.1.243_cudnn7.6.3_0

pytorch-lightning=0.8.4=pypi_0

cudatoolkit=10.1.243=h6bb024c_0

torchvision=0.5.0=pypi_0
 ```

## Usage 

Steps are presented in `v2/eda.ipynb`:

```bash
# step 1.
!python -W ignore trim_data.py

# step 2.
!python -W ignore extract_features.py

# step 3. 
!python -W ignore runner.py

# step 4. 
!python -W ignore submit.py
```

## Setup

All parameters used for these steps is in `config.py`. 

If you using default config, all data must be in `v2/data/`: 

```
# in v2/data - folder 
 
audio_train 
audio_test 
train.csv
sample_submission.csv

# these files are loaded using kaggle api 
```

### Step 1. 

Removing leading/trailing silence, as it doesn't contain useful information. Trimmed data automatically stored as `audio_train_trim`/`audio_test_trim`

### Step 2. 

Precompute log-mel features from raw data. The delta and accelerate of are also calculated. By concatenating log-mel, delta and accelerate we form matrix of (3 x 64 x n), where n is dependant on audio length(n=150 in experiment).  
In the end it will be made `audio_train_logmel`/`audio_test_logmel`

### Step 3. 

Run the whole training pipeline and make predictions for using gmean then. Architecture and training parameters are placed in `config.py`. 

All checkpoints and test predictions are saved in `tb_logs/arch/fold_n` 

### Step 4. 

Make final blending of predictions and create a `sbm.csv` file for submission 

#### Main training features: 

* 5-folds(first is the best, while other worse in 15-20%)
* In dataloader we randomly cut inputs bigger than n-size(150 in experiment) and pad if shorter
* Second trained model(resnet50), we used in dataloader for shorter inputs an augmentation(freq and time masking) and concatenated with the original, making original inputs twice longer. Then, the previous point repeated
* After training on 30-40 epochs, I retrain model on a validation data for 5-10 epochs and saved separately.
* Best checkpoints were saved using pytorch-lightning + tensorboard, val_acc1 was monitored
* For submission, each model used for making 5 test_dataset predictions and averaged with gmean. 
* After gmean used probability calibration that has no significant impact on final score 

#### Results 



`resnext101_32x4d/fold_1` - gave ~89% (MAP@3) on private
`resnext101_32x4d/fold_1_r` - is a retrained version of previous model and mixing preds gave ~90% (MAP@3) on private

`resnet50/fold_1` and `resnet50/fold_1_r` preds were added to them and gave 91.365% on private  
 