#!/usr/bin/env bash

conda init
source ~/.bashrc
conda create -n DCASE2020 python=3.7 -y

eval "$(conda shell.bash hook)"
conda activate DCASE2020

conda install -y -c anaconda tensorflow-gpu=1.15
conda install -y -c conda-forge librosa

pip install zenodo_get 
pip install SoundFile

mkdir Data
cd Data

#DCASE Task 1b
zenodo_get https://zenodo.org/record/3670185#.XmW-ehdLfUp

for z in *.zip; do unzip "$z"; done