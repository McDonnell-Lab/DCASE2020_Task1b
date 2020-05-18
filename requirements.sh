#!/usr/bin/env bash

conda init
source ~/.bashrc
conda create -n DCASE2020 python=3.7 -y

eval "$(conda shell.bash hook)"
conda activate DCASE2020

conda install -y -c anaconda tensorflow-gpu=1.15

pip install zenodo_get

conda activate DCASE2020

#cd /RAID5/mdmcdonn-data/DCASE2020/Temp/

mkdir Data
cd Data

zenodo_get https://zenodo.org/record/3670185#.XmW-ehdLfUp