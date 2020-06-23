# Introduction

This repository contains code aligned with an entry to DCASE 2020 Challenge Task 1B.

Information on the DCASE challenge can be found here: http://dcase.community/challenge2020/task-acoustic-scene-classification

A technical report on the challenge entry can be found at https://github.com/McDonnell-Lab/DCASE2020_Task1b/blob/master/TechnicalReport/McDonnell_USA_Task1b.technical_report.pdf

# Installation

For this repository (https://github.com/McDonnell-Lab/DCASE2020_Task1b), the following are strongly recommended:

1. Ubuntu >= 16.04 
2. a working high end GPU with >= 11 GB RAM and drivers capable of running tensorflow 1.13 or higher 
3. use of Anaconda 
4. use of jupyter notebooks

Example instructions for installing anaconda are here: https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04

This repository contains a script that can automatically download and unzip the *Task 1b* challenge data.

You must have anaconda installed on linux for the script to work.

For example, first download this repositor, e.g. go to https://github.com/McDonnell-Lab/DCASE2020_Task1b and click on "Download zip"  then copy to your desired location and unzi.

Or on ubuntu run: 

>> git clone https://github.com/McDonnell-Lab/DCASE2020_Task1b.git

Open a terminal and run:

>> bash requirements.sh

This will create a new anaconda environment, and download packages needed.

The script makes use of this tool: https://gitlab.com/dvolgyes/zenodo_get for  downloading.

After this has finished, run

>> conda activate DCASE2020

To inspect and download data manually:

-download Task 1b data from here: https://zenodo.org/record/3670185#.XmW-ehdLfUp

# Training and validation

-first run Task1b_make_spectrograms.ipynb
-next try running Source/Notebooks/Task1b_for_Github_train_and_validate.ipynb
-for full "leave one city out" (LOCO) cross-validation, run Task1b_final_design_LOCO.ipynb

