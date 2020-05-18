Info on the DCASE challenge can be found here: http://dcase.community/challenge2020/task-acoustic-scene-classification

For this repository (https://github.com/McDonnell-Lab/DCASE2020), the following are strongly recommended:

1. Ubuntu >= 16.04 
2. a working high end GPU with >= 11 GB RAM and drivers capabale of running tensorflow 1.13 or higher 
3. use of Anaconda 
4. use of jupyter notebooks

See instructions for installing anaconda here: https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04


This repository contains a script that can automatically download and unzip the *Task 1b* challenge data.

You must have anaconda installed on linux for the script to work.

First, download this repository (e.g. go to https://github.com/McDonnell-Lab/DCASE2020 and click on "Download zip"  then copy to your desired location and unzip. (git clone will only work if you have security keys set up).

Open a terminal and run:

>> bash requirements.sh

This will create a new anaconda environment, and download packages nededed.

The script makes use of this tool: https://gitlab.com/dvolgyes/zenodo_get for  downloading  

After this has finished, run

>> conda activate DCASE2020

If anything goes wrong, ask Mark McDonnell for help

TO inspect and download data manually:

-download Task 1a data from here: https://zenodo.org/record/3670167#.XmW_eRdLfUo
-download Task 1b data from here: https://zenodo.org/record/3670185#.XmW-ehdLfUp

