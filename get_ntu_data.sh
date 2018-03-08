#!/bin/bash

clear
mkdir -v datasets
cd datasets
mkdir -v NTU
cd NTU
echo "Downloading the nturgbd_skeletons zip file"
wget http://rose1.ntu.edu.sg/Datasets/actionRecognition/download/nturgbd_skeletons.zip
sudo apt-get install unzip
unzip nturgbd_skeletons.zip
rm nturgbd_skeletons.zip
echo "Downloading dependencies from github"
wget https://raw.githubusercontent.com/yysijie/st-gcn/master/tools/ntu_read_skeleton.py
wget https://raw.githubusercontent.com/yysijie/st-gcn/master/tools/ntu_gendata.py
wget https://raw.githubusercontent.com/shahroudy/NTURGB-D/master/Matlab/samples_with_missing_skeletons.txt
echo "IMPORTANT"
echo "Make changes of line 78 of the program, change w to wb"
nano ntu_gendata.py
echo "Now pickling time!!"
python ntu_gendata.py --data_path nturgb+d_skeletons --ignored_sample_path samples_with_missing_skeletons.txt --out_folder .
