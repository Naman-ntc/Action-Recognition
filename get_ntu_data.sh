#!/bin/bash

clear
mkdir datasets
cd datasets
mkdir NTU
cd NTU
wget http://rose1.ntu.edu.sg/Datasets/actionRecognition/download/nturgbd_skeletons.zip
sudo apt-get install unzip
unzip nturgbd_skeletons.zip
wget https://raw.githubusercontent.com/yysijie/st-gcn/master/tools/ntu_read_skeleton.py
wget https://raw.githubusercontent.com/yysijie/st-gcn/master/tools/ntu_gendata.py
