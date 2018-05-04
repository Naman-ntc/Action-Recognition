# Action-Recognition
Course Project for CS763 : Computer Vision Course, IIT Bombay

## Abstract

### Problem Statement
Given a Video containing Human body Motion you have to recognize the action agent is performing.

### Solution Approaches
We started with Action Recognition from skeleton estimates of Human Body. 
Given 3D ground truth coordinates of Human Body (obtained from Kinect Cameras) we tried to use LSTMS as well as Temporal Convolutions for learning skeleton representation of Human Activity Recognition.
We also tried fancier LSTMs as well where we projected the 3D coordinates onto x-y plane, y-z plane, z-x plane followed by 1D convolutions and subsequently adding the outputs of the 4 LSTMs (x-y, y-z, z-x, 3D). Additionally we tried variants where we chose three out of the four LSTMs and compared performance among different projections.
Then we moved to Action Recognition from Videos.
We used pretrained Hourglass Network to estimate joints at each frame in videos and used similar LSTMs to perform the task of Action Recognition.

## Dependencies
numpy==1.11.0
pandas==0.17.1
matplotlib==1.5.1
keras==2.1.6
torch==0.4.0

## Instructions

## Results

|						Data 				    |	Classifier									|    Results  (Accuracy)		|
|-----------------------------------------------|-----------------------------------------------|-------------------------------|
| Ground-Truth-Skeleton - 5 classes				|	Single LSTM, 3D coordinates					|	75.5%, 79.5% (train)  		|
| Ground-Truth-Skeleton - 5 classes				|	2-Stacked LSTMs, 3D coordinates 			| 	77.1%, 80.4% (train)  		|
| Ground-Truth-Skeleton - 5 classes				|	3-Stacked LSTMs, 3D coordinates 			| 	77.2%, 85.6% (train)  		|
| Ground-Truth-Skeletons - 49 classes			|	2-Stacked LSTMs, 3D coordinates				|	59.7%, 72.5% (train)		|
| Hourglass-Predicted-Skeletons - 8 classes		|	2-Stacked LSTMs, 3D coordinates				|	81.25% 						|
| Hourglass-Predicted-Skeletons - 8 classes 	|	4 LSTMS, outputs fused after 1D conv        |	############## 	 			|
| Hourglass-Predicted-Skeletons - 49 classes 	|	4 LSTMS, outputs fused after 1D conv        |	############## 	 			|

## Inferences

