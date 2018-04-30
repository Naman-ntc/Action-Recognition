import os
import sys
import torch
import numpy as np
import cv2
import pickle
from utils.eval import getPreds

trainKeys = pickle.load(open('trainKeys.pkl','rb'))
trainLabels = pickle.load(open('trainLabels.pkl','rb'))

valKeys = pickle.load(open('valKeys.pkl','rb'))
valLabels = pickle.load(open('valLabels.pkl','rb'))

