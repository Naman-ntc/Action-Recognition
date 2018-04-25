import numpy as np
import cv2
import os

from functions import resizeAndPad, playVideoFromArray, playVideoFromAVI

from ntu_read_skeleton import read_skeleton

class box:
	def __init__(self):
		self.x1 = 1080
		self.x2 = 0
		self.y1 = 1920
		self.y2 = 0

	def __str__(self):
		return "x1:" + str(self.x1) + " x2:" + str(self.x2) + " y1:" + str(self.y1) + " y2:" + str(self.y2) 

	def makeInt(self):
		self.x1 = int(self.x1)
		self.x2 = int(self.x2)
		self.y1 = int(self.y1)
		self.y2 = int(self.y2)

	def extend(self, percentage  = 40):
		fraction = percentage/100.0
		delX = self.x2 - self.x1
		delY = self.y2 - self.y1
		meanX = (self.x1 + self.x2)/2.0
		meanY = (self.y1 + self.y2)/2.0
		delta = max(delX, delY)
		delta += fraction*delta
		self.x1 = max(meanX - delta/2.0, 0)
		self.x2 = min(meanX + delta/2.0, 1080)
		self.y1 = max(meanY - delta/2.0, 0)
		self.y2 = min(meanY + delta/2.0, 1920)
		self.makeInt()
		return int(delta)

os.system("ls > output")
file = open("output", 'r')
videoList = ""
for chunk in file:
	videoList += chunk
file.close()
videoList = videoList.split()
videoList = [x for x in videoList if (x[-3:] == 'avi' and len(x) == 28)]

##### MADE A LIST OF THE FILES IN THE DIRECTORY

file = open("labels", 'w')
count = 0

total = len(videoList)


for video in videoList:
	try:
		label = video[video.find('A')+1:video.find('A') + 4]
		label = int(label) - 1
		cap = cv2.VideoCapture(video)
		bbox = box()
		skeletonFileName = "skeletons/" + video[:-8] + ".skeleton"
		skeleton = read_skeleton(skeletonFileName)
		for frameNo in range(skeleton['numFrame']):
			for jointNo in range(skeleton['frameInfo'][frameNo]['bodyInfo'][0]['numJoint']):
				bbox.x1 = min(bbox.x1, skeleton['frameInfo'][frameNo]['bodyInfo'][0]['jointInfo'][jointNo]['colorY'])
				bbox.x2 = max(bbox.x2, skeleton['frameInfo'][frameNo]['bodyInfo'][0]['jointInfo'][jointNo]['colorY'])
				bbox.y1 = min(bbox.y1, skeleton['frameInfo'][frameNo]['bodyInfo'][0]['jointInfo'][jointNo]['colorX'])
				bbox.y2 = max(bbox.y2, skeleton['frameInfo'][frameNo]['bodyInfo'][0]['jointInfo'][jointNo]['colorX'])

		sideLength = bbox.extend()

		name = "val" + str(count) + ".avi"
		outFile = cv2.VideoWriter(name,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (sideLength, sideLength))


		while (cap.isOpened()):
			ret, frame = cap.read()
			if ret:
				crop = resizeAndPad(frame[bbox.x1:bbox.x2, bbox.y1:bbox.y2, :], (sideLength, sideLength))
				#currentVideo.append(crop)
				outFile.write(crop)
			else:
				break

		outFile.release()

		#np.save(name,currentVideo)

		file.write(str(count)+","+str(label)+"\n")

		if count%10 == 0:
			print(str(count) + " of " + str(total) + " done - " + str(count*100.0/total) + " %")
		
		count = count + 1
	except:
		pass
>>>>>>> e4b335503651c5311f3988e6f424e569d90515d6
	#playVideoFromAVI(name)
#playVideo("S001C001P001R001A001_rgb.avi")

