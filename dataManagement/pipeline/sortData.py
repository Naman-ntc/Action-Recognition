import os
import sys

training_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]


'''
 for filename in os.listdir(data_path):
 if filename in ignored_samples:
 continue
 action_class = int(
 filename[filename.find('A') + 1:filename.find('A') + 4])
 subject_id = int(
 filename[filename.find('P') + 1:filename.find('P') + 4])
 camera_id = int(
filename[filename.find('C') + 1:filename.find('C') + 4])

'''

os.system("ls > output")
file = open("output", 'r')
zips = ""
for chunk in file:
	zips+=chunk

file.close()
zips = zips.split("\n")
zips = [x for x in zips if x[-3:] == "zip"]
for file in zips:
	print("Now working on", file)
	os.system("unzip -q " + file)
	print("Unzipped")
	os.system("ls nturgb+d_rgb > output2")
	file = open("output2", 'r')
	videos = ""
	for chunk in file:
		videos+= chunk
	file.close()
	videos = videos.split("\n")
	for video in videos:
		try:
			subject = int(video[video.find('P') + 1 : video.find('P') + 4])
			if subject in training_subjects:
				os.system("mv nturgb+d_rgb/" + video + " xsub/train")
			else:
				os.system("mv nturgb+d_rgb/" + video + " xsub/val")
		except:
			pass
	os.system("ls nturgb+d_rgb > output2")
	file = open("output2", 'r')
	currentContents = ""
	for chunk in file:
		currentContents += chunk
	print("According to me, we are done with this zip")
	print("The contents of the folder are currently: ")
	print(currentContents)
	file.close()
	os.system("rm -rf nturgb+d_rgb")

