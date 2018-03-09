import numpy as np
import matplotlib.pyplot as plt
import time

a = np.load('temp.npy')

for i in range(300):
	plt.plot(a[5,0,i,:],a[5,1,i,:],'ro', alpha = 0.5)
	for j in range(25):
		plt.text(a[5,0,i,j],a[5,1,i,j], str(j))
	plt.show()
	#time.sleep(0.2)
	print("Frame%d"%(i))