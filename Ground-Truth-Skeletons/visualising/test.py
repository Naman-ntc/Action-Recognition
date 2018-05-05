import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import juggle_axes

"""
If you're on OSX and using the OSX backend, you'll need to change blit=True to blit=False in the FuncAnimation 
initialization below. The OSX backend doesn't fully support blitting. The performance will suffer, but the 
example should run correctly on OSX with blitting disabled
https://stackoverflow.com/questions/9401658/how-to-animate-a-scatter-plot/
"""


def main():
	global ax
	numframes = 100
	numpoints = 10
	#color_data = np.random.random((numframes, numpoints))
	data = np.random.random((numframes, 3, numpoints))*50

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	
	
	scat = ax.scatter(data[0,0,:], data[0,1,:], data[0,2,:], c='r', marker = 'o',alpha=0.5, s=100)

	ani = animation.FuncAnimation(fig, update_plot, frames= range(numframes),
	                              fargs=(data, scat))
	plt.show()


	# for i in range(100):
	# 	scat._offsets3d = juggle_axes(data[i,0,:], data[i,1,:], data[i,2,:], 'z')
	# 	for j in range(25):
	# 		ax.text(data[i,0,:], data[i,1,:], data[i,2,:], '%s' % (j)) 
	# 	fig.canvas.draw()	
def update_plot(i, data, scat):
	global ax
	scat._offsets3d = juggle_axes(data[i,0,:], data[i,1,:], data[i,2,:], 'z')
	# for j in range(25):
	# 	ax.text(data[i,0,:], data[i,1,:], data[i,2,:], '%s' % (j))
	####### THIS ^^^ IS NOT WORKING with ANIMATION :(
	return scat

main()