from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# a = np.load('toyData/valData.npy')

a = np.random.randn(10,3,25) * 50



# index = 50

"""
numframes = 100
numpoints = 25

fig = plt.figure()
scat = plt.scatter()

ani = animation.FuncAnimation(fig, update_plot, frames=xrange(numframes),
                                  fargs=(color_data, scat))
plt.show()


def update_plot(i, data, scat):
    scat.set_array(data[i])
    return scat,

Look at : 
	https://stackoverflow.com/questions/9401658/how-to-animate-a-scatter-plot
"""


for i in range(1):   ####### <<<============== 300
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	# xs = a[index, 0, i, :]
	# ys = a[index, 1, i, :]
	# zs = a[index, 2, i, :]
	xs = a[0,:]
	ys = a[1,:]
	zs = a[2,:]
	#for j in range(25):
		#plt.text(a[5,0,i,j],a[5,1,i,j], a[5,2,i,j], str(j)) 	
	ax.scatter(xs, ys, zs, c = 'r', marker = 'o',alpha=0.5)
	for j in range(25):
		ax.text(xs[j],ys[j],zs[j], '%s' % (j))
	plt.show()

	print("Frame%d"%(i))