from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

a = np.load('toyData/valData.npy')

# for i in range(300):
# 	plt.figure()
# 	plt.plot(a[5,0,i,:],a[5,1,i,:],'ro', alpha = 0.5)
# 	for j in range(25):
# 		plt.text(a[5,0,i,j],a[5,1,i,j], str(j))
# 	plt.draw()
# 	plt.show()
# 	#time.sleep(0.002)
# 	#plt.clf()
	
# 	print("Frame%d"%(i))



index = 50

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


for i in range(300):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	xs = a[index, 0, i, :]
	ys = a[index, 1, i, :]
	zs = a[index, 2, i, :]
	#for j in range(25):
		#plt.text(a[5,0,i,j],a[5,1,i,j], a[5,2,i,j], str(j)) 	
	ax.scatter(xs, ys, zs, c = 'r', marker = 'o',alpha=0.5)
	for i in range(25):
		ax.annotate(i,(xs[i],ys[i],zs[i]))
	## If doesnt work check https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point
	## hopefully it works!!	
	plt.show()

	print("Frame%d"%(i))