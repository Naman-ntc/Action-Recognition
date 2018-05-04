import matplotlib.pyplot as plt
import numpy as np

x, y, z = np.random.random((3, 100))

plt.ion()

fig, ax = plt.subplots()
scat = ax.scatter(x, y, c=z, s=200)

for _ in range(20):
    # Change the colors...
    scat.set_array(np.random.random(100))
    # Change the x,y positions. This expects a _single_ 2xN, 2D array
    scat.set_offsets(np.random.random((100,2)))
    fig.canvas.draw()