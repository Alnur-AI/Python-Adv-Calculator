import numpy as np
from math import *

# BORDERS
x1 = -3.0
x2 = 3.0
## x1 < x < x2
y1 = -3.0
y2 = 3.0
## y1 < y < y2

# m for y and n for x
n = 100
m = 100

def f(x, y):
    return 1/x

#x and y grid
x = np.linspace(x1, x2, n+1)
y = np.linspace(y1, y2, m+1)

#fxy definition
fxy = np.zeros((n+1, m+1))
for i in range(n+1):
	for j in range(m+1):
		fxy[i][j] = f(x[i],y[j])
#min and max of f
fmin = fxy.min
fmax = fxy.max

#///OUTPUT//////////////OUTPUT/////////////////////////////////////
import matplotlib.pyplot as plt
import matplotlib as mpl

from mpl_toolkits.mplot3d.axes3d import Axes3D

fig = plt.figure(figsize = (6, 6))
ax = fig.add_subplot(1, 1, 1, projection = '3d')

X, Y = np.meshgrid(x, y)

surf = ax.plot_surface(X, Y, fxy, rstride = 5, cstride = 5)




plt.show()


