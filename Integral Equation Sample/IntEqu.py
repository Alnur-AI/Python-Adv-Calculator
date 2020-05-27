import numpy as np
from math import *

#input data
b = 17.0 / 16
T = 1.0 / 2
n = 128
eps = 2.5*(10**(-7))
## (1-T <= x <= 1)

#exact solution
def g(x):
    return (b-x) / b / (b-1) * ( (b-2) * np.exp(-x) + np.exp(T-1) )

def phi_a(x):
    return (b-x) * np.exp(-x) / b

#small squares
h = T / n #dx
x = [1-T + i*h for i in range(0,n+1)] #x[i]
x = np.array(x)
#analitical solution
y_a = [phi_a(x[i]) for i in range(0,n+1)]#y_a[i]

#numerical solution
K = 20
phi = np.zeros((K+1,n+1))
for k in range(0,K):
	for i in range(0,n+1):
		A = phi[k][0] / ( 2*(b-x[0]) )
		B = 0
		for j in range(1, i):
			B += phi[k][j] / (b - x[j])
		C = phi[k][i]/(2*(b-x[i]))
		phi[k+1][i] = g(x[i]) - (b-x[i])*h/(b-1)*( A + B + C )

#output
import matplotlib.pyplot as plt
import matplotlib.animation as animation
print(x.shape, phi[k].shape)
def animate ( k ):
	plt.clf ()
	plt.ylim (0 , 0.5)
	plt.title ( "k = " + str ( k ) + " seconds")
	plt.plot (x , y_a, label = 'Numerical')
	plt.plot (x , phi[k], label = 'Analitical')
	plt.legend ()


ani = animation.FuncAnimation ( plt.figure (0) , animate , frames = K , interval = 100)

#plt.plot(x,y_a, label = 'Analitical')
#plt.plot(x,phi[K-5][:])
#plt.legend()
plt.show()




'''
t = np.linspace(0.0, T, m+1)
y = np.zeros((m+1, n+1))
u = np.zeros((n+1, m+1))

#BORDERS 1
for k in range(m+1):
    y[k][0] = ut(t[k])

# BORDERS 2
for i in range(n+1):
    y[0][i] = u0(x[i])

      
#SOLUTION
y[0] = np.vectorize(u0)(x)



for k in range(n+1):
    for i in range(m+1):
        if x[k] > 3*t[i]*t[i]/4:
            u[k][i] = u1(x[k],t[i])
        elif x[k] < 3*t[i]*t[i]/4:
            u[k][i] = u2(x[k],t[i])

print(np.max( np.absolute(u-y) ))
mx=u.max()
mn=u.min()







#///OUTPUT//////////////OUTPUT/////////////////////////////////////
import matplotlib.pyplot as plt
import matplotlib as mpl

from mpl_toolkits.mplot3d.axes3d import Axes3D

Y,X = np.meshgrid(t,x)
fig = plt.figure(figsize=(12, 5.5))
cmap = mpl.cm.get_cmap('RdBu_r')
ax = fig.add_subplot(1,2,1)
c = ax.pcolor(X,Y,u,vmin=mn,vmax=mx, cmap=cmap)
ax.set_xlabel(r"$x_1$", fontsize=14)
ax.set_ylabel(r"$x_2$", fontsize=14)

ax = fig.add_subplot(1, 2, 2, projection='3d')
pl = ax.plot_surface(X, Y, u, vmin=mn,vmax=mx, rstride=3, cstride=3,linewidth=0,cmap=cmap)
ax.set_xlabel(r"$x$", fontsize=14)
ax.set_ylabel(r"$t$", fontsize=14)

cb = plt.colorbar(pl, ax=ax, shrink=0.75)
cb.set_label(r"$u(x_1,x_2)$", fontsize=14)

fig1, ax1 = plt.subplots()
cs = plt.imshow(u, cmap='inferno')
fig1.colorbar(cs)
plt.show()


'''

    
