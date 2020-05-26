# Linear interpolation on an uneven grid
import numpy as np
import scipy.linalg as sla
import time
import matplotlib.pyplot as plt

# Open data files
xFile = open('train.dat', 'r')
yFile = open('train.ans', 'r')
testXfile = open('test.dat', 'r')
testYfile = open('test.ans', 'w')

# Fill arrays with data
n = int(xFile.readline())		
m = int(testXfile.readline())
x = [float(i) for i in xFile.readline().split()]
y = [float(i) for i in yFile.readline().split()]
z = [float(i) for i in testXfile.readline().split()]

print('train dots = ', n)
print('test dots = ', m)
print('x = ', x)
print('y = ', y)
print('z = ', z)

# Find the coefficients for linear interpolation
a = np.zeros(n)
b = np.zeros(n)

for i in range(n-1):
	a[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
	b[i] = y[i]

# Find the values (z; f) with a new set of arguments
f = np.zeros(m)# My approaching

for j in range(m):# For each new argument
	
	for i in range(0,n-1):# On each segment

		if (z[j] < x[0]):# If the new argument is outside the area [x (0); x (n-1)] on the left
			f[j] = a[0] * (z[j] - x[0]) + b[0]

		if (z[j] >= x[n-1]):# If the new argument is outside the area [x (0); x (n-1)] on the right
			f[j] = a[n-2] * (z[j] - x[n-2]) + b[n-2]

		if (x[i] < z[j] and z[j] <= x[i+1]):# If the argument lies in the region [x (i); x (i + 1))
			f[j] = a[i] * (z[j] - x[i]) + b[i]

	testYfile.write(str(f[j]) + ' ')


print('f = ', f)

# Closing files
xFile.close()
yFile.close()
testXfile.close()
testYfile.close()

# Building graphics
plt.plot(x, y) 
plt.plot(x, y, 'o') 
plt.plot(z, f, 'o') 
#plt.plot(z, f) 
plt.show()
