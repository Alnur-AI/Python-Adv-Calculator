import numpy as np
import scipy.linalg as sla
import time
from math import *
import matplotlib.pyplot as plt

# Tridiagonal matrix algorithm
def sweep (n, a, b, c, f):
    alpha = (n + 1) * [0]
    beta = (n + 1) * [0]
    x = n * [0]
    a[0] = 0
    c[n -  1] = 0
    alpha[0] = 0
    beta[0] = 0
    for i in range(0, n):  
        d = a[i] * alpha[i] + b[i]
        alpha [i + 1] = -c[i] / d
        
        beta [i + 1] = (f[i] - (a[i] * beta[i])) / (d)
    x[n - 1] = beta[n]
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i + 1] * x[i + 1] + beta[i + 1]
    return x

# Find A, B, C, D
def generateSpline (x , y):
	n = len(x) - 1
	h = ( x[ n ] - x[0]) / n

	a = np.array ([0] + [1] * ( n - 1) + [0])
	b = np.array ([1] + [4] * ( n - 1) + [1])
	c = np.array ([0] + [1] * ( n - 1) + [0])

	f = np.zeros (n + 1)
	for i in range (1 , n):
		f[ i ] = 3 * ( y [i -1] - 2 * y[i ] + y[i + 1]) / h**2

	print('a = ', a)
	print('b = ', b)
	print('c = ', c)
	print('f = ', f, '\n')

	s = sweep(n + 1, a , b , c , f)
	print('s = ', s)
	
	B = [0] * (n + 1)   
	A = [0] * (n + 1)
	C = [0] * (n + 1)
	D = [0] * (n + 1)

	for i in range (n):
		B [ i ] = s [ i ]
		D [ i ] = y [ i ]
	for i in range(n):
		A[i] = (B[i + 1] - B[i]) / (3 * h)
		C[i] = ((y[i + 1] - y[i]) / h) - ((B[i + 1] + 2 * B[i]) * h) / 3
	return A , B , C , D

def Spline(t, m, x, A, B, C, D):
	f = np.zeros(m)# My approach

	for j in range(m):# For each new argument
		
		for i in range(0,n-1):# On each segment

			if (t[j] < x[0]):# If the new argument is outside the area [x (0); x (n-1)] on the left
				f[j] = A[0]*(t[j]-x[0])**3 + B[0]*(t[j]-x[0])**2 + C[0]*(t[j]-x[0]) + D[0]

			if (t[j] >= x[n-1]):# If the new argument is outside the area [x (0); x (n-1)] on the right
				f[j] = A[n-2]*(t[j]-x[n-2])**3 + B[n-2]*(t[j]-x[n-2])**2 + C[n-2]*(t[j]-x[n-2]) + D[n-2]

			if (x[i] < t[j] and t[j] <= x[i+1]):# If the argument lies in the region [x (i); x (i + 1))
				f[j] = A[i]*(t[j]-x[i])**3 + B[i]*(t[j]-x[i])**2 + C[i]*(t[j]-x[i]) + D[i]

		testYfile.write(str(f[j]) + ' ')
	return f

# Open data files
xFile = open('train.dat', 'r')
yFile = open('train.ans', 'r')
testXfile = open('test.dat', 'r')
testYfile = open('test.ans', 'w')

# Fill arrays with points (x, y)
n = int(xFile.readline())		
m = int(testXfile.readline())
x = [float(i) for i in xFile.readline().split()]
y = [float(i) for i in yFile.readline().split()]

print('train dots = ', n)
print('test dots = ', m)
print('x = ', x)
print('y = ', y, '\n')

# Find A, B, C, D
A,B,C,D = generateSpline (x , y)
print('A = ', A)
print('B = ', B)
print('C = ', C)
print('D = ', D)

# Fill arrays with points (z, f)
z = [float(i) for i in testXfile.readline().split()]
print('z = ', z)
f = Spline(z, len(z), x, A, B, C, D)
print('f = ', f)

# Create a lot of points to form a function
min_xz = min( np.min(x), np.min(z) )
max_xz =  max( np.max(x), np.max(z) )

xnew = np.linspace(min_xz , max_xz, 50 )
ynew = Spline(xnew, len(xnew), x, A, B, C, D)


# Close the files
xFile.close()
yFile.close()
testXfile.close()
testYfile.close()

# Plotting
plt.plot(x, y, 'o', xnew, ynew)
plt.plot(z, f, 'o')
plt.grid(True)
plt.show()
