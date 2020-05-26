import numpy as np
from math import *
import scipy.linalg as sla
import time 
import matplotlib.pyplot as plt

eps = 0.2
size = 10 #size*100 - matrix size
my_time = [0]*size
np_time = [0]*size

def jacobi(n, A, f, x):
	xnew = np.zeros(n)
	for i in range(n):
		s = 0
		for j in range(i):
			s = s + A[i][j] * x[j]
		for j in range(i+1, n):
			s = s + A[i][j] * x[j]
		xnew[i] = (f[i] - s) / A[i][i]
	return xnew

def diff(n, x, y):
	s = 0
	for i in range(n):
		s += (x[i] - y[i]) ** 2
	return sqrt(s)

def solve(n, A, f):
	xnew = np.zeros(n)
	while True:
		x = np.array(xnew)
		xnew = jacobi(n, A, f, x)
		if diff(n, x, xnew) < eps:
			break
	return xnew

for count in range(1,size+1):
	#Input data: n, A, f
	n = count*100		
	print('Make A')
	A = np.random.rand(n,n)
	f = np.random.rand(n)
	
	
	#Create matrix with diagonally dominant type
	print('Make A diagonally dominant type matrix')
	s = np.sum(np.abs(A), axis = 1)
	for i in range(n):
		A[i][i] = A[i][i] + s[i]
	

	#Epsilon and matrix size:
	print('epsilon = ',eps)
	print('n = ', n)

	#My method
	print('Start solving by my method')
	start = time.time()
	x = solve(n, A, f)
	my_time[count - 1] = time.time() - start
	print('My time: ',my_time[count - 1])

	#Numpy solution
	print('Start solving by numpy method')
	start = time.time()
	x_np = np.linalg.solve(A, f)
	np_time[count - 1] = time.time() - start
	print('NP time: ',np_time[count - 1])
	
	#Same solution?
	print('\n||x - x_np|| = ', max(np.absolute(x_np - x)) )
	print ('\n\n\n\n\n')

#Plotting
plt.plot(my_time)
plt.plot(np_time)
plt.show()
