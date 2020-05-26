import numpy as np
import scipy.linalg as sl
import time
import random
import matplotlib.pyplot as plt

#//////ARRAY SIZES/////////#
arr_size = 34
my_time = [0]*arr_size
np_time = [0]*arr_size



#////////Data Input////////#
def input_data(n):
	print('Creating A...')
	A = np.zeros((3, n))
	print('Make diagonals...')
	for a in range(0,3):
		for b in range(0,n):
			A[a][b] = random.random()
		f = np.random.rand(n)
	x = [0] * n
	print('Done!')
	return A,f,x,n


#//////START PROGRAM////////////#
for count in range(1,arr_size+1):


	print('ARRAY_SIZE: ', count*1000)
	A,f,x,n = input_data(count*1000)

	#Numpy solution
	start_time = time.time()

	xx = sl.solve_banded((1,1), A, f)

	np_time[count-1] = time.time() - start_time
	print('\nNP time:', np_time[count-1])
	



	#Start time for my solution
	start_time = time.time()

	#////My solution: creating upper triagular matrix
	#a - Sub-diagonal elements
	#b - Super-diagonal elements
	#c - Diagonal elements
	m = 1;	
	for i in range(1,n):
		m = A[0][i]/A[2][i-1]#m = a[i]/c[i-1];
		A[2][i] = A[2][i] - m*A[1][i-1]#c[i] = c[i] - m*b[i-1]
		f[i] = f[i] - m*f[i-1]#f[i] = f[i] - m*f[i-1]

	#Finding x
	x[n-1] = f[n-1]/A[n-1][n-1];
	for i in range(n - 2, -1, -1):
	  x[i]=(f[i] - A[i][i + 1]*x[i+1]) / A[i][i]

	#Output my time
	my_time[count-1] = time.time() - start_time
	print('My time:', my_time[count-1])

	#Output
	print('\n||x - xx|| = ', max(np.absolute(x-xx)) )
	print ('SAME? ANSWER:', np.allclose(x,xx), '\n\n\n\n\n\n\n\n\n\n\n\n')

plt.plot(my_time)
plt.plot(np_time)
plt.show()
