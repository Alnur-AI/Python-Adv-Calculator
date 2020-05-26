import numpy as np
import time
import matplotlib.pyplot as plt

np_time = [0 , 0 , 0 , 0 , 0 , 0]
my_time = [0 , 0 , 0 , 0 , 0 , 0]

for tm in range (1,6+1):

	#////START TIME COUNT///#
	start_time = time.time()

	#////////Input matrix/////////#
	n = tm * 100
	A = np.random.rand(n, n)
	f = np.random.rand(n)
	x = [0] * n
	#/////////////////////////////#



	#//////Solution from NUMPY//////#
	AA = A
	ff = f
	xx = np.linalg.solve(AA, ff)
	#///////////////////////////////#



	print(tm*100, 'x', tm*100, ' equation\n')
	print('numpy time:', time.time()-start_time, '\n')
	np_time[tm-1] = time.time()-start_time
	start_time = time.time()



	#/Make matrix upper triagular type/#
	for k in range(n):
		f[k] = f[k] / A[k][k]
		A[k] = A[k] / A[k][k]
		for i in range(k + 1, n):
			f[i] = f[i] - f[k] * A[i][k]
			A[i] = A[i] - A[k] * A[i][k]
			A[i][k] = 0
	#//////////////////////////////////#



	#Finding x
	for i in range(n - 1, -1, -1):
		x[i] = f[i]
		for j in range(i + 1, n):
			x[i] = x[i] - A[i][j] * x[j]
	#///////////////////////////////////////////#



	print('My algorithm time:', time.time()-start_time, '\n')
	my_time[tm-1] = time.time()-start_time

	print('Same solutions: ', np.allclose(x,xx))
	print('######################################################')

plt.plot ( np_time )
plt.plot ( my_time )
plt.show ()
