import numpy as np
import matplotlib.pyplot as plt


def lagranz(x,y,t):
	z = 0
	for j in range(len(y)):
		p1 = 1; p2 = 1
		for i in range(len(x)):
			if i == j:
				p1 = p1*1; p2 = p2*1   
			else: 
				p1 = p1*(t - x[i])
				p2 = p2*(x[j] - x[i])
		z = z + y[j]*p1/p2
	return z


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
print('y = ', y)

# Find new points (z, f)
z = [float(i) for i in testXfile.readline().split()]
f = [lagranz(x,y,i) for i in z]

print('z = ', z)
print('f = ', f)

# Create a lot of points to form a function
min_xz = min( np.min(x), np.min(z) )
max_xz =  max( np.max(x), np.max(z) )

xnew = np.linspace(min_xz , max_xz, 50 )
ynew = [lagranz(x,y,i) for i in xnew]

#Saving test.ans
for j in range(0, len(f)):
	testYfile.write(str(f[j]) + ' ')

# Closing files
xFile.close()
yFile.close()
testXfile.close()
testYfile.close()

# Plotting
plt.plot(x, y, 'o', xnew, ynew)
plt.plot(z, f, 'o')
plt.grid(True)
plt.show()
