import numpy as np
import time
from math import *

def asolve(x, y):#Аналитическое решение
    return y*x**3+x*y**2


def f(x, y):
    return -8*x


def approx(n, u, h, f):#возвращает ||Lu- Lu*||
    h2 = h**2
    norm = 0.0
    for i in range(1, n):
        for j in range(1, n):
            t  = (u[i+1][j] - u[i][j]) / h2
            t += (u[i-1][j] - u[i][j]) / h2
            t += (u[i][j+1] - u[i][j]) / h2
            t += (u[i][j-1] - u[i][j]) / h2
            t += f[i][j]
            t  = abs(t)  
            norm = max(norm, t)
    return norm

def err(U, V):
    n = U.shape[0]
    norm = 0
    for i in range(1, n):
        for j in range(1, n):
            b = abs(U[i][j] - V[i][j])
            norm = max(norm, b)
    return norm


def Jacobi(n, u, v, h, f):
    h2 = h**2
    for i in range(1, n):
        for j in range(1, n):
            v[i][j]  = u[i-1][j]
            v[i][j] += u[i+1][j]
            v[i][j] += u[i][j-1]
            v[i][j] += u[i][j+1]
            v[i][j] += h2 * f[i][j]
            v[i][j] /= 4


def Zeidel(n, u, v, h, f):
    h2 = h**2
    for i in range(1, n):
        for j in range(1, n):
            v[i][j]  = v[i-1][j]
            v[i][j] += u[i+1][j]
            v[i][j] += v[i][j-1]
            v[i][j] += u[i][j+1]
            v[i][j] += h2 * f[i][j]
            v[i][j] /= 4

def Sor(n, u, v, h, f, omega):
    h2 = h**2
    for i in range(1, n):
        for j in range(1, n):
            v[i][j]  = u[i+1][j] - u[i][j]
            v[i][j] += v[i-1][j] - u[i][j]
            v[i][j] += u[i][j+1] - u[i][j]
            v[i][j] += v[i][j-1] - u[i][j]
            v[i][j] += h2 * f[i][j]
            v[i][j] *= omega
            v[i][j] /= 4
            v[i][j] += u[i][j]
            







#НАЧАЛО ПРОГРАММЫ

#ВЫБОР МЕТОДА
print('\n What method do you want to use?\n')
print('1 - Jacobi\n2 - Zeidel\n3 - SOR\n4 - EXIT')
flag = int(input())
if flag < 1 or flag > 4:
    print('Incorrect input\n')
    exit(1)
if flag == 4:
    print('Closing...')
    exit(0)
######################################################


#ВЫБОР ЭПСИЛОНА, ЕГО СТЕПЕНИ И ЧИСЛО ДЕЛЕНИЙ ОБЛАСТИ, А ТАК ЖЕ САМО ДЕЛЕНИЕ ОБЛАСТИ
p = 4 # Порядок эпсилона ( eps = 10^(-p) )
eps = 10**(-p)
n = 50 #Число делений области на квадраты
np1 = n + 1

x = np.linspace(0, 1.0, np1)
y = np.linspace(0, 1.0, np1)
h = 1.0 / n
h2 = h**2
######################################################


#МАТРИЦА ТОЧНЫХ РЕШЕНИЙ
Ue = np.zeros((np1, np1))
for i in range(np1):
    for j in range(np1):
        Ue[i][j] = asolve(x[i], y[j])
######################################################


#НАЧАЛЬНОЕ ПРИБЛИЖЕНИЕ
U0 = np.zeros((np1, np1))
for i in range(np1):
    U0[0][i] = Ue[0][i]
    U0[i][0] = Ue[i][0]
    U0[n][i] = Ue[n][i]
    U0[i][n] = Ue[i][n]
######################################################


#МАТРИЦА ЗНАЧЕНИЙ ФУНКЦИИ "f(x,y)", СПЕКТРАЛЬНОГО РАДИУСА
F = np.zeros((n, n)) 
for i in range(1, n):
    for j in range(1, n):
        F[i][j] = f(x[i], y[j])


rs = cos(pi*h) # спектральный радиус
mr = rs / (1.0 - rs)# Коэффициент для оценки для метода Якоби
######################################################


#ОПРЕДЕЛЯЕМ К и рК В ЗАВИСИМОСТИ ОТ ВЫБРАННОГО МЕТОДА
if flag == 1:
    K = ceil(2 * p * log(10) / (pi * h)**2 )
elif flag == 2:
    K = ceil(p * log(10) / (pi * h)**2 )
elif flag == 3:
    omega = 2.0 / (1.0 + sqrt(1.0 - rs**2))  # оптимальный параметр омега для верхней релаксации 
    K = ceil(2 * p * log(10) / (pi * h) )
pk = 0
PK = K
while PK > 9:
	PK = PK // 10
	pk = pk + 1
pk = 10**(pk-1)


res_U0 = err(U0, Ue)
app_Ue = approx(n, Ue, h, F)
app_U0 = approx(n, U0, h, F)

print('\nResidual of zero approx:   || U* - U0 || = ', res_U0)
print('Approx exact solution:   || F - AU* || = ', app_Ue)
print('Approx zero solution:      || F - AU0 || = ', app_U0)
print('Iters: Kmax = ', K)
print('Spectral radius: rs = ', rs, '\n\n')

start = time.time()

u = np.array(U0)
v = np.array(U0)
p_err = 1
c3 = eps
print("  k      F-AUk    F-AUk/F-AU0  Uk-aU   Uk-aU/U0-aU   Uk-Uk-1    Оц погр    rs_exp")

for k in range(1, K + 1):
    if flag == 1:
        Jacobi(n, u, v, h, F) # Выполняется только один выбранный метод
    elif flag == 2:
        Zeidel(n, u, v, h, F)
    elif flag == 3:
        Sor(n, u, v, h, F, omega)
    if k % pk == 0:
        c1 = k
        c2 = approx(n, v, h, F)
        c3 = c2 / app_U0
        c4 = err(v, Ue)
        c5 = c4 / res_U0
        c6 = err(u, v)
        c7 = c6 * mr
        c8 = c6 / p_err
        print("{0:5d} {1:10.5f} {2:10.5f} {3:10.5f} {4:10.5f} {5:10.5f} {6:10.5f} {7:10.5f}".format(k, c2, c3, c4, c5, c6, c7, c8))
    p_err = err(u, v)
    u = v.copy()
    #if c3 < eps:
      # break

if k % pk != 0:
    print("{0:5d} {1:10.5f} {2:10.5f} {3:10.5f} {4:10.5f} {5:10.5f} {6:10.5f} {7:10.5f}".format(k, c2, c3, c4, c5, c6, c7, c8))
    
end = time.time()
print('\nTime: ', end - start)

mx = v.max()
mn = v.min()
print('maxU = ', mx)
print('minU = ', mn)



import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.axes3d import Axes3D


X,Y = np.meshgrid(x,x)
fig = plt.figure(figsize=(12, 5.5))
cmap = mpl.cm.get_cmap('RdBu_r')

ax = fig.add_subplot(1,2,1)
c = ax.pcolor(X,Y,v,vmin=mn,vmax=mx, cmap=cmap)
ax.set_xlabel(r"$x$", fontsize=14)
ax.set_ylabel(r"$y$", fontsize=14)

ax = fig.add_subplot(1, 2, 2, projection='3d')
p = ax.plot_surface(X, Y, v, vmin=mn,vmax=mx, rstride=3, cstride=3,linewidth=0,cmap=cmap)
ax.set_xlabel(r"$x$", fontsize=14)
ax.set_ylabel(r"$y$", fontsize=14)

cb = plt.colorbar(p, ax=ax, shrink=0.75)
cb.set_label(r"$u(x,y)$", fontsize=14)

fig1, ax1 = plt.subplots()
cs = plt.imshow(v, cmap='inferno')
fig1.colorbar(cs)
plt.show()
