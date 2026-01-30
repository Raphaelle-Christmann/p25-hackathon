import numpy as np
import matplotlib.pyplot as plt

dt = 0.01
v,vb= 1.5,1.0
temps = np.arange(0,20,dt)
N = len (temps)
r = np.zeros((N,2)) # tableau des positions [x,y]
r[0] = [0,10]

def f(r,t) :     # définition de la fonction f de l'ED
    x,y = r[0], r[1]
    xB,yB = vb*t,0
    norme = np.sqrt((xB - x)**2 + (yB - y)**2) 
    eps = 1e-10
    norme += eps  # on évite la division par 0
    return  v*np.array([(xB - x)/norme, (yB - y)/norme]) #on renvoie sous forme d'un tableau 

def rk4(r,t,f) : #algorithme de Runge Kutta 4
    
    k1 = f(r, t)
    k2 = f(r + dt/2 *k1, t + dt/2)
    k3 = f(r + dt/2 * k2, t + dt/2)
    k4 = f(r + dt * k3, t + dt)
    
    r_nv = r + (dt/6) * (k1 + 2*k2 + 2*k3 + k4) 
    return r_nv

for i in range(N-1): # remplissage du tableau des positions
    r[i+1] = rk4(r[i], temps[i], f)

plt.figure()
plt.grid()
plt.plot(r[:,0],r[:,1],label="Chien")
plt.plot(vb*temps,np.zeros(N),'m',label="Balle")       
plt.legend()
plt.show()

