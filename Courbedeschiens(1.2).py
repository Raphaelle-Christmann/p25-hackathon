import numpy as np
import matplotlib.pyplot as plt

dt = 0.01
v = 1.5
temps = np.arange(0,20,dt)
N = len (temps)
r = np.zeros((N,7,2)) # tableau des positions des 7 chiens
n = 7
indices = np.arange(n)
angles = 2 * np.pi * indices / n  # répartition des coordonnées de l'heptagone
r[0, :, 0] = np.cos(angles) 
r[0, :, 1] = np.sin(angles) 

def f(r,t) :   
    r_poursuivi = np.roll(r, -1, axis=0) # chaque chien poursuit le suivant
    diff = r_poursuivi - r 
    norme = np.sqrt(np.sum(diff**2, axis=1, keepdims=True)) # norme des vecteurs
    norme += 1e-10  # évite la division par 0
    return v * diff /norme  

def rk4(r,t,f,dt) : # même algorithme de Runge Kutta 4 que pour la partie 1
    k1 = f(r, t)
    k2 = f(r + dt/2 *k1, t + dt/2)
    k3 = f(r + dt/2 * k2, t + dt/2)
    k4 = f(r + dt * k3, t + dt)
    
    r_nv = r + (dt/6) * (k1 + 2*k2 + 2*k3 + k4) 
    return r_nv

for i in range(N-1): # remplissage du tableau des positions
    r[i+1] = rk4(r[i], temps[i], f, dt)

plt.figure()
plt.plot(r[:,:,0],r[:,:,1],linewidth=1) # tracé des trajectoires des 7 chiens
plt.plot(np.append(r[0,:,0],r[0,0,0]),np.append(r[0,:,1],r[0,0,1]),'k-',alpha=0.5) # heptagone initial
plt.axis('equal')   
plt.title("Poursuite mutuelle") 
plt.show()