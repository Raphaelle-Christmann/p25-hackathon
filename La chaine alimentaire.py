import numpy as np
import matplotlib as plt

nb= 10 # nombre d'espèces différentes 
N = 10000 # nombre de points choisis 

p = np.zeros((N+1, nb))  
beta = np.zeros((nb,nb))

import numpy as np
import matplotlib.pyplot as plt

tfinal = 5 # on étudie sur 5 secondes
N = 10000
dt = tfinal / N
p = np.zeros((N+1, nb))  
alpha = np.zeros(nb)
beta = np.zeros((nb, nb))

alpha[0] = 1.0      # proie de base
alpha[1:nb-1] = -0.5 # prédateurs intermédiaires
alpha[-1] = -0.8     # super-prédateur


# Beta : interactions
for i in range(nb):
    for j in range(nb):
        if i == j:
            beta[i,j] = -0.02  # compétition intra-espèce faible
        elif i == j + 1:
            beta[i,j] = 0.1   # prédateur mange sa proie
            beta[j,i] = -0.1  # proie subit prédation
        elif i == nb-1 and j < nb-1:
            beta[i,j] = 0.08  # super-prédateur croît grâce aux autres
            beta[j,i] = -0.05 # toutes les autres subissent super-prédateur
        else:
            beta[i,j] = 0.0   # pas d'interaction

p[0]= np.linspace(5, 15, nb)
temps = np.linspace(0, tfinal, N+1)

def f(p, alpha, beta,t):
    return p * (alpha + beta @ p) 



for i in range(N): # implémentation de la méthode de Runge-Kutta 4.
    k1 = f(p[i], alpha, beta, temps[i])
    k2 = f(p[i] + 0.5*dt*k1, alpha, beta, temps[i] + 0.5*dt)
    k3 = f(p[i] + 0.5*dt*k2, alpha, beta, temps[i] + 0.5*dt)
    k4 = f(p[i] + dt*k3, alpha, beta,temps[i] + dt)
    p[i+1] = p[i] + dt*(k1 + 2*k2 + 2*k3 + k4)/6


for i in range(nb):
    plt.plot(temps, p[:,i])
plt.title("Evolution de la population en fonction du temps")
plt.xlabel("Temps")
plt.ylabel("Population")
plt.grid(True)
plt.legend()
plt.show()

