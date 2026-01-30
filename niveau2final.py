import numpy as np
import matplotlib.pyplot as plt

tfinal = 200 # on étudie sur 200 secondes
N = 10000
p = np.zeros((N+1, 2))  
alpha = 0.1
beta = 0.02
delta = 0.01
gamma = 0.1

def f(p, alpha, beta, delta,gamma,t):
    return np.array([p[0] * (alpha - beta*p[1]),
                     p[1] * (delta*p[0] - gamma)])

dt = tfinal / N
temps = np.linspace(0, tfinal, N+1)
p[0] = [40, 10]  

for i in range(N): # implémentation de la méthode de Runge-Kutta 4.
    k1 = f(p[i], alpha, beta, delta,gamma, temps[i])
    k2 = f(p[i] + 0.5*dt*k1, alpha, beta, delta, gamma, temps[i] + 0.5*dt)
    k3 = f(p[i] + 0.5*dt*k2, alpha, beta, delta, gamma, temps[i] + 0.5*dt)
    k4 = f(p[i] + dt*k3, alpha, beta, delta, gamma,temps[i] + dt)
    p[i+1] = p[i] + dt*(k1 + 2*k2 + 2*k3 + k4)/6


plt.plot(p[:,0], p[:,1])
plt.title("Evolution de la population de renards en fonction celle de lapins")
plt.xlabel("Population de lapins")
plt.ylabel("Population de renards")
plt.grid(True)
plt.legend()
plt.show()

