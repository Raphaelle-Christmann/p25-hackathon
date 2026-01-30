import numpy as np
import matplotlib.pyplot as plt

# Paramètres
nb = 10
N = 20000
tfinal = 50
dt = tfinal / N

p = np.zeros((N+1, nb))
temps = np.linspace(0, tfinal, N+1)

# Croissance naturelle
alpha = np.array([1.0, 0.8, 0.9,    # Proies
                  -0.4, -0.5, -0.3, -0.6,    # Prédateurs intermédiaires
                  -0.7, -0.8, -0.6])  # Super-prédateurs

# Matrice d'interactions (beta)
beta = np.zeros((nb, nb))

# Compétition intra-espèce
np.fill_diagonal(beta, -0.03)

# Prédateurs intermédiaires mangent principalement 1 ou 2 proies
beta[3,0] = 0.1; beta[0,3] = -0.05
beta[3,1] = 0.08; beta[1,3] = -0.04
beta[4,1] = 0.09; beta[1,4] = -0.05
beta[4,2] = 0.08; beta[2,4] = -0.04
beta[5,2] = 0.07; beta[2,5] = -0.03
beta[6,0] = 0.06; beta[0,6] = -0.03

# Super-prédateurs mangent certains prédateurs intermédiaires
beta[7,3] = 0.08; beta[3,7] = -0.04
beta[7,4] = 0.09; beta[4,7] = -0.05
beta[8,5] = 0.07; beta[5,8] = -0.04
beta[8,6] = 0.06; beta[6,8] = -0.03
beta[9,3] = 0.05; beta[3,9] = -0.02
beta[9,4] = 0.05; beta[4,9] = -0.02


beta[3,2] = 0.03; beta[2,3] = -0.015
beta[4,0] = 0.02; beta[0,4] = -0.01
beta[6,2] = 0.02; beta[2,6] = -0.01

# Conditions initiales
p[0] = np.linspace(5, 15, nb)

# Fonction de croissance
def f(p, alpha, beta, t):
    return p * (alpha + beta @ p)

# Runge-Kutta 4
for i in range(N):
    k1 = f(p[i], alpha, beta, temps[i])
    k2 = f(p[i] + 0.5*dt*k1, alpha, beta, temps[i] + 0.5*dt)
    k3 = f(p[i] + 0.5*dt*k2, alpha, beta, temps[i] + 0.5*dt)
    k4 = f(p[i] + dt*k3, alpha, beta, temps[i] + dt)
    p[i+1] = p[i] + dt*(k1 + 2*k2 + 2*k3 + k4)/6
    p[i+1] = np.maximum(p[i+1], 0)

# Visualisation
labels = [f"Espèce {i+1}" for i in range(nb)]
colors = plt.cm.tab10(np.linspace(0,1,nb))

plt.figure(figsize=(14,7))
for i in range(nb):
    plt.plot(temps, p[:,i], label=labels[i], color=colors[i])

plt.title("Écosystème")
plt.xlabel("Temps")
plt.ylabel("Population")
plt.grid(True)
plt.legend(ncol=2)
plt.show()
