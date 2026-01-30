import numpy as np
import matplotlib.pyplot as plt

# Nombre d'espèces
nb = 4  # lapin, renard, poule, loup

# Paramètres
tfinal = 200       # secondes
N = 10000          # nombre de points
dt = tfinal / N

# Vecteur alpha 
# Lapins prolifèrent, renards décroissent sans lapins, poules prolifèrent, loups décroissent sans proies
alpha = np.array([0.2, -0.1, 0.15, -0.05])  

# Matrice beta
# lapin = 0, renard = 1, poule = 2, loup = 3
beta = np.array([
    [0.0, -0.02, -0.01, -0.03],  # lapins sont mangés par renards, poules, loups
    [0.01, 0.0, 0.0, -0.02],     # renards prolifèrent grâce aux lapins, perdent avec les loups
    [0.0, -0.01, 0.0, -0.02],    # poules mangées par renards et loups
    [0.02, 0.01, 0.01, 0.0]      # loups prolifèrent en mangeant toutes les autres
])

p0 = np.array([40, 10, 30, 5])  # lapins, renards, poules, loups

p = np.zeros((N+1, nb))
p[0] = p0
temps = np.linspace(0, tfinal, N+1)


def f(p, alpha, beta):
    return p * (alpha + beta @ p)

# Boucle RK4
for i in range(N):
    k1 = f(p[i], alpha, beta)
    k2 = f(p[i] + 0.5*dt*k1, alpha, beta)
    k3 = f(p[i] + 0.5*dt*k2, alpha, beta)
    k4 = f(p[i] + dt*k3, alpha, beta)
    p[i+1] = p[i] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

# Tracé des populations
plt.figure(figsize=(10,6))
labels = ["Lapins", "Renards", "Poules", "Loups"]
for i in range(nb):
    plt.plot(temps, p[:,i], label=labels[i])
plt.xlabel("Temps")
plt.ylabel("Population")
plt.title("Évolution des populations (Lapins, Renards, Poules, Loups)")
plt.grid(True)
plt.legend()
plt.show()

# Tracé phase plane exemple : Lapins vs Renards
plt.figure(figsize=(6,6))
plt.plot(p[:,0], p[:,1])
plt.xlabel("Population de Lapins")
plt.ylabel("Population de Renards")
plt.title("Phase Plane Lapins vs Renards")
plt.grid(True)
plt.show()
