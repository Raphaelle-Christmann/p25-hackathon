import numpy as np
import matplotlib.pyplot as plt

np.random.seed(999)

# Paramètres
nb = 12        # nombre d'espèces
N = 30000      # nombre de pas de temps
tfinal = 50
dt = tfinal / N

# Initialisation
p = np.zeros((N+1, nb))
alpha = np.random.uniform(-1.0, 1.5, nb)  # croissance/mortalité aléatoire forte
beta = np.zeros((nb, nb))

# Compétition intra-espèce forte pour créer du chaos
np.fill_diagonal(beta, -0.05)

# Réseau trophique anarchique extrême
for i in range(nb):
    n_prey = np.random.randint(2, nb)  # mange 2 à nb-1 autres espèces
    preys = np.random.choice(nb, n_prey, replace=False)
    for j in preys:
        if j != i:
            beta[i,j] = np.random.uniform(0.05, 0.2)   # i croît grâce à j
            beta[j,i] = np.random.uniform(-0.1, -0.01) # j subit prédation

# Conditions initiales
p[0] = np.random.uniform(5, 15, nb)
temps = np.linspace(0, tfinal, N+1)

# Fonction de croissance avec bruit plus fort
def f(p, alpha, beta, t):
    noise = np.random.normal(0, 0.05, size=p.shape)  # bruit plus important
    return p * (alpha + beta @ p) + noise

# Intégration Runge-Kutta 4
for i in range(N):
    k1 = f(p[i], alpha, beta, temps[i])
    k2 = f(p[i] + 0.5*dt*k1, alpha, beta, temps[i] + 0.5*dt)
    k3 = f(p[i] + 0.5*dt*k2, alpha, beta, temps[i] + 0.5*dt)
    k4 = f(p[i] + dt*k3, alpha, beta, temps[i] + dt)
    p[i+1] = p[i] + dt*(k1 + 2*k2 + 2*k3 + k4)/6
    p[i+1] = np.maximum(p[i+1], 0)  # éviter les populations négatives

# Visualisation
labels = [f"Espèce {i+1}" for i in range(nb)]
colors = plt.cm.tab20(np.linspace(0,1,nb))

plt.figure(figsize=(14,7))
for i in range(nb):
    plt.plot(temps, p[:,i], label=labels[i], color=colors[i])

plt.title("Écosystème extrême et chaotique à 12 espèces")
plt.xlabel("Temps")
plt.ylabel("Population")
plt.grid(True)
plt.legend(ncol=2)
plt.show()
