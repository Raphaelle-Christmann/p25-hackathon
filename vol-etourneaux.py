"""Ce module modélise un vol d'étourneaux à n individus, 
à l'aide de numpy, et en strictement moins de 100 lignes ."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Paramètres du modèle

alpha_coh, alpha_rep, alpha_ali, r_coh, r_rep, r_ali = 0.4, 0.25, 0.5, 20.0, 5.0, 15.0

# Définition de la classe qui représente le rgoupe d'étourneaux

class Etourneaux:
    """Classe représentant un groupe d'étourneaux.
    Attributs :
    n : nombre d'étourneaux
    positions : positions des étourneaux (tableau numpy de taille n x 2)
    vitesses : vitesses des étourneaux (tableau numpy de taille n x 2)"""

    def __init__(self, n:int):
        self.n = n # nombre d'étourneaux
        self.positions = np.random.rand(n,2) * 100 # les positions initiales
        self.vitesses = np.random.rand(n,2) * 5 # les vitesses initiales, on ne les met pas trop élevées pour être cohérents

    def voisins(self,i:int) -> tuple:
        """Renvoie les voisinnages associés aux différentes forces pour l'étourneau i.
        Args : i : l'indice de l'étourneau concerné
        Returns : Vci, Vri, Vai les voisinages de cohésion, répulsion et alignement de l'étourneau i"""

        diffs = self.positions - self.positions[i]
        dists = np.linalg.norm(diffs, axis=1)
        Vci = np.where((dists < r_coh)) # voisinage de cohésion de l'étourneau i
        Vri = np.where((dists < r_rep) & (dists > 0)) # voisinage de répulsion de l'étourneau i, on retire i sinon le dénominateur diverge
        Vai = np.where((dists < r_ali)) # voisinage d'alignement de l'étourneau i   
        return Vci, Vri, Vai
    
    def calcule_force(self,i : int) -> np.ndarray:
        """Calcule la somme des forces subies par l'étourneau i en décomposition sur les axes x,y.
        Args i: l'indice de l'étourneau concerné  
        Returns : La somme des forces subies par l'étourneau i"""

        Vci,Vri,Vai = self.voisins(i) # Calcule des voisinages
        r_moy_i, v_moy_i = np.mean(self.positions[Vci], axis = 0), np.mean(self.vitesses[Vai], axis = 0) # Calcul des positions et vitesses moyennes
        fci = - alpha_coh * (self.positions[i] - r_moy_i) # force de cohésion
        match len(Vri[0]):
            case 0:
                fri = 0 # force de répulsion nulle si d'autres oiseaux ne sont pas trop proches
            case _:
                diffs_rep = self.positions[i] - self.positions[Vri]
                fri = alpha_rep * np.sum(diffs_rep / (np.abs(diffs_rep)**2)) # force de répulsion
        fai = - alpha_ali * (self.vitesses[i] - v_moy_i) # force d'alignement
        return fci + fri + fai

    def mise_a_jour(self,dt : float, L =int)-> None:
        """Met à jour les les positions et vitesse des étourneaux à l'aide de la méthode d'Euler."""
        new_v = np.copy(self.vitesses)
        new_r = np.copy(self.positions)
        forces = np.zeros((self.n,2))
        for i in range(self.n):
            forces[i] = self.calcule_force(i)
            new_v[i] += forces[i] * dt  # mise à jour des vitesses
            ri = new_r[i] + new_v[i] * dt  # mise à jour des positions
            x,y = ri[0], ri[1]
            signx = (ri[0] >= L) or (ri[0] < 0)  # indique si on a un dépassement en x ou non
            signy = (ri[1] >= L) or (ri[1] < 0)  # indique si on a un dépassement en y ou non
            x = np.where(x<0,0,x) # dépassement potentiel en 0
            y = np.where(y<0,0,y)
            x = np.where(x>=L,L-1e-5,x) # dépassement potentiel en L avec condition de lissage
            y = np.where(y>=L,L-1e-5,y)
            new_r[i,0], new_r[i,1] = x,y
            new_v[i,0] = np.where(signx, -new_v[i,0]*0.5, new_v[i,0]) # rebond en x, on diminue la vitesse selon x s'il y a un rebond
            new_v[i,1] = np.where(signy, -new_v[i,1]*0.5, new_v[i,1]) # rebond en y, de même
        self.vitesses = new_v
        self.positions = new_r

def simulation_etourneaux(n:int, steps:int, dt:float, L : int) -> tuple:
    """Simule un vol d\'étourneaux
    Args n : le nombre d\'étourneaux
        steps : le nombre d\'itérations de la simulation
    Returns : positions et vitesses des étourneaux à chaque étape de la simulation"""
    etourneaux = Etourneaux(n)
    positions, vitesses = np.zeros((steps, n, 2)), np.zeros((steps, n, 2))
    for i in range(steps):
        etourneaux.mise_a_jour(dt,L)
        positions[i], vitesses[i] = np.copy(etourneaux.positions), np.copy(etourneaux.vitesses)
    return positions, vitesses

def anime_etourneaux(positions:np.ndarray) -> None:
    """Anime la simulation des étourneaux
    Args positions : les positions des étourneaux à chaque étape de la simulation"""
    fig, ax = plt.subplots()
    scat = ax.scatter(positions[0,:,0], positions[0,:,1], marker='v', color="#725a8f")
    def update(frame):
        scat.set_offsets(positions[frame])
        return scat,
    ani = animation.FuncAnimation(fig, update, frames=positions.shape[0], interval=50, blit=True)
    plt.show()

# Crashtest
n = 100
steps = 1000
dt = 0.05
L = 100
positions = simulation_etourneaux(n,steps,dt,L)[0]
anime_etourneaux(positions)