import numpy as np
import matplotlib.pyplot as plt

"Partie 1"
# paramètres
v =  1.5     # vitesse du chien
vB = 1       # vitesse de la balle


def f(r, t):
    """
    on créer un tableau représentant r: array([x, y])
    on retourne dr/dt
    """
    xb = vB * t
    yb = 0
    diff = np.array([xb, yb]) - r
    norme = np.linalg.norm(diff)
    #np.where(condition, valeur_si_vrai, valeur_si_faux)
    return np.where(norme == 0, np.zeros(2),v * diff / norme)



def runge_kutta(f, t0, tf, dt):
    time = np.arange(t0, tf + dt, dt)
    r = np.zeros((len(time), 2))#on creer le tableau de valeurs
    # condition initiale du chien
    r[0] = np.array([0, 10])
    for k in range(len(time) - 1):
        t = time[k]
        rk = r[k]
        k1 = f(rk, t)
        k2 = f(rk + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(rk + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(rk + dt * k3, t + dt)
        r[k + 1] = rk + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    return time, r

time, r = runge_kutta(f, 0, 10, 0.01)

xb = vB * time
yb = np.zeros(len(time))

plt.plot(r[:,0], r[:,1], label="Trajectoire du chien")
plt.plot(xb, yb, "--", label="trajectoire de la balle")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

    

"Partie 2"

def heptagone(n, time):#j'ai besoin du temps en argument pour créer r
    alpha = 2 * np.pi / n
    r = np.zeros((len(time), n, 2))
    for i in range(n):
        r[0, i, 0] = np.cos(i * alpha)
        r[0, i, 1] = np.sin(i * alpha)
    return r

def f2(r, t):
    rnext = np.roll(r, -1, axis=0)   # voisin i+1: on a décalé de 1
    diff = rnext - r #on fait la diff entre les deux tableaux                
    norme = np.linalg.norm(diff, axis=1, keepdims=True)#axis 1 pour faire la norme de chaque vecteur pour chaque temps
    return np.where(norme == 0, np.zeros_like(diff),v * diff / norme)

def runge_kutta2(n, f, t0, tf, dt):
    time = np.arange(t0, tf + dt, dt)
    r = heptagone(n, time)
    for k in range(len(time) - 1):
        t = time[k]
        rk = r[k]
        k1 = f(rk, t)
        k2 = f(rk + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(rk + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(rk + dt * k3, t + dt)
        r[k + 1] = rk + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    return time, r

time, r = runge_kutta2(25, f2, 0, 300, 0.01)

for i in range(25):
    plt.plot(r[:, i, 0], r[:, i, 1], label=f"trajectoire chien {i+1}")

plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.title('course poursuite entre canidés')
plt.legend()
plt.grid()
plt.show()




    







