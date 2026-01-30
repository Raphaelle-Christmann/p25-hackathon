import numpy as np
import matplotlib.pyplot as plt

# def runge_kutta(f, v,t0,tf,dt):
#     time = np.arange(t0, tf + dt, dt)
#     solx = np.zeros(len(time))
#     soly = np.zeros(len(time))
#     solx[0]=1
#     for k in range(len(time) - 1):
#         solx[k + 1] = solx[k] + (1/6)*dt*(fx(solx[k],time[k])+
#          2*fx(solx[k]+(1/2)*dt*f(solx[k], xb[k], soly[k],time[k] + (1/2)*dt)+
#          2*fx(solx[k]+(1/2)*dt*fx(solx[k]+(1/2)*dt* fx(solx[k]+(1/2)*dt*fx(solx[k],time[k]),time[k] + (1/2)*dt), time[k] + (1/2)*dt) +
#          fx(solx[k]+(1/2)*dt*fx(solx[k]+(1/2)*dt*fx(solx[k],time[k]),time[k] + (1/2)*dt)+2*fx(solx[k]+(1/2)*dt*fx(solx[k]+(1/2)*dt* fx(solx[k]+(1/2)*dt*fx(solx[k],time[k]),time[k] + (1/2)*dt), time[k] + (1/2)*dt) , time[k] + (1/2)*dt))))
#         soly[k + 1] = soly[k] + (1/6)*dt*((fy(soly[k],time[k])+
#          2*fy(soly[k]+(1/2)*dt*fy(soly[k],time[k]),time[k] + (1/2)*dt)+
#          2*fy(soly[k]+(1/2)*dt*fy(soly[k]+(1/2)*dt* fy(soly[k]+(1/2)*dt*fy(soly[k],time[k]),time[k] + (1/2)*dt), time[k] + (1/2)*dt) +
#          fy(soly[k]+(1/2)*dt*fy(soly[k]+(1/2)*dt*fy(soly[k],time[k]),time[k] + (1/2)*dt)+2*fy(soly[k]+(1/2)*dt*fy(soly[k]+(1/2)*dt* fy(soly[k]+(1/2)*dt*fy(soly[k],time[k]),time[k] + (1/2)*dt), time[k] + (1/2)*dt) , time[k] + (1/2)*dt)))))
        
#     return solx, soly
   

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

    
    

# def f(x, xb, y, t):
#     norme=np.sqrt((xb-x)**2+y**2)
#     fx=v*(xb-x)/norme
#     fy=v*(-y)/norme
#     return fx, fy

# solx, soly = runge_kutta(f, 10, 0, 10, 0.1)
# time = np.arange(t0, tf + dt, dt)
# xb=v*time
# yb = np.zeros(len(time))


# plt.plot(solx, soly, label="trajectoire du chien")
# plt.plot(xb, yb, label="trajectoire de la balle")
# plt.xlabel("x(t)")
# plt.ylabel("y(t)")
# plt.grid()
# plt.legend()
# plt.show()



