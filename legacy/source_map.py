import numpy as np
import matplotlib.pyplot as plt

# --- Gas properties ---
gamma = 1.4
cpgas = 1005.0
gam1  = gamma - 1
gap1  = gamma + 1          # unused in current script (reserved for choked mdot formula)
rgas  = gam1 * cpgas / gamma


def setStatic(T, P, W, Area):
    """
    Invert stagnation (T, P) + mass flow W through Area -> static (Ps, V).
    T   : stagnation temperature [K]
    P   : stagnation pressure    [kPa]
    W   : mass flow              [kg/s]
    Area: flow area              [m^2]
    Returns (Ps [kPa], V [m/s]).
    """
    # --- Choking check (M = 1 reference state) ---
    Tsch  = T / (1 + 0.5 * gam1)
    Vch   = (gamma * rgas * Tsch) ** 0.5
    rhoch = W / Vch / Area
    Psch  = rhoch * rgas * Tsch / 1000
    Pch   = Psch * (T / Tsch) ** (gamma / gam1)   # P0 required to sustain sonic W at this Area

    if P <= Pch:
        print('choked')
        return Psch, Vch

    # --- Subsonic fixed-point iteration ---
    Ts = 0.9 * T
    err = 1
    niter = 0
    while abs(err / T) > 1e-10 and niter < 100:
        Tsg = Ts
        Ps  = P * (Ts / T) ** (gamma / gam1)
        rho = Ps * 1000 / rgas / Ts
        V   = W / rho / Area
        Ts  = T - (V ** 2) / 2 / cpgas
        err = Ts - Tsg
        niter += 1

    return Ps, V


# --- Station 1 (inlet) / Station 2 (exit) stagnation states ---
PR    = 1.2
etais = 0.9
p1 = 101325.0
t1 = 288.15
p2 = p1 * PR
t2 = t1 * (1 + (PR ** (gam1 / gamma) - 1) / etais)
A1 = 0.1
A2 = 0.1

# --- Analytic choked-flow reference (ideal flow function through A2) ---
ff   = (2 * gamma / gam1 * PR ** (-2 / gamma) * (1 - PR ** (-gam1 / gamma))) ** 0.5
W_ff = ff * A2 * p2 / (rgas * t2) ** 0.5
print(W_ff)

# --- Sweep mass flow, build source-term tables ---
Warr = np.linspace(0, 23, num=100)
Fx, SWx, ecmf = [], [], []

for W in Warr:
    Ps1, u1 = setStatic(t1, p1 / 1000, W, A1)
    Ps2, u2 = setStatic(t2, p2 / 1000, W, A2)
    Fx.append(Ps2 * 1000 * A2 - Ps1 * 1000 * A1 + W * (u2 - u1))   # momentum source
    SWx.append(W * cpgas * (t2 - t1))                               # energy source
    ecmf.append(W * (t2 / 288.15) ** 0.5 / (p2 / 101325))           # corrected flow

Fx  = np.array(Fx)
SWx = np.array(SWx)

# --- Central-difference sensitivities dFx/dW, dSWx/dW (Neumann-extrapolated ends) ---
dFxdW  = (Fx[2:]  - Fx[:-2])  / (Warr[2:] - Warr[:-2])
dSWxdW = (SWx[2:] - SWx[:-2]) / (Warr[2:] - Warr[:-2])
dFxdW  = np.insert(dFxdW,  [0, len(dFxdW)],  [dFxdW[0],  dFxdW[-1]])
dSWxdW = np.insert(dSWxdW, [0, len(dSWxdW)], [dSWxdW[0], dSWxdW[-1]])

# --- Plots ---
fig = plt.figure(); ax = fig.add_subplot(111); ax.plot(Warr, Fx)
fig = plt.figure(); ax = fig.add_subplot(111); ax.plot(Warr, SWx)
fig = plt.figure(); ax = fig.add_subplot(111); ax.plot(Warr, dFxdW)
fig = plt.figure(); ax = fig.add_subplot(111); ax.plot(Warr, dSWxdW)
fig = plt.figure(); ax = fig.add_subplot(111); ax.plot(ecmf, Fx)

np.save('Sources', [Warr, ecmf, Fx, SWx])
