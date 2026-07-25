import matplotlib.pyplot as plt
import numpy as np

##           B                                          B
# GC   0    1     2                    im2   im1   GC
# *----*----*----*----*----*----*----*----*----*----*
#    0  |  1  |  2  |               | im2 | im1 | im
#       f0    f1    f2                    fim1
#

"""
 dcv/dt + df/dx = q
 cv = [rhoA, rhouA, rhoEA]
 f  = [rhouA, (rhou^2+p)A, rhouHA]
 q  = [0, p*dA/dx, 0]

 p = (gamma-1)*rho*(E-u^2/2)
 E = e+u^2/2
 H = E+p/rho
"""

# Constants ============================================================
gamma = 1.4
cpgas = 1005          # J/kg/K
gam1  = gamma-1
gap1  = gamma+1
rgas  = gam1*cpgas/gamma
cvgas = cpgas-rgas
ggm1  = gamma/gam1

# Inputs ================================================================
p1 = 101325           # Pa
t1 = 288.15           # K
p2 = 101325-1e-8      # Pa
t2 = 288.15           # K

cfl     = 2.5
iorder  = 2
limfac  = 1.5
epsentr = 0.05
nrk     = 5
ark     = [0.0695, 0.1602, 0.2898, 0.5060, 1.000]

# Source term maps =======================================================
Wsrc, ecmfsrc, Fxsrc, SWxsrc = np.load('Sources.npy')


def interp1d(arr1, arr2, target):
    if target <= arr1[0]:
        x1 = arr1[0]; x2 = arr1[1]
        y1 = arr2[0]; y2 = arr2[1]
    elif target >= arr1[-1]:
        x1 = arr1[-1]; x2 = arr1[-2]
        y1 = arr2[-1]; y2 = arr2[-2]
    else:
        for index in range(0, len(arr1)):
            if target >= arr1[index] and target <= arr1[index+1]:
                x1 = arr1[index]; x2 = arr1[index+1]
                y1 = arr2[index]; y2 = arr2[index+1]
                break
    return ((y1-y2)*target+x1*y2-x2*y1)/(x1-x2)


# Grid ====================================================================
ncv = 101
im  = ncv-1
im1 = ncv-2
x   = np.linspace(0,1,num=im)
ax  = 0.1+0*x
dx  = x[1:]-x[:-1]
da  = ax[1:]-ax[:-1]
xcv = 0.5*(x[1:]+x[:-1])
a   = 0.1+0*xcv
a   = np.insert(a,[0,im1],[a[0],a[-1]])
vol = 0.5*(ax[1:]+ax[:-1])*dx
vol = np.insert(vol,[0,im1],[vol[0],vol[-1]])

# Initialize Flow Field ===================================================
temp = t1*(p2/p1)**(gam1/gamma)
rho  = p2/(rgas*temp)
mach = (2*(t1/temp-1)/gam1)**0.5
c    = (gamma*p2/rho)**0.5
u    = c*mach
mass = rho*u*ax[0]
e    = cvgas*t1
cv   = np.vstack((rho*a, mass+a*0, rho*e*a))
p    = p2+a*0
c    = np.sqrt(gamma*p/rho)
dv   = np.vstack((rho+a*0, cv[1,:]/cv[0,:], p, c))   # [rho, u, p, c]
du   = np.zeros((3,ncv+1))
rs   = np.zeros((3,ncv-1))
ls   = np.zeros((3,ncv-1))

volref  = 1.0
rhoref  = rho
uref    = u
pref    = p2
limfac3 = limfac**3
rvolref = 1.0/volref**1.5
eps2    = np.zeros(3)
eps2[0] = limfac3*rhoref*rhoref*rvolref
eps2[1] = limfac3*uref*uref*rvolref
eps2[2] = limfac3*pref*pref*rvolref


# Boundary Conditions ======================================================
def BC(dv, p1, t1, p2, t2):

    # Left Boundary --------------------------------------------------
    rhod, ud, pd, cd = dv[0,1], dv[1,1], dv[2,1], dv[3,1]

    if ud>=0:   # Subsonic Inlet (assume ud<cd)
        jm  = ud-2*cd/gam1              # <0
        c02 = gamma*rgas*t1
        dis = -0.5*gam1+gap1*c02/(gam1*jm**2)
        if dis<0: dis = 1e-20
        cb   = -jm*gam1/gap1*(1+dis**0.5)
        tb   = t1*cb**2/c02
        pb   = p1*(tb/t1)**(gamma/gam1)
        rhob = pb/(rgas*tb)
        ub   = (2*cpgas*(t1-tb))**0.5
    else:
        if abs(ud)<cd:   # Subsonic Outlet
            jm   = ud-2*cd/gam1
            pb   = p1
            rhob = rhod*(p1/pd)**(1/gamma)
            tb   = pb/rhob/rgas
            cb   = (gamma*rgas*tb)**0.5
            ub   = jm+2*cb/gam1
        else:
            # Supersonic Outlet
            pb   = pd
            rhob = rhod
            ub   = ud

    out = [rhob*a[0], rhob*ub*a[0], (pb/gam1+0.5*rhob*ub*ub)*a[0], pb]

    # Right Boundary -------------------------------------------------
    rhod, ud, pd, cd = dv[0,im1], dv[1,im1], dv[2,im1], dv[3,im1]

    if ud<0:    # Subsonic Inlet (assume abs(ud)<cd)
        jp  = ud+2*cd/gam1              # >0
        c02 = gamma*rgas*t2
        dis = -0.5*gam1+gap1*c02/(gam1*jp**2)
        if dis<0: dis = 1e-20
        cb   = jp*gam1/gap1*(1+dis**0.5)
        tb   = t2*cb**2/c02
        pb   = p2*(tb/t2)**(gamma/gam1)
        rhob = pb/(rgas*tb)
        ub   = -(2*cpgas*(t2-tb))**0.5
    else:
        if ud<cd:        # Subsonic Outlet
            jp   = ud+2*cd/gam1
            pb   = p2
            rhob = rhod*(p2/pd)**(1/gamma)
            tb   = pb/rhob/rgas
            cb   = (gamma*rgas*tb)**0.5
            ub   = jp-2*cb/gam1
        else:
            # Supersonic Outlet
            pb   = pd
            rhob = rhod
            ub   = ud

    out += [rhob*a[-1], rhob*ub*a[-1], (pb/gam1+0.5*rhob*ub*ub)*a[-1], pb]
    return out


cv[0, 0], cv[1, 0], cv[2, 0], p[0], \
cv[0,-1], cv[1,-1], cv[2,-1], p[-1] = BC(dv, p1, t1, p2, t2)


# Solver ===================================================================
def cons_to_prim(cv, a):
    rho = cv[0,:]/a
    u   = cv[1,:]/cv[0,:]
    p   = gam1/a*(cv[2,:]-0.5*cv[1,:]*cv[1,:]/cv[0,:])
    c   = np.sqrt(gamma*p/rho)
    return np.vstack((rho, u, p, c))


def MUSCL(af, bf, eps):
    return (af*(bf*bf+eps)+bf*(af*af+eps))/(af*af+bf*bf+2.0*eps+1e-30)


def entropy_corr(z, d):
    sc = []
    for z_, d_ in zip(z,d):
        if z_>d_:
            sc.append(z_)
        else:
            sc.append(0.5*(z_*z_+d_*d_)/d_)
    return np.array(sc)


t    = 0
tend = 0.5
tarr = []
uarr = []
while (1):

    cvold = cv.copy()
    diss  = np.zeros(im1)

    # Time step (global CFL)
    lambdac = np.abs(dv[1,:])+dv[3,:]
    dt = cfl*dx/lambdac[1:-1]
    dt = min(dt)

    # R-K stages
    for irk in range(nrk):

        # Left/right reconstructed states
        if iorder == 2:
            du[0,1:-1] = cv[0,1:]/a[1:]-cv[0,:-1]/a[:-1]
            du[1,1:-1] = cv[1,1:]/cv[0,1:]-cv[1,:-1]/cv[0,:-1]
            du[2,1:-1] = p[1:]-p[:-1]
            du[:,0]  = du[:,1]
            du[:,-1] = du[:,-2]

            vola  = (0.5*(vol[1:]+vol[:-1]))**1.5
            eps2n = eps2[0]*vola
            deltr1 = 0.5*MUSCL(du[0,2:], du[0,1:-1], eps2n)
            deltl1 = 0.5*MUSCL(du[0,1:-1], du[0,:-2], eps2n)
            eps2n = eps2[1]*vola
            deltr2 = 0.5*MUSCL(du[1,2:], du[1,1:-1], eps2n)
            deltl2 = 0.5*MUSCL(du[1,1:-1], du[1,:-2], eps2n)
            eps2n = eps2[2]*vola
            deltr3 = 0.5*MUSCL(du[2,2:], du[2,1:-1], eps2n)
            deltl3 = 0.5*MUSCL(du[2,1:-1], du[2,:-2], eps2n)

            rr = cv[0,1:]/a[1:]-deltr1
            ur = cv[1,1:]/cv[0,1:]-deltr2
            pr = p[1:]-deltr3
            rl = cv[0,:-1]/a[:-1]+deltl1
            ul = cv[1,:-1]/cv[0,:-1]+deltl2
            pl = p[:-1]+deltl3
        else:
            rr = cv[0,1:]/a[1:]      # rho
            ur = cv[1,1:]/cv[0,1:]   # u
            pr = p[1:]               # p
            rl = cv[0,:-1]/a[:-1]
            ul = cv[1,:-1]/cv[0,:-1]
            pl = p[:-1]

        # Convective (central) flux
        hl  = ggm1*pl/rl+0.5*ul*ul
        qrl = ul*rl
        hr  = ggm1*pr/rr+0.5*ur*ur
        qrr = ur*rr

        fcav1 = qrl+qrr
        fcav2 = qrl*ul+qrr*ur+pl+pr
        fcav3 = qrl*hl+qrr*hr

        # Roe averages
        rav = (rl*rr)**0.5
        dd  = rav/rl
        dd1 = 1.0/(1.0+dd)
        uav = (ul+dd*ur)*dd1
        hav = (hl+dd*hr)*dd1
        q2a = 0.5*uav*uav
        c2a = gam1*(hav-q2a)
        cav = c2a**0.5
        durl = ur-ul

        h1 = np.abs(uav-cav)
        h2 = np.abs(uav)
        h3 = np.abs(uav+cav)
        delta = epsentr*cav

        eabs1 = entropy_corr(h1, delta)
        eabs2 = entropy_corr(h2, delta)
        eabs3 = entropy_corr(h3, delta)

        h1 = rav*cav*durl
        h2 = eabs1*(pr-pl-h1)/(2*c2a)
        h3 = eabs2*(rr-rl-(pr-pl)/c2a)
        h5 = eabs3*(pr-pl+h1)/(2*c2a)

        fdiss1 = h2+h3+h5
        fdiss2 = h2*(uav-cav)+h3*uav+h5*(uav+cav)
        fdiss3 = h2*(hav-cav*uav)+h3*q2a+h5*(hav+cav*uav)

        f = np.vstack((0.5*(fcav1-fdiss1)*ax,
                       0.5*(fcav2-fdiss2)*ax,
                       0.5*(fcav3-fdiss3)*ax))

        rhs = f[:,1:]-f[:,:-1]

        # Source term (compressor stage injected at cell 50)
        q1 = p[1:-1]*da
        q2 = 0*da
        """
        q1[50] = interp1d(Wsrc, Fxsrc, cvold[1,50])
        q2[50] = interp1d(Wsrc, SWxsrc, cvold[1,50])
        """
        temp = (dv[3,52]**2)/gamma/rgas
        mach = dv[1,52]/dv[3,52]
        tt   = temp*(1+0.5*gam1*mach**2)
        pt   = dv[2,52]*(tt/temp)**(gamma/gam1)
        ecmf = cv[1,52]*((tt/288.15)**0.5)/(pt/101325)
        q1[50] = interp1d(ecmfsrc, Fxsrc, ecmf)
        q2[50] = interp1d(ecmfsrc, SWxsrc, ecmf)

        rhs[1,:] = rhs[1,:]-q1
        rhs[2,:] = rhs[2,:]-q2

        # Update
        cv[:,1:-1] = cvold[:,1:-1]-dt/dx*ark[irk]*rhs
        dv = cons_to_prim(cv, a)
        p  = dv[2,:]

        cv[0, 0], cv[1, 0], cv[2, 0], p[0], \
        cv[0,-1], cv[1,-1], cv[2,-1], p[-1] = BC(dv, p1, t1, p2, t2)

    tarr.append(t)
    uarr.append(dv[1,1])

    t += dt
    print(t)
    if (t>tend): break

rho  = 0.5*(dv[0,1:]+dv[0,:-1])
u    = 0.5*(dv[1,1:]+dv[1,:-1])
p    = 0.5*(dv[2,1:]+dv[2,:-1])
c    = 0.5*(dv[3,1:]+dv[3,:-1])
temp = p/(rgas*rho)
mach = u/c
tt   = temp*(1+0.5*gam1*mach**2)
pt   = p*(tt/temp)**(gamma/gam1)

# Post-Process ==============================================================
fig = plt.figure(); ax = fig.add_subplot(111); ax.plot(x,pt)
fig = plt.figure(); ax = fig.add_subplot(111); ax.plot(x,tt)
fig = plt.figure(); ax = fig.add_subplot(111); ax.plot(dv[2,:])
