import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.special import zeta, gamma

plt.rcParams['text.usetex'] = True

pi = np.arccos(-1.0)
beta = 2.0
dim=3
photon_number=3000
length=np.power(photon_number*gamma(dim/2)*np.power(2,dim-1)*np.power(pi,dim/2)/zeta(dim)*beta**dim /(dim-1)/gamma(dim),1/dim)
mc_steps=2500000
mc_output = 10000
equilibrium = 1250000

np.random.seed(51)

def change_k():
    global photon_ar
    new_n=np.zeros(dim)
    n=np.random.randint(0,photon_number)
    i=np.random.randint(0,dim)
    for d in range(dim):
        if (i==d):
            new_n[i]=photon_ar[n,i]+np.random.randint(1,3)*2-3
        else:
            new_n[d]=photon_ar[n,d]
    old_n2 = 0
    new_n2 = 0
    for d in range(dim):
        old_n2+=photon_ar[n,d]**2
        new_n2+=new_n[d]**2
    delta_omega=(np.sqrt(new_n2)-np.sqrt(old_n2))*2*pi/length
    if (np.random.random()<np.exp(-beta*delta_omega)):
        photon_ar[n,i]=new_n[i]

photon_ar=np.zeros((photon_number,dim))

for i in range(photon_number):
    for j in range(dim):
        photon_ar[i,j]=np.random.randint(-10,11)

axis_photon=np.zeros(photon_number)

def update(frame):
    global fixed_bins,ave_counts
    print(frame)
    for i in range(mc_output):
        change_k()
    for i in range(photon_number):
        for d in range(dim):
            axis_photon[i]+=photon_ar[i,d]**2
        axis_photon[i]=np.sqrt(axis_photon[i])*2*pi/length
    counts, bin = np.histogram(axis_photon,bins=bins)
    for i in range(len(bin)-1):
        counts[i] = counts[i]*(bin[i]+bin[i+1])/2
        if(frame*mc_output==equilibrium):
            fixed_bins = np.copy(bin)
            ave_counts = np.zeros(counts.shape)
        if (frame*mc_output>equilibrium):
            counts2,fixed_bins = np.histogram(axis_photon,bins=fixed_bins)
            ave_counts[i]+=counts2[i]*(bin[i]+bin[i+1])/2
    hist.set_data(counts,bin)
    return hist

axis_omegas=np.linspace(0,15*2*pi*np.sqrt(3),15)

maximum_omega=7
print("maximum omega= "+str(maximum_omega))

bins=80

x_axis=np.linspace(0,7,100)
y_axis=(dim-1)*np.power(x_axis,dim)/(np.exp(beta*x_axis)-1)/np.power(pi,dim/2)/gamma(dim/2)/np.power(2,dim-1)*length**dim*maximum_omega/bins#*bin[1]

for i in range(photon_number):
    for d in range(dim):
        axis_photon[i]+=photon_ar[i,d]**2
    axis_photon[i]=np.sqrt(axis_photon[i])*2*pi/length
counts, bin = np.histogram(axis_photon,bins=bins)
ave_counts=np.zeros(counts.shape)
fixed_bins=np.zeros(bin.shape)
for i in range(len(bin)-1):
    counts[i] = counts[i]*(bin[i]+bin[i+1])/2

fig, ax = plt.subplots()
dist = ax.plot(x_axis,y_axis)
hist = ax.stairs(counts,bin)
ax.set_xlabel(r'$\omega [s^{-1}]$',fontsize=20)
ax.set_ylabel(r'$Vu_\omega\Delta\omega [\hbar s^{-1}]$',fontsize=20)
ax.set_title(r'$3D$ simulation vs theoretical calculation')
ax.legend(('3D theoretical result','MC simulation'),loc="upper right")

anim = FuncAnimation(fig,update, frames=(int)(mc_steps/mc_output), interval=40)
anim.save('movie.mp4')

for i in range((int)(bins)):
    ave_counts[i]*=(mc_output/(mc_steps-equilibrium))

fig2,ax2= plt.subplots()
ax2.plot(x_axis,y_axis)
hist2 = ax2.stairs(ave_counts,fixed_bins)

ax2.set_xlabel(r'$\omega [s^{-1}]$',fontsize=20)
ax2.set_ylabel(r'$Vu_\omega\Delta\omega [\hbar s^{-1}]$',fontsize=20)
ax2.set_title(r'$3D$ simulation vs theoretical calculation')
ax2.legend(('3D theoretical result','MC simulation'),loc="upper right")
fig2.savefig('ave.png')
