"A simulation of the detection of supernova neutrinos assuming coherent conversion on a turbulent magnetic field"
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc

matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman'], 'size':16})
matplotlib.rc('text', usetex=True)

limits = np.loadtxt('Current_Limits.csv', usecols=(1,))
mu_NO = np.loadtxt('nu_mu_limit_NO.txt')*1e-13
mu_IO = np.loadtxt('nu_mu_limit_IO.txt')*1e-13
nus_NO = np.loadtxt('nus_limit_NO.txt')*1e-13
nus_IO = np.loadtxt('nus_limit_IO.txt')*1e-13
mu_limits = [min(mu_NO), min(mu_IO), min(nus_NO), min(nus_IO)]

fig, ax = plt.subplots(figsize=(((1+np.sqrt(5))/2 * 5), 5))

ax.set_yscale('log')
ax.set_ylim(1e-14, 1e-10)
ax.tick_params(
    axis='x',
    bottom=False,
    labelbottom=False)
ax.tick_params(
    axis='y',
    which='both',
    right=True)
lines1 = [[(9, (n*np.pi/0.4039+1)*max(nus_NO)), (9, ((n+1)*np.pi/0.4039-1)*max(nus_NO))] for n in range(1500)]
lines2 = [[(9, (n*np.pi/0.4039+1)*min(nus_NO)), (9, ((n+1)*np.pi/0.4039-1)*min(nus_NO))] for n in range(1500)]
lines3 = [[(10, (n*np.pi/0.4039+1)*max(nus_IO)), (10, ((n+1)*np.pi/0.4039-1)*max(nus_IO))] for n in range(1500)]
lines4 = [[(10, (n*np.pi/0.4039+1)*min(nus_IO)), (10, ((n+1)*np.pi/0.4039-1)*min(nus_IO))] for n in range(1500)]
lines1 = np.reshape(np.array(lines1), (1500, 2, 2))
lines2 = np.reshape(np.array(lines2), (1500, 2, 2))
lines3 = np.reshape(np.array(lines3), (1500, 2, 2))
lines4 = np.reshape(np.array(lines4), (1500, 2, 2))
lc1 = mc.LineCollection(lines1, colors='b', linewidths=np.sqrt(500), capstyle="butt", alpha=0.5)
lc2 = mc.LineCollection(lines2, colors='b', linewidths=np.sqrt(500), capstyle="butt", alpha=0.5)
lc3 = mc.LineCollection(lines3, colors='b', linewidths=np.sqrt(500), capstyle="butt", alpha=0.5)
lc4 = mc.LineCollection(lines4, colors='b', linewidths=np.sqrt(500), capstyle="butt", alpha=0.5)
ax.add_collection(lc2)
ax.add_collection(lc1)
ax.add_collection(lc3)
ax.add_collection(lc4)
for i in range(len(limits)):
    ax.arrow(i, limits[i], 0, -0.2*limits[i], head_length=0.05*limits[i], width = 0.02,
             head_width=0.15, length_includes_head=True, fc="gray", ec="gray")
for i in range(int(len(mu_limits)/2)):
    ax.arrow(i+len(limits), mu_limits[i], 0, -0.2*mu_limits[i], head_length=0.05*mu_limits[i],
             width = 0.02, head_width=0.15, length_includes_head=True, fc="b", ec="b")
    ax.arrow(i+2+len(limits), mu_limits[i+2], 0, -0.2*mu_limits[i+2], head_length=0.05*mu_limits[i+2],
             width = 0.02, head_width=0.15, length_includes_head=True, fc="r", ec="r")
ax.plot([7, 7], [min(mu_NO), max(mu_NO)], linewidth=np.sqrt(500), solid_capstyle="butt", c="b", alpha=0.5)
ax.plot([8, 8], [min(mu_IO), max(mu_IO)], linewidth=np.sqrt(500), solid_capstyle="butt", c="b", alpha=0.5)
# ax.plot([9, 9], [min(nus_NO), max(nus_NO)], linewidth=np.sqrt(500), solid_capstyle="butt", c="c")
# ax.plot([10, 10], [min(nus_IO), max(nus_IO)], linewidth=np.sqrt(500), solid_capstyle="butt", c="c")
ax.scatter(range(len(limits)), limits, s=500, marker='_', color="gray")
ax.scatter(len(limits)+np.arange(len(mu_limits)), mu_limits, s=500, marker='_', color=["b", "b", "r", "r"])
ax.set_ylabel(r'$\mu_\nu (\mu_B)$')
ax.text(0, 1.1*limits[0], "White \n dwarfs", color="gray", ha="center", size=10)
ax.text(1, 1.1*limits[1], "Globular \n Cluster", color="gray", ha="center", size=10)
ax.text(2, 1.1*limits[2], "Borexino", color="gray", ha="center", size=10)
ax.text(3, 1.1*limits[3], "Gemma", color="gray", ha="center", size=10)
ax.text(4, 1.1*limits[4], "PandaX-II", color="gray", ha="center", size=10)
ax.text(5, 1.1*limits[5], "XENON1T", color="gray", ha="center", size=10)
ax.text(6, 1.1*limits[6], "XENONnT", color="gray", ha="center", size=10)
ax.text(7, 0.6*mu_limits[0], r'$\nu_\mu$ only (NO)', color="b", ha="center", size=10)     
ax.text(8, 0.6*mu_limits[1], r'$\nu_\mu$ only (IO)', color="b", ha="center", size=10)     
ax.text(9, 0.6*mu_limits[2], r'$\mu_\nu$''\'s (NO)', color="r", ha="center", size=10)     
ax.text(10, 0.6*mu_limits[3], r'$\mu_\nu$''\'s (IO)', color="r", ha="center", size=10)  
plt.tight_layout()
plt.savefig("limits.pdf")