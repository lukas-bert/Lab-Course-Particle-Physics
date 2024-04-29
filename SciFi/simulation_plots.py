import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat
import scipy.constants as const
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as stds
import pandas as pd

#
# Um simulation.py neu auszuführen: r_exit_cut.pdf in content/pics löschen (runtime ~ 1h)
# Um simulation_plots.py neu auszuführen: rmin_sim_core.pdf in content/pics löschen
#

# Plots for simulation data

df_sim = pd.read_csv("content/data/Simulation/dataframe.csv") # read csv file created earlier

# Plot Angle distribution for photons in core and in cladding

counts, bins, _ = plt.hist(df_sim["Theta"][df_sim["inCladding"] == False], 
                           bins = 30, label = "Kernphoton", color = "cornflowerblue", edgecolor = "royalblue")
_, bins, _ = plt.hist(df_sim["Theta"][df_sim["inCladding"] == True], 
                      bins = 30, label = "Mantelphoton", color = "darkorange", edgecolor = "orangered", alpha = .7)
plt.vlines(21.37, 0, 1.2*max(counts), color = "black", ls = "dashed", label = r"$\Theta_{\mathrm{max}1} = 21,37°$")
plt.vlines(27.44, 0, 1.2*max(counts), color = "red", ls = "dashed", label = r"$\Theta_{\mathrm{max}2} = 27,44°$")
plt.vlines(38.68, 0, 1.2*max(counts), color = "black", ls = "dotted", label = r"$\Theta_{\mathrm{max}3} = 38,68°$")
plt.vlines(46.34, 0, 1.2*max(counts), color = "red", ls = "dotted", label = r"$\Theta_{\mathrm{max}4} = 46,34°$")
plt.vlines(50.94, 0, 1.2*max(counts), color = "green", ls = "dotted", label = r"$\Theta_{\mathrm{max}5} = 50,94°$")
plt.xlabel(r"$\Theta$ [°]")
plt.ylabel("Intensität [a.u.]")
plt.ylim(0, 1.2*max(counts))
plt.xlim(0, 52)
plt.ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
plt.legend(loc = "upper left");
plt.tight_layout()
plt.savefig("content/pics/theta_sim.pdf")
plt.close()

# print maximal angles
print("######################################################################")
print(df_sim.groupby(["inCladding"])["Theta"].max())
print("######################################################################")

# 2d hist theta vs. r_min

# plot for core photons
N_bins = 50
h = plt.hist2d(df_sim["r_min"][df_sim["inCladding"] == False],
               df_sim["Theta"][df_sim["inCladding"] == False],
               bins = [np.linspace(0, 0.125, N_bins), np.linspace(0, 50, N_bins)])
plt.colorbar(h[3]);
plt.xlabel(r"$r_\mathrm{min}$ [mm]")
plt.ylabel(r"$\Theta$ [°]");
plt.tight_layout()
plt.savefig("content/pics/rmin_sim_core.pdf")
plt.close()

# plot for cladding photons
h2 = plt.hist2d(df_sim["r_min"][df_sim["inCladding"] == True],
               df_sim["Theta"][df_sim["inCladding"] == True],
               bins = [h[1], h[2]])
plt.colorbar(h2[3]);
plt.xlabel(r"$r_\mathrm{min}$ [mm]")
plt.ylabel(r"$\Theta$ [°]");
plt.tight_layout()
plt.savefig("content/pics/rmin_sim_cladding.pdf")
plt.close()

# absorption plots

h = plt.hist2d(df_sim["gpsPosX"], df_sim["Theta"], bins = (12, 50))
plt.colorbar(h[3]);
plt.xlabel(r"$x$ [mm]")
plt.ylabel(r"$\Theta$ [°]");
plt.tight_layout()
plt.savefig("content/pics/absorption_sim.pdf")
plt.close()

# plot of absorption coefficient vs theta

bins = h[2]

def exp(x, a, I_0):
    return I_0 * np.exp(-a*x)

params, pcov, theta = [], [], []
x = np.unique(df_sim["gpsPosX"])
#x_ = np.linspace(0, 2000, 10000)

for i in range(len(bins)-1):
    theta.append((bins[i] + bins[i+1])/2)
    y = (df_sim[(bins[i] < df_sim["Theta"]) & (df_sim["Theta"] < bins[i+1])]).groupby("gpsPosX").count()
    y = np.array(y["y_exit"])
    p1, p2 = op.curve_fit(exp, x, y, p0 = [1/100, y[0]])
    params.append(p1)
    pcov.append(p2)
    
params, pcov, theta = np.array(params), np.array(pcov), np.array(theta)

a_avg = np.mean(params[:, 0])

print("-------------------------------------------------------------------")
print(f"a_avg (Simulation) = {a_avg:.4e}")
print("-------------------------------------------------------------------")

plt.plot(theta, params[:, 0], color = "cornflowerblue", label = "Fitparameter")
plt.ylabel(r"$a_0$ [mm$^{-1}$]")
plt.xlabel(r"$\Theta$ [°]")
plt.xlim(0,50)
plt.ylim(1.5e-4, 4.5e-4)
plt.ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))

x = np.linspace(0, 50, 1000)

def f(x, a, b):
    return a/np.cos(x) + b*np.tan(x)

plt.plot(x, f(x/180*np.pi, params[0,0], .5*params[0,0]), label = "Theoriekurve", ls = "dashed", c = "firebrick")
plt.ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("content/pics/absorption_sim2.pdf")
plt.close() 
