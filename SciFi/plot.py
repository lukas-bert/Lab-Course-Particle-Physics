import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat
import scipy.constants as const
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as stds
import pandas as pd

# Plots for simulation data

df_sim = pd.read_csv("content/data/Simulation/dataframe.csv") # read csv file created earlier

# Plot Angle distribution for photons in core and in cladding

counts, bins, _ = plt.hist(df_sim["Theta"][df_sim["inCladding"] == False], 
                           bins = 30, label = "Kernphoton", color = "cornflowerblue", edgecolor = "royalblue")
_, bins, _ = plt.hist(df_sim["Theta"][df_sim["inCladding"] == True], 
                      bins = 30, label = "Mantelphoton", color = "darkorange", edgecolor = "orangered")
plt.vlines(21.37, 0, 1.2*max(counts), color = "red", ls = "dashed", label = r"$\theta_{\mathrm{max}1} = 21,37°$");
plt.vlines(38.68, 0, 1.2*max(counts), color = "black", ls = "dashed", label = r"$\theta_{\mathrm{max}2} = 38.68°$");
plt.vlines(44.77, 0, 1.2*max(counts), color = "green", ls = "dashed", label = r"$\theta_{\mathrm{max}3} = 44,77°$");
plt.xlabel(r"$\theta$ [°]")
plt.ylabel("Intensity [a.u.]")
plt.ylim(0, 1.2*max(counts))
plt.xlim(0, 48)
plt.legend(loc = "upper left");
plt.tight_layout()
plt.savefig("build/theta_sim.pdf")
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
plt.ylabel(r"$\theta$ [°]");
plt.tight_layout()
plt.savefig("build/rmin_sim_core.pdf")
plt.close()

# plot for cladding photons
h2 = plt.hist2d(df_sim["r_min"][df_sim["inCladding"] == True],
               df_sim["Theta"][df_sim["inCladding"] == True],
               bins = [h[1], h[2]])
plt.colorbar(h2[3]);
plt.xlabel(r"$r_\mathrm{min}$ [mm]")
plt.ylabel(r"$\theta$ [°]");
plt.tight_layout()
plt.savefig("build/rmin_sim_cladding.pdf")
plt.close()

# plot for absorption coefficient
h = plt.hist2d(df_sim["gpsPosX"], df_sim["Theta"], bins = 12)
plt.colorbar(h[3]);
plt.xlabel(r"$x$ [mm]")
plt.ylabel(r"$\theta$ [°]")
plt.tight_layout()
plt.savefig("build/absorption_sim.pdf")
plt.close()

#####################################################################################################
# Plots for measurements

# spectrometer measurement with lights on/off

# read data
lam_on, DC_on, C_on = np.genfromtxt("content/data/licht_an.txt", unpack = True)
lam_off, DC_off, C_off = np.genfromtxt("content/data/licht_aus.txt", unpack = True)

# without substracting dark counts
plt.plot(lam_on, C_on, marker = ".", c = "cornflowerblue", label = "Light on", lw = 0)
plt.plot(lam_off, C_off, marker = ".", c = "firebrick", label = "Light off", lw = 0, alpha = .7)
plt.ylabel("Intensity [a.u.]")
plt.xlabel(r"$\lambda$ [nm]")
plt.xlim(min(lam_on), max(lam_on))
plt.ylim(0, 1.1*max(C_on))
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("build/lights_on1.pdf")
plt.close()

# with dark coutnts substracted
plt.plot(lam_on, C_on - DC_on, marker = ".", c = "cornflowerblue", label = "Light on", lw = 0)
plt.plot(lam_off, C_off - DC_off, marker = ".", c = "firebrick", label = "Light off", lw = 0)
plt.ylabel("Intensity [a.u.]")
plt.xlabel(r"$\lambda$ [nm]")
plt.xlim(min(lam_on), max(lam_on))
plt.ylim(0, 1.1*max(C_on - DC_on))
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("build/lights_on2.pdf")
plt.close()

# confirmation of radial symmetry

import itertools

# read all files and store in dataframe
Hs = range(-18, 30, 4)
Vs = range(-6, 34, 4)

data = []

for h, v in itertools.product(Hs, Vs):
    lam, C = np.genfromtxt(f"content/data/radial_2_trohabe/Attenuation_h={h}deg_v={v}deg_x=0mm.txt", unpack = True)
    lam_dc, DC = np.genfromtxt(f"content/data/radial_2_trohabe/DarkCounts_h={h}deg_v={v}deg.txt", unpack = True)
    intensity = np.mean(C-DC) # take mean counts over all lambdas as measure for intensity
    data.append(np.array([h, v, intensity])) 

data = np.array(data)

# plotting
plt.figure(figsize=(5, 4))
plt.hist2d(data[:,0], data[:,1], weights = data[:,2], bins = [np.array(range(-20, 32, 4)), np.array(range(-8, 36, 4))]);
plt.colorbar(label = "intensity");
plt.xlabel('Horizontal Angle [°]')
plt.ylabel('Vertical Angle [°]')
#plt.title('2D Intensity Distribution of Light Source')
plt.tight_layout()
plt.savefig("build/radial.pdf")
plt.close()

# measuremnt of intensity

# read data
Hs = range(0, 40, 4)
Xs = range(0, 2000, 100)

data = []

for h, x in itertools.product(Hs, Xs):
    lam, C = np.genfromtxt(f"content/data/intensity_2_trohabe/Attenuation_h={h}deg_v=0deg_x={x}mm.txt", unpack = True)
    lam_dc, DC = np.genfromtxt(f"content/data/intensity_2_trohabe/DarkCounts_h={h}deg_v=0deg.txt", unpack = True)
    intensity = np.mean(C-DC) # take mean counts over all lambdas as measure for intensity
    data.append(np.array([h, x, intensity])) 
data = np.array(data)
#data[data[:, 1] > 500 ,2] = data[data[:,1] > 500][:,2] * 2.5 # manipulation of data hehe (the fiber was bent at x = 500)

# plotting
plt.hist2d(data[:,1], data[:,0], weights = data[:,2], bins = [np.array(range(-50, 2000, 100)), np.array(range(-2, 40, 4))]);
plt.colorbar(label = "intensity");
plt.xlabel('x [mm]')
plt.ylabel('Vertical Angle [°]')
#plt.title('2D Intensity Distribution of Light Source')
plt.tight_layout()
plt.savefig("build/intensity.pdf")
plt.close()

# intensity distribution of angle

# read data
data = []

for h in range(0, 45):
    lam, C = np.genfromtxt(f"content/data/small_angle_trohabe/Attenuation_h={h}deg_v=0deg_x=0mm.txt", unpack = True)
    lam_dc, DC = np.genfromtxt(f"content/data/small_angle_trohabe/DarkCounts_h={h}deg_v=0deg.txt", unpack = True)
    intensity = np.mean(C-DC) # take mean counts over all lambdas as measure for intensity
    data.append(np.array([h, intensity])) 
data = np.array(data)

# plotting
plt.hist(data[:,0], weights = data[:,1], bins = np.linspace(-0.5, 44.5, 46), color = "cornflowerblue");
plt.ylabel("Intensity [a.u.]")
plt.xlabel("Horizontal Angle [°]");
plt.vlines(data[data[:,1] ==max(data[:,1])][0,0], 0, 100, ls = "dashed", color = "firebrick", label = "Maximum")
plt.ylim(0,100)
plt.xlim(-0.5,44.5)
plt.legend()
plt.tight_layout()
plt.savefig("build/intensity_angle.pdf")
plt.close()
