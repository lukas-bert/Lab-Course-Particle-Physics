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

#####################################################################################################
# Plots for measurements

# spectrometer measurement with lights on/off

# read data
lam_on, DC_on, C_on = np.genfromtxt("content/data/licht_an.txt", unpack = True)
lam_off, DC_off, C_off = np.genfromtxt("content/data/licht_aus.txt", unpack = True)

# without substracting dark counts
plt.plot(lam_on, C_on, marker = ".", c = "cornflowerblue", label = "Licht an", lw = 0)
plt.plot(lam_off, C_off, marker = ".", c = "firebrick", label = "Licht aus", lw = 0, alpha = .7)
plt.ylabel("Intensität [a.u.]")
plt.xlabel(r"$\lambda$ [nm]")
plt.xlim(min(lam_on), max(lam_on))
plt.ylim(0, 1.1*max(C_on))
plt.ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("build/lights_on1.pdf")
plt.close()

# with dark coutnts substracted
plt.plot(lam_on, C_on - DC_on, marker = ".", c = "cornflowerblue", label = "Licht an", lw = 0)
plt.plot(lam_off, C_off - DC_off, marker = ".", c = "firebrick", label = "Licht aus", lw = 0)
plt.ylabel("Intensität [a.u.]")
plt.xlabel(r"$\lambda$ [nm]")
plt.xlim(min(lam_on), max(lam_on))
plt.ylim(0, 1.1*max(C_on - DC_on))
plt.ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
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
plt.colorbar(label = "Intensität");
plt.xlabel('Horizontaler Winkel [°]')
plt.ylabel('Vertikaler Winkel [°]')
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

for h in Hs:
    factor = .97*(data[(data[:,0] == h)][5,2]/data[(data[:,0] == h)][6,2])
    data[(data[:,0] == h) & (data[:,1 ] > 500), 2] *= factor

df = pd.DataFrame(data, columns = ["Theta", "x", "I"])

# plotting
plt.hist2d(df["x"], df["Theta"], weights = df["I"], bins = [np.array(range(-50, 2000, 100)), np.array(range(-2, 40, 4))])
plt.colorbar(label = "Intensität");
plt.xlabel(r'$x$ [mm]')
plt.ylabel('Vertikaler Winkel [°]')
#plt.title('2D Intensity Distribution of Light Source')
plt.tight_layout()
plt.savefig("build/intensity.pdf")
plt.close()

# fitting

import matplotlib.cm as cm
colors = cm.tab10(np.linspace(0, 1, len(Hs)))

def exp(x, a, I_0):
    return I_0 * np.exp(-a*x)

params, pcov = [], []
x_ = np.linspace(0, 2000, 10000)

for i in range(len(Hs)):
    h = Hs[i]
    x = df["x"][df["Theta"] == h]
    y = df["I"][df["Theta"] == h]
    p1, p2 = op.curve_fit(exp, x, y, p0 = [1/100, np.array(y)[0]])
    params.append(p1)
    pcov.append(p2)
    plt.plot(x, y, c = colors[i], label = r"$\Theta$ =" + f"{h}°", lw = 0, marker = ".")
    plt.plot(x, exp(x, *p1), ls = "dashed", c = colors[i]);

plt.ylabel("Intensität [a.u.]")
plt.xlabel(r"$x$ [mm]")
plt.ylim(20, 100)
plt.xlim(0, 1999)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("build/fits_absorption.pdf")
plt.close()

params, pcov = np.array(params), np.array(pcov)  

a_avg = np.mean(params[:, 0])

print("-------------------------------------------------------------------")
print(f"a_avg = {a_avg:.4e}")
print("-------------------------------------------------------------------")

plt.plot(Hs, params[:, 0], color = "cornflowerblue", label = "Fitparamter")

x = np.linspace(0, 36, 1000)

def f(x, a, b):
    return a/np.cos(x) + b*np.tan(x)

plt.plot(x, f(x/180*np.pi, params[0,0], .5*params[0,0]), label = "Theoriekurve", ls = "dashed", c = "firebrick")
plt.ylabel(r"$a_0$ [mm$^{-1}$]")
plt.xlabel(r"$\Theta$ [°]")
plt.xlim(0,36)
plt.ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("build/absorption_coefficient.pdf")
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
plt.ylabel("Intensität [a.u.]")
plt.xlabel("Horizontaler Winkel [°]");
plt.vlines(data[data[:,1] ==max(data[:,1])][0,0], 0, 100, ls = "dashed", color = "firebrick", label = "Maximum")
plt.ylim(0, 100)
plt.xlim(-0.5,44.5)
plt.legend()
plt.tight_layout()
plt.savefig("build/intensity_angle.pdf")
plt.close()
