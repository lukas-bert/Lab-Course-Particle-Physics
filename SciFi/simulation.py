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

# reading data
df_sim = pd.DataFrame()

names = ["y_exit", "z_exit", "x_start", "y_start", "z_start", "px_start", "py_start", "pz_start", "reflCoCl", "reflClCl", "wl", "gpsPosX", "length_core", "length_clad", "rayleighScatterings"]

for i in range(1200): # 1200 files to read
    frame = pd.read_csv(f"content/data/Simulation/job_{i}.txt", sep="\s+", comment = "#", names = names)
    frame["job_nr"] = i
    df_sim = pd.concat([df_sim, frame])

df_sim["index"] = range(0, len(df_sim))# set new index
df_sim.set_index("index", inplace = True)

df_sim["r_exit"] = np.sqrt(df_sim["y_exit"]**2 + df_sim["z_exit"]**2) # calculate exit radius

# prepare data

# 1. remove unphysical events (exit radius > fiber) + plot
_, bins, _ = plt.hist(df_sim["r_exit"], bins = 100, range = [0, 0.5], color = "cornflowerblue", label = "Vor Selektion")

df_sim = df_sim.drop(df_sim[df_sim.r_exit > 0.125].index)

plt.hist(df_sim["r_exit"], bins = bins, color = "firebrick", label = "Nach Selektion", histtype = "step", lw = 2);

plt.xlabel(r"$r_\mathrm{exit}$ [mm]")
plt.ylabel("Intensität [a.u.]")
plt.ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
plt.legend();
plt.tight_layout()
plt.savefig("content/pics/r_exit_cut.pdf")
plt.close()

df_sim = df_sim.drop(df_sim[df_sim["rayleighScatterings"] > 0].index) # drop all events that did rayleigh scattering
df_sim["inCladding"] = df_sim["length_clad"] > 0 # split core and cladding photons
df_sim["Theta"] = np.arccos(df_sim["px_start"])/np.pi*180 # calculate angle theta

# Calculate r_min (distance line-line)

def distance_lineline(v1, v2, p1, p2): # v1, v2, p1, p2 should be numpy arrays
    cross_prod = np.cross(v1, v2)
    if np.all(cross_prod == 0):
        return np.linalg.norm(p1 - p2)
    n = cross_prod / np.linalg.norm(cross_prod)
    return np.abs(np.dot(n, (p1 - p2)))

def r_min(px, py, pz, x, y, z):
    a = np.array([x, y, z])
    p = np.array([px, py, pz])
    return distance_lineline(np.array([1, 0, 0]), p, np.zeros(3), a)

df_sim['r_min'] = df_sim.apply(lambda row: r_min(row['px_start'], row['py_start'], row['pz_start'],
                                                 row['x_start'], row['y_start'], row['z_start']), axis=1)

# save to dataframe which is easier to handle for the plotting
df_sim.to_csv("content/data/Simulation/dataframe.csv")
