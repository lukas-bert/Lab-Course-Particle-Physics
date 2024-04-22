import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat
import scipy.constants as const
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as stds
import pandas as pd

# reading data
df_sim = pd.DataFrame()

names = ["y_exit", "z_exit", "x_start", "y_start", "z_start", "px_start", "py_start", "pz_start", "reflCoCl", "reflClCl", "wl", "gpsPosX", "length_core", "length_clad", "rayleighScatterings"]

for i in range(90, 100): # range(1200)
    frame = pd.read_csv(f"content/data/Simulation/job_{i}.txt", sep="\s+", comment = "#", names = names)
    df_sim = df_sim.append(frame)

print(df_sim.head())

# prepare data

df_sim["r_exit"] = np.sqrt(df_sim["y_exit"]**2 + df_sim["z_exit"]**2)
df_sim = df_sim.drop(df_sim[df_sim.r_exit > 0.125].index)

plt.hist(df_sim["r_exit"], bins = 100)
plt.savefig("build/plot.pdf")
