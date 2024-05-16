import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as stds
import pandas as pd

U_leak, I_leak = np.genfromtxt("content/data/leakage_current.txt", unpack = True)

# ----- Plot depletion voltage -----
fig, ax = plt.subplots()

ax.plot(U_leak, I_leak, "x", label="Data")
ax.vlines(80, ymin = 0.8, ymax=2, linestyles='dashed', colors='grey', label="Depletion voltage")

ax.set_xlabel(r"$U \mathbin{/} \unit{\volt}$")
ax.set_ylabel(r"$I \mathbin{/} \unit{\micro\ampere}$")
ax.legend()

fig.tight_layout()
fig.savefig("build/leakage.pdf")
plt.close()

# ----- Plot pedastals etc. -----

pedestal_adc = np.genfromtxt("content/data/Pedestal.txt", unpack=True, delimiter=";")

#check dimensions of ADC
print(f"Shape ADC: {np.shape(pedestal_adc)}")
print(f"Length ADC: {len(pedestal_adc)}")

pedestal = 1/len(pedestal_adc)*np.sum(pedestal_adc, axis=0)
#pedestal = np.mean(pedestal_adc, axis=0)
common_mode_shift = 1/128*np.sum(pedestal_adc-pedestal, axis=1)
noise = np.sqrt(1/(len(pedestal_adc)-1) * np.sum((((pedestal_adc - pedestal).T - common_mode_shift.T).T)**2, axis=0))

# Plots
channels = np.linspace(1,128,128)

# Pedestal plot
fig, ax = plt.subplots()
ax.bar(channels, pedestal)
ax.set_ylim(min(pedestal)-3, max(pedestal)+3)

ax.set_ylabel("Pedestal")
ax.set_xlabel("Channel")

fig.tight_layout()
fig.savefig(f"build/pedestal.pdf")
plt.close()

# Common mode shift
fig, ax = plt.subplots()
ax.hist(common_mode_shift, density=True, bins=40)

ax.set_xlabel("Common mode shift")

fig.tight_layout()
fig.savefig(f"build/common_mode_shift.pdf")
plt.close()

# Noise
fig, ax = plt.subplots()
ax.bar(channels, noise)
ax.set_ylim(min(noise)-0.2, max(noise)+0.2)

ax.set_ylabel("Noise")
ax.set_xlabel("Channel")

fig.tight_layout()
fig.savefig(f"build/noise.pdf")
plt.close()


# ----- Plot Calib -----

index = [5, 20 , 50, 87, 100]
calib_charge = {}
calib_adc = {}
for i in index:
   calib_charge[f"Channel_{i}"], calib_adc[f"Channel_{i}"] = np.genfromtxt(f"content/data/BertschHacheneyTroska/Calib/{i}.txt", unpack=True)

calib_charge["Channel_100_0"], calib_adc["Channel_100_0"] = np.genfromtxt(f"content/data/BertschHacheneyTroska/Calib/100_0.txt", unpack=True)

fig, ax = plt.subplots()
for i in index:
    ax.plot(calib_charge[f"Channel_{i}"], calib_adc[f"Channel_{i}"], "x", ms=0.8, label = f"Channel {i}")

ax.set_xlabel(r"$\text{Charge} \mathbin{/} \mathrm{e}$")
ax.set_ylabel("ADC counts")
ax.legend()

fig.tight_layout()
fig.savefig("build/calib_all_channels.pdf")
plt.close()

fig, ax = plt.subplots()
ax.plot(calib_charge[f"Channel_100_0"], calib_adc[f"Channel_100_0"], "x")
fig.tight_layout()
fig.savefig(f"build/calib_channel_100_0.pdf")
plt.close()

# Plot means
calib_adc_mean = np.mean(np.array([calib_adc["Channel_5"], calib_adc["Channel_20"], calib_adc["Channel_50"], calib_adc["Channel_87"], calib_adc["Channel_100"]]), axis=0)
calib_adc_std = np.std(np.array([calib_adc["Channel_5"], calib_adc["Channel_20"], calib_adc["Channel_50"], calib_adc["Channel_87"], calib_adc["Channel_100"]]), axis=0)

fig, ax = plt.subplots()
ax.errorbar(calib_charge["Channel_5"], calib_adc_mean, yerr=calib_adc_std, elinewidth=0.2, ls="None", marker="x", markersize=2, markeredgewidth=0.3, capsize=0.2, barsabove=True, ecolor="firebrick", label="Mean values")
ax.plot(calib_charge["Channel_100_0"], calib_adc["Channel_100_0"], marker="x", markersize=2, markeredgewidth=0.3, ls="None", label=r"$U_{\mathrm{bias}} = \qty{0}{\volt}$")

ax.set_xlabel(r"$\text{Charge} \mathbin{/} \mathrm{e}$")
ax.set_ylabel("ADC counts")
ax.legend()

fig.tight_layout()
fig.savefig("build/calib_mean.pdf")
plt.close()

# Polyfit means
cut_fit = 250
def polyfit4(x, a, b, c, d, e):
    return a + b*x + c*x**2 + d*x**3 + e*x**4

pparams, pcov = curve_fit(polyfit4, calib_adc_mean[np.where(calib_adc_mean < cut_fit)], calib_charge["Channel_5"][np.where(calib_adc_mean < cut_fit)], p0=[200,200,1, -0.001,1e-5])
perr = np.sqrt(np.diag(pcov))

params = unp.uarray(pparams, perr)

print("-----------------")
print("Params polyfit")
for i in range(5):
    print(f"Parameter {i}: {params[i]}")
print("-----------------")

fig, ax = plt.subplots()

ax.plot(calib_adc_mean, calib_charge["Channel_5"], ls="None", marker="x", markersize=2, markeredgewidth=1, label="Data")
ax.plot(calib_adc_mean, polyfit4(calib_adc_mean, *noms(params)), label="Fit")
ax.vlines(cut_fit, ymin=0, ymax=250e3, ls="dashed", label="Cut fit", color="grey")

ax.set_ylabel(r"$\text{Charge} \mathbin{/} \mathrm{e}$")
ax.set_xlabel("ADC counts")
ax.legend()

fig.tight_layout()
fig.savefig("build/calib_polyfit.pdf")
plt.close()

# ----- Laserscan -----
# Laser delay scan
# ADD LASER DELAY SCAN HERE, laser_delay.txt needs to be formated
laser_t, laser_time_adc = np.genfromtxt("content/data/laser_delay.txt", unpack=True)

fig, ax = plt.subplots()
ax.plot(laser_t, laser_time_adc, marker="x", ms=8, ls="None", label="Data")

ax.vlines(laser_t[8], 0, 100, ls="dashed", colors="grey", label=r"$t=\qty{64}{\nano\second}$")

print(f"Optimal delay laser scan: {laser_t[8]}")

ax.set_xlabel(r"$t \mathbin{/} \unit{\nano\second}$")
ax.set_ylabel("ADC counts")
ax.legend()

fig.tight_layout()
fig.savefig("build/laser_delay.pdf")
plt.close()

# Laser adc
laser_adc = np.genfromtxt("content/data/Laserscan.txt", unpack=True)

print(f"Laserscan shape adc: {np.shape(laser_adc)}")

plt.pcolormesh(laser_adc)
plt.colorbar(label="ADC counts")
plt.xlabel(r"$\text{Distance} \mathbin{/} \qty{10}{\micro\metre}$")
plt.ylabel("Channel")

plt.tight_layout()
plt.savefig("build/laser_scan.pdf")
plt.close()

#magnification plot

plt.pcolormesh(laser_adc)
plt.colorbar(label="ADC counts")

plt.ylim(77,87)

plt.xlabel(r"$\text{Distance} \mathbin{/} \qty{10}{\micro\metre}$")
plt.ylabel("Channel")

plt.tight_layout()
plt.savefig("build/laser_scan_mag.pdf")
plt.close()

# Plot channel 83
fig, ax = plt.subplots()
ax.plot(np.linspace(0,34,35), laser_adc[82,:], label = "Channel 83", marker="x", ms=8, ls="None")

ax.vlines(17, 0, 140, color="grey", label="Maximum", ls="dashed")
ax.vlines(29, 0, 140, color="grey", ls="dashed")

ax.vlines(12, 0, 140, color="gray", label = "Edge", ls="dotted")
ax.vlines(22, 0, 140, color="gray", ls="dotted")

ax.vlines(24, 0, 140, color="gray", ls="dotted")
ax.vlines(34, 0, 140, color="gray", ls="dotted")

ax.set_xlabel(r"$\text{Distance} \mathbin{/} \qty{10}{\micro\metre}$")
ax.set_ylabel("ADC counts")
ax.legend()

fig.tight_layout()
fig.savefig("build/laser_channel_83.pdf")
plt.close()

# Plot channel 84
fig, ax = plt.subplots()
ax.plot(np.linspace(0,34,35), laser_adc[83,:], label = "Channel 84", marker="x", ms=8, ls="None")

ax.vlines(2, 0, 140, color="grey", label="Maximum", ls="dashed")
ax.vlines(13, 0, 140, color="grey", ls="dashed")

ax.vlines(6, 0, 140, color="gray", label = "Edge", ls="dotted")
ax.vlines(8, 0, 140, color="gray", ls="dotted")
ax.vlines(18, 0, 140, color="gray", ls="dotted")

ax.set_xlabel(r"$\text{Distance} \mathbin{/} \qty{10}{\micro\metre}$")
ax.set_ylabel("ADC counts")
ax.legend()

fig.tight_layout()
fig.savefig("build/laser_channel_84.pdf")
plt.close()

# ----- CCE Laser -----
voltage = np.linspace(0,200, 21, dtype=int)

ccel = np.zeros(shape=(21, 128))
for i in voltage:
    ccel[np.where(voltage==i),:] = np.genfromtxt(f"content/data/{i}CCEL.txt", unpack=True)
    

# Plot heatmaps U-channel-cce
plt.pcolormesh(ccel.T)
plt.colorbar(label="ADC counts")
plt.xlabel(r"$U \mathbin{/} \qty{10}{\volt}$")
plt.ylabel("Channel")

plt.tight_layout()
plt.savefig("build/ccel.pdf")
plt.close()

plt.pcolormesh(ccel.T)
plt.colorbar(label="ADC counts")
plt.xlabel(r"$U \mathbin{/} \qty{10}{\volt}$")
plt.ylabel("Channel")

plt.ylim(78,85)

plt.tight_layout()
plt.savefig("build/ccel_mag.pdf")
plt.close()

# Plot CCE channel 82 

fig, ax = plt.subplots()
ax.plot(voltage, ccel[:,81]/np.mean(ccel[8:,81]), marker="x", ms=8, ls="None", label="Data channel 82")

ax.vlines(80, 0, 140/np.mean(ccel[8:,81]), ls="dashed", colors="grey", label=r"$U_{\mathrm{dep}}$")
ax.plot(voltage[8:], (np.mean(ccel[8:,81])+0*voltage[8:])/np.mean(ccel[8:,81]), label="Mean of plateau")

print(f"Mean of plateau (CCE): {np.mean(ccel[8:,81])}")

# - Fit CCE -
D = 300 # sensor thickness
U_dep = 80 # depletion voltage
def CCE(U, a):
    return (1 - np.exp(- (D*np.sqrt(U/U_dep)) / a)) / (1 - np.exp(- D / a))

pparams_cce, pcov_cce = curve_fit(CCE, voltage[:8], ccel[:8,81]/np.mean(ccel[8:,81]), bounds=(0,300))
perr_cce = np.sqrt(np.diag(pcov_cce))
params_cce = unp.uarray(pparams_cce, perr_cce)

print("-----------------")
print("Params CCEL")
for i in range(len(params_cce)):
    print(f"Parameter {i}: {params_cce[i]} um")
print("-----------------")

voltage_fit = np.linspace(voltage[0], voltage[7], 1000)
ax.plot(voltage_fit, CCE(voltage_fit, noms(params_cce[0])), label="Fit")

ax.set_xlabel(r"$U \mathbin{/} \unit{\volt}$")
ax.set_ylabel("CCE (norm.)")
ax.legend()

fig.tight_layout()
fig.savefig("build/ccel_channel_82.pdf")
plt.close()

# ----- CCEQ -----
cceq = np.zeros(21)
for i in voltage:
    df_temp = pd.read_csv(f"content/data/{i}_Cluster_adc_entries.txt", sep="\t", names = ['{}'.format(i) for i in range(128)], skiprows=1)
    df_temp.fillna(0, inplace=True)
    df_temp.insert(0, "Sum", df_temp.sum(axis=1))

    temp_arr= df_temp["Sum"]
    cceq[np.where(i==voltage)] = np.mean(temp_arr) #.sum().mean()

fig, ax = plt.subplots()
ax.plot(voltage, cceq, marker="x", ms=8, ls="None", label="Data")

ax.vlines(80, 0, 100, ls="dashed", colors="grey", label=r"$U_{\mathrm{dep}}$")

ax.set_xlabel(r"$U \mathbin{/} \unit{\volt}$")
ax.set_ylabel("CCE (mean counts)")
ax.legend()

fig.tight_layout()
fig.savefig("build/cceq.pdf")
plt.close()

# ----- Large source scan
# clusters per event
number_cluster = np.genfromtxt("content/data/number_of_clusters.txt", unpack=True)
fig, ax = plt.subplots()

ax.bar(np.linspace(0,127, 128), number_cluster)

ax.set_xlabel("Number of clusters")
ax.set_ylabel("Counts")

ax.set_xlim(-1,5)

fig.tight_layout()
fig.savefig("build/number_cluster.pdf")
plt.close()

# channels per cluster
number_channels = np.genfromtxt("content/data/cluster_size.txt", unpack=True)
fig, ax = plt.subplots()

ax.bar(np.linspace(0,127, 128), number_channels)

ax.set_xlabel("Channels per Cluster")
ax.set_ylabel("Counts")

ax.set_xlim(-1,9)

fig.tight_layout()
fig.savefig("build/number_channels.pdf")
plt.close()

# Hitmap
hitmap = np.genfromtxt("content/data/hitmap.txt", unpack=True)
fig, ax = plt.subplots()

ax.bar(np.linspace(1,128, 128), number_channels)

ax.set_xlabel("Channel")
ax.set_ylabel("Counts")

fig.tight_layout()
fig.savefig("build/hitmap.pdf")
plt.close()

# Sums up the rows and extracts the row averages in numpy array
df_cluster_adc = pd.read_csv("content/data/Cluster_adc_entries.txt", sep="\t", names=['{}'.format(i) for i in range(128)], skiprows=1)
df_cluster_adc.fillna(0, inplace=True)
df_cluster_adc.insert(0, "Sum", df_cluster_adc.sum(axis=1))

cluster_adc = df_cluster_adc["Sum"]

fig, ax = plt.subplots()
ax.hist(cluster_adc, bins = np.arange(1,350,1))

ax.set_xlabel("ADC counts")
ax.set_ylabel("Events")

fig.tight_layout()
fig.savefig("build/cluster_adc.pdf")
plt.close()

# cluster_adc to energy
energy = polyfit4(cluster_adc, *noms(params))
energy =  3.6*energy*1e-3 #3.6 eV energy for e-/h+ generation
fig, ax = plt.subplots()
ax.hist(energy[energy<350], bins=80, histtype="step", label="Energy")

ax.vlines(np.mean(energy), 0, 55e3, ls="dashed", colors="grey", label = r"$E_{\mathrm{mean}} = \qty{136}{\kilo\electronvolt}$")
ax.vlines(89, 0, 55e3, ls="dotted", colors="gray", label = r"$E_{\mathrm{MPV}} = \qty{89}{\kilo\electronvolt}$")

ax.set_xlabel(r"$\text{Energy} \mathbin{/} \unit{\kilo\electronvolt}$")
ax.set_ylabel("Events")
ax.legend()

fig.tight_layout()
fig.savefig("build/cluster_adc_energy.pdf")
plt.close()

print(f"------ CCEQ ------")
print(f"E_mean = {np.mean(energy)} keV")
print(f"E_MPV = {89} keV")
