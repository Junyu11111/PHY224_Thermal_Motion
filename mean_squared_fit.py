import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

import Analysis_Module
import Analysis_Module as Analysis
import os

import main

def linear_model_function(x, a) -> float:

    return a*x


error_in_position = 0.003 * 10 ** -6
msd, mean_error = main.mean_squared_distance(main.read_files("tracking_data"), error_in_position)
time = np.arange(0, 60, 0.5)

popt_msd, pcov_msd = \
    curve_fit(linear_model_function, time,
              msd, sigma=mean_error,
              absolute_sigma=True)

print(popt_msd)
print(pcov_msd)

plt.errorbar(time, msd , mean_error, marker='o', ls='', label='measured data')
plt.plot(time, linear_model_function(time, *popt_msd), label='linear predicted model')
plt.ylabel("Mean-Squared Distance (m)")
plt.xlabel("Time (s)")
plt.title("Brownian Motion of Beads - Mean-Squared Distance (m) v.s. Time (s)")
plt.legend()
plt.show()
plt.figure()

D = popt_msd/4
print(D)

chi_sq = Analysis_Module.characterize_fit(msd, linear_model_function(time, *popt_msd), mean_error, 1)
print(chi_sq)

resid_error = []

for i in mean_error:
    r_err = float(Analysis_Module.error_prop_addition([i, pcov_msd]))
    resid_error.append(r_err)
# print(resid_error)

plt.errorbar(time, msd - linear_model_function(time, *popt_msd) ,
             resid_error, marker='o', ls='', label='measured data - linear predicted model')
plt.plot(time, np.zeros_like(msd), "g-", label="0 line")
plt.ylabel(" Measured Data Minus Predicted Model of Mean-Squared Distance (m)")
plt.xlabel("Time (s)")
plt.title("Residual: Brownian Motion of Beads - Mean-Squared Distance (m) v.s. Time (s)")
plt.legend()

plt.show()
plt.figure()

