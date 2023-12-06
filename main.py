import numpy as np
import scipy.optimize
from _decimal import Decimal
from matplotlib import pyplot as plt

import Analysis_Module
import Analysis_Module as Analysis
import os


def read_files(data_folder_path):
    data_dict = {}
    pixel_to_m = 0.12048 * 10 ** -6  # in meters
    for path in os.listdir(data_folder_path):
        sample_dict = {}
        for file in os.listdir(os.path.join(data_folder_path, path)):
            key = file[-7:-4]
            sample_dict[key] = np.loadtxt(
                os.path.join(data_folder_path, path, file),
                unpack=True, skiprows=2) * pixel_to_m
        data_dict[path[:-5]] = sample_dict
    return data_dict


def step_length(positions):
    lst = [0]
    for i in range(len(positions) - 1):
        lst.append(positions[i + 1] - positions[i])
    return np.array(lst)


def displacement_2d(coordinates):
    return np.subtract(coordinates[0], coordinates[0][0]) ** 2 + np.subtract(
        coordinates[1], coordinates[1][0]) ** 2


def step_size(coordinates):
    lst = []
    err_lst = []
    for i in range(len(coordinates[0]) - 1):
        dx = np.subtract(coordinates[0][i], coordinates[0][i + 1])
        dy = np.subtract(coordinates[1][i], coordinates[1][i + 1])
        sq_d = dx ** 2 + dy ** 2
        lst.append(np.sqrt(sq_d))
        err_lst.append(
            Analysis.error_prop_exponent(sq_d,
                                         Analysis.error_prop_addition([
                                             Analysis.error_prop_exponent(dx, error_in_position, 2),
                                             Analysis.error_prop_exponent(dy, error_in_position, 2)
                                         ]), 1 / 2))
    return np.array(lst), np.array(err_lst)


def mean_squared_distance(data_dict, uncertainty_in_position):
    lst, uncertainty_list = squared_distance(data_dict, uncertainty_in_position)
    uncertainty_list = np.array(uncertainty_list).transpose()
    mean_err = []
    for i in uncertainty_list:
        mean_err.append(uncertainty_in_mean(i))
    return np.mean(lst, axis=0), np.array(mean_err)


def squared_distance(data_dict, uncertainty_in_position):
    lst = []
    uncertainty_list = []
    for ii in data_dict:
        for jj in data_dict[ii]:
            lst.append(displacement_2d(data_dict[ii][jj]))
            uncertainty_list.append(
                Analysis.error_prop_addition(
                    [Analysis.error_prop_exponent(data_dict[ii][jj][0],
                                                  uncertainty_in_position, 2),
                     Analysis.error_prop_exponent(data_dict[ii][jj][1],
                                                  uncertainty_in_position, 2)]))
    return np.array(lst), np.array(uncertainty_list)


def uncertainty_in_mean(uncertainties):
    return np.sqrt(np.sum(np.square(uncertainties))) / len(uncertainties)


# def flatten_dict_content(full_dict):
#     lst_x = []
#     lst_y = []
#     for ii in full_dict:
#         if isinstance(full_dict[ii], dict):
#             lst_x.append(flatten_dict_content(full_dict[ii]))
#         else:
#             lst.append(full_dict[ii])
#     return np.array(lst).flatten()


def step_length_join(data_dict):
    lst = []
    err_lst = []
    for ii in data_dict:
        for jj in data_dict[ii]:
            step, err = step_size(data_dict[ii][jj])
            lst.append(step)
            err_lst.append(err)

    return np.array(lst).flatten(), np.array(err_lst).flatten()


def step_length_dict(data_dict):
    b_dict = {}
    for ii in data_dict:
        for jj in data_dict[ii]:
            b_dict[ii + jj] = np.sqrt(displacement_2d(data_dict[ii][jj]))
    return b_dict


def step_length_sample(data_dict):
    s_dict = {}
    for ii in data_dict:
        for jj in data_dict[ii]:
            if ii not in s_dict:
                s_dict[ii] = []
            s_dict[ii].append(np.sqrt(displacement_2d(data_dict[ii][jj])))
        s_dict[ii] = np.array(s_dict[ii]).flatten()
    return s_dict


def histogram_plot(data, rounded_range, interval):
    plt.xlabel("r(m)")
    plt.ylabel("Probability Density")
    # round the range to nearest 10 that in include the all the num in data.
    return plt.hist(data,
                    bins=int((rounded_range[1] - rounded_range[0]) / interval),
                    density=True,
                    range=rounded_range
                    , edgecolor='black', label="Data Recorded")


def ceil_dig(num, decimal):
    # Create the quantize rounding pattern
    decimal = int(decimal)
    rounding_pattern = Decimal('1e-{0}'.format(decimal))
    num_d = Decimal(num)
    return float(num_d.quantize(rounding_pattern, rounding='ROUND_CEILING'))


def floor_dig(num, decimal):
    decimal = int(decimal)
    rounding_pattern = Decimal('1e-{0}'.format(decimal))
    num_d = Decimal(num)
    return float(num_d.quantize(rounding_pattern, rounding='ROUND_FLOOR'))


def probability_density_function(x, d):
    x = np.array(x)
    t = 0.5
    dxt = np.multiply(d, t)
    return x * np.exp(-(x ** 2) / (4 * dxt)) / (2 * dxt)


def bead_plot(displacement_dict):
    for i in displacement_dict:
        print(i, displacement_dict)
        plt.figure(i)
        step_size = 5
        dec = -np.log10(step_size)
        step_l = displacement_dict[i]
        step_l_r = (floor_dig(min(step_l), dec),
                    ceil_dig(max(step_l), dec))
        histogram_plot(step_l, step_l_r, 10 ** (-dec))


def bin_edges_to_x(bin_edges):
    x = []
    for i in range(len(bin_edges) - 1):
        x.append((bin_edges[i + 1] + bin_edges[i]) / 2)
    return np.array(x)


def maximum_likelihood_estimate_d(sd, sd_err):
    t = 0.5
    return (1 / (4 * t)) * np.mean(sd), (1 / (4 * t)) * uncertainty_in_mean(sd_err)


def d_to_k(d, viscosity, radius, temp):
    k = d * 6 * np.pi * viscosity * radius / temp
    return k


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 10.5})

    viscosity = 1e-3
    temperature = 296.5
    radius = 1.9e-6 / 2
    error_in_position = 0.003 * 10 ** -6  # in meters
    data_dict = read_files("tracking_data")
    msd, err_msd = mean_squared_distance(data_dict,
                                         error_in_position)
    time = np.arange(0, 60, 0.5)
    plt.figure("msd vs time")
    Analysis.plot_x_vs_y(time, 0, msd, err_msd, "mean squared distance", None)
    popt, pcov, _ = Analysis.curve_fit_plt(time, msd, err_msd, "linear", Analysis.linear_function)

    step_length, err_in_step = step_length_join(data_dict)
    plt.figure("histogram")
    plt.title(r"Brownian Motion of Beads - Distribution of Step Size for $\Delta t = 0.5 s$")
    decimal = 7
    step_length_range = (floor_dig(min(step_length), decimal),
                         ceil_dig(max(step_length), decimal))
    counts, bin_edges, _ = histogram_plot(step_length, step_length_range, 10 ** (-decimal))
    r = bin_edges_to_x(bin_edges)
    sd = np.square(step_length)
    print(sd)
    sd_err = Analysis.error_prop_exponent(step_length, err_in_step, 2)
    d, err_in_d = maximum_likelihood_estimate_d(sd, sd_err)
    print(d, err_in_d)
    prediction_est = probability_density_function(r, d)
    plt.plot(r, prediction_est, label="Rayleigh Distribution with estimated D")
    print(counts)
    uncertainty_in_pd = np.sqrt(counts / (len(step_length) * (bin_edges[1] - bin_edges[0])))
    popt, pcov = scipy.optimize.curve_fit(probability_density_function, r, counts, sigma=uncertainty_in_pd,
                                          absolute_sigma=True, p0=1.84e-13)
    d_fit = popt
    d_fit_err = np.sqrt(np.diag(pcov))
    prediction_fit = probability_density_function(r, popt)
    plt.plot(r, prediction_fit, label="Rayleigh Distribution Best Fit Curve")
    print(d_fit, d_fit_err)
    # popt, _, predictions = Analysis.curve_fit_plt(r, counts, uncertainty_in_pd, "pdf", probability_density_function)
    # print(predictions, popt)
    # bead_plot(step_length_sample(read_files("tracking_data")))
    plt.tight_layout()
    plt.legend()
    plt.savefig("figs/histogram")

    plt.figure("Residual")
    plt.title(r"Residual: Brownian Motion of Beads - Distribution of Step Size for $\Delta t = 0.5 s$", wrap=True)
    plt.xlabel("r(m)")
    plt.ylabel("Residual")
    uncertainty_estimate = np.sqrt(
        ((err_in_d / d) * prediction_est) ** 2 + counts / (len(step_length) * (bin_edges[1] - bin_edges[0])))
    chi_sq = Analysis.plot_residual(r, counts, uncertainty_estimate, prediction_est,
                                    "Rayleigh Distribution with estimated D",
                                    probability_density_function)
    print("chi_sq:", chi_sq)
    uncertainty_fit = np.sqrt(
        ((d_fit_err / d_fit) * prediction_fit) ** 2 + counts / (len(step_length) * (bin_edges[1] - bin_edges[0])))
    chi_sq = Analysis.plot_residual(r, counts, uncertainty_fit, prediction_fit,
                                    "Rayleigh Distribution with best fit D",
                                    probability_density_function)
    plt.tight_layout()
    print("chi_sq:", chi_sq)
    plt.savefig("figs/residual_histogram")

    plt.figure("uncertainty_range")
    k = d_to_k(d, viscosity, radius, temperature)
    k_err = Analysis.error_prop_multiplication(k,
                                               [[d, err_in_d], [1 / 1000, 0.05 / 1000], [296.5, 0.5], [1.9e-6, 0.1e-6]])
    print(k, k_err)
    k_fit = d_to_k(d_fit, viscosity, radius, temperature)
    k_fit_err = Analysis.error_prop_multiplication(
        k, [[d_fit, d_fit_err], [1 / 1000, 0.05 / 1000], [296.5, 0.5], [1.9e-6, 0.1e-6]])

    plt.title("Calculated Boltzmann Constant with uncertainty")
    boltzmann_constant_dict = {"Distribution of r Bestfit": k_fit,
                               "Maximum Likelihood Estimate": k, "Mean Squared Distance Bestfit": 1.297058907916478e-23}
    boltzmann_constant__err_dict = {"Distribution of r Bestfit": k_fit_err,
                                    "Maximum Likelihood Estimate": k_err,
                                    "Mean Squared Distance Bestfit": 9.418583098284825e-25}
    Analysis.plot_data_range(boltzmann_constant_dict, boltzmann_constant__err_dict, "Boltzmann Constant from")
    y_range = [0, 2]
    x = [1.38e-23, 1.38e-23]
    plt.plot(x, y_range)
    plt.legend(loc='center', bbox_to_anchor=(0.45, 0.75))
    plt.xlabel("J/K")
    plt.savefig("figs/Boltzmann Constant")
    print(boltzmann_constant_dict)
    print(boltzmann_constant__err_dict)
    print(200*np.abs(k_fit-1.38e-23)/(k_fit+1.38e-23))
    print(200 * np.abs(k - 1.38e-23) / (k + 1.38e-23))
    plt.show()
