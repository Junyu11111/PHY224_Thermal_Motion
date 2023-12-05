import numpy as np
from _decimal import Decimal
from matplotlib import pyplot as plt

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
    for ii in data_dict:
        for jj in data_dict[ii]:
            lst.append(np.sqrt(displacement_2d(data_dict[ii][jj])))
    return np.array(lst).flatten()


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
    plt.xlabel("r")
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
    return (1 / (4 * t)) * np.mean(sd.flatten()), (1 / (4 * t)) * uncertainty_in_mean(sd_err.flatten())


def d_to_k(d, viscosity, radius, temp):
    k = d * 6 * np.pi * viscosity * radius / temp
    return k


if __name__ == "__main__":
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
    # popt, _, _ = Analysis.curve_fit_plt(time, msd, err_msd, "linear", Analysis.linear_function)
    # print(popt)

    step_length = step_length_join(data_dict)
    plt.figure("histogram")
    decimal = 6
    step_length_range = (floor_dig(min(step_length), decimal),
                         ceil_dig(max(step_length), decimal))
    counts, bin_edges, _ = histogram_plot(step_length, step_length_range, 10 ** (-decimal))
    print(bin_edges)
    r = bin_edges_to_x(bin_edges)
    print(r)
    # sd, sd_err = squared_distance(data_dict, error_in_position)
    # d, err_in_d = maximum_likelihood_estimate_d(sd, sd_err)
    # print(d, err_in_d)
    print(d_to_k(2.23e-13, viscosity, radius, temperature))
    print(counts)
    popt, _, predictions = Analysis.curve_fit_plt(r, counts, np.sqrt(counts), "pdf", probability_density_function)
    print(predictions, popt)
    # bead_plot(step_length_dict(read_files("tracking_data")))
    # plt.show()
    # # bead_plot(step_length_sample(read_files("tracking_data")))
    plt.show()
