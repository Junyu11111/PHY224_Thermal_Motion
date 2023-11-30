import numpy as np
from matplotlib import pyplot as plt

import Analysis_Module as Analysis
import os


def read_files(data_folder_path):
    data_dict = {}
    for path in os.listdir(data_folder_path):
        sample_dict = {}
        for file in os.listdir(os.path.join(data_folder_path, path)):
            key = file[-7:-4]
            sample_dict[key] = np.loadtxt(
                os.path.join(data_folder_path, path, file),
                unpack=True, skiprows=2)
        data_dict[path[:-5]] = sample_dict
    return data_dict


def step_length(positions):
    lst = [0]
    for i in range(len(positions) - 1):
        lst.append(positions[i + 1] - positions[i])
    return np.array(lst)


def displacement_2d(coordinates):
    print(np.subtract(coordinates[0], coordinates[0][0]))
    return np.subtract(coordinates[0], coordinates[0][0]) ** 2 + np.subtract(
        coordinates[1], coordinates[1][0]) ** 2


def mean_squared_distance(data_dict):
    lst = []
    for ii in data_dict:
        for jj in data_dict[ii]:
            print(displacement_2d(data_dict[ii][jj]))
            lst.append(displacement_2d(data_dict[ii][jj]))
    return np.mean(lst, axis=0)


def step_length_join(data_dict):
    lst = []
    for ii in data_dict:
        for jj in data_dict[ii]:
            print(step_length(data_dict[ii][jj]))
            lst.append(np.sqrt(displacement_2d(data_dict[ii][jj])))
    return np.array(lst).flatten()


def step_length_dict(data_dict):
    b_dict = {}
    for ii in data_dict:
        for jj in data_dict[ii]:
            b_dict[ii + jj] = np.sqrt(displacement_2d(data_dict[ii][jj]))
    return b_dict


def histogram_plot(data, rounded_range, interval):
    plt.xlabel("Number of Counts")
    plt.ylabel("Probability Density")
    # round the range to nearest 10 that in include the all the num in data.
    print(rounded_range, (rounded_range[1] - rounded_range[0]) / interval)
    plt.hist(data, bins=int((rounded_range[1] - rounded_range[0]) / interval),
             density=True,
             range=rounded_range
             , edgecolor='black', label="Data Recorded")


def ceil_dig(num, decimal):
    return np.ceil(num * (10 ** decimal)) * 10 ** (-decimal)


def floor_dig(num, decimal):
    return np.floor(num * (10 ** decimal)) * 10 ** (-decimal)


def bead_plot(displacement_dict):
    for i in displacement_dict:
        plt.figure(i)
        dec = 0
        step_l = displacement_dict[i]
        step_l_r = (floor_dig(min(step_l), dec),
                    ceil_dig(max(step_l), dec))
        histogram_plot(step_l, step_l_r, 1 ** (-dec))


if __name__ == "__main__":
    msd = mean_squared_distance(read_files("tracking_data"))
    time = np.arange(0, 60, 0.5)
    plt.figure("msd vs time")
    Analysis.plot_x_vs_y(time, 0, msd, 0, "mean squared distance", None)

    step_length = step_length_join(read_files("tracking_data"))
    print("step_len:", step_length)
    plt.figure("histogram")
    decimal = -1
    step_length_range = (floor_dig(min(step_length), decimal),
                         ceil_dig(max(step_length), decimal))
    histogram_plot(step_length, step_length_range, 10 ** (-decimal))
    bead_plot(step_length_dict(read_files("tracking_data")))
    plt.show()
