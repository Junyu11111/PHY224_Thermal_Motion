import numpy as np
import Analysis_Module as Analysis
import os


def read_files(data_folder_path):
    data_dict = {}
    for path in os.listdir(data_folder_path):
        sample_dict = {}
        for file in os.listdir(os.path.join(data_folder_path, path)):
            key = file[-7:-4]
            sample_dict[key] = np.loadtxt(os.path.join(data_folder_path, path, file),
                                          unpack=True, skiprows=2)
        data_dict[path[:-5]] = sample_dict
    return data_dict


def displacement(positions):
    lst = []
    for i in positions:
        lst.append(i - i[0])
    return np.array(lst)


def displacement_2d(x, y):
    return (x - x[0]) ^ 2 + (y - y[0]) ^ 2


if __name__ == "__main__":
    print(read_files("tracking_data"))
