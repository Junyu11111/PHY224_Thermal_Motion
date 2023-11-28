import inspect
from typing import Iterable, Union, List
import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy.optimize import curve_fit


def linear_function(x, a, b):
    return a * x + b


def measurement_error(data: Union[ndarray, Iterable, int, float], last_digit_error: Union[ndarray, Iterable, int, float]
                      , percentage_error: Union[ndarray, Iterable, int, float]) -> any:
    """
    Calculate measurement error based on the given data, last digit error, and percentage error.
    :param data: data
    :param last_digit_error: last digit error
    :param percentage_error: percentage error
    :return: measurement error
    """
    return ((data * percentage_error) ** 2 + last_digit_error ** 2) ** (
            1 / 2)


def characterize_fit(y: Union[ndarray, Iterable, int, float], prediction: Union[ndarray, Iterable, int, float],
                     uncertainty: Union[ndarray, Iterable, int, float], param_num: int) -> any:
    """
    Calculate chi-sq of the model.
    :param y: the data
    :param prediction: the model prediction
    :param uncertainty: the data uncertainty
    :param param_num: num of parameters of the model
    :return: chi-sq
    """
    return np.sum(((y - prediction) / uncertainty) ** 2) / (len(y) - param_num)


def plot_x_vs_y(x: Union[ndarray, Iterable, int, float], x_error: Union[ndarray, Iterable, int, float],
                y: Union[ndarray, Iterable, int, float], y_error: Union[ndarray, Iterable, int, float],
                graph_name: str, model: Union[callable, None]):
    """
    Plot the data with error bar and the best-fit line of the predicted model.
    :param x: x
    :param x_error: error in x
    :param y: y
    :param y_error: error in y
    :param graph_name: name of the data set
    :param model: the theoretical model
    :return: popt, pcov, y-prediction
    """
    plt.errorbar(x, y, xerr=x_error,
                 yerr=y_error, ls='', lw=1, marker='o', markersize=2.5, capsize=1.5, capthick=0.5,
                 label="{} with error bar".format(graph_name), zorder=0, ecolor="grey")
    plt.legend()
    if model is not None:
        return curve_fit_plt(x, y, y_error, graph_name, model)


def curve_fit_plt(x, y, y_error, graph_name, model):
    popt, pcov = curve_fit(model, x, y, sigma=y_error, absolute_sigma=True)
    step = (x[-1] - x[0]) / 100 * len(x)
    x_c = np.arange(x[0] - step, x[-1] + step, step)
    prediction_c = model(x_c, *popt)
    plt.plot(x_c, prediction_c, label="{} best fit line".format(graph_name))
    prediction = model(x, *popt)
    print("prediction:", prediction)
    plt.legend()
    return popt, pcov, prediction


def plot_residual(x: Union[ndarray, Iterable, int, float], y: Union[ndarray, Iterable, int, float],
                  uncertainty: Union[ndarray, Iterable, int, float], prediction: Union[ndarray, Iterable, int, float],
                  graph_name: str, model: Union[callable, None, int]):
    """
    Plot the residual based on data and the prediction.
    :param model: the theoretical model
    :param x: x
    :param y: y
    :param uncertainty: uncertainty in y
    :param prediction: y-prediction
    :param graph_name: name of the data set
    :return: chi_sq
    """
    plt.errorbar(x, y - prediction, yerr=uncertainty, ls='', lw=0.5, marker='o', markersize=2,
                 capsize=1.5, capthick=0.5, ecolor="grey",
                 label="{} Residual".format(graph_name))
    _, labels = plt.gca().get_legend_handles_labels()
    if "0 line" not in labels:
        plt.plot(x, np.zeros_like(y), "g-", label="0 line")
    plt.legend()
    if model is not None:
        if type(model) == int:
            param_num = model
        else:
            param_num = len(inspect.signature(model).parameters) - 1
        chi_sq = characterize_fit(y, prediction, uncertainty, param_num)
        return chi_sq


def plot_data_range(data_dict, uncertainty_dict, name):
    count = 0
    ax = plt.gca()
    ratio = 0.01
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
    ax.axes.get_yaxis().set_visible(False)
    for k in data_dict:
        plt.errorbar(data_dict[k], count, xerr=uncertainty_dict[k], label="{} {}".format(name, k), marker="x",
                     capsize=3,
                     capthick=1)
        count += 1
    plt.legend(fontsize="6.5", fancybox=True, framealpha=0.3)


def round_sig_fig(x: Union[int, float], sig_figs: int) -> any:
    """
    Round x to sig_fig of significant figures
    :param x: x
    :param sig_figs: number of significant figures
    :return: rounded x
    """
    sig_figs -= int((np.ceil(np.log10(abs(x)))))
    return np.round(x, sig_figs)


def error_prop_multiplication(product, variable_list):
    """
    Calculate the uncertainty through multiplication
    :param product: product of the variables
    :param variable_list: [[variable value],[variable uncertainty]]
    :return: uncertainty of product
    """
    ratio_sum = 0
    for variable in variable_list:
        ratio_sum += (variable[1] / variable[0]) ** 2
    return product * np.sqrt(ratio_sum)


def error_prop_addition(uncertainty_list):
    """
    Calculate the uncertainty through addition
    :param uncertainty_list: [uncertainty]
    :return: uncertainty of sum
    """
    square_sum = 0
    for uncertainty in uncertainty_list:
        square_sum += uncertainty ** 2
    return np.sqrt(square_sum)


def plot_directional_graph(x, y, name, num_of_arrows):
    step = int(np.floor(len(x) / (num_of_arrows + 1)))
    for i in range(1, num_of_arrows + 1):
        plt.quiver(x[i * step], y[i * step], x[i * step + 1] - x[i * step], y[i * step + 1] - y[i * step],
                   scale=1,
                   angles='xy', scale_units='xy', zorder=2, headwidth=5
                   )
    plt.plot(x, y, label=name, zorder=1, linestyle='-', marker='o', markersize=1.5, linewidth=1)


def error_prop_exponent(variable, uncertainty, exponent):
    return exponent * (variable ** exponent) * (uncertainty / variable)
