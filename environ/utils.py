from typing import Literal, Union
import numpy as np


def zipf(num_items, alpha) -> np.ndarray:
    """
    Generate a Zipf distribution for the given number of items and alpha parameter.
    Args:
        num_items (int): Number of items.
        alpha (float): Zipf distribution parameter.
    Returns:
        np.ndarray: Zipf distribution probabilities.
    """
    z = np.arange(1, num_items + 1)
    zipf_dist = 1 / (z**alpha)
    zipf_dist /= np.sum(zipf_dist)
    return zipf_dist


def compute_data_rate(
    allocated_spectrum: float,
    transmission_power: float,
    noise_power: float,  # thermal noise PSD in dBm/Hz, typically -174
    distance: Union[float, np.ndarray],
    path_loss_model: Literal["macro", "micro"] = "macro",
) -> Union[float, np.ndarray]:
    if path_loss_model == "macro":
        path_loss = 128.1 + 37.6 * np.log10(np.maximum(distance * 1e-3, 1e-6))
    elif path_loss_model == "micro":
        path_loss = 140.7 + 36.7 * np.log10(np.maximum(distance * 1e-3, 1e-6))
    else:
        raise ValueError("Invalid path loss model")

    received_power = transmission_power - path_loss  # dBm

    # Scale noise PSD by bandwidth to get total noise power
    noise_power_total_dbm = noise_power + 10 * np.log10(allocated_spectrum)
    noise_power_linear = 10 ** ((noise_power_total_dbm - 30) / 10)  # Watts
    received_power_linear = 10 ** ((received_power - 30) / 10)  # Watts

    snr = np.maximum(received_power_linear / noise_power_linear, 1e-9)
    return allocated_spectrum * np.log2(1 + snr)
