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
    noise_power: float,
    distance: Union[float, np.ndarray],
    path_loss_model: Literal["macro", "micro"] = "macro",
) -> Union[float, np.ndarray]:
    """
    Compute the data rate based on the Shannon-Hartley theorem.
    Args:
        allocated_spectrum (float): Allocated spectrum in Hz.
        transmission_power (float): Transmission power in dBm.
        noise_power (float): Noise power in dBm.
        distance (Union[float, np.ndarray]): Distance in meters.
        path_loss_model (str): Path loss model, either "macro" or "micro".
    Returns:
        float: Data rate in bps.
    """
    if path_loss_model == "macro":
        path_loss = 128.1 + 37.6 * np.log10(max(distance * 1e-3, 1e-6))  # Avoid log(0)
    elif path_loss_model == "micro":
        path_loss = 140.7 + 36.7 * np.log10(max(distance * 1e-3, 1e-6))
    else:
        raise ValueError("Invalid path loss model")

    received_power = transmission_power - path_loss
    noise_power_linear = 10 ** ((noise_power - 30) / 10)
    received_power_linear = 10 ** ((received_power - 30) / 10)

    # Calculate the data rate using Shannon-Hartley theorem
    snr = max(
        received_power_linear / noise_power_linear, 1e-9
    )  # Avoid division by zero
    data_rate = allocated_spectrum * np.log2(1 + snr)
    return data_rate
