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
    distance: float,
    noise_power: float = -174,
    noise_figure_db: float = 9.0,
    path_loss_model: str = "macro",
):
    """
    Compute data rate using Shannon capacity.

    Includes:
        - Path loss
        - Thermal noise
        - Receiver noise figure

    Excludes:
        - Shadowing
        - Fast fading
        - Interference

    Returns:
        data_rate_bps
    """

    distance_km = max(distance * 1e-3, 1e-6)

    if path_loss_model == "macro":
        path_loss_db = 128.1 + 37.6 * np.log10(distance_km)

    elif path_loss_model == "micro":
        path_loss_db = 140.7 + 36.7 * np.log10(distance_km)

    else:
        raise ValueError("Invalid path loss model")

    # Received power (dBm)
    rx_power_dbm = transmission_power - path_loss_db

    # Thermal noise power
    noise_power_dbm = noise_power + 10 * np.log10(allocated_spectrum) + noise_figure_db

    # SNR (dB)
    snr_db = rx_power_dbm - noise_power_dbm

    # SNR (linear)
    snr_linear = 10 ** (snr_db / 10)

    # Shannon capacity
    data_rate_bps = allocated_spectrum * np.log2(1 + snr_linear)

    return data_rate_bps
