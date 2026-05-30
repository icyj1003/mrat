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


def compute_snr(
    transmission_power: float,
    distance: float,
    noise_power: float = -174,
    noise_figure_db: float = 9.0,
    path_loss_model: str = "macro",
):
    """
    Compute SNR in linear scale using path loss and thermal noise.

    Returns:
        snr_linear
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

    # Thermal noise power for a 1 Hz reference bandwidth.
    noise_power_dbm = noise_power + noise_figure_db

    # SNR in dB and linear scale.
    snr_db = rx_power_dbm - noise_power_dbm
    snr_linear = 10 ** (snr_db / 10)
    return snr_linear


def compute_data_rate(
    allocated_spectrum: float,
    snr_linear: float | None = None,
    transmission_power: float | None = None,
    distance: float | None = None,
    noise_power: float = -174,
    noise_figure_db: float = 9.0,
    path_loss_model: str = "macro",
):
    """
    Compute data rate using Shannon capacity.

    Preferred usage:
        compute_data_rate(allocated_spectrum, snr_linear=...)

    Backward-compatible usage:
        compute_data_rate(allocated_spectrum, transmission_power=..., distance=...)
    """

    if snr_linear is None:
        if transmission_power is None or distance is None:
            raise ValueError(
                "compute_data_rate requires snr_linear or (transmission_power and distance)"
            )
        snr_linear = compute_snr(
            transmission_power=transmission_power,
            distance=distance,
            noise_power=noise_power,
            noise_figure_db=noise_figure_db,
            path_loss_model=path_loss_model,
        )

    # Shannon capacity
    data_rate_bps = allocated_spectrum * np.log2(1 + snr_linear)

    return data_rate_bps
