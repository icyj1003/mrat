import torch


def gumbel_sigmoid(logits, tau=1.0):
    """
    Gumbel-Sigmoid function with temperature and hard sampling.
    Args:
        logits (torch.Tensor): Input logits.
        tau (float): Temperature parameter.
        hard (bool): If True, use hard sampling.
    Returns:
        torch.Tensor: Soft action.
    """
    # Compute the Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits))).to(logits.device)

    # Apply the Gumbel trick
    y = (logits + gumbel_noise) / tau

    soft = torch.sigmoid(y)

    return soft


def sigmoid_log_prob(logits, action):
    """
    Get log probability of the sampled output using Gumbel-Sigmoid.
    Args:
        logits (torch.Tensor): Input logits.
        tau (float): Temperature parameter.
    Returns:
        torch.Tensor: Log probability of the sampled output.
    """
    # Compute log-prob manually
    log_prob = action.log() * torch.sigmoid(logits) + (1 - action).log() * (
        1 - torch.sigmoid(logits)
    )

    return log_prob


if __name__ == "__main__":
    # Example usage
    logits = torch.randn(2, 2, 5)
    tau = 0.5

    soft_output, log_prob = gumbel_sigmoid(logits, tau)
    print("Soft Output:", soft_output)
    print("Log Probability:", log_prob)
