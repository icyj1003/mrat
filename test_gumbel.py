import torch
import torch.nn.functional as F


def gumbel_sigmoid(logits, tau=1.0):
    """
    Apply Gumbel-Softmax to the input logits.

    Args:
        logits (torch.Tensor): Input logits.
        tau (float): Temperature parameter for Gumbel-Softmax.

    Returns:
        torch.Tensor: Gumbel-Softmax output.
    """
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    y = logits + gumbel_noise
    return F.softmax(y / tau, dim=-1)


if __name__ == "__main__":
    # Example usage
    logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    tau = 0.5
    output = gumbel_sigmoid(logits, tau)
    print(output)
