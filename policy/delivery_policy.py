import torch


def random_policy(num_agents, num_actions, action_dim, masks, projection=None):
    """
    Generate random actions for the given number of agents, actions, and action dimension.
    num_agents: Number of agents
    num_actions: Number of actions
    action_dim: Dimension of each action
    masks: Mask for the actions [num_agents x num_actions x action_dim]
    """
    random_logits = torch.rand(num_agents, num_actions, action_dim) + masks * -1e10
    distribution = torch.distributions.Categorical(logits=random_logits)
    actions = distribution.sample()
    # If projection is provided, apply it to the actions
    if projection is not None:
        valid_actions = projection(actions)
    else:
        valid_actions = actions

    # Calculate log probabilities of the actions
    log_probs = distribution.log_prob(valid_actions)
    return valid_actions, log_probs
