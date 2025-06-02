import torch


def random_cache_policy(num_edges, num_items, masks, projection=None):
    random_logits = torch.rand(num_edges * num_items) + masks * -1e10
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
