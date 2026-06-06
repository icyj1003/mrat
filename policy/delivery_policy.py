import numpy as np
import torch

from agent.mappo import MAPPO
from environ.utils import compute_data_rate, compute_snr
from .bga import BGA

# Action indices map to RAT combinations in environment order:
# [V2N, V2V, PC5, WiFi].
ACTION_TABLE = torch.tensor(
    [
        [0, 0, 0, 0],  # 0: idle
        [0, 1, 0, 0],  # 1: V2V
        [0, 0, 1, 0],  # 2: PC5
        [0, 0, 0, 1],  # 3: WiFi
        [1, 0, 0, 0],  # 4: V2N
        [0, 1, 1, 0],  # 5: V2V + PC5
        [0, 1, 0, 1],  # 6: V2V + WiFi
        [1, 1, 0, 0],  # 7: V2N + V2V
        [0, 0, 1, 1],  # 8: PC5 + WiFi
        [1, 0, 1, 0],  # 9: V2N + PC5
        [1, 0, 0, 1],  # 10: V2N + WiFi
        [0, 1, 1, 1],  # 11: V2V + PC5 + WiFi
        [1, 1, 1, 0],  # 12: V2N + V2V + PC5
        [1, 1, 0, 1],  # 13: V2N + V2V + WiFi
        [1, 0, 1, 1],  # 14: V2N + PC5 + WiFi
        [1, 1, 1, 1],  # 15: all RATs
    ],
    dtype=torch.long,
)
ACTION_TO_ID = {tuple(action.tolist()): idx for idx, action in enumerate(ACTION_TABLE)}

RAT_SELECTION_TABLE = torch.tensor(
    [
        [0, 0, 0, 0],  # 0: idle
        [1, 0, 0, 0],  # 1: V2N
        [0, 1, 0, 0],  # 2: V2V
        [0, 0, 1, 0],  # 3: PC5
        [0, 0, 0, 1],  # 4: WiFi
    ],
    dtype=torch.long,
)
RAT_SELECTION_TO_ID = {
    tuple(action.tolist()): idx for idx, action in enumerate(RAT_SELECTION_TABLE)
}


def _rat_mask_to_action_mask(masks: torch.Tensor) -> torch.Tensor:
    disabled = masks[:, :, 1] > 0.5
    forced = masks[:, :, 0] > 0.5

    action_bits = ACTION_TABLE.to(device=masks.device).bool().unsqueeze(0)

    invalid_due_to_disable = (disabled.unsqueeze(1) & action_bits).any(dim=-1)
    invalid_due_to_force = (forced.unsqueeze(1) & ~action_bits).any(dim=-1)

    invalid_mask = invalid_due_to_disable | invalid_due_to_force
    return invalid_mask.unsqueeze(1).to(dtype=masks.dtype)


def _action_id_to_rat_vector(action_ids: torch.Tensor) -> torch.Tensor:
    flat_action_ids = action_ids.long().reshape(-1)
    action_vectors = ACTION_TABLE.to(device=action_ids.device).index_select(
        0, flat_action_ids
    )
    return action_vectors.view(*action_ids.shape, ACTION_TABLE.size(1))


def _rat_vector_to_action_id(actions: torch.Tensor) -> torch.Tensor:
    flat_actions = actions.long().reshape(-1, ACTION_TABLE.size(1))
    action_ids = [ACTION_TO_ID[tuple(action.tolist())] for action in flat_actions]
    return torch.tensor(action_ids, dtype=torch.long, device=actions.device).view(
        *actions.shape[:-1], 1
    )


def _selection_mask_to_action_mask(masks: torch.Tensor) -> torch.Tensor:
    disabled = masks[:, :, 1] > 0.5
    forced = masks[:, :, 0] > 0.5

    action_mask = torch.zeros(
        masks.size(0),
        RAT_SELECTION_TABLE.size(0),
        dtype=masks.dtype,
        device=masks.device,
    )
    action_mask[:, 1:] = disabled.to(dtype=masks.dtype)
    action_mask[:, 1:] -= 2 * forced.to(dtype=masks.dtype)
    return action_mask.unsqueeze(1)


def _selection_action_id_to_rat_vector(action_ids: torch.Tensor) -> torch.Tensor:
    flat_action_ids = action_ids.long().reshape(-1)
    action_vectors = RAT_SELECTION_TABLE.to(device=action_ids.device).index_select(
        0, flat_action_ids
    )
    return action_vectors.view(*action_ids.shape, RAT_SELECTION_TABLE.size(1))


def _selection_rat_vector_to_action_id(actions: torch.Tensor) -> torch.Tensor:
    flat_actions = actions.long().reshape(-1, RAT_SELECTION_TABLE.size(1))
    action_ids = [
        RAT_SELECTION_TO_ID[tuple(action.tolist())] for action in flat_actions
    ]
    return torch.tensor(action_ids, dtype=torch.long, device=actions.device).view(
        *actions.shape[:-1], 1
    )


def _sample_projected_actions(
    actor,
    states: torch.Tensor,
    masks: torch.Tensor,
    action_table: torch.Tensor,
    action_to_id: dict,
    projection=None,
    device: torch.device | str = "cpu",
):
    states = torch.stack([state.to(device) for state in states], dim=0)
    masks = torch.stack([mask.to(device) for mask in masks], dim=0)

    active_mask = states[:, -1] > 0.5
    active_indices = torch.where(active_mask)[0]

    actions = torch.zeros(
        states.size(0),
        action_table.size(1),
        dtype=torch.long,
        device=device,
    )
    log_probs = torch.zeros(states.size(0), 1, dtype=torch.float32, device=device)

    if active_indices.numel() == 0:
        return actions.cpu(), log_probs.cpu()

    active_states = states[active_indices]
    active_masks = masks[active_indices]

    logits = actor(active_states, active_masks)
    if logits.dim() == 4 and logits.size(0) == 1:
        logits = logits.squeeze(0)

    dist = torch.distributions.Categorical(logits=logits)
    sampled_action_ids = dist.sample().detach()
    sampled_actions = action_table.to(device=device).index_select(
        0, sampled_action_ids.squeeze(-1)
    )

    full_actions = torch.zeros(
        states.size(0),
        action_table.size(1),
        dtype=sampled_actions.dtype,
        device=device,
    )
    full_actions[active_indices] = sampled_actions

    if projection is not None:
        full_actions = projection(full_actions).to(device=device)

    projected_active_actions = full_actions[active_indices.to(device)]
    projected_action_ids = torch.tensor(
        [
            action_to_id[tuple(action.tolist())]
            for action in projected_active_actions.long()
        ],
        dtype=torch.long,
        device=device,
    ).view(-1, 1)

    sampled_log_probs = dist.log_prob(projected_action_ids).detach()

    actions[active_indices] = projected_active_actions
    log_probs[active_indices] = sampled_log_probs

    return actions.cpu(), log_probs.cpu()


class DeliveryPolicy:
    def __init__(self, *args, **kwargs):
        self.steps = 0

    def _nearest_v2v_distance(self, vehicle_index):
        if not hasattr(self, "env") or not hasattr(self.env, "nearest_cache_idx"):
            return None

        nearest_idx = int(self.env.nearest_cache_idx[vehicle_index])
        nearest_dist = float(self.env.nearest_cache_dist[vehicle_index])
        if nearest_idx < 0 or nearest_dist >= self.env.v2v_pc5_coverage:
            return None

        return nearest_idx, nearest_dist

    def act(self, *args, **kwargs):
        self.steps += 1

    def store_transition(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass

    def model(self, *args, **kwargs):
        pass


class RandomDeliveryPolicy(DeliveryPolicy):
    def __init__(self, num_agents, num_actions, action_dim):
        super().__init__()
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.action_dim = action_dim

    def act(self, states, masks, projection=None):
        super().act()
        logits = (
            torch.rand(self.num_agents, self.num_actions, self.action_dim).to(
                states.device
            )
            + masks * -1e10
        )
        distribution = torch.distributions.Categorical(logits=logits)
        actions = distribution.sample()
        # If projection is provided, apply it to the actions
        if projection is not None:
            valid_actions = projection(actions)
        else:
            valid_actions = actions

        # Calculate log probabilities of the actions
        log_probs = distribution.log_prob(valid_actions)
        return valid_actions, log_probs


class AllLinkDeliveryPolicy(DeliveryPolicy):
    def __init__(self):
        super().__init__()

    def act(self, states, masks, projection=None):
        super().act()
        actions = 1 - masks[:, :, 1]
        if projection is not None:
            valid_actions = projection(actions)
        else:
            valid_actions = actions

        # Calculate log probabilities of the actions
        log_probs = torch.zeros_like(valid_actions)
        return valid_actions, log_probs


class MAPPODeliveryPolicy(DeliveryPolicy):
    def __init__(self, args, env, writer=None):
        super().__init__()
        self.args = args
        self.env = env
        self.num_vehicles = args.num_vehicles
        self.num_rats = env.num_rats
        self.agent = MAPPO(
            name="mappo_delivery",
            num_agents=args.num_vehicles,
            num_actions=1,
            action_dim=ACTION_TABLE.size(0),
            state_dim=env.state_dim,
            hidden_dim=args.hidden_dim,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            num_epochs=args.num_epoch,
            clip_range=args.clip_range,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            tau=args.tau,
            entropy_coeff=args.entropy_coeff,
            penalty_coeff=args.penalty_coeff,
            mini_batch_size=args.mini_batch_size,
            max_grad_norm=args.max_grad_norm,
            device=args.device,
            writer=writer,
        )

    def act(self, states, masks, projection=None):
        super().act()
        action_masks = _rat_mask_to_action_mask(masks)
        if (
            projection is None
            and getattr(self.env, "bandwidth_allocation_scheme", "fair_share")
            == "capacity_limit"
        ):
            projection = self.env.bandwidth_constraints_handler
        return _sample_projected_actions(
            self.agent.actor,
            states,
            action_masks,
            ACTION_TABLE,
            ACTION_TO_ID,
            projection=projection,
            device=self.agent.device,
        )

    def store_transition(
        self,
        states,
        masks,
        actions,
        log_probs,
        rewards,
        next_states,
        dones,
        violations,
        active_masks,
    ):
        super().store_transition()
        action_masks = _rat_mask_to_action_mask(masks)
        action_ids = _rat_vector_to_action_id(actions)
        self.agent.buffer.add(
            states,
            action_masks,
            action_ids,
            log_probs,
            rewards,
            next_states,
            dones,
            violations,
            active_masks,
        )

    def train(self, *args, **kwargs):
        self.agent.update()
        return super().train(*args, **kwargs)

    def model(self):
        return self.agent.actor.state_dict()

    def train_from_episodes(self):
        self.agent.train_from_episodes()
        return super().train()


class RATSelection(DeliveryPolicy):
    def __init__(self, args, env, writer=None):
        super().__init__()
        self.args = args
        self.env = env
        self.num_vehicles = args.num_vehicles
        self.num_rats = env.num_rats
        self.agent = MAPPO(
            name="mappo_delivery",
            num_agents=args.num_vehicles,
            num_actions=1,
            action_dim=RAT_SELECTION_TABLE.size(0),
            state_dim=env.state_dim,
            hidden_dim=args.hidden_dim,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            num_epochs=args.num_epoch,
            clip_range=args.clip_range,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            tau=args.tau,
            entropy_coeff=args.entropy_coeff,
            penalty_coeff=args.penalty_coeff,
            mini_batch_size=args.mini_batch_size,
            max_grad_norm=args.max_grad_norm,
            device=args.device,
            writer=writer,
        )

    def act(self, states, masks, projection=None):
        super().act()
        action_masks = _selection_mask_to_action_mask(masks)
        if (
            projection is None
            and getattr(self.env, "bandwidth_allocation_scheme", "fair_share")
            == "capacity_limit"
        ):
            projection = self.env.bandwidth_constraints_handler
        return _sample_projected_actions(
            self.agent.actor,
            states,
            action_masks,
            RAT_SELECTION_TABLE,
            RAT_SELECTION_TO_ID,
            projection=projection,
            device=self.agent.device,
        )

    def store_transition(
        self,
        states,
        masks,
        actions,
        log_probs,
        rewards,
        next_states,
        dones,
        violations,
        active_masks,
    ):
        super().store_transition()
        action_masks = _selection_mask_to_action_mask(masks)
        action_ids = _selection_rat_vector_to_action_id(actions)
        self.agent.buffer.add(
            states,
            action_masks,
            action_ids,
            log_probs,
            rewards,
            next_states,
            dones,
            violations,
            active_masks,
        )

    def train(self, *args, **kwargs):
        self.agent.update()
        return super().train(*args, **kwargs)

    def model(self):
        return self.agent.actor.state_dict()


class TrueAllLink(DeliveryPolicy):
    def __init__(self, args, env, writer=None):
        super().__init__()
        self.args = args
        self.env = env
        self.num_vehicles = env.num_vehicles
        self.num_rats = env.num_rats

    def _to_numpy(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        return tensor

    def act(self, states, masks, projection=None):
        super().act()

        states_np = self._to_numpy(states)
        masks_np = self._to_numpy(masks)
        actions = np.ones((self.num_vehicles, self.num_rats), dtype=np.int64)

        # Force disable actions based on masks
        for i in range(self.num_vehicles):
            for j in range(self.num_rats):
                if masks_np[i, j, 1] == 1:
                    actions[i, j] = 0  # Force disable this RAT
                elif masks_np[i, j, 0] == 1:
                    actions[i, j] = 1  # Force enable this RAT

        # Convert back to tensor
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        log_probs = torch.zeros_like(actions_tensor, dtype=torch.float32)
        return actions_tensor, log_probs

    def store_transition(self, *args, **kwargs):
        return super().store_transition(*args, **kwargs)

    def train(self, *args, **kwargs):
        return super().train(*args, **kwargs)

    def model(self, *args, **kwargs):
        return super().model(*args, **kwargs)


class CheapSel(DeliveryPolicy):
    def __init__(self, args, env, writer=None):
        super().__init__()
        self.args = args
        self.env = env
        self.num_vehicles = env.num_vehicles
        self.num_rats = env.num_rats

    def _to_numpy(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        return tensor

    def act(self, states, masks, projection=None):
        super().act()

        masks_np = self._to_numpy(masks)
        actions = np.zeros((self.num_vehicles, self.num_rats), dtype=np.int64)

        for vehicle_index in range(self.num_vehicles):
            costs = (
                np.ones((4,), dtype=np.float32) * np.inf
            )  # Initialize costs for each RAT with a large number
            # get the requested item index
            requested_item = np.where(self.env.requests_matrix[vehicle_index] == 1)[0]

            # ignore if request has been satisfied
            if self.env.delivery_done[vehicle_index] == 1:
                continue

            # download with v2n
            if masks_np[vehicle_index, 0, 1] == 0:  # if v2n is not restricted
                # compute the distance from the vehicle to the BS
                distance = self.env.bs_distance[vehicle_index]

                # compute v2n data rate with macro path loss model
                data_rate = compute_data_rate(
                    allocated_spectrum=self.env.v2n_bandwidth,
                    transmission_power=self.env.v2n_transmission_power,
                    noise_power=self.env.noise_power,
                    distance=distance,
                    path_loss_model="macro",
                )

                # compute the number of segments that can be transfered
                v2n_transfered_segment = np.floor(
                    data_rate * self.env.dt / self.env.code_size
                )

                # accumulate the cost
                costs[0] = (
                    self.env.v2n_cost * v2n_transfered_segment * self.env.code_size
                )

            # download with v2v
            if masks_np[vehicle_index, 1, 1] == 0:  # if v2v is not restricted
                nearest_v2v = self._nearest_v2v_distance(vehicle_index)
                if nearest_v2v is not None:
                    _, min_distance = nearest_v2v
                    v2v_snr = compute_snr(
                        transmission_power=self.env.v2v_transmission_power,
                        noise_power=self.env.noise_power,
                        distance=min_distance,
                        path_loss_model="micro",
                    )
                    data_rate = compute_data_rate(
                        allocated_spectrum=self.env.v2v_bandwidth,
                        snr_linear=v2v_snr,
                    )

                    # compute the number of segments that can be transfered
                    v2v_transfered_segment = np.floor(
                        data_rate * self.env.dt / self.env.code_size
                    )

                    # accumulate the cost
                    costs[1] = (
                        self.env.v2v_cost * v2v_transfered_segment * self.env.code_size
                    )

            # download with v2i pc5 and vehicle is not out of the road
            if masks_np[vehicle_index, 2, 1] == 0 and self.env.out[vehicle_index] == 0:
                # compute the distance from the vehicle to its local edge
                distance = self.env.local_edge_distance[vehicle_index]

                # compute v2i pc5 data rate with micro path loss model
                data_rate = compute_data_rate(
                    allocated_spectrum=self.env.v2i_pc5_bandwidth,
                    transmission_power=self.env.v2i_pc5_transmission_power,
                    noise_power=self.env.noise_power,
                    distance=distance,
                    path_loss_model="micro",
                )

                # check if the edge has the requested item
                if (
                    self.env.cache[
                        int(self.env.local_of[vehicle_index]), requested_item
                    ]
                    == 1
                ):
                    # compute the number of segments that can be transfered directly from the local edge
                    v2i_pc5_transfered_segment = np.floor(
                        data_rate * self.env.dt / self.env.code_size
                    )

                    # accumulate the collected segments
                    costs[2] = (
                        self.env.v2i_pc5_cost
                        * v2i_pc5_transfered_segment
                        * self.env.code_size
                    )
                # if the edge does not have the requested item
                else:
                    # check for the nearest neighbor edge (by hop count) that has the requested item
                    hop_distance = 99
                    for edge_index in range(self.env.num_edges):
                        if self.env.cache[
                            edge_index, requested_item
                        ] == 1 and edge_index != int(self.env.local_of[vehicle_index]):
                            hop_distance = min(
                                hop_distance,
                                abs(edge_index - int(self.env.local_of[vehicle_index])),
                            )

                    # if there is a neighbor edge that has the requested item
                    if hop_distance < 99 and not self.env.remove_edge_cooperation:
                        v2i_pc5_transfered_segment = np.floor(
                            self.env.dt
                            * self.env.i2i_data_rate
                            * data_rate
                            / (
                                self.env.code_size
                                * (data_rate * hop_distance + self.env.i2i_data_rate)
                            )
                        )
                        # accumulate the cost
                        costs[2] += (
                            v2i_pc5_transfered_segment
                            * self.env.code_size
                            * (self.env.i2i_cost + self.env.v2i_pc5_cost)
                        )

                    # if there is no neighbor edge that has the requested item, use backhaul link
                    else:
                        v2i_pc5_transfered_segment = np.floor(
                            self.env.dt
                            * self.env.i2n_data_rate
                            * data_rate
                            / (
                                self.env.code_size
                                * (data_rate + self.env.i2n_data_rate)
                            )
                        )
                        # accumulate the cost: i2n + v2i_pc5
                        costs[2] += (
                            v2i_pc5_transfered_segment
                            * self.env.code_size
                            * (self.env.i2n_cost + self.env.v2i_pc5_cost)
                        )

            # download with v2i wifi and vehicle is not out of the road
            if masks_np[vehicle_index, 3, 1] == 0 and self.env.out[vehicle_index] == 0:
                # check if the vehicle is within the coverage of the edge wifi
                distance = self.env.local_edge_distance[vehicle_index]

                if distance < self.env.v2i_wifi_coverage:
                    # compute v2i wifi data rate with micro path loss model
                    data_rate = compute_data_rate(
                        allocated_spectrum=self.env.v2i_wifi_bandwidth,
                        transmission_power=self.env.v2i_wifi_transmission_power,
                        noise_power=self.env.noise_power,
                        distance=distance,
                        path_loss_model="micro",
                    )

                    # check if the edge has the requested item
                    if (
                        self.env.cache[
                            int(self.env.local_of[vehicle_index]), requested_item
                        ]
                        == 1
                    ):
                        # compute the number of segments that can be transfered directly from the local edge
                        v2i_wifi_transfered_segment = np.floor(
                            data_rate * self.env.dt / self.env.code_size
                        )

                        # accumulate the collected segments
                        costs[3] = (
                            self.env.v2i_wifi_cost
                            * v2i_wifi_transfered_segment
                            * self.env.code_size
                        )
                    # if the edge does not have the requested item
                    else:
                        # check for the nearest neighbor edge (by hop count) that has the requested item
                        hop_distance = 99
                        for edge_index in range(self.env.num_edges):
                            if self.env.cache[
                                edge_index, requested_item
                            ] == 1 and edge_index != int(
                                self.env.local_of[vehicle_index]
                            ):
                                hop_distance = min(
                                    hop_distance,
                                    abs(
                                        edge_index
                                        - int(self.env.local_of[vehicle_index])
                                    ),
                                )

                        # if there is a neighbor edge that has the requested item
                        if hop_distance < 99 and not self.env.remove_edge_cooperation:
                            v2i_wifi_transfered_segment = np.floor(
                                self.env.dt
                                * self.env.i2i_data_rate
                                * data_rate
                                / (
                                    self.env.code_size
                                    * (
                                        data_rate
                                        + self.env.i2i_data_rate * hop_distance
                                    )
                                )
                            )
                            # accumulate the cost
                            costs[3] = (
                                v2i_wifi_transfered_segment
                                * self.env.code_size
                                * (self.env.i2i_cost + self.env.v2i_wifi_cost)
                            )

                        # if there is no neighbor edge that has the requested item, use backhaul link
                        else:
                            v2i_wifi_transfered_segment = np.floor(
                                self.env.dt
                                * self.env.i2n_data_rate
                                * data_rate
                                / (
                                    self.env.code_size
                                    * (data_rate + self.env.i2n_data_rate)
                                )
                            )
                            # accumulate the cost
                            costs[3] = (
                                self.env.i2n_cost
                                * v2i_wifi_transfered_segment
                                * self.env.code_size
                                + self.env.v2i_wifi_cost
                                * v2i_wifi_transfered_segment
                                * self.env.code_size
                            )
            temp_actions = np.zeros((self.num_rats,), dtype=np.int64)
            temp_actions[torch.argmin(torch.tensor(costs))] = 1
            actions[vehicle_index] = temp_actions

        log_probs = torch.zeros_like(
            torch.tensor(actions, dtype=torch.float32), dtype=torch.float32
        )
        return torch.tensor(actions, dtype=torch.long), log_probs

    def store_transition(self, *args, **kwargs):
        return super().store_transition(*args, **kwargs)

    def train(self, *args, **kwargs):
        return super().train(*args, **kwargs)

    def model(self, *args, **kwargs):
        return super().model(*args, **kwargs)


class GreedyDeliveryPolicy(DeliveryPolicy):
    def __init__(self, args, env, writer=None):
        super().__init__()
        self.args = args
        self.env = env
        self.num_vehicles = env.num_vehicles
        self.num_rats = env.num_rats

    def _to_numpy(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        return tensor

    def _requested_item(self, vehicle_index):
        return int(np.where(self.env.requests_matrix[vehicle_index] == 1)[0][0])

    def _link_info(self, vehicle_index, rat_index):
        requested_item = self._requested_item(vehicle_index)

        if rat_index == 0:
            distance = self.env.bs_distance[vehicle_index]
            data_rate = compute_data_rate(
                allocated_spectrum=self.env.v2n_bandwidth,
                transmission_power=self.env.v2n_transmission_power,
                noise_power=self.env.noise_power,
                distance=distance,
                path_loss_model="macro",
            )
            transferred_segment = np.floor(data_rate * self.env.dt / self.env.code_size)
            cost = self.env.v2n_cost * transferred_segment * self.env.code_size
            return data_rate, cost

        if rat_index == 1:
            nearest_v2v = self._nearest_v2v_distance(vehicle_index)
            if nearest_v2v is None:
                return None

            _, min_distance = nearest_v2v
            v2v_snr = compute_snr(
                transmission_power=self.env.v2v_transmission_power,
                noise_power=self.env.noise_power,
                distance=min_distance,
                path_loss_model="micro",
            )
            data_rate = compute_data_rate(
                allocated_spectrum=self.env.v2v_bandwidth,
                snr_linear=v2v_snr,
            )
            transferred_segment = np.floor(data_rate * self.env.dt / self.env.code_size)
            cost = self.env.v2v_cost * transferred_segment * self.env.code_size
            return data_rate, cost

        if rat_index == 2:
            if self.env.out[vehicle_index] == 1:
                return None

            distance = self.env.local_edge_distance[vehicle_index]
            data_rate = compute_data_rate(
                allocated_spectrum=self.env.v2i_pc5_bandwidth,
                transmission_power=self.env.v2i_pc5_transmission_power,
                noise_power=self.env.noise_power,
                distance=distance,
                path_loss_model="micro",
            )

            if (
                self.env.cache[int(self.env.local_of[vehicle_index]), requested_item]
                == 1
            ):
                transferred_segment = np.floor(
                    data_rate * self.env.dt / self.env.code_size
                )
                cost = self.env.v2i_pc5_cost * transferred_segment * self.env.code_size
                return data_rate, cost

            hop_distance = 99
            for edge_index in range(self.env.num_edges):
                if self.env.cache[
                    edge_index, requested_item
                ] == 1 and edge_index != int(self.env.local_of[vehicle_index]):
                    hop_distance = min(
                        hop_distance,
                        abs(edge_index - int(self.env.local_of[vehicle_index])),
                    )

            if hop_distance < 99 and not self.env.remove_edge_cooperation:
                transferred_segment = np.floor(
                    self.env.dt
                    * self.env.i2i_data_rate
                    * data_rate
                    / (
                        self.env.code_size
                        * (data_rate * hop_distance + self.env.i2i_data_rate)
                    )
                )
                cost = (
                    transferred_segment
                    * self.env.code_size
                    * (self.env.i2i_cost + self.env.v2i_pc5_cost)
                )
                return data_rate, cost

            transferred_segment = np.floor(
                self.env.dt
                * self.env.i2n_data_rate
                * data_rate
                / (self.env.code_size * (data_rate + self.env.i2n_data_rate))
            )
            cost = (
                transferred_segment
                * self.env.code_size
                * (self.env.i2n_cost + self.env.v2i_pc5_cost)
            )
            return data_rate, cost

        if rat_index == 3:
            if self.env.out[vehicle_index] == 1:
                return None

            distance = self.env.local_edge_distance[vehicle_index]
            if distance >= self.env.v2i_wifi_coverage:
                return None

            data_rate = compute_data_rate(
                allocated_spectrum=self.env.v2i_wifi_bandwidth,
                transmission_power=self.env.v2i_wifi_transmission_power,
                noise_power=self.env.noise_power,
                distance=distance,
                path_loss_model="micro",
            )

            if (
                self.env.cache[int(self.env.local_of[vehicle_index]), requested_item]
                == 1
            ):
                transferred_segment = np.floor(
                    data_rate * self.env.dt / self.env.code_size
                )
                cost = self.env.v2i_wifi_cost * transferred_segment * self.env.code_size
                return data_rate, cost

            hop_distance = 99
            for edge_index in range(self.env.num_edges):
                if self.env.cache[
                    edge_index, requested_item
                ] == 1 and edge_index != int(self.env.local_of[vehicle_index]):
                    hop_distance = min(
                        hop_distance,
                        abs(edge_index - int(self.env.local_of[vehicle_index])),
                    )

            if hop_distance < 99 and not self.env.remove_edge_cooperation:
                transferred_segment = np.floor(
                    self.env.dt
                    * self.env.i2i_data_rate
                    * data_rate
                    / (
                        self.env.code_size
                        * (data_rate + self.env.i2i_data_rate * hop_distance)
                    )
                )
                cost = (
                    transferred_segment
                    * self.env.code_size
                    * (self.env.i2i_cost + self.env.v2i_wifi_cost)
                )
                return data_rate, cost

            transferred_segment = np.floor(
                self.env.dt
                * self.env.i2n_data_rate
                * data_rate
                / (self.env.code_size * (data_rate + self.env.i2n_data_rate))
            )
            cost = (
                transferred_segment
                * self.env.code_size
                * (self.env.i2n_cost + self.env.v2i_wifi_cost)
            )
            return data_rate, cost

        return None

    def _vehicle_urgency(self, vehicle_index):
        if self.env.delivery_done[vehicle_index] == 1:
            return 0.0

        remaining_deadline = float(self.env.remaining_deadline[vehicle_index, 0])
        remaining_segments = float(self.env.remaining_segments[vehicle_index, 0])

        if remaining_segments <= 0:
            return 0.0

        return remaining_segments / max(remaining_deadline, 1e-6)

    def _build_action(self, vehicle_index, masks_np, target_rate=None):
        actions = np.zeros((self.num_rats,), dtype=np.int64)
        enabled_rate = 0.0

        candidate_links = []
        for rat_index in range(self.num_rats):
            if masks_np[vehicle_index, rat_index, 1] == 1:
                continue

            link_info = self._link_info(vehicle_index, rat_index)
            if link_info is None:
                continue

            data_rate, cost = link_info
            if masks_np[vehicle_index, rat_index, 0] == 1:
                actions[rat_index] = 1
                enabled_rate += data_rate
            else:
                candidate_links.append((cost, data_rate, rat_index))

        if target_rate is None:
            for _, data_rate, rat_index in candidate_links:
                actions[rat_index] = 1
                enabled_rate += data_rate
            return actions, enabled_rate

        if enabled_rate >= target_rate:
            return actions, enabled_rate

        for _, data_rate, rat_index in sorted(
            candidate_links, key=lambda item: item[0]
        ):
            actions[rat_index] = 1
            enabled_rate += data_rate
            if enabled_rate >= target_rate:
                break

        return actions, enabled_rate

    def act(self, states, masks, projection=None):
        super().act()

        masks_np = self._to_numpy(masks)
        actions = np.zeros((self.num_vehicles, self.num_rats), dtype=np.int64)

        urgencies = np.array(
            [self._vehicle_urgency(i) for i in range(self.num_vehicles)]
        )
        sorted_vehicle_indices = list(np.argsort(-urgencies))

        if (
            len(sorted_vehicle_indices) == 0
            or urgencies[sorted_vehicle_indices[0]] <= 0
        ):
            valid_actions = torch.tensor(actions, dtype=torch.long)
            log_probs = torch.zeros_like(valid_actions, dtype=torch.float32)
            return valid_actions, log_probs

        reference_vehicle = sorted_vehicle_indices[0]
        reference_actions, reference_rate = self._build_action(
            reference_vehicle, masks_np, target_rate=None
        )
        actions[reference_vehicle] = reference_actions
        reference_urgency = max(urgencies[reference_vehicle], 1e-6)

        for vehicle_index in sorted_vehicle_indices[1:]:
            target_rate = reference_rate * urgencies[vehicle_index] / reference_urgency
            vehicle_actions, _ = self._build_action(
                vehicle_index,
                masks_np,
                target_rate=target_rate,
            )
            actions[vehicle_index] = vehicle_actions

        valid_actions = torch.tensor(actions, dtype=torch.long)
        if projection is not None:
            valid_actions = projection(valid_actions)

        log_probs = torch.zeros_like(valid_actions, dtype=torch.float32)
        return valid_actions, log_probs

    def store_transition(self, *args, **kwargs):
        return super().store_transition(*args, **kwargs)

    def train(self, *args, **kwargs):
        return super().train(*args, **kwargs)

    def model(self, *args, **kwargs):
        return super().model(*args, **kwargs)


class GA(DeliveryPolicy):
    def __init__(self, args, env, writer=None):
        super().__init__()
        self.args = args
        self.env = env
        self.num_vehicles = env.num_vehicles
        self.num_rats = env.num_rats

    def _to_numpy(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        return tensor

    def convert_action(self, flattened_actions):
        return flattened_actions.reshape(self.num_vehicles, self.num_rats)

    def project_action(self, flattened_actions, projection, masks):
        actions = self.convert_action(flattened_actions)

        for vehicle_index in range(self.num_vehicles):
            # if force disable, set to 0
            # if force enable, set to 1
            for rat_index in range(self.num_rats):
                if masks[vehicle_index, rat_index, 1] == 1:
                    actions[vehicle_index, rat_index] = 0
                elif masks[vehicle_index, rat_index, 0] == 1:
                    actions[vehicle_index, rat_index] = 1

        if projection is not None:
            actions = projection(actions)

        return actions

    def act(self, states, masks, projection=None):
        super().act()
        actions = np.zeros((self.num_vehicles, self.num_rats), dtype=np.int64)

        def values(arr):
            actions = self.convert_action(arr)
            actions = torch.tensor(actions, dtype=torch.float32)
            for vehicle_index in range(self.num_vehicles):
                # if force disable, set to 0
                # if force enable, set to 1
                for rat_index in range(self.num_rats):
                    if masks[vehicle_index, rat_index, 1] == 1:
                        actions[vehicle_index, rat_index] = 0
                    elif masks[vehicle_index, rat_index, 0] == 1:
                        actions[vehicle_index, rat_index] = 1

            if projection is not None:
                arr = projection(actions)

            costs, collected = self.get_info(arr)
            return np.mean(collected / (costs + 1e-10)).item()

        bga = BGA(
            pop_shape=(20, self.num_vehicles * self.num_rats),
            method=values,
            max_round=15,
            p_m=0.02,
        )

        best_solution, best_value = bga.run()

        actions = self.project_action(
            torch.tensor(best_solution, dtype=torch.float32), projection, masks
        )
        log_probs = torch.zeros_like(actions, dtype=torch.float32)
        return actions, log_probs

    def get_info(self, actions):
        costs = np.zeros(self.num_vehicles)
        collected = np.zeros(self.num_vehicles)

        for vehicle_index in range(self.num_vehicles):
            # get the requested item index
            requested_item = np.where(self.env.requests_matrix[vehicle_index] == 1)[0]

            # ignore if request has been satisfied
            if self.env.delivery_done[vehicle_index] == 1:
                continue

            # download with v2n
            if actions[vehicle_index, 0] == 1:  # if v2n is not restricted
                # compute the distance from the vehicle to the BS
                distance = self.env.bs_distance[vehicle_index]

                # compute v2n data rate with macro path loss model
                data_rate = compute_data_rate(
                    allocated_spectrum=self.env.v2n_bandwidth,
                    transmission_power=self.env.v2n_transmission_power,
                    noise_power=self.env.noise_power,
                    distance=distance,
                    path_loss_model="macro",
                )

                # compute the number of segments that can be transfered
                v2n_transfered_segment = np.floor(
                    data_rate * self.env.dt / self.env.code_size
                )
                collected[vehicle_index] += v2n_transfered_segment

                # accumulate the cost
                costs[vehicle_index] += (
                    self.env.v2n_cost * v2n_transfered_segment * self.env.code_size
                )

            # download with v2v
            if actions[vehicle_index, 1] == 1:  # if v2v is not restricted
                nearest_v2v = self._nearest_v2v_distance(vehicle_index)
                if nearest_v2v is not None:
                    _, min_distance = nearest_v2v
                    v2v_snr = compute_snr(
                        transmission_power=self.env.v2v_transmission_power,
                        noise_power=self.env.noise_power,
                        distance=min_distance,
                        path_loss_model="micro",
                    )
                    data_rate = compute_data_rate(
                        allocated_spectrum=self.env.v2v_bandwidth,
                        snr_linear=v2v_snr,
                    )

                    # compute the number of segments that can be transfered
                    v2v_transfered_segment = np.floor(
                        data_rate * self.env.dt / self.env.code_size
                    )
                    collected[vehicle_index] += v2v_transfered_segment

                    # accumulate the cost
                    costs[vehicle_index] += (
                        self.env.v2v_cost * v2v_transfered_segment * self.env.code_size
                    )

            # download with v2i pc5 and vehicle is not out of the road
            if actions[vehicle_index, 2] == 1 and self.env.out[vehicle_index] == 0:
                # compute the distance from the vehicle to its local edge
                distance = self.env.local_edge_distance[vehicle_index]

                # compute v2i pc5 data rate with micro path loss model
                data_rate = compute_data_rate(
                    allocated_spectrum=self.env.v2i_pc5_bandwidth,
                    transmission_power=self.env.v2i_pc5_transmission_power,
                    noise_power=self.env.noise_power,
                    distance=distance,
                    path_loss_model="micro",
                )

                # check if the edge has the requested item
                if (
                    self.env.cache[
                        int(self.env.local_of[vehicle_index]), requested_item
                    ]
                    == 1
                ):
                    # compute the number of segments that can be transfered directly from the local edge
                    v2i_pc5_transfered_segment = np.floor(
                        data_rate * self.env.dt / self.env.code_size
                    )

                    # accumulate the collected segments
                    collected[vehicle_index] += v2i_pc5_transfered_segment
                    costs[vehicle_index] += (
                        self.env.v2i_pc5_cost
                        * v2i_pc5_transfered_segment
                        * self.env.code_size
                    )
                # if the edge does not have the requested item
                else:
                    # check for the nearest neighbor edge (by hop count) that has the requested item
                    hop_distance = 99
                    for edge_index in range(self.env.num_edges):
                        if self.env.cache[
                            edge_index, requested_item
                        ] == 1 and edge_index != int(self.env.local_of[vehicle_index]):
                            hop_distance = min(
                                hop_distance,
                                abs(edge_index - int(self.env.local_of[vehicle_index])),
                            )

                    # if there is a neighbor edge that has the requested item
                    if hop_distance < 99 and not self.env.remove_edge_cooperation:
                        v2i_pc5_transfered_segment = np.floor(
                            self.env.dt
                            * self.env.i2i_data_rate
                            * data_rate
                            / (
                                self.env.code_size
                                * (data_rate * hop_distance + self.env.i2i_data_rate)
                            )
                        )
                        collected[vehicle_index] += v2i_pc5_transfered_segment
                        # accumulate the cost
                        costs[vehicle_index] += (
                            v2i_pc5_transfered_segment
                            * self.env.code_size
                            * (self.env.i2i_cost + self.env.v2i_pc5_cost)
                        )

                    # if there is no neighbor edge that has the requested item, use backhaul link
                    else:
                        v2i_pc5_transfered_segment = np.floor(
                            self.env.dt
                            * self.env.i2n_data_rate
                            * data_rate
                            / (
                                self.env.code_size
                                * (data_rate + self.env.i2n_data_rate)
                            )
                        )
                        collected[vehicle_index] += v2i_pc5_transfered_segment
                        # accumulate the cost: i2n + v2i_pc5
                        costs[vehicle_index] += (
                            v2i_pc5_transfered_segment
                            * self.env.code_size
                            * (self.env.i2n_cost + self.env.v2i_pc5_cost)
                        )

            # download with v2i wifi and vehicle is not out of the road
            if actions[vehicle_index, 3] == 1 and self.env.out[vehicle_index] == 0:
                # check if the vehicle is within the coverage of the edge wifi
                distance = self.env.local_edge_distance[vehicle_index]

                if distance < self.env.v2i_wifi_coverage:
                    # compute v2i wifi data rate with micro path loss model
                    data_rate = compute_data_rate(
                        allocated_spectrum=self.env.v2i_wifi_bandwidth,
                        transmission_power=self.env.v2i_wifi_transmission_power,
                        noise_power=self.env.noise_power,
                        distance=distance,
                        path_loss_model="micro",
                    )

                    # check if the edge has the requested item
                    if (
                        self.env.cache[
                            int(self.env.local_of[vehicle_index]), requested_item
                        ]
                        == 1
                    ):
                        # compute the number of segments that can be transfered directly from the local edge
                        v2i_wifi_transfered_segment = np.floor(
                            data_rate * self.env.dt / self.env.code_size
                        )

                        collected[vehicle_index] += v2i_wifi_transfered_segment

                        # accumulate the collected segments
                        costs[vehicle_index] = (
                            self.env.v2i_wifi_cost
                            * v2i_wifi_transfered_segment
                            * self.env.code_size
                        )
                    # if the edge does not have the requested item
                    else:
                        # check for the nearest neighbor edge (by hop count) that has the requested item
                        hop_distance = 99
                        for edge_index in range(self.env.num_edges):
                            if self.env.cache[
                                edge_index, requested_item
                            ] == 1 and edge_index != int(
                                self.env.local_of[vehicle_index]
                            ):
                                hop_distance = min(
                                    hop_distance,
                                    abs(
                                        edge_index
                                        - int(self.env.local_of[vehicle_index])
                                    ),
                                )

                        # if there is a neighbor edge that has the requested item
                        if hop_distance < 99 and not self.env.remove_edge_cooperation:
                            v2i_wifi_transfered_segment = np.floor(
                                self.env.dt
                                * self.env.i2i_data_rate
                                * data_rate
                                / (
                                    self.env.code_size
                                    * (
                                        data_rate
                                        + self.env.i2i_data_rate * hop_distance
                                    )
                                )
                            )
                            collected[vehicle_index] += v2i_wifi_transfered_segment
                            # accumulate the cost
                            costs[vehicle_index] = (
                                v2i_wifi_transfered_segment
                                * self.env.code_size
                                * (self.env.i2i_cost + self.env.v2i_wifi_cost)
                            )

                        # if there is no neighbor edge that has the requested item, use backhaul link
                        else:
                            v2i_wifi_transfered_segment = np.floor(
                                self.env.dt
                                * self.env.i2n_data_rate
                                * data_rate
                                / (
                                    self.env.code_size
                                    * (data_rate + self.env.i2n_data_rate)
                                )
                            )
                            collected[vehicle_index] += v2i_wifi_transfered_segment
                            # accumulate the cost
                            costs[vehicle_index] = (
                                self.env.i2n_cost
                                * v2i_wifi_transfered_segment
                                * self.env.code_size
                                + self.env.v2i_wifi_cost
                                * v2i_wifi_transfered_segment
                                * self.env.code_size
                            )

        return costs, collected

    def store_transition(self, *args, **kwargs):
        return super().store_transition(*args, **kwargs)

    def train(self, *args, **kwargs):
        return super().train(*args, **kwargs)

    def model(self, *args, **kwargs):
        return super().model(*args, **kwargs)
