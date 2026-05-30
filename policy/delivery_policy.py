import numpy as np
import torch

from agent.mappo import MAPPO
from environ.utils import compute_data_rate
from .bga import BGA


class DeliveryPolicy:
    def __init__(self, *args, **kwargs):
        self.steps = 0

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
        self.agent = MAPPO(
            name="mappo_delivery",
            num_agents=args.num_vehicles,
            num_actions=env.num_rats,
            action_dim=2,
            state_dim=env.state_dim,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
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
        return self.agent.act(states, masks)

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
    ):
        super().store_transition()
        self.agent.buffer.add(
            states,
            masks,
            actions,
            log_probs,
            rewards,
            next_states,
            dones,
            violations,
        )

    def train(self, *args, **kwargs):
        self.agent.update()
        return super().train(*args, **kwargs)

    def model(self):
        return self.agent.actor.state_dict()


class RATSelection(DeliveryPolicy):
    def __init__(self, args, env, writer=None):
        super().__init__()
        self.args = args
        self.num_vehicles = args.num_vehicles
        self.num_rats = env.num_rats
        self.agent = MAPPO(
            name="mappo_delivery",
            num_agents=args.num_vehicles,
            num_actions=1,
            action_dim=self.num_rats + 1,  # +1 for idling action
            state_dim=env.state_dim,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
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

    def mask_convertion(self, masks):
        # Convert mask from (num_agents, num_rats, 2) to (num_agents, 5)
        # where 1 indicates the action is restricted
        # Action 0 (idling) is always available, actions 1-4 correspond to RATs 0-3
        batch_size = masks.size(0)
        new_masks = torch.zeros(
            batch_size, self.num_rats + 1, dtype=masks.dtype, device=masks.device
        )
        # Action 0 (idling) is always available (mask = 0)
        new_masks[:, 0] = 0
        # Actions 1-4 correspond to RATs 0-3

        # force disable if masks[:, :, 1] is 1
        new_masks[:, 1:] = masks[:, :, 1]  #  = 1 later * with -1e10 in actor

        # force enable if masks[:, :, 0] is 1
        new_masks[:, 1:] = new_masks[:, 1:] - masks[:, :, 0] * 2  #

        return new_masks

    def action_convertion(self, actions):
        """
        Convert from RAT selection format to one-hot combination format
        Input: actions (num_agents, 1) - values 0 to num_rats
               - 0 = idle (no RAT selected)
               - 1 to num_rats = select RAT 0 to num_rats-1
        Output: one_hot_actions (num_agents, num_rats) - binary encoding
               - [0,0,0,0] = idle
               - [1,0,0,0] = RAT 0 selected
               - [0,1,0,0] = RAT 1 selected, etc.
        """
        batch_size = actions.size(0)
        one_hot_actions = torch.zeros(
            batch_size, self.num_rats, dtype=torch.long, device=actions.device
        )

        # Go through each agent
        for i in range(batch_size):
            action_value = actions[i, 0].item()  # Get the action value

            if action_value == 0:
                # Idle: keep all zeros [0,0,0,0]
                pass  # one_hot_actions[i] is already zeros
            else:
                # Select RAT: convert action 1-4 to RAT index 0-3
                rat_index = action_value - 1
                one_hot_actions[i, rat_index] = 1

        return one_hot_actions

    def combination_to_action_convertion(self, one_hot_actions):
        """
        Convert from one-hot combination format back to RAT selection format
        Input: one_hot_actions (num_agents, num_rats) - binary encoding
               - [0,0,0,0] = idle
               - [1,0,0,0] = RAT 0 selected
               - [0,1,0,0] = RAT 1 selected, etc.
        Output: actions (num_agents, 1) - values 0 to num_rats
               - 0 = idle (no RAT selected)
               - 1 to num_rats = select RAT 0 to num_rats-1
        """
        batch_size = one_hot_actions.size(0)
        actions = torch.zeros(
            batch_size, 1, dtype=torch.long, device=one_hot_actions.device
        )

        # Go through each agent
        for i in range(batch_size):
            # Check if any RAT is selected (sum > 0)
            if one_hot_actions[i].sum() == 0:
                # All zeros = idle
                actions[i, 0] = 0
            else:
                # Find which RAT is selected (index of the 1)
                rat_index = torch.argmax(one_hot_actions[i]).item()
                actions[i, 0] = rat_index + 1  # Convert RAT index 0-3 to action 1-4

        return actions

    def act(self, states, masks, projection=None):
        super().act()
        masks = self.mask_convertion(masks)

        actions = []
        dists = []

        # Step 1: Get logits from actor and sample actions (RAT selection format)
        for i in range(self.num_vehicles):
            # send to device
            state = states[i].to(self.agent.device)
            mask = masks[i].to(self.agent.device)

            # get the raw logits from the actor
            logit = self.agent.actor(state, mask).squeeze(
                0
            )  # 1 x num_actions x action_dim

            # save dist
            dists.append(torch.distributions.Categorical(logits=logit))

            # Step 2: Sample actions (in RAT selection format)
            actions.append(dists[i].sample())

        actions = torch.stack(actions, dim=0).detach()  # num_agents x num_actions

        # Step 3: Convert to combination form (one-hot format)
        combination_actions = self.action_convertion(actions)

        # Step 4: Apply projection if provided
        if projection is not None:
            valid_combination_actions = projection(combination_actions)
        else:
            valid_combination_actions = combination_actions

        # Step 5: Convert back to RAT selection format (for debugging/verification)
        valid_actions = self.combination_to_action_convertion(valid_combination_actions)

        # Step 6: Calculate log probs using the original sampled actions (before projection)
        log_probs = []
        for i in range(self.num_vehicles):
            log_probs.append(dists[i].log_prob(actions[i]))

        log_probs = torch.stack(log_probs, dim=0).detach()  # num_agents x num_actions

        return valid_combination_actions, log_probs

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
    ):
        super().store_transition()
        self.agent.buffer.add(
            states,
            self.mask_convertion(masks),
            self.combination_to_action_convertion(actions),
            log_probs,
            rewards,
            next_states,
            dones,
            violations,
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
                nearby_vehicles = []

                # search all vehicles in communication range
                nearby_vehicles = [
                    (
                        nearby_vehicle_index,
                        self.env.vehicle_distance[vehicle_index, nearby_vehicle_index],
                    )
                    for nearby_vehicle_index in range(self.env.num_vehicles)
                    if vehicle_index != nearby_vehicle_index
                    and self.env.vehicle_distance[vehicle_index, nearby_vehicle_index]
                    < self.env.v2v_pc5_coverage
                    and self.env.cache[
                        self.env.num_edges + nearby_vehicle_index, requested_item
                    ]
                    == 1
                ]

                # search nearby vehicles that have the requested item

                if len(nearby_vehicles) > 0:
                    min_distance = self.env.v2v_pc5_coverage

                    for nearby_vehicle_index, distance in nearby_vehicles:
                        if distance < min_distance:
                            min_distance = distance

                    # compute the v2v data rate with micro path loss model
                    data_rate = compute_data_rate(
                        allocated_spectrum=self.env.v2v_bandwidth,
                        transmission_power=self.env.v2v_transmission_power,
                        noise_power=self.env.noise_power,
                        distance=min_distance,
                        path_loss_model="micro",
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
            nearby_vehicles = [
                (
                    nearby_vehicle_index,
                    self.env.vehicle_distance[vehicle_index, nearby_vehicle_index],
                )
                for nearby_vehicle_index in range(self.env.num_vehicles)
                if vehicle_index != nearby_vehicle_index
                and self.env.vehicle_distance[vehicle_index, nearby_vehicle_index]
                < self.env.v2v_pc5_coverage
                and self.env.cache[
                    self.env.num_edges + nearby_vehicle_index, requested_item
                ]
                == 1
            ]

            if len(nearby_vehicles) == 0:
                return None

            min_distance = min(distance for _, distance in nearby_vehicles)
            data_rate = compute_data_rate(
                allocated_spectrum=self.env.v2v_bandwidth,
                transmission_power=self.env.v2v_transmission_power,
                noise_power=self.env.noise_power,
                distance=min_distance,
                path_loss_model="micro",
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
                nearby_vehicles = []

                # search all vehicles in communication range
                nearby_vehicles = [
                    (
                        nearby_vehicle_index,
                        self.env.vehicle_distance[vehicle_index, nearby_vehicle_index],
                    )
                    for nearby_vehicle_index in range(self.env.num_vehicles)
                    if vehicle_index != nearby_vehicle_index
                    and self.env.vehicle_distance[vehicle_index, nearby_vehicle_index]
                    < self.env.v2v_pc5_coverage
                    and self.env.cache[
                        self.env.num_edges + nearby_vehicle_index, requested_item
                    ]
                    == 1
                ]

                # search nearby vehicles that have the requested item

                if len(nearby_vehicles) > 0:
                    min_distance = self.env.v2v_pc5_coverage

                    for nearby_vehicle_index, distance in nearby_vehicles:
                        if distance < min_distance:
                            min_distance = distance

                    # compute the v2v data rate with micro path loss model
                    data_rate = compute_data_rate(
                        allocated_spectrum=self.env.v2v_bandwidth,
                        transmission_power=self.env.v2v_transmission_power,
                        noise_power=self.env.noise_power,
                        distance=min_distance,
                        path_loss_model="micro",
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
