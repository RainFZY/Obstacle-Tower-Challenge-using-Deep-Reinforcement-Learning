import torch
import random
import numpy as np


class ExperienceMemory:
    def __init__(self, num_envs, memory_size, action_size, device):
        self._init_memory(num_envs, memory_size, action_size, device)
        self.memory_size = memory_size
        self.action_size = action_size
        self.num_envs = num_envs
        self.memory_pointer = 0
        self._last_hidden_state = None
        self.device = device

    @property
    def last_hidden_state(self):
        return self._last_hidden_state

    @last_hidden_state.setter
    def last_hidden_state(self, value):
        self._last_hidden_state = value

    def _init_memory(self, num_envs, memory_size, action_size, device):
        self.frame = (
            torch.zeros((memory_size, num_envs, 3, 84, 84)).type(torch.uint8).to(device)
        )
        self.reward = torch.zeros((memory_size, num_envs))
        self.done_state = torch.zeros((memory_size, num_envs))
        self.value = torch.zeros((memory_size, num_envs)).to(device)
        self.action_indices = torch.zeros((memory_size, num_envs, action_size)).to(
            device
        )
        self.policy_values = torch.zeros((memory_size, num_envs, action_size)).to(
            device
        )
        self.state_f = torch.zeros((memory_size, num_envs, 288)).to(device)
        self.new_state_f = torch.zeros((memory_size, num_envs, 288)).to(device)

    def empty(self):
        self.frame[0].copy_(self.frame[-1])
        self.memory_pointer = 0

    def add_experience(
        self,
        new_state,
        old_state,
        reward,
        action_encoding,
        done,
        predicted_value,
        policy,
        state_f,
        new_state_f,
    ):

        self.frame[self.memory_pointer].copy_(new_state)
        self.reward[self.memory_pointer].copy_(reward)
        self.value[self.memory_pointer].copy_(predicted_value)
        self.action_indices[self.memory_pointer].copy_(action_encoding)
        self.done_state[self.memory_pointer].copy_(done)
        self.policy_values[self.memory_pointer].copy_(policy)
        self.state_f[self.memory_pointer].copy_(state_f)
        self.new_state_f[self.memory_pointer].copy_(new_state_f)

    def increase_frame_pointer(self):
        self.memory_pointer += 1

    def last_states(self):
        return self.frame[self.memory_pointer - 1]

    def a2c_policy_sampling(self, running_reward_std):
        """
        Sample experiences from one randomly chosen environment.
        """

        env = random.randint(0, self.num_envs - 1)
        last_element = self.memory_pointer - 1

        states = self.frame[:last_element, env, :, :, :]
        rewards = self.reward[:last_element, env] / running_reward_std[env]
        values = self.value[:last_element, env]
        action_indices = self.action_indices[:last_element, env, :]
        dones = self.done_state[:last_element, env]
        state_features = self.state_f[:last_element, env, :]
        new_state_features = self.new_state_f[:last_element, env, :]

        return (
            states,
            action_indices,
            rewards,
            values,
            dones,
            state_features,
            new_state_features,
        )

    def ppo_policy_sampling(self, minibatch_size, running_reward_std):
        """
        Sample experiences from number(minibatch_size) environments
        and concatenate those experiences.
        Done specifically for the PPO training.
        """

        batched_value = []
        batched_states = []
        batched_reward = []
        batched_action_indices = []
        batched_policy = []
        batched_dones = []
        batched_state_f = []
        batched_new_state_f = []

        for _ in range(minibatch_size):
            env = random.randint(0, self.num_envs - 1)
            last_element = self.memory_pointer - 1

            states = self.frame[:last_element, env, :, :, :]
            rewards = self.reward[:last_element, env] / running_reward_std[env]
            values = self.value[:last_element, env]
            action_indices = self.action_indices[:last_element, env, :]
            policies = self.policy_values[:last_element, env, :]
            dones = self.done_state[:last_element, env]
            state_features = self.state_f[:last_element, env, :]
            new_state_features = self.new_state_f[:last_element, env, :]

            batched_value.append(values)
            batched_states.append(states)
            batched_reward.append(rewards)
            batched_action_indices.append(action_indices)
            batched_policy.append(policies)
            batched_dones.append(dones)
            batched_state_f.append(state_features)
            batched_new_state_f.append(new_state_features)

        return (
            torch.cat(batched_states, dim=0),
            torch.cat(batched_action_indices, dim=0),
            torch.cat(batched_policy, dim=0),
            batched_reward,
            batched_value,
            batched_dones,
            torch.cat(batched_state_f, dim=0),
            torch.cat(batched_new_state_f, dim=0),
        )

    def compute_returns(self, rewards, values, dones, discount=0.99):
        """
        Calculate discounted returns (rewards from Tn to T0 timestep).
        """

        num_steps = rewards.shape[0]
        masks = 1 - dones

        returns = np.zeros((num_steps))
        if not dones[-1]:
            returns[-1] = values[-1]

        for step in reversed(range(num_steps - 1)):
            returns[step] = (
                rewards[step] + discount * returns[step + 1] * masks[step + 1]
            )

        return returns
