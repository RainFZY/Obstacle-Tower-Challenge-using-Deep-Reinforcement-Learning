import cv2
import torch
import numpy as np
import itertools

from tqdm import tqdm
from obstacle_tower_env import ObstacleTowerEnv
from agent import definitions


def prepare_state(state):
    """
    Downsize, convert array to pytorch.Tensor and reshape it as (C, H, W)
    """

    frame = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)
    height, width, channel = frame.shape
    frame = frame * 255
    reshaped_frame = np.reshape(frame.astype(np.uint8), (channel, height, width))
    return reshaped_frame


def create_action_space():
    actions = itertools.product(
        definitions.ACTION_MOVE,
        definitions.ACTION_STRAFE,
        definitions.ACTION_TURN,
        definitions.ACTION_JUMP,
    )
    action_space = [list(action) for action in actions]
    return action_space


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def observation_mean_and_std(num_steps, config):
    """
    Collect experiences from environment for num_steps and calculate
    observation mean and standard deviation.
    Used for state normalization during training and inference phases.
    """

    env_path = definitions.OBSTACLE_TOWER_PATH
    env = ObstacleTowerEnv(env_path, config=config, retro=False,
                           realtime_mode=False, worker_id=20)
    env.reset()

    observations = []
    for _ in tqdm(range(num_steps)):
        act = env.action_space.sample()
        obs, _, done, _ = env.step(act)
        state, key, time, _ = obs
        if done:
            env.reset()

        observations.append(prepare_state(state))

    env.close()
    stacked = np.stack(observations)
    return np.mean(stacked), np.std(stacked)
