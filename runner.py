import os
import torch
import argparse

import agent.definitions as definitions

from obstacle_tower_env import ObstacleTowerEnv
from agent.tower_agent import TowerAgent
from agent.utils import create_action_space, observation_mean_and_std
from agent.parallel_environment import prepare_state


def sample_action(action_space, policy):
    probs = torch.distributions.Categorical
    index = probs(probs=policy).sample()
    return action_space[index], index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Obstacle Tower Agent")
    parser.add_argument(
        "--model_name",
        type=str,
        default="model_0.bin",
        help="Name of model to use. E.g. model_(num_of_update).bin",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Environment can use seed. Default seed is 0.",
    )
    parser.add_argument(
        "--observation_stack_size",
        type=int,
        default=10000,
        help="Number of collected observations before calculating mean and std.",
    )
    parser.add_argument(
        "--first_person", type=bool, default=False, help="Use first person camera."
    )
    parser.add_argument(
        "--use_cuda",
        type=bool,
        default=True,
        help="Use GPU for inference phase. This will transfer model and tensors to VRAM.",
    )

    args = parser.parse_args()

    if args.first_person:
        config = {"agent-perspective": 0}
    else:
        config = {"agent-perspective": 1}

    inference_envs = 1
    env_path = definitions.OBSTACLE_TOWER_PATH
    model_name = os.path.join(definitions.MODEL_PATH, args.model_name)
    observation_mean, observation_std = observation_mean_and_std(
        args.observation_stack_size, config
    )

    env = ObstacleTowerEnv(env_path, config=config, retro=False, realtime_mode=True)
    env.seed(args.seed)
    env.reset()

    network_configuration = definitions.network_configuration
    actions = create_action_space()
    action_size = len(actions)

    agent = TowerAgent(
        action_size,
        inference_envs,
        network_configuration["first_filters"],
        network_configuration["second_filters"],
        network_configuration["convolution_output"],
        network_configuration["hidden_state_size"],
        network_configuration["feature_extraction_filters"],
        network_configuration["feature_output_size"],
        network_configuration["forward_model_layer"],
        network_configuration["inverse_model_layer"],
        observation_mean,
        observation_std,
    )

    agent.load_state_dict(torch.load(model_name))

    if args.use_cuda:
        agent.to_cuda()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    frame, key, time, _ = env.reset()
    state = torch.Tensor(prepare_state(frame)).unsqueeze(0).to(device)

    value, policy, rhs = agent.act(state)
    action, action_index = sample_action(actions, policy)
    while True:
        for _ in range(definitions.FRAME_SKIP_SIZE):
            obs, reward, done, _ = env.step(action)
            frame, _, _, _ = obs

        state = torch.Tensor(prepare_state(frame)).unsqueeze(0).to(device)
        if done:
            break

        value, policy, rhs = agent.act(state, rhs)
        action, action_index = sample_action(actions, policy)
