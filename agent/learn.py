import torch
import argparse
import multiprocessing

import agent.definitions as definitions

from agent.trainer import Trainer
from agent.tower_agent import TowerAgent
from agent.experience_memory import ExperienceMemory
from agent.parallel_environment import ParallelEnvironment
from agent.utils import create_action_space, observation_mean_and_std

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Obstacle Tower Agent")

    parser.add_argument(
        "--num_envs",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel environment to train on.",
    )
    parser.add_argument(
        "--experience_memory", type=int, default=128, help="Size of experience memory."
    )
    parser.add_argument(
        "--timesteps", type=int, default=5000000, help="Number of training steps."
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Number of steps per update"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of updates once the experience memory is filled.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=4e-4, help="Learning rate."
    )
    parser.add_argument(
        "--observation_stack_size",
        type=int,
        default=10000,
        help="Number of collected observations before calculating mean and std."
    )
    parser.add_argument(
        "--first_person", type=bool, default=False, help="Use first person camera.")
    parser.add_argument(
        "--ppo", type=bool, default=False, help="Use '\' algorithm for training."
    )
    parser.add_argument("--use_cuda", type=bool, default=True, help="Use GPU training.")

    args = parser.parse_args()

    if args.first_person:
        config = {'agent-perspective': 0}
    else:
        config = {'agent-perspective': 1}

    network_configuration = definitions.network_configuration

    actions = create_action_space()
    action_size = len(actions)

    env_path = definitions.OBSTACLE_TOWER_PATH
    env = ParallelEnvironment(env_path, args.num_envs, config)
    env.start_parallel_execution()
    observation_mean, observation_std = observation_mean_and_std(
        args.observation_stack_size, config)

    agent = TowerAgent(
        action_size,
        args.num_envs,
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
    agent.to_cuda()
    print("Use cuda:", args.use_cuda)
    if args.use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    memory = ExperienceMemory(
        args.num_envs, args.experience_memory, action_size, device
    )

    trainer = Trainer(
        env,
        memory,
        agent,
        actions,
        args.num_envs,  # 8
        args.experience_memory,  # 128
        args.batch_size,  # 128
        args.epochs,  # 4
        args.timesteps,  # 500,0000
        args.learning_rate,
        device,
        args.ppo,
    )
    trainer.train()
