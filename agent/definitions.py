import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
OBSTACLE_TOWER_DIR = os.path.join(ROOT_DIR, "ObstacleTower")
OBSTACLE_TOWER_PATH = os.path.join(OBSTACLE_TOWER_DIR, "obstacletower")

# MODEL_PATH = os.path.join(ROOT_DIR, "models")
MODEL_PATH = os.path.join(ROOT_DIR, "models\\7_timesteps1000000")

# forward/backward/no-move
ACTION_MOVE = [0, 1, 2]

# left/right/no-move
ACTION_STRAFE = [0, 1, 2]

# clock/counterclock
ACTION_TURN = [0, 1, 2]

# no-op/jump
ACTION_JUMP = [0, 1]

# frame-skipping
FRAME_SKIP_SIZE = 6

# number of updates after certain amoung of timesteps during PPO training
# total updates = num of update x num of epoches
UPDATE_CYCLES = 3

network_configuration = {
    "first_filters": 32,
    "second_filters": 64,
    "convolution_output": 512,
    "hidden_state_size": 512,
    "feature_extraction_filters": 32,
    "feature_output_size": 288,
    "forward_model_layer": 342,
    "inverse_model_layer": 576,
}
