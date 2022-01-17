import torch


def _init_module_weights(module, gain="relu"):
    gain_init = 1 if gain == "constant" else torch.nn.init.calculate_gain(gain)
    torch.nn.init.orthogonal_(module.weight.data, gain=gain_init)
    torch.nn.init.constant_(module.bias.data, 0)
    return module


def _init_gru(gru_module):
    for name, param in gru_module.named_parameters():
        if "bias" in name:
            torch.nn.init.constant_(param, 0)
        elif "weight" in name:
            torch.nn.init.orthogonal_(param)
    return gru_module


class BaseNetwork(torch.nn.Module):
    def __init__(self, first_layer_filters, second_layer_filters, out_features, observation_mean, observation_std):
        super(BaseNetwork, self).__init__()

        self.conv1 = _init_module_weights(
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=first_layer_filters,
                kernel_size=8,
                stride=4
            ),
            gain="leaky_relu",
        )
        self.conv2 = _init_module_weights(
            torch.nn.Conv2d(
                in_channels=first_layer_filters,
                out_channels=second_layer_filters,
                kernel_size=4,
                stride=2,
            ),
            gain="leaky_relu",
        )
        self.conv3 = _init_module_weights(
            torch.nn.Conv2d(
                in_channels=second_layer_filters,
                out_channels=second_layer_filters,
                kernel_size=3,
                stride=1,
            ),
            gain="leaky_relu",
        )
        self.bn1 = torch.nn.BatchNorm2d(first_layer_filters)
        self.bn2 = torch.nn.BatchNorm2d(second_layer_filters)

        self.mean = observation_mean
        self.std = observation_std

        self.fully_connected = _init_module_weights(
            torch.nn.Linear(64 * 7 * 7, out_features), gain="leaky_relu"
        )
        self.lrelu = torch.nn.LeakyReLU(inplace=True)

    def forward(self, inputs):
        new_input = inputs.type(torch.float32)
        new_input = (new_input - self.mean) / (self.std + 1e-6)

        conv1_out = self.conv1(new_input)
        self.lrelu(conv1_out)
        conv1_out = self.bn1(conv1_out)

        conv2_out = self.conv2(conv1_out)
        self.lrelu(conv2_out)
        conv2_out = self.bn2(conv2_out)

        conv3_out = self.conv3(conv2_out)
        self.lrelu(conv3_out)
        conv3_out = self.bn2(conv3_out)

        fc_input = conv3_out.view(conv3_out.size(0), -1)

        linear_out = self.fully_connected(fc_input)
        self.lrelu(linear_out)

        return linear_out


class GRUNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_state_size, action_size, num_envs):
        super(GRUNetwork, self).__init__()

        self.gru = _init_gru(
            torch.nn.GRU(
                input_size=input_size,
                hidden_size=hidden_state_size,
                num_layers=1,
                batch_first=True,
            )
        )
        self.num_envs = num_envs

    def forward(self, inputs, last_hidden_state):
        """
        During training process, when obtaining next state, pass inputs as batch through GRU cells.
        Obtaining value and policy on stacked observation requires processing as a sequence.
        """

        if inputs.size(0) == self.num_envs:
            batch_seq = inputs.unsqueeze(1)
        else:
            batch_seq = inputs.unsqueeze(0)

        output, hidden_state = self.gru(batch_seq, last_hidden_state)

        if inputs.size(0) == self.num_envs:
            output = output.squeeze(1)
        else:
            output = output.squeeze(0)

        return output, hidden_state


class ValueNetwork(torch.nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()

        self.value = _init_module_weights(
            torch.nn.Linear(in_features=512, out_features=1), gain="constant"
        )

    def forward(self, inputs):
        """
        Return estimated value V(s).
        """

        # create Tensor([8]) out of Tensor(8 x 1)
        value = torch.squeeze(self.value(inputs))
        return value


class PolicyNetwork(torch.nn.Module):
    def __init__(self, action_size):
        super(PolicyNetwork, self).__init__()

        self.fully_connected = _init_module_weights(
            torch.nn.Linear(in_features=512, out_features=action_size), gain="constant"
        )

        self.policy = torch.nn.Softmax(dim=1)

    def forward(self, inputs):
        """
        Return action probabilities.
        """

        fc_out = self.fully_connected(inputs)
        policy = self.policy(fc_out)
        return policy


class FeatureExtractor(torch.nn.Module):
    def __init__(self, num_of_filters, output_size, obs_mean, obs_std):
        super(FeatureExtractor, self).__init__()

        self.conv_f = torch.nn.Conv2d(3, num_of_filters, kernel_size=8, stride=4)
        self.conv_s = torch.nn.Conv2d(
            num_of_filters, num_of_filters * 2, kernel_size=4, stride=2
        )
        self.conv_t = torch.nn.Conv2d(
            num_of_filters * 2, num_of_filters * 2, kernel_size=3, stride=1
        )
        self.linear = torch.nn.Linear(in_features=64 * 7 * 7, out_features=288)

        self.lrelu = torch.nn.LeakyReLU(inplace=True)

        self.bn1 = torch.nn.BatchNorm2d(num_of_filters)
        self.bn2 = torch.nn.BatchNorm2d(num_of_filters * 2)
        self.bn3 = torch.nn.BatchNorm2d(num_of_filters * 2)
        self.mean = obs_mean
        self.std = obs_std

        torch.nn.init.xavier_uniform_(self.conv_f.weight)
        torch.nn.init.xavier_uniform_(self.conv_s.weight)
        torch.nn.init.xavier_uniform_(self.conv_t.weight)
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, state):
        """
        Create feature representation.
        """

        state = state.type(torch.float32)
        state = (state - self.mean) / (self.std + 1e-6)

        f_output = self.conv_f(state)
        self.lrelu(f_output)
        f_output = self.bn1(f_output)

        s_output = self.conv_s(f_output)
        self.lrelu(s_output)
        s_output = self.bn2(s_output)

        t_output = self.conv_t(s_output)
        self.lrelu(t_output)
        t_output = self.bn3(t_output)

        flatten = t_output.view(t_output.size(0), -1)
        self.lrelu(flatten)
        features = self.linear(flatten)

        return features


class ForwardModel(torch.nn.Module):
    def __init__(self, f_layer_size):
        super(ForwardModel, self).__init__()

        self.first_layer = torch.nn.Linear(f_layer_size, 256)
        self.hidden = torch.nn.Linear(256, 256 * 2)
        self.second_layer = torch.nn.Linear(256 * 2, 288)
        self.lrelu = torch.nn.LeakyReLU(inplace=True)

        torch.nn.init.xavier_uniform_(self.first_layer.weight)
        torch.nn.init.xavier_uniform_(self.hidden.weight)
        torch.nn.init.xavier_uniform_(self.second_layer.weight)

    def forward(self, features, action_indices):
        concat_features = torch.cat((features, action_indices), dim=1)

        intermediate_res = self.first_layer(concat_features)
        self.lrelu(intermediate_res)

        hidden_f = self.hidden(intermediate_res)
        self.lrelu(hidden_f)

        predicted_state = self.second_layer(hidden_f)

        return predicted_state


class InverseModel(torch.nn.Module):
    def __init__(self, f_layer_size, action_size):
        super(InverseModel, self).__init__()

        self.f_layer = torch.nn.Linear(f_layer_size, 256)
        self.hidden_1 = torch.nn.Linear(256, 256 * 2)
        self.hidden_2 = torch.nn.Linear(256 * 2, 256 * 2)
        self.s_layer = torch.nn.Linear(256 * 2, action_size)

        self.lrelu = torch.nn.LeakyReLU(inplace=True)
        self.softmax = torch.nn.Softmax(dim=1)

        torch.nn.init.xavier_uniform_(self.f_layer.weight)
        torch.nn.init.xavier_uniform_(self.hidden_1.weight)
        torch.nn.init.xavier_uniform_(self.hidden_2.weight)
        torch.nn.init.xavier_uniform_(self.s_layer.weight)

    def forward(self, state_features, new_state_features):
        concat_features = torch.cat((state_features, new_state_features), dim=1)

        f_output = self.f_layer(concat_features)
        self.lrelu(f_output)

        hidden_1_out = self.hidden_1(f_output)
        self.lrelu(hidden_1_out)

        hidden_2_out = self.hidden_2(hidden_1_out)
        self.lrelu(hidden_2_out)

        softmax_output = self.s_layer(hidden_2_out)
        predicted_action = self.softmax(softmax_output)

        return predicted_action
