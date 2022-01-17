import torch

from agent import base_networks


class TowerAgent(torch.nn.Module):
    def __init__(
        self,
        action_size,
        num_envs,
        first_layer_filters,
        second_layer_filters,
        conv_output_size,
        hidden_state_size,
        feature_ext_filters,
        feature_output_size,
        forward_model_f_layer,
        inverse_model_f_layer,
        obs_mean,
        obs_std,
        entropy_coeff=0.001,
        value_coeff=0.5,
        ppo_epsilon=0.2,
        beta=0.8,
        isc_lambda=0.8,
    ):

        super(TowerAgent, self).__init__()

        self.conv_network = base_networks.BaseNetwork(
            first_layer_filters,
            second_layer_filters,
            conv_output_size,
            obs_mean,
            obs_std,
        )
        self.lstm_network = base_networks.GRUNetwork(
            conv_output_size, hidden_state_size, action_size, num_envs
        )
        self.feature_extractor = base_networks.FeatureExtractor(
            feature_ext_filters, feature_output_size, obs_mean, obs_std
        )
        self.forward_model = base_networks.ForwardModel(forward_model_f_layer)
        self.inverse_model = base_networks.InverseModel(
            inverse_model_f_layer, action_size
        )

        self.value = base_networks.ValueNetwork()
        self.policy = base_networks.PolicyNetwork(action_size)
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.mse_loss = torch.nn.MSELoss()
        self.ent_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.ppo_epsilon = ppo_epsilon
        self.beta = beta
        self.isc_lambda = isc_lambda

    def to_cuda(self):
        self.conv_network.cuda()
        self.lstm_network.cuda()
        self.value.cuda()
        self.policy.cuda()
        self.feature_extractor.cuda()
        self.forward_model.cuda()
        self.inverse_model.cuda()

    def act(self, state, last_hidden_state=None):
        """
        Run batch of states (3-channel images) through network to get
        estimated value and policy logs.
        """

        conv_features = self.conv_network(state)
        features, hidden_state = self.lstm_network(conv_features, last_hidden_state)

        value = self.value(features)
        policy = self.policy(features)

        return value, policy, hidden_state

    def icm_act(self, state, new_state, action_indices, eta=0.1):
        """
        Run batch of states (3-channel images) through network to get intrinsic reward.
        Intrinsic reward calculation:
            eta/2 * mean((F'(St+1) - F(St+1))^2)
        """

        state_features = self.feature_extractor(state)
        new_state_features = self.feature_extractor(new_state)

        pred_state = self.forward_model(state_features, action_indices)

        intrinsic_reward = (eta / 2) * self.mse_loss(pred_state, new_state_features)
        return intrinsic_reward, state_features, new_state_features

    def forward_act(self, batch_state_features, batch_action_indices):
        return self.forward_model(batch_state_features, batch_action_indices)

    def inverse_act(self, batch_state_features, batch_new_state_features):
        return self.inverse_model(batch_state_features, batch_new_state_features)

    def ppo_loss(
        self,
        old_policy,
        new_policy,
        advantage,
        returns,
        values,
        action_indices,
        new_state_features,
        new_state_predictions,
        action_predictions,
    ):
        policy_loss = self.ppo_policy_loss(
            old_policy, new_policy, advantage, action_indices
        )
        value_loss = self.value_loss(returns, values)
        entropy = self.entropy(new_policy)

        loss = policy_loss + self.value_coeff * value_loss - self.ent_coeff * entropy
        forward_loss = self.forward_loss(new_state_features, new_state_predictions)
        inverse_loss = self.inverse_loss(action_predictions, action_indices.detach())

        agent_loss = (
            self.isc_lambda * loss
            + (1 - self.beta) * inverse_loss
            + self.beta * forward_loss
        )

        return agent_loss, policy_loss, value_loss, entropy, forward_loss, inverse_loss

    def a2c_loss(
        self,
        policy,
        advantage,
        returns,
        values,
        action_indices,
        new_state_features,
        new_state_predictions,
        action_predictions,
    ):

        policy_loss = self.policy_loss(policy, advantage, action_indices)
        value_loss = self.value_loss(returns, values)
        entropy = self.entropy(policy)

        loss = policy_loss + self.value_coeff * value_loss - self.ent_coeff * entropy
        forward_loss = self.forward_loss(new_state_features, new_state_predictions)
        inverse_loss = self.inverse_loss(action_predictions, action_indices.detach())

        agent_loss = (
            self.isc_lambda * loss
            + (1 - self.beta) * inverse_loss
            + self.beta * forward_loss
        )

        return agent_loss, policy_loss, value_loss, entropy, forward_loss, inverse_loss

    def forward_loss(self, new_state_features, new_state_pred):
        forward_loss = 0.5 * self.mse_loss(new_state_pred, new_state_features)
        return forward_loss

    def inverse_loss(self, pred_acts, action_indices):
        inverse_loss = self.cross_entropy(pred_acts, torch.argmax(action_indices, dim=1))
        return inverse_loss

    def value_loss(self, returns, values):
        return 0.5 * self.mse_loss(values, returns)

    def entropy(self, policy):
        dist = torch.distributions.Categorical
        return dist(probs=policy).entropy().mean()

    def policy_loss(self, policy, advantage, action_indices):
        """
        A2C policy loss calculation: -1/n * sum(advantage * log(policy)).
        """

        policy_logs = torch.log(torch.clamp(policy, 1e-20, 1.0))

        # only take policies for taken actions
        pi_logs = torch.sum(torch.mul(policy_logs, action_indices.cuda()), 1)
        policy_loss = -torch.mean(advantage * pi_logs)
        return policy_loss

    def ppo_policy_loss(self, old_policy, new_policy, advantage, action_indices):
        """
        PPO policy loss calculation: mean(-min(ratio, cut_term)).

        Ratio: Advantage * (new_policy / old_policy).

        Cut term: 1 - epsilon <= Ratio <= 1 + epsilon.
        """

        new_policy = torch.log(torch.clamp(new_policy, 1e-20, 1.0))
        old_policy = torch.log(torch.clamp(old_policy, 1e-20, 1.0))

        # only take polices for taken actions
        policy_logs = torch.sum(torch.mul(new_policy, action_indices), 1)
        old_policy_logs = torch.sum(torch.mul(old_policy, action_indices), 1)

        ratio = torch.exp(policy_logs - old_policy_logs)
        ratio_term = ratio * advantage
        clamp = torch.clamp(ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon)
        clamp_term = clamp * advantage

        policy_loss = -torch.min(ratio_term, clamp_term).mean()
        return policy_loss
