import torch, random
import numpy as np

class RolloutUnit:
    def __init__(self, args, conv_module, encoder_module, env_module, policy_module):
        self.n_rollouts = args.n_rollouts
        self.rollout_depth = args.rollout_depth
        self.num_actions = args.num_actions
        self.input_size = args.input_size
        self.conv_output_size = args.conv_output_size
        self.rollout_lstm_input_size = args.rollout_lstm_input_size
        self.conv_module = conv_module
        self.encoder_module = encoder_module
        self.policy_module = policy_module
        self.env_module = env_module

    def make_rollout_encoding(self, state):
        concat_encodings = torch.empty(0)
        for i in range(self.n_rollouts):
            encoding = self.make_rollout(state)
            concat_encodings = torch.cat((concat_encodings, encoding))
        return concat_encodings.flatten()

    def make_rollout(self, state):
        rollout_states = []
        rollout_values = []
        cur_state = state

        for i in range(self.rollout_depth):
            action = random.sample(range(self.num_actions), 1)[0]
            cur_state, value = self.env_module((cur_state.squeeze(0), action))
            rollout_states.append(cur_state)
            rollout_values.append(value)

        hx, cx = None, None
        for rollout_state, rollout_value in zip(reversed(rollout_states), reversed(rollout_values)):
            processed_state = self.conv_module(rollout_state).flatten()
            processed_value = rollout_value.repeat(self.input_size).flatten()
            encoder_input = torch.cat((processed_state, processed_value)).view(-1, self.rollout_lstm_input_size)
            hx, cx = self.encoder_module((encoder_input, hx, cx))
        return hx