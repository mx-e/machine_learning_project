import torch, random
import numpy as np
import torch.nn.functional as F

class RolloutUnit:
    def __init__(self, args, input_conv_module, rollout_conv_module, encoder_module, env_module, policy_output_module, gpu, cpu):
        self.n_rollouts = args.n_rollouts
        self.rollout_depth = args.rollout_depth
        self.num_actions = args.num_actions
        self.input_size = args.input_size
        self.conv_output_size = args.conv_output_size
        self.rollout_lstm_input_size = args.rollout_lstm_input_size
        self.rollout_conv_module = rollout_conv_module
        self.input_conv_module = input_conv_module
        self.encoder_module = encoder_module
        self.env_module = env_module
        self.policy_output_module = policy_output_module
        self.gpu = gpu
        self.cpu = cpu
        self.cuda = args.cuda

    def make_rollout_encoding(self, state):
        concat_encodings = torch.empty(0)
        for i in range(self.n_rollouts):
            encoding = self.make_rollout(state, n_rollout = i)
            concat_encodings = torch.cat((concat_encodings, encoding))
        return concat_encodings.flatten()

    def make_rollout(self, state, n_rollout):
        rollout_states = []
        rollout_values = []
        cur_state = state

        for j in range(self.rollout_depth):
            if j == 0:
                action = range(0,self.num_actions)[n_rollout]
            else:
                logits = self.policy_output_module(self.input_conv_module(state))
                action = torch.exp(logits).multinomial(num_samples=1).data[0] 
            cur_state, value = self.env_module((cur_state.squeeze(0), action))
            rollout_states.append(cur_state)
            rollout_values.append(value)

        hx, cx = None, None
        for rollout_state, rollout_value in zip(reversed(rollout_states), reversed(rollout_values)):
            if self.cuda: rollout_state, rollout_value = rollout_state.to(self.gpu), rollout_value.to(self.gpu)
            processed_state = self.rollout_conv_module(rollout_state).flatten()
            processed_value = rollout_value.repeat(self.input_size).flatten()
            encoder_input = torch.cat((processed_state, processed_value)).view(-1, self.rollout_lstm_input_size)
            hx, cx = self.encoder_module((encoder_input, hx, cx))
            if self.cuda: hx = hx.to(self.cpu)
        return hx