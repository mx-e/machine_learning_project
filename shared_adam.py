"""
Shared optimizer, the parameters in the optimizer will shared in the multiprocessors.
"""

import torch


class SharedAdam(torch.optim.RMSprop):
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False):
        super(SharedAdam, self).__init__(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['square_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['square_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

