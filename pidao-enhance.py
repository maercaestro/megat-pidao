"""
By Abu Huzaifah Bidin
Enhance PIDAO as suggested by the Paper. 

"""

import torch
from torch.optim.optimizer import Optimizer


class EnhancedPIDAO(Optimizer):
    def __init__(self, params, lr=1e-3, kp=1.0, ki=0.1, kd=0.01, a=0.1, c=0.1):
        """
        Enhanced PIDAO Optimizer

        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            a: Damping coefficient
            c: Velocity coefficient (for enhanced formulation)
        """
        defaults = dict(lr=lr, kp=kp, ki=ki, kd=kd, a=a, c=c)
        super(EnhancedPIDAO, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step using the Enhanced PIDAO optimizer.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            kp = group['kp']
            ki = group['ki']
            kd = group['kd']
            a = group['a']
            c = group['c']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Initialize state if not already done
                if 'velocity' not in state:
                    state['velocity'] = torch.zeros_like(p.data)
                    state['integral'] = torch.zeros_like(p.data)
                    state['prev_grad'] = torch.zeros_like(p.data)

                velocity = state['velocity']
                integral = state['integral']
                prev_grad = state['prev_grad']

                # Update integral and derivative terms
                integral += grad
                derivative = grad - prev_grad

                # Enhanced PIDAO Update Rule
                update = kp * grad + ki * integral + kd * derivative
                velocity = a * velocity - lr * update  # Standard PID contribution
                enhanced_update = c * velocity + grad  # Add velocity term for Enhanced PIDAO

                # Parameter update
                p.data -= lr * enhanced_update

                # Save state
                state['velocity'] = velocity
                state['integral'] = integral
                state['prev_grad'] = grad

        return loss