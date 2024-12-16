"""
By Abu Huzaifah Bidin
PIDAO Base in reference to PIDAO paper published by China University
which you can referred to in this github.

"""
import torch
from torch.optim.optimizer import Optimizer

class PIDAO(Optimizer):
    def __init__(self,params, lr=1e-3, kp =1, ki=0.1, kd=0.01, a=0.1):
        """
        PIDAO (Proportional Integral Derivative Accelerated Optimizer

        This optimizer includes PID algorithm on top of classical physical
        model optimizer. Based on the paper by several scientists from
        China

        Args:
        params: Iterable of parameters to optimize
        lr = learning rate
        kp: Proportional gain
        ki = integral gain
        kd: derivative gain
        a: Damping coefficient

        """
        defaults = dict(lr=lr, kp=kp, ki=ki, kd=kd, a=a)
        super(PIDAO,self).__init__(params,defaults)


    @torch.no_grad()
    def step(self,closure = None):
        """
        Performs a single optimization step using PIDAO optimizer for each time this function was called

        Args:
            closure: A closure that re-evaluates the model and return the loss

        """
        #1. update the loss function
        loss = None #initialize the loss
        if closure is not None:
            with torch.enable_grad():
                loss = closure() #update the loss based on the closure

        #2. update the parameters
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

        #3. state initialization
                if len(state) == 0:
                    state['velocity'] = torch.zeros_like(p.data)
                    state['integral'] = torch.zeros_like(p.data)
                    state['prev_grad'] = torch.zeros_like(p.data)

        #4. retrieve the state variables
                velocity = state['velocity']
                integral = state['integral']
                prev_grad = state['prev_grad']

        #5. get the hyperparameters
                lr = group['lr']
                kp = group['kp']
                ki = group['ki']
                kd = group['kd']
                a = group['a']
                
                
         #6. update all the PID parameters
                integral += grad

                
                derivative = grad - prev_grad

                # PID control update
                update = kp * grad + ki * integral + kd * derivative

                # Velocity update (momentum + damping)
                velocity = a * velocity - lr * update

                # Parameter update
                p.data += velocity

                # Save state
                state['velocity'] = velocity
                state['integral'] = integral
                state['prev_grad'] = grad

        return loss