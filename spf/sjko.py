import torch
from collections.abc import Callable
from .utils import clampedlog, euclidean_cost, euclidean_cost_batch
from .optimizers import EulerianOptimizer, LagrangianOptimizer
# from geomloss import SamplesLoss
# import numpy as np

class EulerianSPF:
    def __init__(self, X: torch.Tensor, potential: Callable[[torch.Tensor], torch.Tensor] | torch.Tensor,
                 eps:float, c: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | torch.Tensor = euclidean_cost):
        n = X.size(0)
        if callable(c):
            self.cost_matrix = c(X, X)
            if self.cost_matrix.size() != torch.Size([n, n]):
                raise RuntimeError("c(X,X) should return a matrix of size (X.size(0), X.size(0)).")
        elif type(c) is torch.Tensor:
            if c.size() != torch.Size([n]*2):
                raise RuntimeError("Cost matrix c should be of size (X.size(0), X.size(0)).")
            self.cost_matrix = c

        if callable(potential):
            self.potential_array = potential(X)
        else:
            if potential.size() != torch.Size([X.size(0)]):
                raise RuntimeError("potential array size should be X.size(0).")
            self.potential_array = potential
        self.X = X
        self.eps = eps
        self.optimizer = None
        # self.S_eps = SamplesLoss("sinkhorn", blur=np.sqrt(eps), backend='tensorized', cost=euclidean_cost_batch)

    def set_optimizer(self, opt: EulerianOptimizer):
        self.optimizer = opt

    def SJKO_step(self, µ0: torch.Tensor, tau:float | torch.Tensor, f0: torch.Tensor | None = None, max_optim_steps=100,
                  max_sinkhorn_steps=100, sinkhorn_tol=1e-3, optim_tol=1e-3):
        if self.optimizer is None:
            raise RuntimeError("Optimizer not set. Use set_optimizer(opt) to set an optimizer.")
        self.optimizer.reset()
        µ1 = self.optimizer.step(µ0, 2*tau*self.potential_array)
        schrodinger_potentials = [f0]*2 if f0 is not None else None
        f1 = f0
        k = 1
        d = optim_tol + torch.ones(1)
        while k < max_optim_steps and (d > optim_tol).any():
            f10, g10 = sinkhorn_loop(
                µ1,
                µ0,
                self.cost_matrix,
                self.cost_matrix,
                [self.eps]*max_sinkhorn_steps,
                init=schrodinger_potentials,
                tol=sinkhorn_tol,
            )
            f1 = self_transport(µ1, self.cost_matrix, self.eps, init=f1, tol=sinkhorn_tol,
                                  max_steps=max_sinkhorn_steps)
            schrodinger_potentials = [f10, g10]
            # self.S_eps()
            grad = f10 - f1 + 2*tau*self.potential_array
            µ1 = self.optimizer.step(µ1, grad)
            k += 1
            d = µ1 @ grad - grad
        return µ1
    
    def integrate(self, µ0: torch.Tensor, t: torch.Tensor, return_ft: bool = False, print_progress: bool = False,
                  **kwargs):
        
        f = self_transport(µ0, self.cost_matrix, self.eps)
        µt = µ0[None,:]
        if return_ft:
            ft = f[None,:]
        appended_str = ""
        for i, tau in enumerate(torch.diff(t)):
            if print_progress:
                print(f'\r SJKO step {i+1}/{t.size(0)-1}...'+appended_str, end='')
            µ = self.SJKO_step(µt[-1,:], tau, f0=f, **kwargs)
            f = self_transport(µ, self.cost_matrix, self.eps, init=f, tol=kwargs.get('sinkhorn_tol', 1e-4),
                               max_steps=kwargs.get('max_sinkhorn_steps', 10))
            if return_ft:
                ft = torch.cat((ft, f[None,:]), dim=0)
            µt = torch.cat((µt, µ[None,:]), dim=0)
            appended_str = f"(finished last step in {self.optimizer.current_iter} outer loop iterations)"
        
        if return_ft:
            return µt, ft
        return µt


class LagrangianSPF:

    def __init__(self, potential: Callable[[torch.Tensor], torch.Tensor], eps: float,
                 c: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = euclidean_cost):
        self.potential = potential
        self.eps = eps
        self.c = c
        self.µ = None
        self.optimizer = None

    def set_optimizer(self, opt: LagrangianOptimizer):
        self.optimizer = opt
    
    def SJKO_step(self, x0: torch.Tensor, tau: float | torch.Tensor, f0: torch.Tensor | None = None, max_optim_steps=100,
                  max_sinkhorn_steps=100, sinkhorn_tol=1e-3):
        if self.optimizer is None:
            raise RuntimeError("Optimizer not set. Use set_optimizer(opt) to set an optimizer.")
        if self.µ is None:
            self.µ = torch.ones(x0.size(0)) / x0.size(0)
        self.optimizer.reset()
        x1 = x0.clone()
        x1.requires_grad = True
        grad = torch.autograd.grad(2*tau*self.potential(x1).mean(), x1)[0]
        x1 = self.optimizer.step(x1, grad)
        k = 1
        f1 = f0
        schrodinger_potentials = [f0]*2 if f0 is not None else None
        while k < max_optim_steps:
            f10, g10 = sinkhorn_loop(
                self.µ,
                self.µ,
                self.c(x1, x0.detach()),
                self.c(x0, x1.detach()),
                [self.eps]*max_sinkhorn_steps,
                init=schrodinger_potentials,
                tol=sinkhorn_tol,
            )
            f1 = self_transport(self.µ, self.c(x1, x1.detach()), self.eps, init=f1, tol=sinkhorn_tol,
                                  max_steps=max_sinkhorn_steps)
            schrodinger_potentials = [g10, f10]
            loss = (f10 - f1 + g10 + 2*tau*self.potential(x1)).mean()
            grad = torch.autograd.grad(loss, x1)[0]
            x1 = self.optimizer.step(x1, grad)
            k+=1
        return x1.detach()
    
    def integrate(self, x0: torch.Tensor, t: torch.Tensor, print_progress: bool = False, **kwargs):
        self.µ = torch.ones(x0.size(0)) / x0.size(0)
        f = self_transport(self.µ, self.c(x0, x0.detach()), self.eps, tol=kwargs.get('sinkhorn_tol', 1e-4),
                           max_steps=kwargs.get('max_sinkhorn_steps', 10))
        xt = x0[None,:]
        
        for i, tau in enumerate(torch.diff(t)):
            if print_progress:
                print(f'\r SJKO step {i+1}/{t.size(0)-1}...', end='')
            x = self.SJKO_step(xt[-1,:], tau, f0=f, **kwargs)
            f = self_transport(self.µ, self.c(x, x.detach()), self.eps, init=f)
            xt = torch.cat((xt, x[None,:]), dim=0)
        return xt


# Code was adapted from the geomloss package, see https://github.com/jeanfeydy/geomloss 
def softmin(eps, c, f):
    return -eps * (f.view(1, -1) - c / eps).logsumexp(1).view(-1)

def sinkhorn_loop(µ1, µ2, c_xy, c_yx, eps_list, init=None, tol=1e-4):
    log_µ1 = clampedlog(µ1)
    log_µ2 = clampedlog(µ2)
    torch.autograd.set_grad_enabled(False)
    eps = eps_list[0]
    if init is None:
        f12 = softmin(eps, c_xy, log_µ2)
        g12 = softmin(eps, c_yx, log_µ1)
    else:
        f12, g12 = init
    # count = 0
    for eps in eps_list:
        f12_ = softmin(eps, c_xy, log_µ2 + g12 / eps)
        g12_ = softmin(eps, c_yx, log_µ1 + f12 / eps)
        # count+=1
        if max(abs(f12_ - f12).max().item(),
               abs(g12_ - g12).max().item()) < tol:
            break
        f12, g12 = 0.5 * (f12 + f12_), 0.5 * (g12 + g12_) 

    torch.autograd.set_grad_enabled(True)
    # print(count, end='\r')
    f12, g12 = (
        softmin(eps, c_xy, (log_µ2 + g12 / eps).detach()),
        softmin(eps, c_yx, (log_µ1 + f12 / eps).detach()),
    )
    return f12, g12

def self_transport(µ, c, eps, init=None, tol=1e-4, max_steps=10):
    logµ = clampedlog(µ)
    torch.autograd.set_grad_enabled(False)
    if init is None:
        f_µ = softmin(eps, c, logµ)
    else:
        f_µ = init
    k = 0
    d = tol + 1
    while k < max_steps and d > tol:
        k += 1
        f_µ_ = f_µ.clone()
        f_µ = .5*(f_µ + softmin(eps, c, logµ + f_µ / eps))
        d = abs(f_µ-f_µ_).max().item()
    
    torch.autograd.set_grad_enabled(True)
    f_µ = softmin(eps, c, (logµ + f_µ / eps).detach())
    return f_µ