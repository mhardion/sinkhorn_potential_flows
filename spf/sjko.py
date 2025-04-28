import torch
from collections.abc import Callable
from .utils import clampedlog, sqnorm, euclidean_cost

class EulerianSPF:
    def __init__(self, X: torch.Tensor, potential: Callable[[torch.Tensor], torch.Tensor] | torch.Tensor,
                 eps:float, c: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | torch.Tensor = euclidean_cost,
                 store_b: bool = False):
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
            if potential.size() != torch.Size(X.size(0)):
                raise RuntimeError("potential array size should be X.size(0).")
            self.potential_array = potential
        self.X = X
        self.n = n
        self.eps = eps
        self.schrodinger_potentials = None
        self.optimizer = None
        self.mass_flow = None
    
    def set_init(self, µ0):
        self.mass_flow = µ0[None,:]

    def set_optimizer(self, opt):
        self.optimizer = opt

    def SJKO_step(self, µ0: torch.Tensor, tau:float, max_optim_steps=100, max_sinkhorn_steps=100, sinkhorn_tol=1e-3):
        µ1 = µ0.clone()
        self.optimizer.reset()
        k = 0
        while k < max_optim_steps: # TODO: and (k==0 or ((grad - µ @ grad) < -descent_tol).any()):
            f_11, g_00, g_10, f_01 = sinkhorn_loop(
                µ1,
                µ0,
                self.cost_matrix,
                self.cost_matrix,
                self.cost_matrix,
                [self.eps]*max_sinkhorn_steps,
                init=self.schrodinger_potentials,
                tol=sinkhorn_tol,
            )
            self.schrodinger_potentials = [f_11, g_00, g_10, f_01]
            grad = (f_01 - f_11).flatten() + 2*tau*self.potential_array
            µ1 = self.optimizer.step(µ1, grad)
            k += 1
        return µ1
    
    def integrate(self, µ0: torch.Tensor, t: torch.Tensor, print_progress=False, **kwargs):
        µ_t = µ0[None,:]
        for i, tau in enumerate(torch.diff(t)):
            if print_progress:
                print(f'\r SJKO step {i+1}...', end='')
            µ_t = torch.cat((µ_t, self.SJKO_step(µ_t[-1,:], tau, **kwargs)[None,:]), dim=0)
        return µ_t



# Code was adapted from the geomloss package, see https://github.com/jeanfeydy/geomloss 
def softmin(eps, c, f):
    return -eps * (f.view(1, -1) - c / eps).logsumexp(1).view(-1)

def sinkhorn_loop(µ1, µ2, c_xx, c_yy, c_xy, eps_list, init=None, tol=1e-5):
    c_yx = c_xy.T
    logµ1 = clampedlog(µ1)
    logµ2 = clampedlog(µ2)
    torch.autograd.set_grad_enabled(False)
    eps = eps_list[0]
    if init is None:
        g_12 = softmin(eps, c_yx, logµ1)
        f_21 = softmin(eps, c_xy, logµ2)
        f_11 = softmin(eps, c_xx, logµ1)
        g_22 = softmin(eps, c_yy, logµ2)
    else:
        f_11, g_22, g_12, f_21 = init

    for eps in eps_list:
        ft_21 = softmin(eps, c_xy, logµ2 + g_12 / eps)
        gt_12 = softmin(eps, c_yx, logµ1 + f_21 / eps)
        ft_11 = softmin(eps, c_xx, logµ1 + f_11 / eps)
        gt_22 = softmin(eps, c_yy, logµ2 + g_22 / eps)
        
        if max(abs(ft_21 - f_21).max().item(),
               abs(gt_12 - g_12).max().item()) < tol:
            break

        f_21, g_12 = 0.5 * (f_21 + ft_21), 0.5 * (g_12 + gt_12) 
        f_11, g_22 = 0.5 * (f_11 + ft_11), 0.5 * (g_22 + gt_22) 

    torch.autograd.set_grad_enabled(True)

    f_21, g_12 = (
        softmin(eps, c_xy, (logµ2 + g_12 / eps).detach()),
        softmin(eps, c_yx, (logµ1 + f_21 / eps).detach()),
    )
    f_11 = softmin(eps, c_xx, (logµ1 + f_11 / eps).detach())
    g_22 = softmin(eps, c_yy, (logµ2 + g_22 / eps).detach())

    return f_11, g_22, g_12, f_21

def self_transport(µ, c, eps, init = None, tol=1e-4, maxiter=5, iter_counter=torch.zeros(1)):
    logµ = clampedlog(µ)
    if init is None:
        f_µ = softmin(eps, c, logµ)
    else:
        f_µ = init
    k = 0
    d = tol + 1
    while k < maxiter and d > tol:
        k+=1
        f_µ_ = f_µ.clone()
        f_µ = .5*(f_µ_ + softmin(eps, c, logµ + f_µ / eps))
        d = sqnorm(f_µ-f_µ_).item()
    iter_counter[0] += k
    return f_µ