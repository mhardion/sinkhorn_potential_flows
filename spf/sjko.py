import torch
from collections.abc import Callable
from .utils import clampedlog, euclidean_cost
from .optimizers import EulerianOptimizer, LagrangianOptimizer

class EulerianSPF:
    """Class used to compute Sinkhorn potential flows with an Eulerian discretization.

    Attributes
    ----------
    X : torch.Tensor
        The discrete space of size (n,d) with n the number of points and d the dimension.
    potential_array : torch.Tensor
        The values of the potential on all the points of X (thus of size n).
    eps : float
        The value of epsilon to be used.
    cost_matrix : torch.Tensor
        The matrix of pairwise costs between elements of X (thus of size (n, n)).
    optimizer : EulerianOptimizer
        Gradient based optimizer to be used to compute each SJKO step.
    
    Methods
    -------
    set_optimizer(opt)
        Sets the gradient based optimizer to be used at each SJKO step.
    SJKO_step(µ0, tau, ...)
        Computes one SJKO step from µ0 with step size tau (and additional optimization parameters).
        
    """
    def __init__(self, X: torch.Tensor, potential: Callable[[torch.Tensor], torch.Tensor] | torch.Tensor,
                 eps:float, c: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | torch.Tensor = euclidean_cost):
        """
        Parameters
        ----------
        X : torch.Tensor
            Discrete space of size (n, d).
        potential : torch.Tensor or callable
            Potential to compute the flow of. Can be specified as a callable taking as input a (n, d) tensor (n points
            in dimension d) and returning a (n,) tensor (it will be called on X), or can directly be given as a (n,) tensor.
        eps : float
            Value of epsilon.
        c : torch.Tensor or callable, optional
            Ground cost to be considered. Can be specified as a callable taking a (n, d) tensor and a (n, d) tensor,
            returning the pairwise costs in a (n, n) tensor (will be called on (X, X)), or can be directly given as a
            (n, n) tensor. Default is the square euclidean cost.
        """
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

    def set_optimizer(self, opt: EulerianOptimizer):
        self.optimizer = opt

    def SJKO_step(self, µ0: torch.Tensor, tau:float | torch.Tensor, f0: torch.Tensor | None = None, max_optim_steps=100,
                  max_sinkhorn_steps=100, sinkhorn_tol=1e-5, optim_tol=1e-5):
        """Perform one SJKO step.
        Parameters
        ----------
        µ0 : torch.Tensor
            Initial measure weights. If X is of size (n, d), µ0 is of size (n,).
        tau : float
            Step size.
        f0 : torch.Tensor, optional
            Initialization for the self-transport potential of µ0 (thus of size (n,)). Defaults to None.
        max_optim_steps : int, optional
            Maximum number of steps of the gradient based optimizer. Defaults to 100 but should be much larger depending
            on n.
        max_sinkhorn_steps : int, optional
            Maximum number of steps of the Sinkhorn algorithm to compute the Schrödinger potentials. Defaults to 100 but
            may need to be larger for more precision.
        sinkhorn_tol : float, optional
            Stops the Sinkhorn algorithm when the distance between consecutive iterations is under sinkhorn_tol. Defaults
            to 1e-5.
        optim_tol : float, optional
            Stops the gradient based optimization when the optimality conditions are verified up to optim_tol. Defaults
            to 1e-5.

        Returns
        -------
        µ1 : torch.Tensor
            The result of the SJKO step.
        """
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
            grad = f10 - f1 + 2*tau*self.potential_array
            µ1 = self.optimizer.step(µ1, grad)
            k += 1
            d = µ1 @ grad - grad
        return µ1
    
    def integrate(self, µ0: torch.Tensor, t: torch.Tensor, return_ft: bool = False, print_progress: bool = False,
                  **kwargs):
        """Integrate the SJKO scheme to compute the flow across multiple timestamps.
        Parameters
        ----------
        µ0 : torch.Tensor
            Initial measure (size (n,) if X is of size (n, d)).
        t : torch.Tensor
            Timestamps where to compute the flow. Should start at 0.
        return_ft : bool, optional
            Whether to return the Schrödinger potentials. Defaults to False.
        print_progress : bool, optional
            Whether to print the progress of the computation on the console. Defaults to False.
        **kwargs
            Arguments to be passed to SJKO_step (e.g. max_optim_steps, max_sinkhorn_steps...)
        
        Returns
        -------
            (µt, ft) if return_ft is set to True, µt otherwise, where µt is a (len(t), n) tensor containing the computed
            flow at all timestamps of t, and ft is the corresponding (len(t), n) tensor of Schrödinger potentials.
        """
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
    """Class used to compute Sinkhorn potential flows with a Lagrangian discretization.

    Attributes
    ----------
    potential : callable
        The potential of which the flow is computed.
    eps : float
        The value of epsilon to be used.
    cost : callable
        Cost function such that for x and y (n,d) tensors, c(x,y) gives the (n,n) tensor of pairwise costs. 
    optimizer : EulerianOptimizer
        Gradient based optimizer to be used to compute each SJKO step.
    
    Methods
    -------
    set_optimizer(opt)
        Sets the gradient based optimizer to be used at each SJKO step.
    SJKO_step(x0, tau, ...)
        Computes one SJKO step from x0 with step size tau (and additional optimization parameters).
    """
    def __init__(self, potential: Callable[[torch.Tensor], torch.Tensor], eps: float,
                 c: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = euclidean_cost):
        """
        Parameters
        ----------
        potential : callable
            Potential to compute the flow of. Is specified as a callable taking as input a (n, d) tensor (n points
            in dimension d) and returning a (n,) tensor (it will be called on X).
        eps : float
            Value of epsilon.
        c : callable, optional
            Ground cost to be considered. Is specified as a callable taking a (n, d) tensor and a (n, d) tensor,
            returning the pairwise costs in a (n, n) tensor (will be called on (X, X)). Default is the square euclidean cost.
        """
        self.potential = potential
        self.eps = eps
        self.c = c
        self.µ = None
        self.optimizer = None

    def set_optimizer(self, opt: LagrangianOptimizer):
        self.optimizer = opt
    
    def SJKO_step(self, x0: torch.Tensor, tau: float | torch.Tensor, f0: torch.Tensor | None = None, max_optim_steps=100,
                  max_sinkhorn_steps=100, sinkhorn_tol=1e-3):
        """Perform one SJKO step.
        Parameters
        ----------
        x0 : torch.Tensor
            Initial measure positions. Is of size (n, d) (n points in dimension d).
        tau : float
            Step size.
        f0 : torch.Tensor, optional
            Initialization for the self-transport potential of µ0 (thus of size (n,)). Defaults to None.
        max_optim_steps : int, optional
            Maximum number of steps of the gradient based optimizer. Defaults to 100 but should be much larger depending
            on n.
        max_sinkhorn_steps : int, optional
            Maximum number of steps of the Sinkhorn algorithm to compute the Schrödinger potentials. Defaults to 100 but
            may need to be larger for more precision.
        sinkhorn_tol : float, optional
            Stops the Sinkhorn algorithm when the distance between consecutive iterations is under sinkhorn_tol. Defaults
            to 1e-5.

        Returns
        -------
        x1 : torch.Tensor
            The result of the SJKO step.
        """
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
        """Integrate the SJKO scheme to compute the flow across multiple timestamps.
        Parameters
        ----------
        x0 : torch.Tensor
            Initial measure positions (size (n, d)).
        t : torch.Tensor
            Timestamps where to compute the flow. Should start at 0.
        print_progress : bool, optional
            Whether to print the progress of the computation on the console. Defaults to False.
        **kwargs
            Arguments to be passed to SJKO_step (e.g. max_optim_steps, max_sinkhorn_steps...)
        
        Returns
        -------
            xt : torch.Tensor
                (len(t), n, d) tensor containing the computed particle positions at all timestamps of t.
        """
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
    for eps in eps_list:
        f12_ = softmin(eps, c_xy, log_µ2 + g12 / eps)
        g12_ = softmin(eps, c_yx, log_µ1 + f12 / eps)
        if max(abs(f12_ - f12).max().item(),
               abs(g12_ - g12).max().item()) < tol:
            break
        f12, g12 = 0.5 * (f12 + f12_), 0.5 * (g12 + g12_) 

    torch.autograd.set_grad_enabled(True)
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