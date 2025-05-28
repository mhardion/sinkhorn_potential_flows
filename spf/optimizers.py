import torch
from abc import ABC, abstractmethod
from collections.abc import Callable
from .utils import simplex_proj

class EulerianOptimizer(ABC):

    @abstractmethod
    def step(self, µ: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def reset(self):
        pass

class PGD(EulerianOptimizer):

    def __init__(self, lr: float | Callable[[int], float]):
        super().__init__()
        if callable(lr):
            self.lr = lr
        else:
            self.lr = lambda k: lr
        self.current_iter = 0
    
    def step(self, µ, grad):
        self.current_iter += 1
        return simplex_proj(µ - self.lr(self.current_iter)*grad)
    
    def reset(self):
        self.current_iter = 0

class FrankWolfe(EulerianOptimizer):

    def __init__(self):
        super().__init__()
        self.current_iter = 0
    
    def step(self, µ, grad):
        self.current_iter += 1
        µ_ = torch.zeros_like(µ)
        µ_[grad.argmin()] = 1
        return µ + (2/(self.current_iter+2))*(µ_ - µ)
    
    def reset(self):
        self.current_iter = 0

class ExpGD(EulerianOptimizer):

    def __init__(self, lr: float | Callable[[int], float]):
        super().__init__()
        self.current_iter = 0
        if callable(lr):
            self.lr = lr
        else:
            self.lr = lambda k: lr
    
    def step(self, µ, grad):
        self.current_iter += 1
        µ1 = µ*torch.exp(-self.lr(self.current_iter)*grad)
        µ1 /= µ1.sum()
        return µ1
    
    def reset(self):
        self.current_iter = 0



class LagrangianOptimizer(ABC):

    @abstractmethod
    def step(self, x: torch.Tensor, grad:torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def reset(self):
        pass

class GD(LagrangianOptimizer):

    def __init__(self, lr: float | Callable[[int], float]):
        super().__init__()
        if callable(lr):
            self.lr = lr
        else:
            self.lr = lambda k: lr
        self.current_iter = 0

    def step(self, x: torch.Tensor, grad:torch.Tensor):
        self.current_iter += 1
        return x - self.lr(self.current_iter)*grad

    def reset(self):
        self.current_iter = 0

class NesterovGD(LagrangianOptimizer):

    def __init__(self, lr: float | Callable[[int], float]):
        super().__init__()
        if callable(lr):
            self.lr = lr
        else:
            self.lr = lambda k: lr
        self.a = torch.tensor(1.0)
        self.current_iter = 0
        self.v = None

    def step(self, x: torch.Tensor, grad: torch.Tensor):
        if self.v is None:
            self.v = torch.zeros_like(x)
        self.current_iter += 1
        a_ = (1+torch.sqrt(4*self.a*self.a+1))/2
        momentum = (self.a-1)/a_
        self.a = a_
        self.v =  momentum * self.v - self.lr(self.current_iter) * grad
        return x + momentum * self.v - self.lr(self.current_iter) * grad

    def reset(self):
        self.current_iter = 0
        self.v = None