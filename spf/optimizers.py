import torch
from abc import ABC, abstractmethod
from collections.abc import Callable
from .utils import simplex_proj

class EulerianOptimizer(ABC):

    @abstractmethod
    def step(self, µ: torch.Tensor, grad: torch.Tensor):
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
        µ_[µ.argmin()] = 1
        return µ + (2/(self.current_iter+2))*(µ_ - µ)
    
    def reset(self):
        self.current_iter = 0