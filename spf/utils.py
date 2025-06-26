import torch

def sqnorm(x: torch.Tensor):
    return (x*x).sum(dim=-1)

def euclidean_cost(x, y):
    normx = sqnorm(x)
    normy = sqnorm(y)
    return normx[:,None] + normy[None,:] - 2*x@y.T

def euclidean_cost_batch(x, y):
    normx = sqnorm(x)
    normy = sqnorm(y)
    dotprod = torch.einsum('bik,bjk->bij', x, y)
    return normx[...,None] + normy[:,None,:] - 2*dotprod

def clampedlog(a):
    loga = torch.log(a)
    loga[a <= 0] = -1e5
    return loga

class Grid:
    def __init__(self) -> None:
        pass

    def __getitem__(self, key):
        tensors = []
        for k in range(len(key)):
            tensors.append(torch.arange(key[k].start, key[k].stop, key[k].step))
        return torch.cat([xx[...,None] for xx in torch.meshgrid(tensors, indexing='xy')], dim=-1)

grid = Grid()

def apply_to_eig(M, *args):
    lam, U = torch.linalg.eigh(M)
    res = [U @ torch.diag(f(lam)) @ U.T for f in args]
    if not len(res) - 1:
        return res[0]
    return res


def xy(x):
    return [x[...,i] for i in range(x.size(-1))]

def xyz(x):
    return dict(x=x[...,0], y=x[...,1], z=x[...,2])

def unif(a, b, X):
    assert X.size(1) == 1
    Y = X.flatten()
    µ = torch.zeros_like(Y)
    µ[(a<Y) & (Y<b)] = 1.
    return µ/µ.sum()

def diracs(domain, index):
    i = torch.tensor(index)
    a = torch.zeros(domain.size()[:-1], device=domain.device)
    a[tuple(i.T)] = 1/len(index)
    return a

def simplex_proj(v):
    d = v.size(0)
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0)
    rho = torch.nonzero(u > (cssv - 1) / torch.arange(1, d+1, dtype=v.dtype, device=v.device)).max()
    theta = (cssv[rho] - 1) / (rho + 1)
    w = torch.clamp(v - theta, min=0)
    return w