from plotly import graph_objects as go
import torch
from .utils import xyz, apply_to_eig, sqnorm
import numpy as np

def fig_anim_3d(data, fig_side_px=700, dt=100, axisrange=[-1, 1], axisvisible=True, title=""):
    fig = go.Figure(data=data[0],
                     layout=go.Layout(
                         height=fig_side_px,
                         width=fig_side_px,
                         scene=dict(
                         xaxis=dict(range=axisrange, autorange=False, visible=axisvisible),
                         yaxis=dict(range=axisrange, autorange=False, visible=axisvisible),
                         zaxis=dict(range=axisrange, autorange=False, visible=axisvisible),
                         aspectmode='cube'),
                         title=title,
                         updatemenus=[dict(type="buttons",
                                           showactive=False,
                                           buttons=[dict(label='Play',
                                                         method='animate',
                                                         args=[[None], dict(frame=dict(duration=dt, redraw=True),
                                                                            fromcurrent=True)]),
                                                    dict(label='Pause',
                                                         method='animate',
                                                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                                            mode='immediate',
                                                                            transition=dict(duration=0))])])]
                                     ),
                    )
    fig.frames = [go.Frame(data=d, layout=go.Layout(xaxis=dict(range=axisrange, autorange=False),
                                                      yaxis=dict(range=axisrange, autorange=False))) for d in data]
    return fig

def fig_anim_2d(data, fig_side_px=700, dt=100, axisrange=[-1, 1], axisvisible=True, title=""):
    fig = go.Figure(data=data[0],
                    layout=go.Layout(
                        height=fig_side_px,
                        width=fig_side_px,
                        xaxis=dict(range=axisrange, autorange=False, visible=axisvisible),
                        yaxis=dict(range=axisrange, autorange=False, visible=axisvisible),
                        title=title,
                        updatemenus=[dict(type="buttons",
                                          buttons=[dict(label='Play',
                                                        method='animate',
                                                        args=[None, dict(frame=dict(duration=dt, redraw=True),
                                                                           fromcurrent=True)]),
                                                   dict(label='Pause',
                                                        method='animate',
                                                        args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                                           mode='immediate',
                                                                           transition=dict(duration=0))])])]
                    ),
            )
    fig.frames = [go.Frame(data=d, layout=go.Layout(xaxis=dict(range=axisrange, autorange=False),
                                                    yaxis=dict(range=axisrange, autorange=False))) for d in data]
    return fig

def go_sphere(heatmap=None, **kwargs):
    theta = torch.linspace(0, 2*torch.pi, 100)
    phi = torch.linspace(0, torch.pi, 100//2)
    x = torch.outer(torch.cos(theta), torch.sin(phi))
    y = torch.outer(torch.sin(theta), torch.sin(phi))
    z = torch.outer(torch.ones(theta.size()), torch.cos(phi))
    if heatmap is not None:
        xyz = torch.cat((x[...,None], y[...,None], z[...,None]), dim=-1)
        h = heatmap(xyz)
        return go.Surface(x=x, y=y, z=z, surfacecolor=h, **kwargs)
    return go.Surface(x=x, y=y, z=z, **kwargs)

def traj_3d(traj, **kwargs):
    color = kwargs.pop('color', 'blue')
    line = kwargs.pop('line', {})
    line['color'] = color
    marker = kwargs.pop('marker', {})
    size = marker.get('size', 10)
    marker['color'] = color
    flow_lines = [go.Scatter3d(xyz(traj[:i+1]),
                               marker=dict(marker, size=i*[0]+[size]),
                               line=line,
                               **kwargs) for i in range(traj.size(0))]
    return flow_lines

def mass_flow(X, µ_t, **kwargs):
    if X.size(1) > 2:
        raise NotImplementedError
    if X.size(1) == 2:
        zmax = µ_t.max().item()
        return [go.Heatmap(x=X[:,0], y=X[:,1], z=µ, zmin=0, zmax=zmax, **kwargs) for µ in µ_t]
    return [go.Bar(x=X.flatten(), y=µ, **kwargs) for µ in µ_t]

def b_flow_sphere(cost_matrix, eps, fµ_t, potential_array, B_kwargs={}, rotation_lines_kwargs={}, sphere_kwargs={}, flow_kwargs={}):
    b_t = torch.exp(-fμ_t/eps)
    if b_t.size(1) != 3:
        raise NotImplementedError
    Hc = torch.exp(-cost_matrix/eps)
    P, Q = apply_to_eig(Hc, lambda s:1/torch.sqrt(s), torch.sqrt)
    t = torch.linspace(0, 1, 100)[:,None]
    t_ = 1-t
    B = torch.cat([t_*Q[:,i] + t*Q[:,(i+1)%3] for i in range(3)])
    B /= torch.sqrt((B*B).sum(-1)[:,None])
    Bkwargs = dict(dict(mode='lines', line=dict(width=1, color='red')), **B_kwargs)
    rkwargs = dict(dict(mode='lines', line=dict(width=1, color='black')), **rotation_lines_kwargs)
    flow_kwargs = dict(mode='lines', **flow_kwargs)
    spheredata = [go.Scatter3d(xyz(B), **Bkwargs)]
    V = torch.diag(potential_array)
    PVQ = P @ V @ Q
    spheredata.append(go_sphere(heatmap=lambda b: torch.einsum('ijk,kk,ijk->ij', b, PVQ, b), **dict(showscale=False, **sphere_kwargs))) # type: ignore
    A = (2/eps)*(Q @ V @ P - P @ V @ Q)
    a = torch.tensor([A[2, 1], A[0, 2], A[1, 0]])
    a /= torch.sqrt(sqnorm(a))
    Id = torch.eye(3)
    k = sqnorm(Id-a).argmax()
    e = Id[k,:]
    v = a - e
    R = Id - (2/sqnorm(v))*torch.outer(v, v)
    mask = torch.ones(3).type(torch.bool)
    mask[k] = False
    basis = R[mask,:]
    basis /= torch.sqrt(sqnorm(basis))[:,None]
    altitudes = torch.cos(torch.linspace(0, torch.pi, 20))
    radii = torch.sqrt(1-altitudes**2)
    theta = torch.linspace(0, 2*torch.pi, 100)[:,None]
    c, s = torch.cos(theta), torch.sin(theta)
    for z, r in zip(altitudes, radii):
        circ = z*a + r*c*basis[0] + r*s*basis[1]
        spheredata.append(go.Scatter3d(xyz(circ), legendgroup=1, **rkwargs))
        rkwargs["showlegend"] = False # type: ignore
    traj = b_t @ P
    return spheredata, traj_3d(traj, **flow_kwargs)

def particle_flow(Xt, **kwargs):
    marker = kwargs.pop('marker', {})
    marker['size'] = marker.get('size', 5)
    if Xt.size(2) > 2:
        raise NotImplementedError
    if Xt.size(2) == 2:
        return [go.Scatter(x=x[:,0], y=x[:,1], mode='markers', marker=marker, **kwargs) for x in Xt]
    return [go.Scatter(x=x[:,0], y=torch.zeros(x.size(0)),  mode='markers', marker=marker, **kwargs) for x in Xt]

def particles_to_bars(xt, X, width, **kwargs):
    data = []
    bins = torch.cat((X.flatten()-width/2, X[-1]+width/2))
    for x in xt:
        µ, _ = np.histogram(x, bins)
        µ = µ/µ.sum()
        data.append(go.Bar(x=X.flatten(), y=µ, width=width, **kwargs))
    return data

def potential_heatmap(V, domain, grid_size=50, **kwargs):
    xm, xM, ym, yM = domain
    x = torch.linspace(xm, xM, grid_size)
    y = torch.linspace(ym, yM, grid_size)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    xxx = torch.cat((xx[...,None], yy[...,None]), dim=-1)
    return go.Heatmap(x=x, y=y, z=V(xxx.reshape(-1, 2)).reshape(grid_size, grid_size), **kwargs)