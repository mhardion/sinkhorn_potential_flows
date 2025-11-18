"""Module used to compute Sinkhorn potential flows.
Submodules
----------
sjko : used to compute the SJKO steps in both Eulerian and Lagrangian discretizations
optimizers : catalog of gradient based optimizers in both discretizations
visualize : Tools to display and animate the flow with plotly
utils : additional useful functions
"""

from . import utils, optimizers, visualize
from .sjko import EulerianSPF, LagrangianSPF