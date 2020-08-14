"""
    Sine-Gordon equation:
    u''[y]=sin(2*pi*u[y])
    u[0]=0, u[1]=d, d=t*delta
"""

from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
import random

# =======================================
# function bank
class LoadStepper:

    """
    Time "integrator" for a problem with no time derivatives.  This 
    is basically just to keep track of a parameter, ``self.t``, that
    can be used to parameterize external loading.
    """

    def __init__(self,DELTA_T,t=0.0):
        """
        Initializes the ``LoadStepper`` with a (pseudo)time step ``DELTA_T``
        and initial time ``t``, which defaults to zero.
        """
        self.DELTA_T = DELTA_T
        self.tval = t
        self.t = Expression("t",t=self.tval,degree=0)
        self.advance()

    def advance(self):
        """
        Increments the loading.
        """
        self.tval += float(self.DELTA_T)
        self.t.t = self.tval

# =======================================
# paramter

# ---------------------------------------
# mesh
xmin = 0.0
xmax = 1.0
nel = 10000
p_order = 2
# ---------------------------------------
# material system (nondimensional)
d_NonDim = 0.5 # d/H
wl_NonDim = 0.1  # lambda/H
fe_NonDim = 0.005  # lambda*tao/k
# ---------------------------------------
# loading step
N_STEPS = 20
DELTA_T = 1.0/float(N_STEPS)
stepper = LoadStepper(DELTA_T)
# stepper.t can be called



