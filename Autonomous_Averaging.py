import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import scipy
import sympy
#import webcolors
import statistics
from matplotlib import cm
from matplotlib.ticker import LinearLocator

# Parameters

d = 1 # Half-dimension of the problem

# Averaging for autonomous systems

def J(d):
    """Canonical symplectic matrix.
    Input:
    - d: Int - Half-dimension of the problem"""
    JJ = np.zeros((2*d,2*d))
    I = np.eye(d,d)
    JJ[0:d,d:2*d] , JJ[d:2*d,0:d] = I , -I
    return JJ

def B(d):
    """Matrix of the ODE associated to slow variable (grobal drift).
    Input:
    - d: Int - Half-dimension of the problem"""
    BB = -2*np.diag(np.ones(2*d),0) + np.diag(np.ones(2*d-1),1) + np.diag(np.ones(2*d-1),-1)
    return BB

def f(eps,y):
    """Vector field for the studied ODE.
    Inputs:
    - eps: Float - High oscillation parameter
    - y: Array of shape (2d,) - Space variable"""
    y = np.array(y).reshape(2*d,)
    return (-1/eps*J(d)+B(d))@y

def g(tau,y):
    """Vector field for the studied ODE under averaging form.
    Inputs:
    - tau: Float - Time variable
    - y: Array of shape (2d,) - Space variable"""
    y = np.array(y).reshape(2*d,)
    exp_J = scipy.linalg.expm(tau*J(d))
    return (exp_J@B(d)@exp_J.T)@y

def Integrate(T,h,eps):
    """Integrates and Plots the integration of the highly oscillatory system.
    Inputs:
    - T: Float - Time for ODE integration
    - h: Float - Time step for printing
    - eps: Float - High oscillation parameter"""
    
    def ff(t,y):
        """Vector field for ODE integration.
        Inputs:
        - t: Float - Time variable
        - y: Array of shape (2*d,) - Space variable
        """
        #return f(eps,y)
        return g(t/eps,y)
    

    Y = solve_ivp(fun = ff, t_span = (0,T), y0 = 0.5*np.ones(2*d) , method = "DOP853", t_eval = np.arange(0,T,h) , atol = 1e-8 , rtol = 1e-8).y
    
    plt.figure(figsize=(15,5))
    
    plt.subplot(1,2,1)
    plt.plot(Y[0,:],Y[1,:],color="green")
    plt.grid()
    plt.xlabel("$y_1$")
    plt.ylabel("$y_2$")
    plt.axis("equal")
    
    plt.subplot(1,2,2)
    plt.plot(np.arange(0,T,h),Y[0,:] , label = "$y_1$")
    plt.plot(np.arange(0,T,h),Y[1,:] , label = "$y_2$")
    plt.grid()
    plt.xlabel("$t$")
    plt.ylabel("$y$")
    plt.legend()
    plt.show()
    
    pass

