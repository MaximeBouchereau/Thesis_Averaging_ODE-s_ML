# noinspection PyInterpreter
import sys
from time import sleep
import warnings

import scipy.optimize

warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
# print("No Warning Shown")

import torch
import torch.optim as optim
import torch.nn as nn
import copy

#import autograd.numpy as np
import numpy as npp
import numpy as np
#import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.integrate import solve_ivp
from scipy.optimize import fixed_point
from itertools import product
import statistics

# import autograd.numpy as np
# from autograd import grad
# from autograd import jacobian
# from autograd import hessian
# from torchdiffeq import odeint

import time
import datetime
from datetime import datetime as dtime

import webcolors

# Using of Neural Networks in order to integrate highly oscillatory ODE's


# Maths parameters [adjust]

dyn_syst = "VDP"         # Choice between "Linear", "VDP" (Van der Pol oscillator), "I_Pendulum" and "Logistic"
num_meth = "Forward_Euler"      # Numerical method selected for modified averaged field learning: choice between "Forward_Euler" and "MidPoint"
step_h = [0.001,0.1]            # Interval where time step is selected
step_eps = [0.001, 1]         # Interval where high oscillation parameter is selected
T_simul = 1                     # Time for ODE's simulation
h_simul = 0.01                  # Time step used for ODE's simulation
eps_simul = 0.1                 # High oscillation parameter used for ODE's simulation

# AI parameters [adjust]

K_data = 800                  # Quantity of data
R = 2                          # Amplitude of data in space (i.e. space data will be selected in the box [-R,R]^d)
p_train = 0.8                  # Proportion of data for training
HL = 2                         # Hidden layers per MLP for the first Neural Network
zeta = 200                     # Neurons per hidden layer of the first Neural Network
alpha = 2e-3                   # Learning rate for gradient descent
Lambda = 1e-9                  # Weight decay
BS = 100                       # Batch size (for mini-batching) for first training
N_epochs = 200                 # Epochs
N_epochs_print = 20            # Epochs between two prints of the Loss value


print(150 * "-")
print("Learning of solution of highly oscillatory ODE's")
print(150 * "-")

print("   ")
print(150 * "-")
print("Parameters:")
print(150 * "-")
print('    # Maths parameters:')
print("        - Dynamical system:", dyn_syst)
print("        - Numerical method used:", num_meth)
print("        - Interval where time step is selected:", step_h)
print("        - Interval where high oscillation parameter is selected:", step_eps)
print("        - Time for ODE's simulation:", T_simul)
print("        - Time step used for ODE's simulation:", h_simul)
print("        - High oscillation parameter used for ODE's simulation:", eps_simul)
print("    # AI parameters:")
print("        - Data's number for first training:", K_data)
print("        - Amplitude of data in space:", R)
print("        - Proportion of data for training:", format(p_train, '.1%'))
print("        - Hidden layers per MLP's for the first Neural Network:", HL)
print("        - Neurons on each hidden layer for the first Neural Network:", zeta)
print("        - Learning rate:", format(alpha, '.2e'))
print("        - Weight decay:", format(Lambda, '.2e'))
print("        - Batch size (mini-batching) for first training:", BS)
print("        - Epochs for first training:", N_epochs)
print("        - Epochs between two prints of the Loss value for first training:", N_epochs_print)

# Dimension of the problem

if dyn_syst == "Logistic":
    d = 1
else:
    d = 2

def y0start(dyn_syst):
    """Gives the initial data (vector) for ODE's integration and study of trajectories"""
    if d == 2:
        if dyn_syst == "Linear":
            return np.array([1.0,0.0])
        if dyn_syst == "VDP":
            return np.array([0.5,0.5])
        if dyn_syst == "I_Pendulum":
            return np.array([0.5,-0.5])
    if d == 1:
        return np.array([1.1])

class NA:
    """Numerical analysis class, contains:
     - Function which describes the dynamical system
     - Function which creates data for training/test
     - ODE Integrator"""

    def f(tau,y):
        """Describes the dynamics of the studied ODE
         - t : Float - Time variable
         - y : Array of shape (d,) - Space variable
         Returns an array of shape (d,)"""
        y = np.reshape(np.array(y) , (d,))
        if dyn_syst == "Linear":
            return np.array([ ( -1 + np.cos(tau) )*y[0] - np.sin(tau)*y[1]  , (1+np.sin(tau))*y[0] + np.cos(tau)*y[1] ])
        if dyn_syst == "VDP":
            z = np.zeros(d)
            z[0] = - np.sin(tau)*(1/4 - (np.cos(tau) * y[0] + np.sin(tau) * y[1]) ** 2) * (-np.sin(tau) * y[0] + np.cos(tau) * y[1])
            z[1] = np.cos(tau)*(1/4 - (np.cos(tau) * y[0] + np.sin(tau) * y[1]) ** 2) * (-np.sin(tau) * y[0] + np.cos(tau) * y[1])
            return z
        if dyn_syst == "I_Pendulum":
            y1, y2 = y[0], y[1]
            z = np.zeros_like(y)
            z[0] = y2 + np.sin(tau) * np.sin(y1)
            z[1] = np.sin(y1) - (1 / 2) * np.sin(tau) ** 2 * np.sin(2 * y1) - np.sin(tau) * np.cos(y1) * y2
            return z
        if dyn_syst == "Logistic":
            z = np.zeros(d)
            z[0] = y[0]*(1-y[0]) + np.sin(tau)
            return z

    def f_av(t,y):
        """Gives the averaged vector field of the studied ODE
         - t : Float - Time variable
         - y : Array of shape (d,) - Space variable"""
        #nb_coeff = 1
        #for s in y.shape:
        #    nb_coeff = nb_coeff * s
        #y = torch.tensor(y).reshape(d, int(nb_coeff / d))
        y = y.reshape(d,)
        z = npp.zeros_like(y)
        if dyn_syst == "Linear":
            z[0,:] = -y[0,:]
            z[1,:] = y[0,:]
        if dyn_syst == "VDP":
            y1, y2 = y[0], y[1]
            z[0] = (-1 / 8) * (y1 ** 2 + y2 ** 2 - 1) * y1
            z[1] = (-1 / 8) * (y1 ** 2 + y2 ** 2 - 1) * y2
        if dyn_syst == "I_Pendulum":
            y1, y2 = y[0], y[1]
            z[0] = y2
            z[1] = np.sin(y1) - (1 / 4) * np.sin(2 * y1)
        if dyn_syst == "Logistic":
            z[0, :] = y[0, :]*(1-y[0, :])
        return z.reshape(d,)

    def F_av_1(t,y,epsilon):
        """Gives the averaged Field F^{epsilon} at order 1 (used for integration at stroboscopic times)
         - t : Float - Time variable
         - y : Array of shape (d,) - Space variable
         - epsilon : Float - High oscillation parameter
         CAUTION: Only for the Linear system and Logistic equation!!!"""
        y = np.reshape(np.array(y), (d,))
        if dyn_syst == "Linear":
            return np.array([(-1 + epsilon) * y[0] + (epsilon-epsilon**2) * y[1] , (1+epsilon) * y[0] - epsilon*(1+epsilon) * y[1]])/(1+epsilon**2)
        if dyn_syst == "Logistic":
            return np.array([ y[0]*(1-y[0])+ epsilon*(1-2*y[0])  - (3/2)*epsilon**2 ])

    def f_av_NN(y):
        """Returns the averaged vector field associated to the corresponding dynamical system, useful for neural network
        Inputs:
        - y: Tensor of shape (d,1) - Space variable"""

        nb_coeff = 1
        for s in y.shape:
            nb_coeff = nb_coeff * s
        y = torch.tensor(y).reshape(d, int(nb_coeff / d))
        z = torch.zeros_like(y)

        if dyn_syst == "Linear":
            z[0, :] = -y[0, :]
            z[1, :] = y[0, :]
        if dyn_syst == "VDP":
            z[0, :] = (-1/8)*(y[0, :]**2+y[1, :]**2-1)*y[0, :]
            z[1, :] = (-1/8)*(y[0, :]**2+y[1, :]**2-1)*y[1, :]
        if dyn_syst == "Logistic":
            z[0, :] = y[0, :]*(1-y[0, :])

        return z

    def ODE_Solve(ti , tf , h  , epsilon , Yinit):
        """Solves the highly oscillatory ODE:
               y' = f(t/epsilon , y)
        by using a DOP853 method which approximates the exact flow, from time t_0 to time t_0 + h.
         - ti : Float - Starting time of integration
         - tf : Float - Ending time of integration
         - h: Float - Duration of integration
         - epsilon : Float - Parameter of high oscillation
         - Yinit : Array of shape (d,) -Initial condition"""

        Yinit = np.reshape( np.array(Yinit) , (d,) )

        def ff(t, y):
            """Describes the dynamics of the studied ODE with high oscillations
             - t : Float - Time variable
             - y : Array of shape (d,) - Space variable
             Returns an array of shape (d,)"""
            y = np.reshape(np.array(y), (d,))
            return NA.f(t/epsilon , y)

        S = solve_ivp(fun = ff , t_span = (ti , tf + 2*h) , y0 = Yinit , method = "DOP853" , atol = 1e-8 , rtol = 1e-8 , t_eval = np.arange(ti , tf + h , h) )
        return S.y

class DataCreate:
    """Data class, for Data creation, contains function for Data creation"""

    def Data(K):
     """Function for Data creation, computes solutions at times t = 2*pi*epsilon where epsilon is randommly
     selected in the interval eps for the ODE
            y' = f( t/epsilon , y)
     - K : Integer - Number of data created
     Returns a tuple containing:
     - Initial data for training and test
     - Solutions at times t = 2*pi*epsilon for training and test
     - Epsilons for training and test
     All the componants of the tuple are tensors of shape (d,K0)/(d,K-K0) for solutions and shape (1,K0)/(1,K-K0)
     for epsilons, where K0 is the number of data used for training."""

     print(" ")
     print(150 * "-")
     print("Data creation...")
     print(150 * "-")

     start_time_data = time.time()

     Y0 , Y1 = np.random.uniform(low = -R , high = R , size = (d,K)) , np.zeros((d,K))
     T0 = np.random.uniform(low = 0 , high = 2*np.pi , size = (1,K))
     H = np.exp(np.random.uniform(low = np.log(step_h[0]) , high = np.log(step_h[1]) , size = (1,K)))
     EPS = np.exp(np.random.uniform(low = np.log(step_eps[0]) , high = np.log(step_eps[1]) , size = (1,K)))

     if dyn_syst == "Logistic":
         Y0 = np.abs(Y0)

     pow = max([int(np.log10(K) - 1), 3])
     pow = min([pow, 6])

     for k in range(K):
         end_time_data = start_time_data + (K / (k + 1)) * (time.time() - start_time_data)
         end_time_data = datetime.datetime.fromtimestamp(int(end_time_data)).strftime(' %Y-%m-%d %H:%M:%S')
         print(" Loading :  {} % \r".format(str(int(10 ** (pow) * (k + 1) / K) / 10 ** (pow - 2)).rjust(3)), " Estimated time for ending : " + end_time_data, " - ", end="")
         Y1[:,k] = NA.ODE_Solve(ti = T0[0,k] , tf = T0[0,k] + H[0,k] , h = H[0,k] , epsilon = EPS[0,k] , Yinit = Y0[:,k])[:,1]

     K0 = int(p_train*K)
     Y0_train = torch.tensor(Y0[:, 0:K0])
     Y0_test = torch.tensor(Y0[:, K0:K])
     Y1_train = torch.tensor(Y1[:, 0:K0])
     Y1_test = torch.tensor(Y1[:, K0:K])
     T0_train = torch.tensor(T0[:, 0:K0])
     T0_test = torch.tensor(T0[:, K0:K])
     H_train = torch.tensor(H[:, 0:K0])
     H_test = torch.tensor(H[:, K0:K])
     EPS_train = torch.tensor(EPS[:, 0:K0])
     EPS_test = torch.tensor(EPS[:, K0:K])

     print("Computation time for data creation (h:min:s):",
           str(datetime.timedelta(seconds=int(time.time() - start_time_data))))
     return (Y0_train , Y0_test , Y1_train , Y1_test , T0_train , T0_test , H_train , H_test , EPS_train , EPS_test)

class NN(nn.Module, NA):
    def __init__(self):
        super().__init__()
        self.F_eps = nn.ModuleList([nn.Linear(d + 1, zeta), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta, zeta), nn.Tanh()] + [nn.Linear(zeta, d, bias=True)])
        self.F_h = nn.ModuleList([nn.Linear(d + 2, zeta), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta, zeta), nn.Tanh()] + [nn.Linear(zeta, d, bias=True)])
        self.Phi_Pert_p_1 = nn.ModuleList([nn.Linear(d + 3, zeta), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta, zeta), nn.Tanh()] + [nn.Linear(zeta, d, bias=True)])
        self.Phi_Pert_m_1 = nn.ModuleList([nn.Linear(d + 3, zeta), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta, zeta), nn.Tanh()] + [nn.Linear(zeta, d, bias=True)])

    def forward(self, y, t, h, eps , y1 = None):
        """Structured Neural Network.
        Inputs:
         - y: Tensor of shape (d,n) - space variable
         - t: Tensor of shape (1,n) - Time variable
         - h: Tensor of shape (1,n) - Time step
         - eps: Tensor of shape (1,n) - High oscillation parameter
         - y1: Tensor of shape (d,n) - Space variable (for implicit methods). Default: None"""

        y = y.T
        y = y.float()

        if num_meth == "MidPoint":
            y1 = y1.T
            y1 = y1.float()
        t = torch.tensor(t).T
        h = torch.tensor(h).T
        eps = torch.tensor(eps).T
        tau = t/eps
        tau_1 = (t+h)/eps
        ZERO = torch.zeros_like(tau)
        ONE = torch.ones_like(tau)

        # First part: Structure of the equation

        if num_meth == "Forward_Euler":

            y0 = y
            x_Phi_Pert_m_1 = torch.cat((torch.cos(tau), torch.sin(tau), y0, eps), dim=1)
            x_Phi_Pert_m_1_0 = torch.cat((ONE, ZERO, y0, eps), dim=1)
            for i, module in enumerate(self.Phi_Pert_m_1):
                x_Phi_Pert_m_1 = module(x_Phi_Pert_m_1)
                x_Phi_Pert_m_1_0 = module(x_Phi_Pert_m_1_0)
            y0_bis = y0 + eps*(x_Phi_Pert_m_1 - x_Phi_Pert_m_1_0)

            #x_F_eps = torch.cat((y0_bis, eps), dim=1)
            x_F_h = torch.cat((y0_bis , h, eps), dim=1)
            #for i, module in enumerate(self.F_eps):
            #    x_F_eps = module(x_F_eps)
            for i, module in enumerate(self.F_h):
                x_F_h = module(x_F_h)

            x_F = y0_bis + h*NA.f_av_NN(y0_bis.T).T + h*x_F_h
            #x_F = y0_bis + h*NA.f_av_NN(y0_bis.T).T + h*eps*x_F_eps + h**2*x_F_h

            x_Phi_Pert_p_1 = torch.cat((torch.cos(tau_1), torch.sin(tau_1), x_F, eps), dim=1)
            x_Phi_Pert_p_1_0 = torch.cat((ONE, ZERO, x_F, eps), dim=1)
            for i, module in enumerate(self.Phi_Pert_p_1):
                x_Phi_Pert_p_1 = module(x_Phi_Pert_p_1)
                x_Phi_Pert_p_1_0 = module(x_Phi_Pert_p_1_0)
            y1_hat = x_F + eps*(x_Phi_Pert_p_1 - x_Phi_Pert_p_1_0)


        if num_meth == "MidPoint":

            y0 , y0__1 = y , y1
            x_Phi_Pert_m_1__0 = torch.cat((torch.cos(tau), torch.sin(tau), y0, eps), dim=1)
            x_Phi_Pert_m_1_0__0 = torch.cat((ONE, ZERO, y0, eps), dim=1)
            x_Phi_Pert_m_1__1 = torch.cat((torch.cos(tau), torch.sin(tau), y0__1, eps), dim=1)
            x_Phi_Pert_m_1_0__1 = torch.cat((ONE, ZERO, y0__1, eps), dim=1)
            for i, module in enumerate(self.Phi_Pert_m_1):
                x_Phi_Pert_m_1__0 = module(x_Phi_Pert_m_1__0)
                x_Phi_Pert_m_1_0__0 = module(x_Phi_Pert_m_1_0__0)
                x_Phi_Pert_m_1__1 = module(x_Phi_Pert_m_1__1)
                x_Phi_Pert_m_1_0__1 = module(x_Phi_Pert_m_1_0__1)
            y0_bis__0 = y0 + eps * (x_Phi_Pert_m_1__0 - x_Phi_Pert_m_1_0__0)
            y0_bis__1 = y0__1 + eps * (x_Phi_Pert_m_1__1 - x_Phi_Pert_m_1_0__1)

            x_F_h = torch.cat(((y0_bis__0 + y0_bis__1)/2, h, eps), dim=1)
            for i, module in enumerate(self.F_h):
                x_F_h = module(x_F_h)

            x_F = y0_bis__0 + h * NA.f_av_NN(((y0_bis__0 + y0_bis__1)/2).T).T + h * x_F_h

            x_Phi_Pert_p_1 = torch.cat((torch.cos(tau_1), torch.sin(tau_1), x_F, eps), dim=1)
            x_Phi_Pert_p_1_0 = torch.cat((ONE, ZERO, x_F, eps), dim=1)
            for i, module in enumerate(self.Phi_Pert_p_1):
                x_Phi_Pert_p_1 = module(x_Phi_Pert_p_1)
                x_Phi_Pert_p_1_0 = module(x_Phi_Pert_p_1_0)
            y1_hat = x_F + eps * (x_Phi_Pert_p_1 - x_Phi_Pert_p_1_0)




        # Second part: Autoencoder structure, first direction

        x_Phi_Pert_p_1 = torch.cat((torch.cos(tau), torch.sin(tau), y0, eps), dim=1)
        x_Phi_Pert_p_1_0 = torch.cat((ONE, ZERO, y0, eps), dim=1)
        for i, module in enumerate(self.Phi_Pert_p_1):
            x_Phi_Pert_p_1 = module(x_Phi_Pert_p_1)
            x_Phi_Pert_p_1_0 = module(x_Phi_Pert_p_1_0)
        y_p_m = y0 + eps * (x_Phi_Pert_p_1 - x_Phi_Pert_p_1_0)

        x_Phi_Pert_m_1 = torch.cat((torch.cos(tau), torch.sin(tau), y_p_m, eps), dim=1)
        x_Phi_Pert_m_1_0 = torch.cat((ONE, ZERO, y_p_m, eps), dim=1)
        for i, module in enumerate(self.Phi_Pert_m_1):
            x_Phi_Pert_m_1 = module(x_Phi_Pert_m_1)
            x_Phi_Pert_m_1_0 = module(x_Phi_Pert_m_1_0)
        y_p_m = y_p_m + eps * (x_Phi_Pert_m_1 - x_Phi_Pert_m_1_0)


        # Third part: Autoencoder structure, second direction

        x_Phi_Pert_m_1 = torch.cat((torch.cos(tau), torch.sin(tau), y0, eps), dim=1)
        x_Phi_Pert_m_1_0 = torch.cat((ONE, ZERO, y0, eps), dim=1)
        for i, module in enumerate(self.Phi_Pert_m_1):
            x_Phi_Pert_m_1 = module(x_Phi_Pert_m_1)
            x_Phi_Pert_m_1_0 = module(x_Phi_Pert_m_1_0)
        y_m_p = y0 + eps*(x_Phi_Pert_m_1-x_Phi_Pert_m_1_0)

        x_Phi_Pert_p_1 = torch.cat((torch.cos(tau), torch.sin(tau), y_m_p, eps), dim=1)
        x_Phi_Pert_p_1_0 = torch.cat((ONE, ZERO, y_m_p, eps), dim=1)
        for i, module in enumerate(self.Phi_Pert_p_1):
            x_Phi_Pert_p_1 = module(x_Phi_Pert_p_1)
            x_Phi_Pert_p_1_0 = module(x_Phi_Pert_p_1_0)
        y_m_p = y_m_p + eps * (x_Phi_Pert_p_1 - x_Phi_Pert_p_1_0)


        # Fourth part: Modified averaged field

        y_F_eps = torch.cat((y,eps),dim=1)
        y_F_h = torch.cat((y,h,eps),dim=1)

        for i, module in enumerate(self.F_eps):
            y_F_eps = module(y_F_eps)
        for i, module in enumerate(self.F_h):
            y_F_h = module(y_F_h)

        y_F = NA.f_av_NN(y.T).T + y_F_h
        #y_F = NA.f_av_NN(y.T).T + eps * y_F_eps + h * y_F_h



        # Fifth part: Variable change (high oscillions generator)


        y_Phi_Pert_p_1 = torch.cat((torch.cos(tau), torch.sin(tau), y, eps), dim=1)
        y_Phi_Pert_p_1_0 = torch.cat((ONE, ZERO, y, eps), dim=1)
        for i, module in enumerate(self.Phi_Pert_p_1):
            y_Phi_Pert_p_1 = module(y_Phi_Pert_p_1)
            y_Phi_Pert_p_1_0 = module(y_Phi_Pert_p_1_0)
        y_phi = y + eps * (y_Phi_Pert_p_1 - y_Phi_Pert_p_1_0)


        return (y1_hat).T , (y_p_m).T , (y_m_p).T , (y_F).T , (y_phi).T

class Train(NN, NA):
    """Training of the neural network, depends on the numerical method chosen
    Choice of the numerical method:
        - Forward Euler
        - MidPoint"""

    def Loss(self, Y0, Y1, T0, H, EPS, model):
        """Computes the Loss function between two series of data Y0 and Y1 according to the numerical method
        Inputs:
        - Y0: Tensor of shape (d,n)
        - Y1: Tensor of shape (d,n)
        - EPS: Tensor of shape (1,n)
        - model: Neural network which will be optimized
        - meth: Character string - Numerical method used in order to compute predicted values
        Computes a predicted value Y1hat which is a tensor of shape (d,n) and returns the mean squared error between Y1hat and Y1
        => Returns a tensor of shape (1,1)"""
        Y0 = torch.tensor(Y0, dtype=torch.float32)
        Y0.requires_grad = True
        Y1hat = torch.zeros_like(Y0)
        Y1hat.requires_grad = True
        T0 = torch.tensor(T0, dtype=torch.float32)
        T0.requires_grad = True
        H = torch.tensor(H, dtype=torch.float32)
        H.requires_grad = True
        EPS = torch.tensor(EPS, dtype=torch.float32)
        EPS.requires_grad = True

        YY = model(Y0 , T0 , H, EPS, y1 = Y1)
        #YY_0 = model(Y0 , T0 , 0*H)[0]
        Y1_hat = YY[0]
        Y0_p_m_hat = YY[1]
        Y0_m_p_hat = YY[2]
        #print("Loss [ODE]:",format(((Y1_hat - Y1) ** 2).mean(),'.4E') , "-" , "Loss [AE-1]:",format(((Y0_p_m_hat - Y0) ** 2).mean(),'.4E') , "-" , "Loss [AE-2]:",format(((Y0_m_p_hat - Y0) ** 2).mean(),'.4E'))
        loss = (((Y1_hat - Y1)).abs() ** 2).mean() + ((Y0_p_m_hat - Y0) ** 2).mean() + ((Y0_m_p_hat - Y0) ** 2).mean()# + ((YY_0 - Y0) ** 2).mean()
        #loss = (((Y1_hat - Y1)).abs() ** 1).mean() + ((Y0_p_m_hat - Y0).abs() ** 1).mean() + ((Y0_m_p_hat - Y0).abs() ** 1).mean()# + ((YY_0 - Y0) ** 2).mean()
        #loss = (((Y1_hat - Y1)).abs() ** 1).mean() + ((Y0_p_m_hat - Y0).abs() ** 1).mean()  #
        #loss = (((Y1_hat - Y1)).abs() ** 2).mean()  + ((Y0_m_p_hat - Y0).abs() ** 2).mean()# + ((YY_0 - Y0) ** 2).mean()

        return loss

    def train(self, model, Data):
        """Makes the training on the data
        Inputs:
        - model: Neural network which will be optimized
        - Data: Tuple of tensors - Set of data created
        => Returns the lists Loss_train and Loss_test of the values of the Loss respectively for training and test,
        and best_model, which is the best apporoximation of the modified field computed"""

        start_time_train = time.time()

        print(" ")
        print(150 * "-")
        print("First training...")
        print(150 * "-")

        Y0_train = Data[0]
        Y0_test = Data[1]
        Y1_train = Data[2]
        Y1_test = Data[3]
        T0_train = Data[4]
        T0_test = Data[5]
        H_train = Data[6]
        H_test = Data[7]
        EPS_train = Data[8]
        EPS_test = Data[9]

        optimizer = optim.AdamW(model.parameters(), lr=alpha, betas=(0.9, 0.999), eps=1e-8, weight_decay=Lambda,amsgrad=True)  # Algorithm AdamW
        best_model, best_loss_train, best_loss_test = model, np.infty, np.infty  # Selects the best minimizer of the Loss function
        Loss_train = [] # list for loss_train values
        Loss_test = []  # List for loss_test values

        for epoch in range(N_epochs + 1):
            for ixs in torch.split(torch.arange(Y0_train.shape[1]), BS):
                optimizer.zero_grad()
                model.train()
                Y0_batch = Y0_train[:, ixs]
                Y1_batch = Y1_train[:, ixs]
                T0_batch = T0_train[:, ixs]
                H_batch = H_train[:, ixs]
                EPS_batch = EPS_train[:, ixs]
                loss_train = self.Loss(Y0_batch, Y1_batch, T0_batch, H_batch, EPS_batch, model)
                loss_train.backward()
                optimizer.step()  # Optimizer passes to the next epoch for gradient descent

            loss_test = self.Loss(Y0_test, Y1_test, T0_test , H_test, EPS_test, model)

            if loss_train < best_loss_train:
                best_loss_train = loss_train
                best_loss_test = loss_test
                best_model = copy.deepcopy(model)
                # best_model = model

            Loss_train.append(loss_train.item())
            Loss_test.append(loss_test.item())

            if epoch % N_epochs_print == 0:  # Print of Loss values (one print each N_epochs_print epochs)
                end_time_train = start_time_train + ((N_epochs + 1) / (epoch + 1)) * (time.time() - start_time_train)
                end_time_train = datetime.datetime.fromtimestamp(int(end_time_train)).strftime(' %Y-%m-%d %H:%M:%S')
                print('    Step', epoch, ': Loss_train =', format(loss_train, '.4E'), ': Loss_test =', format(loss_test, '.4E'), " -  Estimated end:", end_time_train)

        print("Loss_train (final)=", format(best_loss_train, '.4E'))
        print("Loss_test (final)=", format(best_loss_test, '.4E'))

        print("Computation time for training (h:min:s):", str(datetime.timedelta(seconds=int(time.time() - start_time_train))))

        return (Loss_train, Loss_test, best_model)

class Integrate(Train, NA):

    def integrate(self, model, name, save_fig):
        """Prints the values of the Loss along the epochs, trajectories and errors.
        Inputs:
        - Ltr: List containing the values of Loss_train along the epochs
        - Lte: List containing the values of Loss_test along the epochs
        - model: Best model learned during training, Loss_train and Loss_test
        - name: Character string - Potential name of the graph
        - save_fig: Boolean - Saves or not the figure"""

        def write_size():
            """Changes the size of writings on all windows"""
            axes = plt.gca()
            axes.title.set_size(7)
            axes.xaxis.label.set_size(7)
            axes.yaxis.label.set_size(7)
            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)
            plt.legend(fontsize=7)
            pass

        def write_size3D():
            """Changes the size of writings on all windows - 3d variant"""
            axes = plt.gca()
            axes.title.set_size(7)
            axes.xaxis.label.set_size(7)
            axes.yaxis.label.set_size(7)
            axes.zaxis.label.set_size(7)
            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)
            axes.zaxis.set_tick_params(labelsize=7)
            plt.legend(fontsize=7)
            pass

        start_time_integrate = time.time()

        Model , Loss_train , Loss_test = model[0] , model[1] , model[2]

        print(" ")
        print(100 * "-")
        print("Integration...")
        print(100 * "-")

        fig = plt.figure()

        ax = fig.add_subplot(2, 1, 2)
        plt.plot(range(len(Loss_train)), Loss_train, color='green', label='$Loss_{train}$')
        plt.plot(range(len(Loss_test)), Loss_test, color='red', label='$Loss_{test}$')
        plt.grid()
        plt.legend()
        plt.yscale('log')
        plt.title('Evolution of the Loss function (MLP)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        write_size()

        def Flow_hat(y , t , h , eps):
            """Vector field learned with the neural network
            Inputs:
            - y: Array of shape (d,) - Space variable
            - t: Float - Time
            - h: Float - Step size
            - eps: Float - High oscillation parameter"""
            y = torch.tensor(y).reshape(d, 1)
            y.requires_grad = True
            t_tensor = torch.tensor([[t]]).float()
            h_tensor = torch.tensor([[h]]).float()
            eps_tensor = torch.tensor([[eps]]).float()
            z = Model(y , t_tensor , h_tensor , eps_tensor)[0]
            z = z.detach().numpy()
            z = np.array(z, dtype=np.float64)
            return z.reshape(d, )

        def F_hat(y , t , h , eps):
            """Learned averaged field.
            Inputs:
            - y: Array of shape (d,) - Space variable
            - t: Float - Time
            - h: Float - Step size
            - eps: Float - High oscillation parameter"""
            y = torch.tensor(y).reshape(d, 1)
            y.requires_grad = True
            t_tensor = torch.tensor([[t]]).float()
            h_tensor = torch.tensor([[h]]).float()
            eps_tensor = torch.tensor([[eps]]).float()
            if num_meth == "Forward_Euler":
                z = Model(y, t_tensor, h_tensor, eps_tensor)[3]
            if num_meth == "MidPoint":
                z = Model(y, t_tensor, h_tensor, eps_tensor , y1 = y)[3]
            z = z.detach().numpy()
            z = np.array(z, dtype=np.float64)
            return z.reshape(d, )

        def Phi_hat(y , t , h , eps):
            """Learned variable change (high oscillations generator).
            Inputs:
            - y: Array of shape (d,) - Space variable
            - t: Float - Time
            - h: Float - Step size
            - eps: Float - High oscillation parameter"""
            y = torch.tensor(y).reshape(d, 1)
            y.requires_grad = True
            t_tensor = torch.tensor([[t]]).float()
            h_tensor = torch.tensor([[h]]).float()
            eps_tensor = torch.tensor([[eps]]).float()
            if num_meth == "Forward_Euler":
                z = Model(y, t_tensor, h_tensor, eps_tensor)[4]
            if num_meth == "MidPoint":
                z = Model(y, t_tensor, h_tensor, eps_tensor, y1=y)[4]
            z = z.detach().numpy()
            z = np.array(z, dtype=np.float64)
            return z.reshape(d, )

        TT = np.arange(0, T_simul+h_simul, h_simul)

        # Integration with DOP853 (good approximation of exact flow)
        start_time_exact = time.time()
        Y_exact = NA.ODE_Solve(ti = 0 ,  tf = T_simul , h = h_simul , epsilon = eps_simul , Yinit = y0start(dyn_syst))
        print("Integration time of ODE with DOP853 (one trajectory - h:min:s):", datetime.timedelta(seconds=time.time() - start_time_exact))

        # Integration with learned flow
        start_time_app = time.time()
        Y_app = np.zeros_like(Y_exact)
        #Y_app[:,0] = y0start(dyn_syst)
        #for n in range(np.size(TT)-1):
        #    Y_app[:,n+1] = Flow_hat(Y_app[:,n] , TT[n] , h_simul , eps_simul)

        Psi_app = np.zeros_like(Y_exact)
        Psi_app[:,0] = y0start(dyn_syst)
        for n in range(np.size(TT)-1):
            if num_meth == "Forward_Euler":
                Psi_app[:,n+1] = Psi_app[:,n] + h_simul*F_hat(Psi_app[:,n] , TT[n] , h_simul , eps_simul)
            if num_meth == "MidPoint":
                yy = Psi_app[:,n]
                N_iter = 5
                for k in range(N_iter):
                    yy = Psi_app[:,n] + h_simul*F_hat((Psi_app[:,n]+yy)/2 , TT[n] , h_simul , eps_simul)
                Psi_app[:,n+1] = yy
        for n in range(np.size(TT)):
            Y_app[:,n] = Phi_hat(Psi_app[:,n] , TT[n] , h_simul , eps_simul)

        print("Integration time of ODE with learned flow (h:min:s):", str(datetime.timedelta(seconds=time.time() - start_time_app)))

        print("   ")
        # Error computation between trajectory ploted with f for DOP853 and F_app for numerical method at stroboscopic times
        norm_exact = np.linalg.norm(np.array([np.linalg.norm((Y_exact)[:, i]) for i in range((Y_exact).shape[1])]) , np.infty) # Norm of the exact solution
        err_app = np.array([np.linalg.norm((Y_exact - Y_app)[:, i]) for i in range((Y_exact - Y_app).shape[1])])
        Err_app = np.linalg.norm(err_app, np.infty)/norm_exact
        print("Relative error between trajectories ploted with exact flow and learned flow:", format(Err_app, '.4E'))

        if d == 2:
            plt.subplot(2, 2, 1)
            plt.title("Trajectories")
            plt.axis('equal')
            plt.plot(Y_exact[0, :], Y_exact[1, :], color='black', linestyle='dashed',label=r"$\varphi_{t}^f(y_0)$")
            plt.plot(Y_app[0, :], Y_app[1, :], color='green', label="$y^{\epsilon}_{\\theta}(t)$")
            plt.xlabel("$y_1$")
            plt.ylabel("$y_2$")
            plt.legend()
            plt.grid()
            write_size()
            plt.subplot(2, 2, 2)
            plt.title("Comparison of local errors")
            plt.ylim(min(err_app[1:]),max(err_app[1:]))
            plt.yscale('log')
            plt.plot(TT, err_app, color="orange", label="$|$"+r"$y^{\epsilon}_{\theta}(t) - y^{\epsilon}(t) |$")
            plt.xlabel("t")
            plt.ylabel("Local error")
            plt.legend()
            plt.grid()
            write_size()

            f = plt.gcf()
            dpi = f.get_dpi()
            h, w = f.get_size_inches()
            f.set_size_inches(h * 1.7, w * 1.7)
            if save_fig == True:
                plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
            else:
                plt.show()

            if dyn_syst == "VDP":
                Y_exact_VC, Y_app_VC = np.zeros_like(Y_exact), np.zeros_like(Y_app)
                for n in range(np.size(TT)):
                    VC = np.array([[np.cos(TT[n] / eps_simul), np.sin(TT[n] / eps_simul)], [-np.sin(TT[n] / eps_simul), np.cos(TT[n] / eps_simul)]])
                    Y_exact_VC[:, n], Y_app_VC[:, n] = VC @ Y_exact[:, n], VC @ Y_app[:, n]

                plt.figure()
                plt.plot(np.squeeze(Y_exact_VC[0, :]), np.squeeze(Y_exact_VC[1, :]), label="Exact solution",  color="black", linestyle="dashed")
                plt.plot(np.squeeze(Y_app_VC[0, :]), np.squeeze(Y_app_VC[1, :]), label="Learned flow", color="green")
                plt.grid()
                plt.legend(loc = 'upper right')
                plt.xlabel("$q$")
                plt.ylabel("$p$")
                plt.title("$\epsilon = $" + str(eps_simul))
                plt.axis("equal")
                if save_fig == True:
                    plt.savefig(name + "_Variable_change" + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
                else:
                    plt.show()

        if d == 1:
            plt.subplot(2, 2, 1)
            plt.title("Trajectories")
            Y_exact = Y_exact.reshape(np.size(Y_exact),)
            Y_app = Y_app.reshape(np.size(Y_app), )
            plt.plot(TT, Y_exact, color='black', linestyle='dashed', label=r"$\varphi_{t}^f(y_0)$")
            plt.plot(TT, Y_app, color="green", label="$(\Phi_{\\theta,h,t_n}^{f})^n(y_0)$")
            plt.ylim(np.min([np.min(Y_app), np.min(Y_exact)]),np.max([np.max(Y_app), np.max(Y_exact)]))
            plt.xlabel("$t$")
            plt.ylabel("$y$")
            plt.legend()
            plt.grid()
            write_size()
            plt.subplot(2, 2, 2)
            plt.title("Comparison of local errors")
            plt.ylim(min(err_app[1:]), max(err_app[1:]))
            plt.yscale('log')
            plt.plot(TT, err_app, color="orange", label="$|$" + r"$\varphi_{2n\pi\epsilon}^{f}(y_0) - (\phi_{2\pi\epsilon}^{F^{\epsilon}_{app}})^n(y_0) |$")
            plt.xlabel("t")
            plt.ylabel("Local error")
            plt.legend()
            plt.grid()
            write_size()

        #plt.show()
        #f = plt.gcf()
        dpi = f.get_dpi()
        #h, w = f.get_size_inches()
        #f.set_size_inches(h * 1.7, w * 1.7)

        #if save_fig == True:
        #    plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
        #else:
        #    plt.show()

        print("Computation time for integration (h:min:s):", str(datetime.timedelta(seconds=int(time.time() - start_time_integrate))))

        pass

    def integrate_MM(self, model, name, save_fig):
        """Prints the values of the Loss along the epochs, trajectories with Micro-Macro scheme and errors.
        Inputs:
        - model: Best model learned during training, Loss_train and Loss_test
        - name: Character string - Potential name of the graph
        - save_fig: Boolean - Saves or not the figure"""

        def write_size():
            """Changes the size of writings on all windows"""
            axes = plt.gca()
            axes.title.set_size(7)
            axes.xaxis.label.set_size(7)
            axes.yaxis.label.set_size(7)
            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)
            plt.legend(fontsize=7)
            pass

        def write_size3D():
            """Changes the size of writings on all windows - 3d variant"""
            axes = plt.gca()
            axes.title.set_size(7)
            axes.xaxis.label.set_size(7)
            axes.yaxis.label.set_size(7)
            axes.zaxis.label.set_size(7)
            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)
            axes.zaxis.set_tick_params(labelsize=7)
            plt.legend(fontsize=7)
            pass

        start_time_integrate = time.time()

        Model , Loss_train , Loss_test = model[0] , model[1] , model[2]

        print(" ")
        print(100 * "-")
        print("Integration...")
        print(100 * "-")

        fig = plt.figure()

        ax = fig.add_subplot(2, 1, 2)
        plt.plot(range(len(Loss_train)), Loss_train, color='green', label='$Loss_{train}$')
        plt.plot(range(len(Loss_test)), Loss_test, color='red', label='$Loss_{test}$')
        plt.grid()
        plt.legend()
        plt.yscale('log')
        plt.title('Evolution of the Loss function (MLP)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        write_size()

        def Flow_hat(y , t , h , eps):
            """Vector field learned with the neural network
            Inputs:
            - y: Array of shape (d,) - Space variable
            - t: Float - Time
            - h: Float - Step size
            - eps: Float - High oscillation parameter"""
            y = torch.tensor(y).reshape(d, 1)
            y.requires_grad = True
            t_tensor = torch.tensor([[t]]).float()
            h_tensor = torch.tensor([[h]]).float()
            eps_tensor = torch.tensor([[eps]]).float()
            z = Model(y , t_tensor , h_tensor , eps_tensor)[0]
            z = z.detach().numpy()
            z = np.array(z, dtype=np.float64)
            return z.reshape(d, )

        def F_hat(y , t , h , eps):
            """Learned averaged field.
            Inputs:
            - y: Array of shape (d,) - Space variable
            - t: Float - Time
            - h: Float - Step size
            - eps: Float - High oscillation parameter"""
            y = torch.tensor(y).reshape(d, 1)
            y.requires_grad = True
            t_tensor = torch.tensor([[t]]).float()
            h_tensor = torch.tensor([[h]]).float()
            eps_tensor = torch.tensor([[eps]]).float()
            if num_meth == "Forward_Euler":
                z = Model(y, t_tensor, h_tensor, eps_tensor)[3]
            if num_meth == "MidPoint":
                z = Model(y, t_tensor, h_tensor, eps_tensor , y1 = y)[3]
            z = z.detach().numpy()
            z = np.array(z, dtype=np.float64)
            return z.reshape(d, )

        def Phi_hat(y , t , h , eps):
            """Learned variable change (high oscillations generator).
            Inputs:
            - y: Array of shape (d,) - Space variable
            - t: Float - Time
            - h: Float - Step size
            - eps: Float - High oscillation parameter"""
            y = torch.tensor(y).reshape(d, 1)
            y.requires_grad = True
            t_tensor = torch.tensor([[t]]).float()
            h_tensor = torch.tensor([[h]]).float()
            eps_tensor = torch.tensor([[eps]]).float()
            if num_meth == "Forward_Euler":
                z = Model(y, t_tensor, h_tensor, eps_tensor)[4]
            if num_meth == "MidPoint":
                z = Model(y, t_tensor, h_tensor, eps_tensor, y1=y)[4]
            z = z.detach().numpy()
            z = np.array(z, dtype=np.float64)
            return z.reshape(d, )

        def g_hat(w , v , t , h , eps):
            """Learned vector field associated to the Micro part in the Micro-Macro scheme.
            Inputs:
            - w: Array of shape (d,) - Space variable
            - v: Array of shape (d,) - Sopurce term coming from Macro part
            - t: Float - Time variable
            - h: Float - Step size
            - eps: Float - High oscillation parameter."""
            eta = 1e-5  # Small parameter for finite difference
            return (NA.f(t/eps , Phi_hat(v,t,0,eps) + w) - (1/(2*eta))*(Phi_hat(v + eta*F_hat(v,t,0,eps) , t , 0 , eps) - Phi_hat(v - eta*F_hat(v,t,0,eps) , t , 0 , eps))    -      (1/(2*eta))*(Phi_hat(v , t + eta , 0 , eps) - Phi_hat(v , t - eta , 0 , eps)))
            #return (NA.f(t/eps , Phi_hat(v,t,0,eps) + w) - (1/(2*eta))*(Phi_hat(v + eta*F_hat(v,t,h,eps) , t , 0 , eps) - Phi_hat(v - eta*F_hat(v,t,h,eps) , t , 0 , eps))    -      (1/(2*eta))*(Phi_hat(v , t + eta , 0 , eps) - Phi_hat(v , t - eta , 0 , eps)))
            #return h*NA.f(t / eps, Phi_hat(v, t, 0, eps) + w) - (Phi_hat(v, t + h, 0, eps) - Phi_hat(v, t, 0, eps))

        TT = np.arange(0, T_simul+h_simul, h_simul)

        # Integration with DOP853 (good approximation of exact flow)
        start_time_exact = time.time()
        Y_exact = NA.ODE_Solve(ti = 0 ,  tf = T_simul , h = h_simul , epsilon = eps_simul , Yinit = y0start(dyn_syst))
        print("Integration time of ODE with DOP853 (one trajectory - h:min:s):", datetime.timedelta(seconds=time.time() - start_time_exact))

        # Integration with learned flow
        start_time_app = time.time()
        Y_app = np.zeros_like(Y_exact)
        #Y_app[:,0] = y0start(dyn_syst)
        #for n in range(np.size(TT)-1):
        #    Y_app[:,n+1] = Flow_hat(Y_app[:,n] , TT[n] , h_simul , eps_simul)

        # Standard integration with Slow-Fast decomposition

        V_app = np.zeros_like(Y_exact)
        V_app[:,0] = y0start(dyn_syst)
        for n in range(np.size(TT)-1):
            if num_meth == "Forward_Euler":
                V_app[:,n+1] = V_app[:,n] + h_simul*F_hat(V_app[:,n] , TT[n] , h_simul , eps_simul)
            if num_meth == "MidPoint":
                yy = V_app[:,n]
                N_iter = 5
                for k in range(N_iter):
                    yy = V_app[:,n] + h_simul*F_hat((V_app[:,n]+yy)/2 , TT[n]+h_simul/2 , h_simul , eps_simul)
                V_app[:,n+1] = yy
        for n in range(np.size(TT)):
            Y_app[:,n] = Phi_hat(V_app[:,n] , TT[n] , h_simul , eps_simul)

        # Adding Micro-Macro correcting term

        W_app = np.zeros_like(Y_exact)
        W_app[:,0] = np.zeros_like(y0start(dyn_syst))
        for n in range(np.size(TT)-1):
            if num_meth == "Forward_Euler":
                W_app[:,n+1] = W_app[:,n] + h_simul*g_hat(W_app[:,n],V_app[:,n],TT[n],h_simul,eps_simul)
                #W_app[:,n+1] = W_app[:,n] + g_hat(W_app[:,n],V_app[:,n],TT[n],h_simul,eps_simul)
            if num_meth == "MidPoint":
                yy = W_app[:,n]
                N_iter = 5
                for k in range(N_iter):
                    yy = W_app[:,n] + h_simul*g_hat( (W_app[:,n]+yy)/2 , (V_app[:,n]+V_app[:,n+1])/2 , TT[n] + h_simul/2 , h_simul , eps_simul)
                    #yy = W_app[:,n] + (h_simul/2)*( g_hat( yy , V_app[:,n] , TT[n] , eps_simul)  +   g_hat( yy , V_app[:,n+1] , TT[n+1] , eps_simul)  )
                W_app[:,n+1] = yy


        Y_app = Y_app + W_app

        print("Integration time of ODE with learned flow (h:min:s):", str(datetime.timedelta(seconds=time.time() - start_time_app)))

        print("   ")
        # Error computation between trajectory ploted with f for DOP853 and F_app for numerical method at stroboscopic times
        norm_exact = np.linalg.norm(np.array([np.linalg.norm((Y_exact)[:, i]) for i in range((Y_exact).shape[1])]) , np.infty) # Norm of the exact solution
        err_app = np.array([np.linalg.norm((Y_exact - Y_app)[:, i]) for i in range((Y_exact - Y_app).shape[1])])
        Err_app = np.linalg.norm(err_app, np.infty)/norm_exact
        print("Relative error between trajectories ploted with exact flow and learned flow:", format(Err_app, '.4E'))

        if d == 2:
            plt.subplot(2, 2, 1)
            plt.title("Trajectories")
            plt.axis('equal')
            plt.plot(Y_exact[0, :], Y_exact[1, :], color='black', linestyle='dashed',label=r"$\varphi_{t}^f(y_0)$")
            plt.plot(Y_app[0, :], Y_app[1, :], color='green', label="$y^{\epsilon}_{\\theta}(t)$")
            plt.xlabel("$y_1$")
            plt.ylabel("$y_2$")
            plt.legend()
            plt.grid()
            write_size()
            plt.subplot(2, 2, 2)
            plt.title("Comparison of local errors")
            plt.ylim(min(err_app[1:]),max(err_app[1:]))
            plt.yscale('log')
            plt.plot(TT, err_app, color="orange", label="$|$"+r"$y^{\epsilon}_{\theta}(t) - y^{\epsilon}(t) |$")
            plt.xlabel("t")
            plt.ylabel("Local error")
            plt.legend()
            plt.grid()
            write_size()

            f = plt.gcf()
            dpi = f.get_dpi()
            h, w = f.get_size_inches()
            f.set_size_inches(h * 1.7, w * 1.7)
            if save_fig == True:
                plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
            else:
                plt.show()

            if dyn_syst == "VDP":
                Y_exact_VC, Y_app_VC = np.zeros_like(Y_exact), np.zeros_like(Y_app)
                for n in range(np.size(TT)):
                    VC = np.array([[np.cos(TT[n] / eps_simul), np.sin(TT[n] / eps_simul)], [-np.sin(TT[n] / eps_simul), np.cos(TT[n] / eps_simul)]])
                    Y_exact_VC[:, n], Y_app_VC[:, n] = VC @ Y_exact[:, n], VC @ Y_app[:, n]

                plt.figure()
                plt.plot(np.squeeze(Y_app_VC[0, :]), np.squeeze(Y_app_VC[1, :]), label="Learned flow", color="green", linewidth=2)
                plt.plot(np.squeeze(Y_exact_VC[0, :]), np.squeeze(Y_exact_VC[1, :]), label="Exact solution",  color="black", linestyle="dashed", linewidth=2)
                #plt.plot(np.squeeze(Y_app_VC[0, :]), np.squeeze(Y_app_VC[1, :]), label="Learned flow", color="green")
                plt.grid()
                plt.legend(loc = 'upper right')
                plt.xlabel("$q$")
                plt.ylabel("$p$")
                plt.title("$\epsilon = $" + str(eps_simul))
                plt.axis("equal")
                if save_fig == True:
                    plt.savefig(name + "_Variable_change" + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
                else:
                    plt.show()

        if d == 1:
            plt.subplot(2, 2, 1)
            plt.title("Trajectories")
            Y_exact = Y_exact.reshape(np.size(Y_exact),)
            Y_app = Y_app.reshape(np.size(Y_app), )
            plt.plot(TT, Y_exact, color='black', linestyle='dashed', label=r"$\varphi_{t}^f(y_0)$")
            plt.plot(TT, Y_app, color="green", label="$(\Phi_{\\theta,h,t_n}^{f})^n(y_0)$")
            plt.ylim(np.min([np.min(Y_app), np.min(Y_exact)]),np.max([np.max(Y_app), np.max(Y_exact)]))
            plt.xlabel("$t$")
            plt.ylabel("$y$")
            plt.legend()
            plt.grid()
            write_size()
            plt.subplot(2, 2, 2)
            plt.title("Comparison of local errors")
            plt.ylim(min(err_app[1:]), max(err_app[1:]))
            plt.yscale('log')
            plt.plot(TT, err_app, color="orange", label="$|$" + r"$\varphi_{2n\pi\epsilon}^{f}(y_0) - (\phi_{2\pi\epsilon}^{F^{\epsilon}_{app}})^n(y_0) |$")
            plt.xlabel("t")
            plt.ylabel("Local error")
            plt.legend()
            plt.grid()
            write_size()

        #plt.show()
        #f = plt.gcf()
        dpi = f.get_dpi()
        #h, w = f.get_size_inches()
        #f.set_size_inches(h * 1.7, w * 1.7)

        #if save_fig == True:
        #    plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
        #else:
        #    plt.show()

        print("Computation time for integration (h:min:s):", str(datetime.timedelta(seconds=int(time.time() - start_time_integrate))))

        pass

class Trajectories(Integrate):
    """Class for the study of convergence of trajectories"""

    def traj(self, model, name, save_fig):
        """Prints the global errors according to the step of the numerical method
        Inputs:
        - model: Model learned during training
        - name: Character string - Potential name of the graph
        - save_fig: Boolean - Saves or not the figure"""

        EEPS = np.exp(np.linspace(np.log(step_eps[0]), np.log(step_eps[1]), 7))
        HH = np.exp(np.linspace(np.log(step_h[0]) , np.log(step_h[1]) , 9 ))
        ERR_app = np.zeros((np.size(EEPS), np.size(HH)))

        Model = model[0]

        for i in range(np.size(EEPS)):
            eeps = EEPS[i]
            for j in range(np.size(HH)):
                hh = HH[j]
                print(" - h = {} \r".format(format(hh, '.4E')),"eps = "+format(format(eeps, '.4E')), end="")

                def F_hat(y, t, h, eps):
                    """Learned modified averaged field.
                    Inputs:
                    - y: Array of shape (d,) - Space variable
                    - t: Float - Time
                    - h: Float - Step size
                    - eps: Float - High oscillation parameter"""
                    y = torch.tensor(y).reshape(d, 1)
                    y.requires_grad = True
                    t_tensor = torch.tensor([[t]]).float()
                    h_tensor = torch.tensor([[h]]).float()
                    eps_tensor = torch.tensor([[eps]]).float()
                    if num_meth == "Forward_Euler":
                        z = Model(y, t_tensor, 0*h_tensor, eps_tensor)[3]
                    if num_meth == "MidPoint":
                        z = Model(y, t_tensor, h_tensor, eps_tensor, y)[3]
                    z = z.detach().numpy()
                    z = np.array(z, dtype=np.float64)
                    return z.reshape(d, )

                def Phi_hat(y, t, h, eps):
                    """Learned variable change (high oscillations generator).
                    Inputs:
                    - y: Array of shape (d,) - Space variable
                    - t: Float - Time
                    - h: Float - Step size
                    - eps: Float - High oscillation parameter"""
                    y = torch.tensor(y).reshape(d, 1)
                    y.requires_grad = True
                    t_tensor = torch.tensor([[t]]).float()
                    h_tensor = torch.tensor([[h]]).float()
                    eps_tensor = torch.tensor([[eps]]).float()
                    if num_meth == "Forward_Euler":
                        z = Model(y, t_tensor, h_tensor, eps_tensor)[4]
                    if num_meth == "MidPoint":
                        z = Model(y, t_tensor, h_tensor, eps_tensor, y)[4]
                    z = z.detach().numpy()
                    z = np.array(z, dtype=np.float64)
                    return z.reshape(d, )

                # Integration with DOP853 (approximation of the exact flow)
                Y_exact = NA.ODE_Solve(ti=0, tf=T_simul, h=hh, epsilon=eeps, Yinit=y0start(dyn_syst))

                # Integration with learned flow
                TT = np.arange(0, T_simul + hh, hh)
                Y_app = np.zeros_like(Y_exact)

                Psi_app = np.zeros_like(Y_exact)
                Psi_app[:, 0] = y0start(dyn_syst)

                for n in range(np.size(TT) - 1):
                    if num_meth == "Forward_Euler":
                        Psi_app[:, n + 1] = Psi_app[:, n] + hh * F_hat(Psi_app[:, n], TT[n], hh, eeps)
                    if num_meth == "MidPoint":
                        yy = Psi_app[:, n]
                        N_iter = 5
                        for k in range(N_iter):
                            yy = Psi_app[:, n] + hh * F_hat((Psi_app[:, n] + yy) / 2, TT[n], hh, eeps)
                        Psi_app[:, n + 1] = yy
                for n in range(np.size(TT)):
                    Y_app[:, n] = Phi_hat(Psi_app[:, n], TT[n], hh, eeps)


                # Computation of the norms of the exact solutions for the computation of relative errors
                norm_exact = np.linalg.norm(np.array([np.linalg.norm((Y_exact)[:, i]) for i in range((Y_exact).shape[1])]), np.infty)  # Norm of the exact solution
                err_app = np.array([np.linalg.norm((Y_exact - Y_app)[:, i]) for i in range((Y_exact - Y_app).shape[1])])
                Err_app = np.linalg.norm(err_app, np.infty) / norm_exact

                ERR_app[i,j] = Err_app


        plt.figure()
        plt.title("Error between trajectories")
        for i in range(np.size(EEPS)):
            cmap = plt.get_cmap("hsv")
            L_eps = np.size(EEPS)
            Colors = [cmap(i / L_eps) for i in range(L_eps)]
            plt.plot(HH, ERR_app[i, :], linestyle="dashed" , marker="s", color=Colors[i] , label = str(format(EEPS[i],'.4E')))
        plt.legend()
        plt.grid()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Step size $h$")
        plt.ylabel("Global error")
        if save_fig == True:
            plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
        else:
            plt.show()
        pass

    def traj_MM(self, model, name, save_fig):
        """Prints the global errors according to the step of the numerical method + Micro-Macro corrective term
        Inputs:
        - model: Model learned during training
        - name: Character string - Potential name of the graph
        - save_fig: Boolean - Saves or not the figure"""

        EEPS = np.exp(np.linspace(np.log(step_eps[0]), np.log(step_eps[1]), 7))
        HH = np.exp(np.linspace(np.log(step_h[0]) , np.log(step_h[1]) , 9 ))
        ERR_app = np.zeros((np.size(EEPS), np.size(HH)))

        Model = model[0]

        for i in range(np.size(EEPS)):
            eeps = EEPS[i]
            for j in range(np.size(HH)):
                hh = HH[j]
                print(" - h = {} \r".format(format(hh, '.4E')),"eps = "+format(format(eeps, '.4E')), end="")

                def F_hat(y, t, h, eps):
                    """Learned modified averaged field.
                    Inputs:
                    - y: Array of shape (d,) - Space variable
                    - t: Float - Time
                    - h: Float - Step size
                    - eps: Float - High oscillation parameter"""
                    y = torch.tensor(y).reshape(d, 1)
                    y.requires_grad = True
                    t_tensor = torch.tensor([[t]]).float()
                    h_tensor = torch.tensor([[h]]).float()
                    eps_tensor = torch.tensor([[eps]]).float()
                    if num_meth == "Forward_Euler":
                        z = Model(y, t_tensor, h_tensor, eps_tensor)[3]
                    if num_meth == "MidPoint":
                        z = Model(y, t_tensor, h_tensor, eps_tensor, y)[3]
                    z = z.detach().numpy()
                    z = np.array(z, dtype=np.float64)
                    return z.reshape(d, )

                def Phi_hat(y, t, h, eps):
                    """Learned variable change (high oscillations generator).
                    Inputs:
                    - y: Array of shape (d,) - Space variable
                    - t: Float - Time
                    - h: Float - Step size
                    - eps: Float - High oscillation parameter"""
                    y = torch.tensor(y).reshape(d, 1)
                    y.requires_grad = True
                    t_tensor = torch.tensor([[t]]).float()
                    h_tensor = torch.tensor([[h]]).float()
                    eps_tensor = torch.tensor([[eps]]).float()
                    if num_meth == "Forward_Euler":
                        z = Model(y, t_tensor, h_tensor, eps_tensor)[4]
                    if num_meth == "MidPoint":
                        z = Model(y, t_tensor, h_tensor, eps_tensor, y)[4]
                    z = z.detach().numpy()
                    z = np.array(z, dtype=np.float64)
                    return z.reshape(d, )

                def g_hat(w, v, t, h, eps):
                    """Learned vector field associated to the Micro part in the Micro-Macro scheme.
                    Inputs:
                    - w: Array of shape (d,) - Space variable
                    - v: Array of shape (d,) - Sopurce term coming from Macro part
                    - t: Float - Time variable
                    - h: Float - Step size
                    - eps: Float - High oscillation parameter."""
                    eta = 1e-5  # Small parameter for finite difference
                    return (NA.f(t / eps, Phi_hat(v, t, 0, eps) + w) - (1 / (2 * eta)) * (Phi_hat(v + eta * F_hat(v, t, 0, eps), t, 0, eps) - Phi_hat(v - eta * F_hat(v, t, 0, eps), t, 0, eps)) - (1 / (2 * eta)) * ( Phi_hat(v, t + eta, 0, eps) - Phi_hat(v, t - eta, 0, eps)))
                    #return (NA.f(t / eps, Phi_hat(v, t, 0, eps) + w) - (1 / (2 * eta)) * (Phi_hat(v + eta * F_hat(v, t, h, eps), t, 0, eps) - Phi_hat(v - eta * F_hat(v, t, 0, eps), t, 0, eps)) - (1 / (2 * eta)) * ( Phi_hat(v, t + eta, 0, eps) - Phi_hat(v, t - eta, 0, eps)))
                    #return h*NA.f(t / eps, Phi_hat(v, t, 0, eps) + w) - (Phi_hat(v, t+h, eps) - Phi_hat(v, t, 0, eps))

                # Integration with DOP853 (approximation of the exact flow)
                Y_exact = NA.ODE_Solve(ti=0, tf=T_simul, h=hh, epsilon=eeps, Yinit=y0start(dyn_syst))

                # Integration with learned flow
                TT = np.arange(0, T_simul + hh, hh)
                Y_app = np.zeros_like(Y_exact)

                # Integration with learned flow
                start_time_app = time.time()
                Y_app = np.zeros_like(Y_exact)
                # Y_app[:,0] = y0start(dyn_syst)
                # for n in range(np.size(TT)-1):
                #    Y_app[:,n+1] = Flow_hat(Y_app[:,n] , TT[n] , h_simul , eps_simul)

                # Standard integration with Slow-Fast decomposition

                V_app = np.zeros_like(Y_exact)
                V_app[:, 0] = y0start(dyn_syst)
                for n in range(np.size(TT) - 1):
                    if num_meth == "Forward_Euler":
                        V_app[:, n + 1] = V_app[:, n] + hh * F_hat(V_app[:, n], TT[n], hh, eeps)
                    if num_meth == "MidPoint":
                        yy = V_app[:, n]
                        N_iter = 5
                        for k in range(N_iter):
                            yy = V_app[:, n] + hh * F_hat((V_app[:, n] + yy) / 2, TT[n], hh, eeps)
                        V_app[:, n + 1] = yy
                for n in range(np.size(TT)):
                    Y_app[:, n] = Phi_hat(V_app[:, n], TT[n], hh, eeps)

                # Adding Micro-Macro correcting term

                W_app = np.zeros_like(Y_exact)
                W_app[:, 0] = np.zeros_like(y0start(dyn_syst))
                for n in range(np.size(TT) - 1):
                    if num_meth == "Forward_Euler":
                        W_app[:, n + 1] = W_app[:, n] + hh*g_hat(W_app[:, n], V_app[:, n], TT[n], h_simul, eeps)
                    if num_meth == "MidPoint":
                        yy = W_app[:, n]
                        N_iter = 5
                        for k in range(N_iter):
                            yy = W_app[:,n] + hh*g_hat( (W_app[:,n]+yy)/2 , (V_app[:,n]+V_app[:,n+1])/2 , TT[n] + hh/2 , hh , eeps)
                            #yy = W_app[:, n] + (hh / 2) * (g_hat(yy, V_app[:, n], TT[n], eeps) + g_hat(yy, V_app[:, n + 1], TT[n + 1], eeps))
                        W_app[:, n + 1] = yy
                Y_app = Y_app + W_app


                # Computation of the norms of the exact solutions for the computation of relative errors
                norm_exact = np.linalg.norm(np.array([np.linalg.norm((Y_exact)[:, i]) for i in range((Y_exact).shape[1])]), np.infty)  # Norm of the exact solution
                err_app = np.array([np.linalg.norm((Y_exact - Y_app)[:, i]) for i in range((Y_exact - Y_app).shape[1])])
                Err_app = np.linalg.norm(err_app, np.infty) / norm_exact

                ERR_app[i,j] = Err_app


        plt.figure()
        plt.title("Error between trajectories")
        for i in range(np.size(EEPS)):
            cmap = plt.get_cmap("hsv")
            L_eps = np.size(EEPS)
            Colors = [cmap(i / L_eps) for i in range(L_eps)]
            plt.plot(HH, ERR_app[i, :], linestyle="dashed" , marker="s", color=Colors[i] , label = str(format(EEPS[i],'.4E')))
        plt.legend()
        plt.grid()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Step size $h$")
        plt.ylabel("Global error")
        if save_fig == True:
            plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
        else:
            plt.show()
        pass

class Trajectories_UA(Integrate):
    """Class for the study of convergence of trajectories w.r.t. eps in order to see UA property or not."""

    def traj_UA(self, model, name, save_fig):
        """Prints the global errors according to the eps parameter of the numerical method
        Inputs:
        - model: Model learned during training
        - name: Character string - Potential name of the graph
        - save_fig: Boolean - Saves or not the figure"""

        EEPS = np.exp(np.linspace(np.log(step_eps[0]), np.log(step_eps[1]), 7))
        HH = np.exp(np.linspace(np.log(step_h[0]) , np.log(step_h[1]) , 9 ))
        ERR_app = np.zeros((np.size(EEPS), np.size(HH)))

        Model = model[0]

        for i in range(np.size(HH)):
            hh = HH[i]
            for j in range(np.size(EEPS)):
                eeps = EEPS[j]
                print(" - h = {} \r".format(format(hh, '.4E')),"eps = "+format(format(eeps, '.4E')), end="")

                def F_hat(y, t, h, eps):
                    """Learned modified averaged field.
                    Inputs:
                    - y: Array of shape (d,) - Space variable
                    - t: Float - Time
                    - h: Float - Step size
                    - eps: Float - High oscillation parameter"""
                    y = torch.tensor(y).reshape(d, 1)
                    y.requires_grad = True
                    t_tensor = torch.tensor([[t]]).float()
                    h_tensor = torch.tensor([[h]]).float()
                    eps_tensor = torch.tensor([[eps]]).float()
                    if num_meth == "Forward_Euler":
                        z = Model(y, t_tensor, h_tensor, eps_tensor)[3]
                    if num_meth == "MidPoint":
                        z = Model(y, t_tensor, h_tensor, eps_tensor, y)[3]
                    z = z.detach().numpy()
                    z = np.array(z, dtype=np.float64)
                    return z.reshape(d, )

                def Phi_hat(y, t, h, eps):
                    """Learned variable change (high oscillations generator).
                    Inputs:
                    - y: Array of shape (d,) - Space variable
                    - t: Float - Time
                    - h: Float - Step size
                    - eps: Float - High oscillation parameter"""
                    y = torch.tensor(y).reshape(d, 1)
                    y.requires_grad = True
                    t_tensor = torch.tensor([[t]]).float()
                    h_tensor = torch.tensor([[h]]).float()
                    eps_tensor = torch.tensor([[eps]]).float()
                    if num_meth == "Forward_Euler":
                        z = Model(y, t_tensor, h_tensor, eps_tensor)[4]
                    if num_meth == "MidPoint":
                        z = Model(y, t_tensor, h_tensor, eps_tensor, y)[4]
                    z = z.detach().numpy()
                    z = np.array(z, dtype=np.float64)
                    return z.reshape(d, )

                # Integration with DOP853 (approximation of the exact flow)
                Y_exact = NA.ODE_Solve(ti=0, tf=T_simul, h=hh, epsilon=eeps, Yinit=y0start(dyn_syst))

                # Integration with learned flow
                TT = np.arange(0, T_simul + hh, hh)
                Y_app = np.zeros_like(Y_exact)

                Psi_app = np.zeros_like(Y_exact)
                Psi_app[:, 0] = y0start(dyn_syst)

                for n in range(np.size(TT) - 1):
                    if num_meth == "Forward_Euler":
                        Psi_app[:, n + 1] = Psi_app[:, n] + hh * F_hat(Psi_app[:, n], TT[n], hh, eeps)
                    if num_meth == "MidPoint":
                        yy = Psi_app[:, n]
                        N_iter = 5
                        for k in range(N_iter):
                            yy = Psi_app[:, n] + hh * F_hat((Psi_app[:, n] + yy) / 2, TT[n], hh, eeps)
                        Psi_app[:, n + 1] = yy
                for n in range(np.size(TT)):
                    Y_app[:, n] = Phi_hat(Psi_app[:, n], TT[n], hh, eeps)


                # Computation of the norms of the exact solutions for the computation of relative errors
                norm_exact = np.linalg.norm(np.array([np.linalg.norm((Y_exact)[:, i]) for i in range((Y_exact).shape[1])]), np.infty)  # Norm of the exact solution
                err_app = np.array([np.linalg.norm((Y_exact - Y_app)[:, i]) for i in range((Y_exact - Y_app).shape[1])])
                Err_app = np.linalg.norm(err_app, np.infty) / norm_exact

                ERR_app[j,i] = Err_app


        plt.figure()
        plt.title("Error between trajectories")
        for i in range(np.size(HH)):
            cmap = plt.get_cmap("hsv")
            L_h = np.size(HH)
            Colors = [cmap(i / L_h) for i in range(L_h)]
            plt.plot(EEPS, ERR_app[:, i], linestyle="dashed" , marker="s", color=Colors[i] , label = str(format(HH[i],'.4E')))
        plt.legend(loc='upper left')
        plt.grid()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("$\epsilon$")
        plt.ylabel("Global error")
        if save_fig == True:
            plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
        else:
            plt.show()
        pass

    def traj_UA_MM(self, model, name, save_fig):
        """Prints the global errors according to the step of the numerical method + Micro-Macro corrective term
        Inputs:
        - model: Model learned during training
        - name: Character string - Potential name of the graph
        - save_fig: Boolean - Saves or not the figure"""

        EEPS = np.exp(np.linspace(np.log(step_eps[0]), np.log(step_eps[1]), 7))
        HH = np.exp(np.linspace(np.log(step_h[0]) , np.log(step_h[1]) , 9 ))
        ERR_app = np.zeros((np.size(EEPS), np.size(HH)))

        Model = model[0]

        for i in range(np.size(HH)):
            hh = HH[i]
            for j in range(np.size(EEPS)):
                eeps = EEPS[j]
                print(" - h = {} \r".format(format(hh, '.4E')),"eps = "+format(format(eeps, '.4E')), end="")

                def F_hat(y, t, h, eps):
                    """Learned modified averaged field.
                    Inputs:
                    - y: Array of shape (d,) - Space variable
                    - t: Float - Time
                    - h: Float - Step size
                    - eps: Float - High oscillation parameter"""
                    y = torch.tensor(y).reshape(d, 1)
                    y.requires_grad = True
                    t_tensor = torch.tensor([[t]]).float()
                    h_tensor = torch.tensor([[h]]).float()
                    eps_tensor = torch.tensor([[eps]]).float()
                    if num_meth == "Forward_Euler":
                        z = Model(y, t_tensor, h_tensor, eps_tensor)[3]
                    if num_meth == "MidPoint":
                        z = Model(y, t_tensor, h_tensor, eps_tensor, y)[3]
                    z = z.detach().numpy()
                    z = np.array(z, dtype=np.float64)
                    return z.reshape(d, )

                def Phi_hat(y, t, h, eps):
                    """Learned variable change (high oscillations generator).
                    Inputs:
                    - y: Array of shape (d,) - Space variable
                    - t: Float - Time
                    - h: Float - Step size
                    - eps: Float - High oscillation parameter"""
                    y = torch.tensor(y).reshape(d, 1)
                    y.requires_grad = True
                    t_tensor = torch.tensor([[t]]).float()
                    h_tensor = torch.tensor([[h]]).float()
                    eps_tensor = torch.tensor([[eps]]).float()
                    if num_meth == "Forward_Euler":
                        z = Model(y, t_tensor, h_tensor, eps_tensor)[4]
                    if num_meth == "MidPoint":
                        z = Model(y, t_tensor, h_tensor, eps_tensor, y)[4]
                    z = z.detach().numpy()
                    z = np.array(z, dtype=np.float64)
                    return z.reshape(d, )

                def g_hat(w, v, t, h, eps):
                    """Learned vector field associated to the Micro part in the Micro-Macro scheme.
                    Inputs:
                    - w: Array of shape (d,) - Space variable
                    - v: Array of shape (d,) - Sopurce term coming from Macro part
                    - t: Float - Time variable
                    - h: Float - Step size
                    - eps: Float - High oscillation parameter."""
                    eta = 1e-5  # Small parameter for finite difference
                    return (NA.f(t / eps, Phi_hat(v, t, 0, eps) + w) - (1 / (2 * eta)) * (Phi_hat(v + eta * F_hat(v, t, 0, eps), t, 0, eps) - Phi_hat(v - eta * F_hat(v, t, 0, eps), t, 0, eps)) - (1 / (2 * eta)) * ( Phi_hat(v, t + eta, 0, eps) - Phi_hat(v, t - eta, 0, eps)))
                    #return (NA.f(t / eps, Phi_hat(v, t, 0, eps) + w) - (1 / (2 * eta)) * (Phi_hat(v + eta * F_hat(v, t, h, eps), t, 0, eps) - Phi_hat(v - eta * F_hat(v, t, 0, eps), t, 0, eps)) - (1 / (2 * eta)) * ( Phi_hat(v, t + eta, 0, eps) - Phi_hat(v, t - eta, 0, eps)))
                    #return h*NA.f(t / eps, Phi_hat(v, t, 0, eps) + w) - (Phi_hat(v, t+h, eps) - Phi_hat(v, t, 0, eps))

                # Integration with DOP853 (approximation of the exact flow)
                Y_exact = NA.ODE_Solve(ti=0, tf=T_simul, h=hh, epsilon=eeps, Yinit=y0start(dyn_syst))

                # Integration with learned flow
                TT = np.arange(0, T_simul + hh, hh)
                Y_app = np.zeros_like(Y_exact)

                # Integration with learned flow
                start_time_app = time.time()
                Y_app = np.zeros_like(Y_exact)
                # Y_app[:,0] = y0start(dyn_syst)
                # for n in range(np.size(TT)-1):
                #    Y_app[:,n+1] = Flow_hat(Y_app[:,n] , TT[n] , h_simul , eps_simul)

                # Standard integration with Slow-Fast decomposition

                V_app = np.zeros_like(Y_exact)
                V_app[:, 0] = y0start(dyn_syst)
                for n in range(np.size(TT) - 1):
                    if num_meth == "Forward_Euler":
                        V_app[:, n + 1] = V_app[:, n] + hh * F_hat(V_app[:, n], TT[n], hh, eeps)
                    if num_meth == "MidPoint":
                        yy = V_app[:, n]
                        N_iter = 5
                        for k in range(N_iter):
                            yy = V_app[:, n] + hh * F_hat((V_app[:, n] + yy) / 2, TT[n], hh, eeps)
                        V_app[:, n + 1] = yy
                for n in range(np.size(TT)):
                    Y_app[:, n] = Phi_hat(V_app[:, n], TT[n], hh, eeps)

                # Adding Micro-Macro correcting term

                W_app = np.zeros_like(Y_exact)
                W_app[:, 0] = np.zeros_like(y0start(dyn_syst))
                for n in range(np.size(TT) - 1):
                    if num_meth == "Forward_Euler":
                        W_app[:, n + 1] = W_app[:, n] + hh*g_hat(W_app[:, n], V_app[:, n], TT[n], h_simul, eeps)
                    if num_meth == "MidPoint":
                        yy = W_app[:, n]
                        N_iter = 5
                        for k in range(N_iter):
                            yy = W_app[:,n] + hh*g_hat( (W_app[:,n]+yy)/2 , (V_app[:,n]+V_app[:,n+1])/2 , TT[n] + hh/2 , hh , eeps)
                            #yy = W_app[:, n] + (hh / 2) * (g_hat(yy, V_app[:, n], TT[n], eeps) + g_hat(yy, V_app[:, n + 1], TT[n + 1], eeps))
                        W_app[:, n + 1] = yy
                Y_app = Y_app + W_app


                # Computation of the norms of the exact solutions for the computation of relative errors
                norm_exact = np.linalg.norm(np.array([np.linalg.norm((Y_exact)[:, i]) for i in range((Y_exact).shape[1])]), np.infty)  # Norm of the exact solution
                err_app = np.array([np.linalg.norm((Y_exact - Y_app)[:, i]) for i in range((Y_exact - Y_app).shape[1])])
                Err_app = np.linalg.norm(err_app, np.infty) / norm_exact

                ERR_app[j,i] = Err_app


        plt.figure()
        plt.title("Error between trajectories")
        for i in range(np.size(HH)):
            cmap = plt.get_cmap("hsv")
            L_h = np.size(HH)
            Colors = [cmap(i / L_h) for i in range(L_h)]
            plt.plot(EEPS, ERR_app[:, i], linestyle="dashed" , marker="s", color=Colors[i] , label = str(format(HH[i],'.4E')))
        plt.legend(loc='upper right')
        plt.grid()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("$\epsilon$")
        plt.ylabel("Global error")
        if save_fig == True:
            plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
        else:
            plt.show()
        pass

class Convergence(Integrate):
    def curves(self, model, name, save_fig):
        """Prints the curves of convergence
        Inputs:
        - model: Best model learned during training
        - name: Character string - Potential name of the graph
        - save_fig: Boolean - Saves or not the figure"""

        n_disc = 31 # Number of points of discretizations for error computations
        n_eps = 11 # Number of epsilon's selected for error computation
        n_Gauss = 10 # Number of Gauss points in the Gauss quadrature for approximation of integrals
        Model = model[0]

        def Phi_hat(y, t, h, eps):
            """Learned variable change (high oscillations generator).
            Inputs:
            - y: Array of shape (d,) - Space variable
            - t: Float - Time
            - h: Float - Step size
            - eps: Float - High oscillation parameter"""
            y = torch.tensor(y).reshape(d, 1)
            y.requires_grad = True
            t_tensor = torch.tensor([[t]]).float()
            h_tensor = torch.tensor([[h]]).float()
            eps_tensor = torch.tensor([[eps]]).float()
            if num_meth == "Forward_Euler":
                z = Model(y, t_tensor, h_tensor, eps_tensor)[4]
            if num_meth == "MidPoint":
                z = Model(y, t_tensor, h_tensor, eps_tensor, y)[4]
            z = z.detach().numpy()
            #z = y.detach().numpy()
            z = np.array(z, dtype=np.float64)
            return z.reshape(d, )

        def F_hat(y, t, h, eps):
            """Learned modified averaged field.
            Inputs:
            - y: Array of shape (d,) - Space variable
            - t: Float - Time
            - h: Float - Step size
            - eps: Float - High oscillation parameter"""
            y = torch.tensor(y).reshape(d, 1)
            y.requires_grad = True
            t_tensor = torch.tensor([[t]]).float()
            h_tensor = torch.tensor([[h]]).float()
            eps_tensor = torch.tensor([[eps]]).float()
            if num_meth == "Forward_Euler":
                z = Model(y, t_tensor, h_tensor, eps_tensor)[3]
            if num_meth == "MidPoint":
                z = Model(y, t_tensor, h_tensor, eps_tensor, y)[3]
            z = z.detach().numpy()
            z = np.array(z, dtype=np.float64)
            return z.reshape(d, )

        def Phi_1(tau , y , eps):
            """High oscillation generator at order 1.
            Inputs:
            - tau: Float - Space variable
            - y: Array of shape (d,) - Space variable
            - eps: Float - High oscillation parameter"""
            yy = npp.zeros_like(y)
            xi, omega = npp.polynomial.legendre.leggauss(n_Gauss)
            ttau = [tau / 2 * ksi + tau / 2 for ksi in xi]
            F = [tau / 2 * omega[i] * (NA.f(ttau[i], y) - NA.f_av(0,y)) for i in range(len(xi))]
            for i in range(len(xi)):
               yy = yy + eps * F[i]
            return y + yy

        def F_1(y , eps):
            """Averaged field at order 1.
            Inputs:
            - y: Array of shape (d,) - Space variable
            - eps: Float - High oscillation parameter"""
            yy = npp.zeros_like(y)
            e1 , e2 = np.array([1,0]) , np.array([0,1])
            eta = 1e-5 # Small parameter for finite difference to approximate derivarives

            def Phi_1_avg(y , eps):
                """Average high oscillation generator at order 1.
                Inputs:
                - tau: Float - Space variable
                - y: Array of shape (d,) - Space variable
                - eps: Float - High oscillation parameter"""
                yy = npp.zeros_like(y)
                xi, omega = npp.polynomial.legendre.leggauss(n_Gauss)

                S1 = [npp.pi * ksi + npp.pi for ksi in xi]
                F = [npp.pi * omega[i] * (Phi_1(S1[i] , y , eps)) for i in range(len(xi))]
                for i in range(len(xi)):
                    yy = yy + F[i]
                return yy

            def f_Phi_1_avg(y , eps):
                """f composed with high oscillation generator at order 1.
                Inputs:
                - y: Array of shape (d,) - Space variable
                - eps: Float - High oscillation parameter"""
                yy = npp.zeros_like(y)
                xi, omega = npp.polynomial.legendre.leggauss(n_Gauss)

                S1 = [npp.pi * ksi + npp.pi for ksi in xi]
                F = [npp.pi * omega[i] * (NA.f(S1[i] , Phi_1(S1[i] , y , eps))) for i in range(len(xi))]
                for i in range(len(xi)):
                    yy = yy + F[i]
                return yy

            D_Phi1_avg = (1/(2*eta))*npp.concatenate( ( (Phi_1_avg(y+eta*e1 , eps) - Phi_1_avg(y-eta*e1 , eps)).reshape(d,1) , (Phi_1_avg(y+eta*e2 , eps) - Phi_1_avg(y-eta*e2 , eps)).reshape(d,1) ) , axis = 1)

            yy = np.linalg.solve(D_Phi1_avg , f_Phi_1_avg(y,eps))
            return yy

        EEPS = np.exp(np.linspace(np.log(10*step_eps[0]), np.log(step_eps[1]), n_eps))
        EEPSsq = EEPS ** 2
        EEPScu = EEPS ** 3
        EEPSqu = EEPS ** 4

        XX = list(product(np.linspace(-R, R, n_disc), repeat=d))
        XX_T = list(product(np.linspace(0,2*np.pi,n_disc),XX))

        err_phi_1 = []
        err_phi_2 = []
        err_F_1 = []
        err_F_2 = []
        for eeps in EEPS:
            print("  eps= {} \r".format(format(eeps, '.4E')), end="")
            ListDiff_phi_1 = max([np.linalg.norm(np.array(x[1]).reshape(d, 1) - Phi_hat(np.array(x[1]), x[0] * eeps, 0, eeps).reshape(d, 1)) for x in XX_T])
            ListDiff_phi_2 = max([np.linalg.norm(Phi_1(x[0], np.array(x[1]), eeps).reshape(d, 1) - Phi_hat(np.array(x[1]), x[0] * eeps, 0, eeps).reshape(d, 1)) for x in XX_T])

            ListDiff_F_1 = max([np.linalg.norm( NA.f_av(0,np.array(x[1])).reshape(d, 1) - F_hat(np.array(x[1]), x[0] * eeps, 0, eeps).reshape(d, 1)) for x in XX_T])
            ListDiff_F_2 = max([np.linalg.norm(F_1(np.array(x[1]), eeps).reshape(d, 1) - F_hat(np.array(x[1]), x[0] * eeps, 0, eeps).reshape(d, 1)) for x in XX_T])

            err_phi_1 = err_phi_1 + [ListDiff_phi_1]
            err_phi_2 = err_phi_2 + [ListDiff_phi_2]

            err_F_1 = err_F_1 + [ListDiff_F_1]
            err_F_2 = err_F_2 + [ListDiff_F_2]

        plt.figure()
        plt.title("Error between learned $\phi_{\\theta}$ and $\phi^\\varepsilon$ at first orders - " + num_meth)
        plt.scatter(EEPS, err_phi_1, label="Order 1", marker="s")
        plt.scatter(EEPS, err_phi_2, label="Order 2", marker="s")
        plt.plot(EEPS, err_phi_1[-1]*EEPS, label="$\\varepsilon \mapsto C\\varepsilon$", linestyle='dashed')
        plt.plot(EEPS, err_phi_2[-1]*EEPSsq, label="$\\varepsilon \mapsto C\\varepsilon^2$", linestyle='dashed')
        plt.legend()
        plt.grid()
        plt.xscale('log')
        plt.yscale('log')

        if save_fig == True:
            plt.savefig(name + "_phi_" + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
        else:
            plt.show()

        plt.figure()
        plt.title("Error between learned $F_{\\theta}(\cdot,0,\cdot)$ and $F^\\varepsilon$ at first orders - " + num_meth)
        plt.scatter(EEPS, err_F_1, label="Order 1", marker="s")
        plt.scatter(EEPS, err_F_2, label="Order 2", marker="s")
        plt.plot(EEPS, err_F_1[-1]*EEPS, label="$\\varepsilon \mapsto C\\varepsilon$", linestyle='dashed')
        plt.plot(EEPS, err_F_2[-1]*EEPSsq, label="$\\varepsilon \mapsto C\\varepsilon^2$", linestyle='dashed')
        plt.legend()
        plt.grid()
        plt.xscale('log')
        plt.yscale('log')

        if save_fig == True:
            plt.savefig(name + "_F_" + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
        else:
            plt.show()

        pass





def ExData(name_data="DataODE_FLow_Averaging_AE"):
    """Creates data y0, y1 with the function solvefEDOData
    with the chosen vector field at the beginning of the program
    Input:
    - name_data: Character string - Name of the registered tuple containing the data (default: "DataEDO")"""
    DataODE = DataCreate.Data(K_data)
    torch.save(DataODE, name_data)
    pass

def ExTrain(name_model='model_Flow_Averaging_AE', name_data='DataODE_Flow_Averaging_AE'):
    """Launches training and computes Loss_train, loss_test and best_model with the function Train().train
    Saves the files Loss_train, Loss_test and best_model with a given name
    Inputs (character strings):
    - name_model: Name of the file saved for best_model (default: "best_model")
    - name_data: Name of the file containing the created data (default: "DataEDO") used for training"""
    DataODE = torch.load(name_data)
    Loss_train, Loss_test, best_model = Train().train(model=NN(), Data=DataODE)
    torch.save((best_model,Loss_train,Loss_test), name_model)
    pass

def ExIntegrate(name_model="model_Flow_Averaging_AE", name_graph="Simulation_Flow_Averaging_AE", save=False):
    """Launches integration of the main equation and modified equation with the chosen model
    Inputs:
    - name_model: Character string - Name of the model made with neural network chosen for integration with F_app, and Loss_train/Loss_test
    - name_graph: Character string - Name of the graph which will be registered
    - save: Boolean - Saves the figure or not (default: False)"""
    Lmodel = torch.load(name_model)
    Integrate().integrate(model=Lmodel, name=name_graph,save_fig=save)
    pass

def ExIntegrate_MM(name_model="model_Flow_Averaging_AE", name_graph="Simulation_Flow_Averaging_AE_MM", save=False):
    """Launches integration with Micro-Macro scheme of the main equation and modified equation with the chosen model
    Inputs:
    - name_model: Character string - Name of the model made with neural network chosen for integration with F_app, and Loss_train/Loss_test
    - name_graph: Character string - Name of the graph which will be registered
    - save: Boolean - Saves the figure or not (default: False)"""
    Lmodel = torch.load(name_model)
    Integrate().integrate_MM(model=Lmodel, name=name_graph,save_fig=save)
    pass

def ExTraj(name_model="model_Flow_Averaging_AE", name_graph="Simulation_Convergence_Trajectories_FLow_Averaging_AE", save=False):
    """plots the curves of convergence between the trajectories integrated with f and F_app with the numerical method chosen
    Inputs:
    - name_model: Character string - Name of the model made with neural network chosen for integration with F_app
    - name_graph: Character string - Name of the graph which will be registered
    - save: Boolean - Saves the figure or not (default: False)"""
    Lmodel = torch.load(name_model)
    Trajectories().traj(model=Lmodel, name=name_graph, save_fig=save)
    pass

def ExTraj_MM(name_model="model_Flow_Averaging_AE", name_graph="Simulation_Convergence_Trajectories_FLow_Averaging_AE_MM", save=False):
    """plots the curves of convergence between the trajectories integrated with f and F_app with the numerical method chosen + Micro-Macro corrective term
    Inputs:
    - name_model: Character string - Name of the model made with neural network chosen for integration with F_app
    - name_graph: Character string - Name of the graph which will be registered
    - save: Boolean - Saves the figure or not (default: False)"""
    Lmodel = torch.load(name_model)
    Trajectories().traj_MM(model=Lmodel, name=name_graph, save_fig=save)
    pass

def ExTraj_UA(name_model="model_Flow_Averaging_AE", name_graph="Simulation_Convergence_Trajectories_UA_FLow_Averaging_AE", save=False):
    """plots the curves of convergence between the trajectories integrated with f and F_app with the numerical method chosen
    Inputs:
    - name_model: Character string - Name of the model made with neural network chosen for integration with F_app
    - name_graph: Character string - Name of the graph which will be registered
    - save: Boolean - Saves the figure or not (default: False)"""
    Lmodel = torch.load(name_model)
    Trajectories_UA().traj_UA(model=Lmodel, name=name_graph, save_fig=save)
    pass

def ExTraj_UA_MM(name_model="model_Flow_Averaging_AE", name_graph="Simulation_Convergence_Trajectories_UA_FLow_Averaging_AE_MM", save=False):
    """plots the curves of convergence between the trajectories integrated with f and F_app with the numerical method chosen + Micro-Macro corrective term
    Inputs:
    - name_model: Character string - Name of the model made with neural network chosen for integration with F_app
    - name_graph: Character string - Name of the graph which will be registered
    - save: Boolean - Saves the figure or not (default: False)"""
    Lmodel = torch.load(name_model)
    Trajectories_UA().traj_UA_MM(model=Lmodel, name=name_graph, save_fig=save)
    pass

def ExConv(name_model="best_model", name_graph="Simulation_Convergence", save=False):
    """Plots the curves of convergence between the learned field and the modified field
    Inputs:
    - name_model: Character string - Name of the model made with neural network chosen for integration with f_app
    - name_graph: Character string - Name of the graph which will be registered
    - save: Boolean - Saves the figure or not (default: False)"""
    best_model = torch.load(name_model)
    Convergence().curves(model=best_model, name=name_graph, save_fig=save)
    pass
