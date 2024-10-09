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

import numpy as np
import numpy as np
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

dyn_syst = "VDP"                # Choice between "VDP" (Van der Pol oscillator)
num_meth = "Forward_Euler"      # Numerical method selected for modified averaged field learning: choice between "Forward_Euler" and "MidPoint"
step_h = [0.001,0.1]            # Interval where time step is selected
step_eps = [0.001, 0.2]         # Interval where high oscillation parameter is selected
T_simul = 1                     # Time for ODE's simulation
h_simul = 0.01                  # Time step used for ODE's simulation
eps_simul = 0.001                 # High oscillation parameter used for ODE's simulation

# AI parameters [adjust]

K_data = 100000                  # Quantity of data
R = 2                          # Amplitude of data in space (i.e. space data will be selected in the box [-R,R]^d)
p_train = 0.8                  # Proportion of data for training
HL = 2                         # Hidden layers per MLP for the first Neural Network
zeta = 200                     # Neurons per hidden layer of the first Neural Network
alpha = 2e-3                   # Learning rate for gradient descent
Lambda = 1e-9                  # Weight decay
BS = 100                       # Batch size (for mini-batching) for training
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
print("        - Interval where time step is selected:", step_h)
print("        - Interval where high oscillation parameter is selected:", step_eps)
print("        - Time for ODE's simulation:", T_simul)
print("        - Time step used for ODE's simulation:", h_simul)
print("        - High oscillation parameter used for ODE's simulation:", eps_simul)
print("    # AI parameters:")
print("        - Data's number for training:", K_data)
print("        - Amplitude of data in space:", R)
print("        - Proportion of data for training:", format(p_train, '.1%'))
print("        - Hidden layers per MLP's for the first Neural Network:", HL)
print("        - Neurons on each hidden layer for the first Neural Network:", zeta)
print("        - Learning rate:", format(alpha, '.2e'))
print("        - Weight decay:", format(Lambda, '.2e'))
print("        - Batch size (mini-batching) for training:", BS)
print("        - Epochs for training:", N_epochs)
print("        - Epochs between two prints of the Loss value for training:", N_epochs_print)

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

    def A(eps):
        """Matrix wich generates high oscillations.
        Input:
        - eps: Float -High oscillation parameter"""
        if dyn_syst == "VDP":
            return (1/eps)*np.array([[0,1],[-1,0]])

    def g(y):
        """Nonlinear part of the system.
        Input:
        - y: Array of shape (d,) - Space variable"""
        if dyn_syst == "VDP":
            y1 , y2 = y[0] , y[1]
            return np.array([0,(1/4-y1**2)*y2])

    def f(t,y,eps):
        """Describes the dynamics of the studied ODE
         - t : Float - Time variable
         - y : Array of shape (d,) - Space variable
         - eps: Float - High oscillation parameter
         Returns an array of shape (d,)"""
        y = np.reshape(np.array(y) , (d,))
        return NA.A(eps)@y + NA.g(y)

    def A_NN(eps):
        """Tensor form of the linear generator of high oscillations.
        Input:
        - eps: Float - High oscillation parameter"""
        return torch.tensor(NA.A(eps))

    def g_NN(y):
        """Nonlinear part of the system, adapted for neural network.
        Input:
        - y: Tensor of shape (d,n) - Space variable"""
        nb_coeff = 1
        for s in y.shape:
            nb_coeff = nb_coeff * s
        y = torch.tensor(y).reshape(d, int(nb_coeff / d))
        z = torch.zeros_like(y)
        if dyn_syst == "VDP":
            y1 , y2 = y[0,:] , y[1,:]
            z = y
            z[0,:] = 0*y[0,:]
            z[1,:] = (1/4-y1**2)*y2
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
            return NA.A(epsilon)@y + NA.g(y)

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
     H = np.exp(np.random.uniform(low = np.log(step_h[0]) , high = np.log(step_h[1]) , size = (1,K)))
     EPS = np.exp(np.random.uniform(low = np.log(step_eps[0]) , high = np.log(step_eps[1]) , size = (1,K)))
     tt0 = np.random.uniform(low = 0 , high = 0*np.pi , size = (1,K))

     if dyn_syst == "Logistic":
         Y0 = np.abs(Y0)

     pow = max([int(np.log10(K) - 1), 3])
     pow = min([pow, 6])

     for k in range(K):
         end_time_data = start_time_data + (K / (k + 1)) * (time.time() - start_time_data)
         end_time_data = datetime.datetime.fromtimestamp(int(end_time_data)).strftime(' %Y-%m-%d %H:%M:%S')
         print(" Loading :  {} % \r".format(str(int(10 ** (pow) * (k + 1) / K) / 10 ** (pow - 2)).rjust(3)), " Estimated time for ending : " + end_time_data, " - ", end="")
         Y1[:,k] = NA.ODE_Solve(ti = tt0[0,k] , tf = tt0[0,k] + H[0,k] , h = H[0,k] , epsilon = EPS[0,k] , Yinit = Y0[:,k])[:,1]

     K0 = int(p_train*K)
     Y0_train = torch.tensor(Y0[:, 0:K0])
     Y0_test = torch.tensor(Y0[:, K0:K])
     Y1_train = torch.tensor(Y1[:, 0:K0])
     Y1_test = torch.tensor(Y1[:, K0:K])
     H_train = torch.tensor(H[:, 0:K0])
     H_test = torch.tensor(H[:, K0:K])
     EPS_train = torch.tensor(EPS[:, 0:K0])
     EPS_test = torch.tensor(EPS[:, K0:K])
     tt0_train = torch.tensor(tt0[:, 0:K0])
     tt0_test = torch.tensor(tt0[:, K0:K])

     print("Computation time for data creation (h:min:s):",
           str(datetime.timedelta(seconds=int(time.time() - start_time_data))))
     return (Y0_train , Y0_test , Y1_train , Y1_test , H_train , H_test , EPS_train , EPS_test , tt0_train , tt0_test)

class NN(nn.Module, NA):
    def __init__(self):
        super().__init__()
        self.R_Phi_g = nn.ModuleList([nn.Linear(d + 2, zeta), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta, zeta), nn.Tanh()] + [nn.Linear(zeta, d, bias=True)])
        self.R_Phi_A = nn.ModuleList([nn.Linear(d + 3, zeta), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta, zeta), nn.Tanh()] + [nn.Linear(zeta, d, bias=True)])

    def forward(self, t, y, h, eps):
        """Structured Neural Network.
        Inputs:
         - t: Tensor of shape (1,n) - Time variable
         - y: Tensor of shape (d,n) - space variable
         - h: Tensor of shape (1,n) - Step size
         - eps: Tensor of shape (1,n) - High oscillation parameter"""

        y = y.T
        y = y.float()
        t = torch.tensor(t).T
        h = torch.tensor(h).T
        eps = torch.tensor(eps).T

        # Structure of the solution of the equation

        y1 = y

        x_Phi_g = torch.cat((y , h , eps) , dim = 1)
        x_Phi_g_0 = torch.cat((y , 0*h , eps) , dim = 1)
        for i, module in enumerate(self.R_Phi_g):
            x_Phi_g = module(x_Phi_g)
            x_Phi_g_0 = module(x_Phi_g_0)

        y1 = y1 + h*(x_Phi_g-x_Phi_g_0)

        x_Phi_A = torch.cat((torch.cos((t+h)/eps) , torch.sin((t+h)/eps) , y1 , eps) , dim = 1)
        x_Phi_A_0 = torch.cat((torch.ones_like((t+h)/eps) , torch.zeros_like((t+h)/eps) , y1 , eps) , dim = 1)
        for i, module in enumerate(self.R_Phi_A):
            x_Phi_A = module(x_Phi_A)
            x_Phi_A_0 = module(x_Phi_A_0)

        y1 = y1 + (x_Phi_A-x_Phi_A_0)

        # Commutativity of the flows

        y2 = y

        x_Phi_A_2 = torch.cat((torch.cos((t+h)/eps) , torch.sin((t+h)/eps) , y , eps) , dim = 1)
        x_Phi_A_2_0 = torch.cat((torch.ones_like((t+h)/eps) , torch.zeros_like((t+h)/eps) , y , eps) , dim = 1)
        for i, module in enumerate(self.R_Phi_A):
            x_Phi_A_2 = module(x_Phi_A_2)
            x_Phi_A_2_0 = module(x_Phi_A_2_0)

        y2 = y2 + (x_Phi_A_2-x_Phi_A_2_0)

        x_Phi_g_2 = torch.cat((y2, h, eps), dim=1)
        for i, module in enumerate(self.R_Phi_g):
            x_Phi_g_2 = module(x_Phi_g_2)

        y2 = y2 + h * x_Phi_g_2


        # Flow associated to high oscillation part

        x_Phi_A_1 = torch.cat((torch.cos((t) / eps), torch.sin((t) / eps), y, eps), dim=1)
        x_Phi_A_0_1 = torch.cat((torch.ones_like((t) / eps), torch.zeros_like((t) / eps), y, eps), dim=1)
        for i, module in enumerate(self.R_Phi_A):
            x_Phi_A_1 = module(x_Phi_A_1)
            x_Phi_A_0_1 = module(x_Phi_A_0_1)

        y_Phi_A = y + (x_Phi_A_1 - x_Phi_A_0_1)

        # Flow associated to non oscillation part

        x_Phi_g_1 = torch.cat((y, h, eps), dim=1)
        for i, module in enumerate(self.R_Phi_g):
            x_Phi_g_1 = module(x_Phi_g_1)

        y_Phi_g = y + h*(x_Phi_g_1)



        return (y1).T , (y1-y2).T , (y_Phi_A).T , (y_Phi_g).T

class Train(NN, NA):
    """Training of the neural network, depends on the numerical method chosen
    Choice of the numerical method:
        - Forward Euler
        - MidPoint"""

    def Loss(self, Y0, Y1, T0, H, EPS, model):
        """Computes the Loss function between two series of data Y0 and Y1 according to the numerical method
        Inputs:
        - Y0: Tensor of shape (d,n): Initial data
        - Y1: Tensor of shape (d,n): Exact data
        - T0: Tensor of shape (1,n): Time variable
        - H: Tensor of shape (1,n): Step size
        - EPS: Tensor of shape (1,n): High oscillation parameter
        - model: Neural network which will be optimized
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

        YY = model(0*T0, Y0 , H, EPS)[0]
        Y1_hat = YY

        loss_ODE = (((Y1_hat - Y1)).abs() ** 2).mean()
        loss_flow_commute = ((model(0*T0, Y0 , H , EPS)[1])**2).mean()

        #print(loss_ODE , loss_flow_commute)

        loss = loss_ODE + loss_flow_commute


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
        print("Training...")
        print(150 * "-")

        Y0_train = Data[0]
        Y0_test = Data[1]
        Y1_train = Data[2]
        Y1_test = Data[3]
        H_train = Data[4]
        H_test = Data[5]
        EPS_train = Data[6]
        EPS_test = Data[7]
        T0_train = Data[8]
        T0_test = Data[9]

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

            loss_test = self.Loss(Y0_test, Y1_test ,T0_test, H_test, EPS_test, model)

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

        def Flow_hat(y , h , eps):
            """Learned (exact) flow associated to slow part of the vector field
            Inputs:
            - y: Array of shape (d,) - Space variable
            - h: Float - Step size
            - eps: Float - High oscillation parameter"""
            y = torch.tensor(y).reshape(d, 1)
            y.requires_grad = True
            h_tensor = torch.tensor([[h]]).float()
            eps_tensor = torch.tensor([[eps]]).float()
            z = Model(y , h_tensor , eps_tensor)[0]
            z = z.detach().numpy()
            z = np.array(z, dtype=np.float64)
            return z.reshape(d, )

        def Flow_hat_g(t, y , h , eps):
            """Learned (exact) flow associated to slow part of the vector field
            Inputs:
            - t: Float - Tima variable
            - y: Array of shape (d,) - Space variable
            - h: Float - Step size
            - eps: Float - High oscillation parameter"""
            y = torch.tensor(y).reshape(d, 1)
            y.requires_grad = True
            t_tensor = torch.tensor([[t]]).float()
            h_tensor = torch.tensor([[h]]).float()
            eps_tensor = torch.tensor([[eps]]).float()
            z = Model(t_tensor , y , h_tensor , eps_tensor)[3]
            z = z.detach().numpy()
            z = np.array(z, dtype=np.float64)
            return z.reshape(d, )

        def Flow_hat_A(t , y , h , eps):
            """Learned (exact) periodic flow associated to high oscillation part of the vector field
            Inputs:
            - t: Float - Time variable
            - y: Array of shape (d,) - Space variable
            - h: Float - Step size
            - eps: Float - High oscillation parameter"""
            y = torch.tensor(y).reshape(d, 1)
            y.requires_grad = True
            t_tensor = torch.tensor([[t]]).float()
            h_tensor = torch.tensor([[h]]).float()
            eps_tensor = torch.tensor([[eps]]).float()
            z = Model(t_tensor , y , h_tensor , eps_tensor)[2]
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
        Y_app[:,0] = y0start(dyn_syst)
        for n in range(np.size(TT)-1):
            Y_app[:,n+1] = Flow_hat_g(TT[n] , Y_app[:,n] , h_simul , eps_simul)
        #    Y_app[:,n+1] = Flow_hat(Y_app[:,n] , h_simul , eps_simul)
        for n in range(np.size(TT)):
            Y_app[:,n] = Flow_hat_A(TT[n] , Y_app[:,n] , h_simul , eps_simul)

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
            plt.plot(Y_exact[0, :], Y_exact[1, :], color='black', linestyle='dashed',label=r"$\varphi_{t_n}^f(y_0)$")
            plt.plot(Y_app[0, :], Y_app[1, :], color='green', label="$(\Phi_{\\theta,h,t_n}^{f})^n(y_0)$")
            plt.xlabel("$y_1$")
            plt.ylabel("$y_2$")
            plt.legend()
            plt.grid()
            write_size()
            plt.subplot(2, 2, 2)
            plt.title("Comparison of local errors")
            plt.ylim(min(err_app[1:]),max(err_app[1:]))
            plt.yscale('log')
            plt.plot(TT, err_app, color="orange", label="$|$"+r"$\varphi_{t_n}^f(y_0) - (\Phi_{\theta,h,t_n}^{f})^n(y_0) |$")
            plt.xlabel("t")
            plt.ylabel("Local error")
            plt.legend()
            plt.grid()
            write_size()

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
        f = plt.gcf()
        dpi = f.get_dpi()
        h, w = f.get_size_inches()
        f.set_size_inches(h * 1.7, w * 1.7)

        if save_fig == True:
            plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
        else:
            plt.show()

        print("Computation time for integration (h:min:s):",
              str(datetime.timedelta(seconds=int(time.time() - start_time_integrate))))

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

                def Flow_hat(y, h, eps):
                    """Learned (exact) flow associated to slow part of the vector field
                    Inputs:
                    - y: Array of shape (d,) - Space variable
                    - h: Float - Step size
                    - eps: Float - High oscillation parameter"""
                    y = torch.tensor(y).reshape(d, 1)
                    y.requires_grad = True
                    h_tensor = torch.tensor([[h]]).float()
                    eps_tensor = torch.tensor([[eps]]).float()
                    z = Model(y, h_tensor, eps_tensor)[0]
                    z = z.detach().numpy()
                    z = np.array(z, dtype=np.float64)
                    return z.reshape(d, )

                def Flow_hat_g(t, y, h, eps):
                    """Learned (exact) flow associated to slow part of the vector field
                    Inputs:
                    - t: Float - Tima variable
                    - y: Array of shape (d,) - Space variable
                    - h: Float - Step size
                    - eps: Float - High oscillation parameter"""
                    y = torch.tensor(y).reshape(d, 1)
                    y.requires_grad = True
                    t_tensor = torch.tensor([[t]]).float()
                    h_tensor = torch.tensor([[h]]).float()
                    eps_tensor = torch.tensor([[eps]]).float()
                    z = Model(t_tensor, y, h_tensor, eps_tensor)[3]
                    z = z.detach().numpy()
                    z = np.array(z, dtype=np.float64)
                    return z.reshape(d, )

                def Flow_hat_A(t, y, h, eps):
                    """Learned (exact) periodic flow associated to high oscillation part of the vector field
                    Inputs:
                    - t: Float - Time variable
                    - y: Array of shape (d,) - Space variable
                    - h: Float - Step size
                    - eps: Float - High oscillation parameter"""
                    y = torch.tensor(y).reshape(d, 1)
                    y.requires_grad = True
                    t_tensor = torch.tensor([[t]]).float()
                    h_tensor = torch.tensor([[h]]).float()
                    eps_tensor = torch.tensor([[eps]]).float()
                    z = Model(t_tensor, y, h_tensor, eps_tensor)[2]
                    z = z.detach().numpy()
                    z = np.array(z, dtype=np.float64)
                    return z.reshape(d, )

                # Integration with DOP853 (approximation of the exact flow)
                Y_exact = NA.ODE_Solve(ti=0, tf=T_simul, h=hh, epsilon=eeps, Yinit=y0start(dyn_syst))

                TT = np.arange(0, T_simul + hh, hh)

                # Integration with learned flow
                start_time_app = time.time()
                Y_app = np.zeros_like(Y_exact)
                Y_app[:, 0] = y0start(dyn_syst)
                for n in range(np.size(TT) - 1):
                    Y_app[:, n + 1] = Flow_hat_g(TT[n] , Y_app[:, n], hh, eeps)
                    #Y_app[:, n + 1] = Flow_hat(Y_app[:, n], hh, eeps)
                for n in range(np.size(TT)):
                    Y_app[:, n] = Flow_hat_A(TT[n], Y_app[:, n], hh , eeps)


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




def ExData(name_data="DataODE_Autonomous_FLow_Averaging"):
    """Creates data y0, y1 with the function solvefEDOData
    with the chosen vector field at the beginning of the program
    Input:
    - name_data: Character string - Name of the registered tuple containing the data (default: "DataEDO")"""
    DataODE = DataCreate.Data(K_data)
    torch.save(DataODE, name_data)
    pass

def ExTrain(name_model='model_Autonomous_Flow_Averaging', name_data='DataODE_Autonomous_Flow_Averaging'):
    """Launches training and computes Loss_train, loss_test and best_model with the function Train().train
    Saves the files Loss_train, Loss_test and best_model with a given name
    Inputs (character strings):
    - name_model: Name of the file saved for best_model (default: "best_model")
    - name_data: Name of the file containing the created data (default: "DataEDO") used for training"""
    DataODE = torch.load(name_data)
    Loss_train, Loss_test, best_model = Train().train(model=NN(), Data=DataODE)
    torch.save((best_model,Loss_train,Loss_test), name_model)
    pass

def ExIntegrate(name_model="model_Autonomous_Flow_Averaging", name_graph="Simulation_Autonomous_Flow_Averaging", save=False):
    """Launches integration of the main equation and modified equation with the chosen model
    Inputs:
    - name_model: Character string - Name of the model made with neural network chosen for integration with F_app, and Loss_train/Loss_test
    - name_graph: Character string - Name of the graph which will be registered
    - save: Boolean - Saves the figure or not (default: False)"""
    Lmodel = torch.load(name_model)
    Integrate().integrate(model=Lmodel, name=name_graph,save_fig=save)
    pass

def ExTraj(name_model="model_Autonomous_Flow_Averaging", name_graph="Simulation_Convergence_Trajectories_Autonomous_FLow_Averaging", save=False):
    """plots the curves of convergence between the trajectories integrated with f and F_app with the numerical method chosen
    Inputs:
    - name_model: Character string - Name of the model made with neural network chosen for integration with F_app
    - name_graph: Character string - Name of the graph which will be registered
    - save: Boolean - Saves the figure or not (default: False)"""
    Lmodel = torch.load(name_model)
    Trajectories().traj(model=Lmodel, name=name_graph, save_fig=save)
    pass

#ExData()
#ExTrain()

