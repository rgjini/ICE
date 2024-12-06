import numpy as np
from numpy.linalg import cholesky, solve, norm, svd
import EnsembleMethods as EKA
import l96.L96_model as l96
import matplotlib.pyplot as plt

path = 'l96/'
np.random.seed(42)  #initialize random seed

def model_setup():
    nx = 40
    gamma = 8.0
    t = 0.01 
    T = 0.2

    H = np.loadtxt(path + 'data/H.txt', delimiter = ',') #sampling matrix

    y = np.loadtxt(path + 'data/y.txt', delimiter = ',')
    R = np.loadtxt(path + 'data/R.txt', delimiter = ',')
    mu = np.loadtxt(path + 'data/mu.txt', delimiter = ',')
    B = np.loadtxt(path + 'data/B.txt', delimiter = ',')

    return nx, t, T, H, gamma, y, R, mu, B



nx, t, T, H, gamma, y, R, mu, B = model_setup()

#Intitializing EKI ensemble 
K = 100         #number of ensemble members
max_iter = 1   #set a maximum number of runs 
N_t = nx       

y_len = len(y)
u = np.zeros((max_iter + 1, N_t, K))    #initialize parameter ensemble
u[0] = np.random.normal(0, 1, size = (N_t,K))

#Compute needed matrices
Rsq_inv = EKA.matrix_inv_sqrt(R)
Bsq = EKA.matrix_sqrt(B)

#Date vector
z = Rsq_inv@y
z_arr = np.tile(z, (K,1)).T
z_len = len(z)

#Identity matrix (new data error matrix)
I = np.identity(z_len)
#Ensemble mean and initial RMSE
u_bar = np.mean(u[0], axis = 1)

#ensemble model output
i = 0
g = np.zeros((z_len, K))
for j in range(0, K): 
    g[:y_len,j] = Rsq_inv@l96.G(Bsq@u[i,:,j] + mu, t, T, H, gamma)
g_bar = np.mean(g, axis = 1)
#Calculating covariances 
Cug = ((u[i].T - u_bar).T@(g.T - g_bar))/(K - 1)   #Cxy
Cgg = np.cov(g)                                    #Cyy

plt.figure()
plt.imshow(Cgg)
plt.title('Ensemble Empirical Estimate')
plt.colorbar()
plt.show(block = False)

import gpflow
import tensorflow as tf
tf.experimental.numpy.experimental_enable_numpy_behavior()

model = gpflow.models.GPR(
    (u[0].T, g.T),
    kernel=gpflow.kernels.Matern52(), noise_variance=1e-3
)
# Fit the model
opt = gpflow.optimizers.Scipy()
opt.minimize(model.training_loss, model.trainable_variables)


xs = np.random.normal(0, 1, size = (N_t,100000))
ys, _ = model.predict_f(xs.T, full_output_cov = False)  #full_output_cov = True not supported??

print(ys.shape)

GP_Cgg = np.cov(ys.T)

plt.figure()
plt.imshow(GP_Cgg)
plt.title('GP Empirical Estimate')
plt.colorbar()
plt.show()


# #Note to self - check to make sure that you can switch the format of the lorenz function
# (Bsq@u[i + 1] + mu[:,np.newaxis]), (i+1)*K, Cgg, exit




#TEKI Test 
# u_out, f_out, _ = EKA.TEKI(l96.G, u, (t, T, H, gamma),  
#                     y, R, mu, B, method = "rmse", 
#                     min_rmse = 1, tol_x = 1e-4, tol_f = 1e-4, max_iter = max_runs)

