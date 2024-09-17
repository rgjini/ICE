'''
Ensemble methods to test covariance corrections
By: Rebecca Gjini and Kyle Ivey
09/14/2024

This script stores all the ensemble algorithms for testing our covariance correction method
'''

'''
Notes: 
Hey Kyle, I included just TEKI in here (and I have code for ETKI, UKI, and IEKF), but 
we should obviously also include ESMDA and any other ensemble smoother methods
'''

import numpy as np 
from numpy.linalg import cholesky, solve, norm, svd

def matrix_inv(A): 
    '''
    Take the inverse of a matrix using SVD

    A: list of floats 2D array
        Matrix to take the inverse of 

    Returns 
    -------

    A_inv_sqrt : list of floats 2D array
        inverse square of matrix A
    '''
    [u_a, l_a, q_a] = svd(A)  
    return (u_a/l_a)@u_a.T

def matrix_inv_sqrt(A): 
    '''
    Take the inverse sqrt of a matrix using SVD

    A: list of floats 2D array
        Matrix to take the inverse sqrt of 

    Returns 
    -------

    A_inv_sqrt : list of floats 2D array
        inverse square root of matrix A
    '''
    [u_a, l_a, q_a] = svd(A)  
    return (u_a/np.sqrt(l_a))@u_a.T  #put tolerance so you can decide how much variance for each covarinace matrix

def matrix_sqrt(A): 
    '''
    Take the sqrt of a matrix using SVD

    A: list of floats 2D array
        Matrix to take the sqrt of 

    Returns 
    -------

    A_sqrt : list of floats 2D array
        square root of matrix A
    '''
    [u_a, l_a, q_a] = svd(A)  
    return (u_a*np.sqrt(l_a))@u_a.T
    
def TEKI(func, u_0, args, y, R, mu, B, min_rmse = 1, tol_x = 0.0001, tol_f = 0.0001, 
         method = "all", max_iter = 10e2):
    '''
    TEKI algorithm from Chada et al. 2019

    func: function 
        forward model function that is compared to the data
    u_0: list of floats 2D array
        array of intial ensemble members to be updated throught the algorithm (n,k)
        must be distributed mean zero, covariance I (whiten before running)
        n -> number of parameters 
        k -> number of ensemble members
    args: tuple of variables?
        tuple of additional variables that are passed into the function
    y: list of floats array
        data vector with dimensions (m, 1)
    R: list of floats 2D array
        data covariance matrix with dimensions (m, m)
    mu: list of floats array
        prior mean with dimensions (n, 1)
    B: list of floats 2D array
        prior covariance matrix with dimensions (n, n)
    min_rmse: float
        target RMSE value
    tol_x: float
        target tolerance between x_n and x_(n+1)
    tol_f: float
        target tolerance between func_n and func_(n+1)
    method: String 
        method of convergence
        choose between ('all', 'rmse', 'tol_x', 'tol_f')
    max_iter: int
        maximum number of iterations before the algorithm stops
    
    Returns 
    -------
    u[i], i*K, exit 
    
    u[i]: list of floats 2D array
        ensemble at final iteration
    (i+1)*K: float
        number of foward runs
    exit: int 
        if method = "all", exit criteria is returned 
        0 - hit max number of iterations
        1 - hit rmse tol
        2 - hit tol_x 
        3 - hit tol_f
    ''' 
    exit = 0 # exit criteria 

    (N_t, K) = u_0.shape  #parameter space size, ensemble size
    y_len = len(y)
    u = np.zeros((max_iter + 1, N_t, K))    #initialize parameter ensemble
    u[0] = u_0
    #Compute needed matrices
    Rsq_inv = matrix_inv_sqrt(R)
    Bsq = matrix_sqrt(B)
    #Date vector
    z = np.concatenate((Rsq_inv@y, np.zeros(len(mu))))
    z_arr = np.tile(z, (K,1)).T
    z_len = len(z)
    #Identity matrix (new data error matrix)
    I = np.identity(z_len)
    #Ensemble mean and initial RMSE
    u_bar = np.mean(u[0], axis = 1)
    RMSE_u = norm(Rsq_inv@(y - func(Bsq@u_bar + mu, *args)))/np.sqrt(y_len)

    #Algorithm:
    for i in range(0,max_iter): 
        #ensemble model output
        g = np.zeros((z_len, K))
        for j in range(0, K): 
            g[:y_len,j] = Rsq_inv@func(Bsq@u[i,:,j] + mu, *args)
            g[y_len:,j] = u[i,:,j] 
        g_bar = np.mean(g, axis = 1)
        #Calculating covariances 
        Cug = ((u[i].T - u_bar).T@(g.T - g_bar))/(K - 1)   #Cxy
        Cgg = np.cov(g)                                    #Cyy
        #Update
        update = Cug@solve(Cgg + I, (np.random.normal(0, 1, size = (z_len, K)) - 
                                    g) + z_arr)
        u[i + 1] = u[i] + update
        #Calculate ensemble mean
        u_bar_1 = u_bar
        u_bar = np.mean(u[i + 1], axis = 1)

        if method == "all": 
            RMSE_u_1 = RMSE_u
            RMSE_u = norm(Rsq_inv@(y - func(Bsq@u_bar + mu, *args)))/np.sqrt(y_len)
            if RMSE_u < min_rmse:  #Convergence Criteria
                exit = 1
                break
            mean_diff = norm(u_bar - u_bar_1)/norm(u_bar_1)
            if ((mean_diff < tol_x) and (norm(u_bar_1) > 1e-5) and (i > 0)):  #Convergence Criteria
                exit = 2
                break
            func_diff = np.abs((RMSE_u - RMSE_u_1)/RMSE_u_1)
            if func_diff < tol_f: #Convergence Criteria
                exit = 3
                break
        elif method == "rmse":
            RMSE_u = norm(Rsq_inv@(y - func(Bsq@u_bar + mu, *args)))/np.sqrt(y_len)
            #print('RMSE: %s Iter: %s' % (np.round(RMSE_u, 5), i + 1))
            if RMSE_u < min_rmse:  #Convergence Criteria
                exit = 1
                break
        elif method == "tol_x": 
            mean_diff = norm(u_bar - u_bar_1)/norm(u_bar_1)
            #print('Mean Difference: %s Iter: %s' % (np.round(mean_diff, 5), i + 1))
            if (mean_diff < tol_x) and (norm(u_bar_1) > 1e-5) and (i > 0):  #Convergence Criteria
                exit = 2
                break
        else: #tol_f 
            RMSE_u_1 = RMSE_u
            RMSE_u = norm(Rsq_inv@(y - func(Bsq@u_bar + mu, *args)))/np.sqrt(y_len)
            func_diff = np.abs((RMSE_u - RMSE_u_1)/RMSE_u_1)
            #print('Function Difference: %s Iter: %s' % (np.round(func_diff, 5), i + 1))
            if func_diff < tol_f: #Convergence Criteria
                exit = 3
                break
    #Note to self - check to make sure that you can switch the format of the lorenz function
    return (Bsq@u[i + 1] + mu[:,np.newaxis]), (i+1)*K, exit
