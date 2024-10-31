'''
Lorenz 96 model 
By: Rebecca Gjini 
Date: 04/10/2024

This version of the L96 model is meant to be used for testing EKI algorithms
-For parameter estimation of 40 parameters
'''
import numpy as np
import numba 
from numba import jit, njit

@njit
def lorenz(x, gamma = 8.0): 
    '''
    This method computes the lorenz 96 system and ouputs the ode vector x
    
    x : array of floats
        Vector of the conditions to simulate through the lorenz system (i.e initial coniditions)
    gamma : double 
        Forcing for the lorenz system (default value is 8)

    Returns
    --------

    1D array of floats : 
        Vector of the solution to the L96 equation
    '''
    N = len(x)
    out_x = np.zeros((N))
    for kk in range(0, N): 
        out_x[kk] = - x[kk] + (x[(kk + 1)%N] - x[(kk - 2)])*x[(kk - 1)] + gamma
    return out_x

@njit
def runge_kutta_v(x_0, t, T, gamma):
     # x_0: inital state, t: time step, T: runtime
    '''
    This method simulates the lorenz system over time using a 4th order runge-kutta scheme
    
    x_0 : array of floats
        Vector of the initial conitions to simulate through the lorenz system (i.e initial coniditions)
    t : double 
        time step
    T : int
        total time to simulate through
    gamma : double 
        forcing constant 

    Returns
    -------

    solv : 2d array of floats 
        rows are values at each time step and columns are the parameter vector of the 
        lorenz 96 model
    '''
    solv = np.empty((int(T / t) + 1, len(x_0))) #array to store the solution
    solv[0] = x_0
    for i in range(0, len(solv) - 1, 1): 
        #calculating k values for xn + 1
        k1 = lorenz(solv[i], gamma)
        k2 = lorenz(solv[i] + t*0.5*k1, gamma)
        k3 = lorenz(solv[i] + t*0.5*k2, gamma)
        k4 = lorenz(solv[i] + t*k3, gamma)
        #calculating the x_{n+1} value 
        solv[i + 1] = solv[i] + (1.0/6.0)*t*(k1 + 2*k2 + 2*k3 + k4)
    return solv

def G(x_0, t, T, H, gamma): 
    '''
    This function is the forward model for the L96 problem, outputting the every other state 
    after simulating for a certain amount of time
    
    x_0 : array of floats 
        initial condition (i.e. parameters) of the l96 system 
    t : double 
        time step
    T : int
        total time to simulate through
    H : 2D array of ints
        sampling matrix
    gamma : double 
        forcing constant

    Returns
    -------

    array of floats 
        state of the l96 system after a certain amount of time T and sampled using H
    '''

    #run runge_kutta scheme, select last value at the end
    return (runge_kutta_v(x_0, t, T, gamma)[-1])

def r(x_0, t, T, H, gamma, y, Rinv_sqrt, mu, Bsqrt): 
    '''
    r(x) function (vector form of cost function) for the levenburg-marquard algorithm with 
    finite differencing
    
    x_0 : tuple of doubles 
        rho and beta parameters for the lorenz 63 system
    t : double 
        time step
    T : int
        total time to simulate through
    H : 2D array of ints
        sampling matrix
    gamma : double 
        forcing constant
    y : array of floats 
        the data
    Rin_sqrt : 2D array of floats
        Inverse squareroot of the data covaraince matrix
    mu : array of floats 
        prior mean
    Bsqrt : 2D array of floats
        Squareroot of the prior covaraince matrix

    Returns
    -------

    array of floats 
        Cost function vector
    
    '''
    return (1/np.sqrt(2))*np.concatenate((Rinv_sqrt@(G(Bsqrt@x_0 + mu, t, T, H, gamma) - y), x_0))


