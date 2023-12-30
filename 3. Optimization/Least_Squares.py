import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


# define the true parameters
A, B, C = 1.2, 2.3, 2.8
Y = np.array([A, B, C])


# define the function to approximate
def fun_obj(
        x: np.ndarray,
        a: float,
        b: float,
        c: float
    ):
    '''
    replicate a sinusoide
    '''

    y = a + b*np.sin(c*x)
    return y


def create_data(
        a: float,
        b: float,
        c: float,
        n: int=100,
        noise: int=1
    ):
    '''
    build a dataset with the objective function and a noise

    parameters
    ----------
        a, b, c: parameters to use
        n: number of datapoints to generate
        noise: random component to add to each datapoint

    returns
    ---------
        x, y: datapoints following the function pattern with an additional random component
    '''

    # generate the data
    x = np.linspace(-1, 5, n)
    y = fun_obj(x, a, b, c) + noise*np.random.random(n)

    return x, y


def compute_square_dist(
        x: float,
        y: float,
        a: float,
        b: float,
        c: float
    ):
    '''
    compute the square of the distance from the predicted y to the true y
    
    parameters
    ----------
        x, y: datapoints following the function pattern with an additional random component
        a, b, c: parameters to use for the prediction
    
    returns
    ----------
        d_sq: squared distance from the predicted to the true value
    '''

    pred = fun_obj(x, a, b, c) # predict the value
    d_sq = np.mean((y-pred)**2) # compute the square distance

    return d_sq


def loss(
        params: np.ndarray,
        x: float,
        y: float
    ):
    '''
    compute the loss function for the optimization

    parameters
    ----------
        params: parameters to optimize
        x, y: datapoints following the function pattern with an additional random component
    
    returns
    ----------
        ls: loss function value
    '''
    
    ls = compute_square_dist(x, y, params[0], params[1], params[2]) # compute the loss function as the square of the distance
    return ls


def norm(v: np.ndarray):
    '''
    compute the norm of a vector
    
    parameters
    ----------
        v: vector
    
    returns
    ----------
        norm: norm of the vector
    '''

    norm = np.sqrt(v@v)
    return norm


def finite_diff(
        fun,
        p: np.ndarray,
        delta: float=1e-5
    ):
    '''
    compute the finite difference version for the gradient of a given dimensional function in a specific point
    
    parameters
    ----------
        fun: function to optimize
        p: point in the dimensional space
        delta: displacement to compute the finite differencing

    returns
    ----------
        grad: gradient of the function in the finite difference version
    '''

    n = len(p)

    grad = np.zeros(n) # initialie the gradient
    # loop over the dimensions
    for i in range(n):
        p0 = p[i] # get the starting point in the current dimension

        # compute the function on the left
        p[i] = p0-delta
        fun_left = fun(p)

        # compute the function on the right
        p[i] = p0+delta
        fun_right = fun(p)

        grad[i] = (fun_right-fun_left)/(2*delta) # compute the finite difference as an average of the two
        p[i] = p0 # set the new starting point

    return grad


def grad_descent(
        fun,
        x0: np.ndarray,
        grad_fun=None,
        nesterov: bool=False,
        max_steps: int=1000,
        alpha: float=0.01,
        beta: float=0.5,
        epsi: float=1e-3,
        keep_steps: bool=True,
        verbosity: int=0
    ):
    '''
    compute a number of gradient descent optimization steps on the given function from a specific starting point

    parameters
    ----------
        fun: function to optimize
        x0: starting point in the dimensional space
        grad_fun: function to use as gradient of the function
        nesterov: boolean for the use of Nesterov's gradient
        max_steps: maximum number of optimization steps to run
        alpha, beta: weights of the gradient function and the Nesterov's gradient
        epsi: minimum threshold to stop the optimization 
        keep_steps: boolean for the creation of a list of visited points during the optimization
        verbosity: level of verbosity of the function
    
    returns
    ----------
        res: results of the gradient descent optimization 
    '''

    # check that the gradient function is provided and use the finite difference function otherwise
    if grad_fun is None:
        grad_fun = lambda z: finite_diff(fun, z, 1e-3)
    
    x, xs = x0, [x0] # initialize the starting point and the list of visited points
    converged = False # initialize the convergence status

    # check that Nesterov's gradient is being used and initialize it eventually
    if nesterov:
        v_nest = np.zeros(len(x0))
    
    # loop over the steps
    for i in range(max_steps+1):

        # check that Nesterov's gradient is being used and combine the two gradients eventually
        if nesterov:
            v_nest = beta*v_nest + alpha*grad_fun(x+beta*v_nest)
            dx = v_nest

        # use the gradient function only otherwise
        else:
            dx = grad_fun(x)

        # check that the visited points are being stored and append the current one to the list eventually
        if keep_steps:
            x = x-dx if nesterov else x-alpha*dx
            xs.append(x)

        # don't remember it otherwise
        else:
            x -= dx if nesterov else alpha*dx
        
        # control the verbosity of the function
        verb = max_steps/np.arange(1, verbosity+1)
        if verbosity != 0 and i in verb:
            s = f'iter: {i}, x: {np.round(x, 3)}, f(x): {round(fun(x), 3)}, dx: {round(norm(dx), 3)}'
            print(s)
        
        # check that the function has converged and stop it otherwise
        if norm(dx) < epsi:
            converged = True
            break

    # check that the last point is not already in the list and add it eventually
    if not keep_steps:
        xs.append(x)

    return x, xs, converged


x_train, y_train = create_data(A, B, C) # produce some data points
guesses = np.array ([0.8, 1.5, 3.1]) # define the starting parameters

plt.plot(x_train, y_train, 'r.') # plot the values to fit

plt.plot(x_train, fun_obj(x_train, guesses[0], guesses[1], guesses[2])) # plot the values before the optimization

# optimize the parameters and plot the new values
p_opt, p_list, converged = grad_descent(
    lambda z: loss(z, x_train, y_train),
    x0=guesses,
    max_steps=100000,
    alpha=0.01)
plt.plot(x_train, fun_obj(x_train, p_opt[0], p_opt[1], p_opt[2]))

# use scipy for the optimization
p_list = [guesses]
opt_res = minimize(lambda z: loss(z, x_train, y_train), guesses, callback=lambda x: p_list.append(x))
plt.plot(x_train, fun_obj(x_train, opt_res.x[0], opt_res.x[1], opt_res.x[2]), '.')

plt.legend(['true', 'no opt', 'opt', 'scipy'])
plt.show()
