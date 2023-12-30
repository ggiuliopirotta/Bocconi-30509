import matplotlib.pyplot as plt
import numpy as np


class Gd_Results:
    '''
    store the results of a gradient descent optimization
    
    parameters
    ----------
        fun: function to optimize
        x, y: coordinates of the strating point for the optimization
        xs, ys: list of coordinates of the visited points during the optimization
        iters: iterations run to optimize the function
        converged: boolean to indicate whether the optimization converged or not
    '''

    def __init__(
            self,
            fun,
            x: np.ndarray,
            xs: list,
            iters: int,
            converged: bool
        ):

        self.fun = fun # store the function

        # store the coordinates of the starting point
        self.x = x
        self.y = fun(x)

        # store the coordinates of the visited points
        self.xs = np.array(xs)
        self.ys = fun(xs)

        # store other variables
        self.iters = iters
        self.converged = converged

    def __repr__(self):
        '''
        represent the main pieces of information regarding the optimization
        '''

        s = f'\nx_min: {np.round(self.x, 3)}\ny_min: {round(self.y, 3)}\niters: {self.iters}\nconverged: {self.converged}'
        return s


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
        delta: float=1e-3
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
        keep_steps: bool=False,
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
        verb = max_steps/np.arange(verbosity+1)
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

    res = Gd_Results(fun, x, xs, i, converged) # store the results

    return res


def rosenbrock(x: np.ndarray):
    '''
    replicate Rosenbrock's function
    '''

    y = 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
    return y


# define the axes
x_int = np.linspace(-7, 5, 1000)
y_int = np.linspace(-4, 2, 500)

x, y = np.meshgrid(x_int, y_int) # create the 2d grid

x0 = np.array([-1.2, 1]) # define starting point

sol = grad_descent(
    fun=rosenbrock,
    x0=x0,
    max_steps=1000,
    alpha=0.003,
    keep_steps=True,
    verbosity=5
) # optimize the function
print(sol)

ros_xy = rosenbrock((x, y))

# plot
plt.contour(x, y, ros_xy, 20, cmap='RdGy')
plt.plot(sol.xs[:, 0], sol.xs[:, 1])
plt.show()
