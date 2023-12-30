import numpy as np


g_test = [
    np.array([[6., 8., 13.]]),
    np.array([[9., 15., np.inf], [8., 10., 12.], [np.inf, 8., 7.]]),
    np.array([[15., np.inf], [20., 8.], [np.inf, 7.]]),
    np.array([[3.], [4.]])
] # design the test for debugging


def backward(
        c: np.ndarray,
        whence: np.ndarray
    ):
    '''
    back-track the shortest path and the relative minimum cost to get from the starting to the ending node of a graph

    parameters
    ----------
        c: cost matrix containing the minimum cost to get to each cell
        whence: matrix containing the predecessor of each node

    returns
    ----------
        path: shortest path to go through the graph
        cost: minimum cost associated to the shortest path
    '''

    layer, node = len(whence)-1, 0 # set the layer and node index to the last node

    path = [] # initialize the path
    # loop to reach the starting layer
    while layer != 0:
        path.append((layer, node)) # append the current node

        node = whence[layer][node] # shift to the predecessor
        layer -= 1 # get one layer back

    path.append((0, 0)) # append the starting point
    path.reverse() # sort the nodes in the correct order

    cost = round(c[-1, -1], 2) # get the minimum cost
    return cost, path


def forward(g: np.ndarray):
    '''
    compute the minimum cost to get to each node in the given graph and store each node's predecessor

    parameters
    ----------
        g: matrix representing a uni-directed and weighted graph

    returns
    ----------
        c: cost matrix containing the minimum cost to get to each node
        whence: matrix containing the predecessor of each node
    '''

    # initialize the cost matrix
    c = [np.array([np.inf for _ in range(len(i))], dtype=float) for i in g]
    c.append(np.array([np.inf], dtype=float))
    c[0][0] = 0 # set the cost of the starting layer as a terminal value

    # initialize the predecessor matrix
    whence = [np.array([-1 for _ in range(len(i))], dtype=int) for i in g]
    whence.append(np.array([-1], dtype=int))

    # loop from the first layers
    for i in range(1, len(c)):

        # chose the shortest path to get to the current layer among all possible edges
        c_new = np.repeat(c[i-1].reshape(len(c[i-1]), 1), len(g[i-1][0]), axis=1) + g[i-1]
        c[i] = np.min(c_new, axis=0)

        whence[i] = np.argmin(c_new, axis=0) # keep track of the direction followed

    return c, whence


c, whence = forward(g_test) # compute the forward pass

path, cost = backward(
    c=c,
    whence=whence
) # compute the backward pass

print(g_test)
print(f'shortest path: {path}\nminimum cost: {cost}')
