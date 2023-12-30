import numpy as np


# design the tests for debugging
w1 = np.array([
    [0.0, -9.0, 3.0, 1.0],
    [-0.5, 0.0, -0.8, 0.1],
    [0.1, 3.3, 0.0, 2.2],
    [np.inf, np.inf, np.inf, 0.0]
])

w2 = np.array([
    [0.0, 20.0, 10.0, 63.0, 72.0, np.inf],
    [np.inf, 0.0, 0.0, 40.0, np.inf, 70.0],
    [np.inf, 5.0, 0.0, 40.0, 34.0, 100.0],
    [np.inf, np.inf, -20.0, 0.0, -5.0, 36.0],
    [np.inf, -31.0, np.inf, 5.0, 0.0, 80.0],
    [np.inf, np.inf, np.inf, np.inf, np.inf, 0.0]
])


class Floyd_Warshall:
    '''
    apply the **Floyd Warshall algorithm**

    the algorithm computes the shortest path and the relative minimum cost in a directed, acycled and weighted graph

    parameters
    ----------
        w: matrix representing a directed, acycled and weighted graph
        c: cost to get from each node to another
        whence: matrix containing the predecessor of each node

    methods
    ----------
        backward: back-track the shortest path to get from a node to another
        forward: compute the minimum cost to get to each node
    '''

    def __init__(
            self,
            w: np.ndarray
        ):

        self.w = w # store the graph matrix

        # initialize the variables and the structures for the algorithm
        self.c = None
        self.whence = None

    def backward(
            self,
            i: int,
            j: int
        ):
        '''
        back-track the shortest path and the relative minimum cost to get from a node to another

        parameters
        ----------
            i: starting node
            j: ending node

        returns
        ----------
            path: shotest path to get from a node to another
            c: minimum cost to get from a node to another
        '''
        
        cost, whence = self.c, self.whence

        idx = whence[i, j] # get the predecessor of the node

        # check that the node has a predecessor and recover the full path eventually
        if idx != -1:
            path = [j]
            
            # loop to reach the starting node
            while idx != i:
                path.append(idx) # append the current node
                idx = whence[i, idx] # shift to the predecessor

            path.append(i) # add the starting node
            path.reverse() # sort the nodes in the correct order

        # set an empty path otherwise
        else:
            path = []
        
        c = cost[i, j] # get the minimum cost
        
        return path, c

    def forward(self):
        '''
        compute the minimum cost to get to each node and store each node's predecessor
        '''

        w = self.w
    
        n = w.shape[0]
        nn = np.arange(n)

        # check that the graph is acycled and print an error message otherwise
        if not np.array_equal(w[nn, nn], np.zeros(n)):
            raise Exception('the graph provided is not acycled')

        self.c = w.copy() # initialize a matrix to keep track of the cumulative cost

        # set up the matrix for the predecessors
        self.whence = np.arange(n).reshape((n, 1)) + np.zeros(n, dtype=int)
        no_pred = self.c == np.inf
        self.whence[no_pred] = -1 # set a sentinel value for the nodes without a predecessor

        # loop over all the nodes
        for i in range(n):
            c_new = self.c[:, i].reshape((n, 1)) + self.c[i, :] # sum the cost of connecting the current node with each of the other
            acc = c_new < self.c
            self.c[acc] = c_new[acc] # accept only the new connections with a lower cost

            # set the current node as the predecessor for the accepted nodes
            pred_new = np.repeat(self.whence[i, :].reshape(1, n), n, axis=0)
            self.whence[acc] = pred_new[acc]

            # check that there are no loops and print an error message otherwise
            if (self.c[nn, nn] < 0).any():
                err = int(np.argmin(self.c[nn, nn])) # get the error node index
                cycle = self.backward(err, err)
                raise Exception (f'cycle detected: {cycle}')


# solve a graph with the Floyd Warshall algorithm
fw = Floyd_Warshall(w2)
fw.forward()
path, c = fw.backward(0, 5)

print(f'optimal path: {path}')
print(f'minimum cost: {c}')
