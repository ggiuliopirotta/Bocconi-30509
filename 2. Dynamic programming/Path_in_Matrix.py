import numpy as np


test = np.array([
    [1, 1, 1, 1],
    [2, 1, 2, 3],
    [0, 1, 2, 0],
    [0, 0, 0.5, 3],
    [2, 1, 1, 0]
    ], dtype=float
) # design the test for debugging


def backward(
        mat: np.ndarray,
        c: float,
        whence: np.ndarray
    ):
    '''
    back-track the shortest path and the relative minimum cost to get from the top-left to the bottom-right of a matrix

    parameters
    ----------
        mat: matrix to go through
        c: cost matrix containing the minimum cost to get to each cell
        whence: matrix containing the predecessor of each cell

    returns
    ----------
        path: shortest path to go through the matrix
        cost: minimum cost associated to the shortest path
    '''
    
    # set the row and column value to the bottom-right cell
    row, col = mat.shape
    row -= 1
    col -= 1

    path = [] # initialize the path
    # loop to reach the starting cell
    while row != 0 or col != 0:
        path.append((row, col)) # append the current cell

        idx = whence[row, col] # get the direction to reach the predecessor
        # shift to the predecessor accordingly
        if idx == 0:
            row -= 1
        else:
            col -= 1
    
    path.append((row, col)) # append the starting point
    path.reverse() # sort the cells in the correct order

    cost = round(c[-1, -1], 2) # get the minimum cost
    return path, cost


def forward(mat: np.ndarray):
    '''
    compute the shortest path to get to any cell in the given matrix moving only down or right
    
    parameters
    ----------
        mat: matrix to go through
    
    returns
        c: cost matrix containing the minimum cost to get to each cell
        whence: matrix containing the predecessor of each node
    '''

    n, m = mat.shape # get the shape of the matrix
 
    c, whence = np.full((n, m), np.inf, dtype=float), -np.ones((n, m), dtype=int) # initialize the cost and the predecessor matrix
    c[0], c[:, 0] = np.cumsum(mat[0]), np.cumsum(mat[:, 0]) # set the cost to get to the first row or left column as the cumulative sum

    # initialize the structures to keep track of the cost from the top
    c_top = np.zeros((n, m), dtype=float)
    c_top[1:, 0] = np.cumsum(mat[:n-1, 0]) # set the cost to get to the first column as the cumulative sum
    c_top[0] = np.inf # set the cost to get to the first row as terminal value

    # initialize the structure to keep track of the cost from the left
    c_left = np.zeros((n, m), dtype=float)
    c_left[:, 0] = np.inf # set the cost to get to the left column as terminal value
    c_left[0, 1:] = np.cumsum(mat[0, :m-1]) # set the cost to get to the first row as the cumulative sum

    # initialize the structure to keep track of the minimum cost
    c_min = np.zeros((n, m), dtype=float)
    c_min[0] = c_left[0]
    c_min[:, 0] = c_top[:, 0]

    # loop over diagonal slices until half of the matrix
    for i in range(1, min(n, m)):
        mask_row, mask_col = np.arange(i), np.arange(i-1, -1, -1) # create the arrays for the slicing
        
        # slice to get the correct previous frames
        c_top[1:i+1, 1:i+1][mask_row, mask_col] = c[:i, 1:i+1][mask_row, mask_col]
        c_left[1:i+1, 1:i+1][mask_row, mask_col] = c[1:i+1, :i][mask_row, mask_col]

        c_min[1:i+1, 1:i+1][mask_row, mask_col] = np.min(
            (c_top[1:i+1, 1:i+1][mask_row, mask_col], c_left[1:i+1, 1:i+1][mask_row, mask_col]),
            axis=0
        ) # get the minimum between the top and left value

        c[1:i+1, 1:i+1][mask_row, mask_col] = c_min[1:i+1, 1:i+1][mask_row, mask_col] + mat[1:i+1, 1:i+1][mask_row, mask_col] # paste the minimum value to the cost

    # loop over diagonal slices from half of the matrix
    for j in range(-min(n, m)+1, 0, 1):
        mask_row, mask_col = np.arange(-j-1, -1, -1), np.arange(-j) # create the arrays for the slicing

        # slice to get the correct previous frames
        c_top[-1:j-1:-1, -1:j-1:-1][mask_row, mask_col] = c[-2:j-2:-1, -1:j-1:-1][mask_row, mask_col]
        c_left[-1:j-1:-1, -1:j-1:-1][mask_row, mask_col] = c[-1:j-1:-1, -2:j-2:-1][mask_row, mask_col]

        c_min[-1:j-1:-1, -1:j-1:-1][mask_row, mask_col] = np.min(
            (c_top[-1:j-1:-1, -1:j-1:-1][mask_row, mask_col], c_left[-1:j-1:-1, -1:j-1:-1][mask_row, mask_col]),
            axis=0
        ) # get the minimum between the top and left value

        c[-1:j-1:-1, -1:j-1:-1][mask_row, mask_col] = c_min[-1:j-1:-1, -1:j-1:-1][mask_row, mask_col] + mat[-1:j-1:-1, -1:j-1:-1][mask_row, mask_col] # paste the minimum value to the cost

    # keep track of whether to go down or left
    whence[c_min == c_top] = 0
    whence[c_min == c_left] = 1

    return c, whence


# design a random matrix to solve
n, m = np.random.choice(np.arange(7, 10), 2, replace=True)
mat = np.around(5*np.random.random(size=(n, m)), 1)

c, whence = forward(mat) # compute the forward pass

path, cost = backward(
    mat=mat,
    c=c,
    whence=whence
) # compute the backward pass

print(mat)
print(f'shortest path: {path}\nminimum cost: {cost}')
