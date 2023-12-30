import numpy as np


# design the test for debugging
img_test = np.array([
    [1.0, 2.0, 1.0, 1.0],
    [2.0, 0.0, 0.0, 3.0],
    [1.0, 1.0, 2.0, 0.0],
    [2.0, 2.0, 0.5, 1.0],
    [2.0, 1.0, 1.0, 0.0]
])

grad_test = np.array([
    [1.0, 1.0 , 0.5, 0.0],
    [2.0, 1.0 , 1.5, 3.0],
    [0.0, 0.5 , 1.5, 2.0],
    [0.0, 0.75, 1.0, 0.5],
    [1.0, 0.5 , 0.5, 1.0]
])

seam_test = np.array([2, 1, 0, 0, 1])

img_red_test = np.array([
    [1.0, 2.0, 1.0],
    [2.0, 0.0, 3.0],
    [1.0, 2.0, 0.0],
    [2.0, 0.5, 1.0],
    [2.0, 1.0, 0.0]
])


def test_fun(
        fun,
        obj_pre: np.ndarray,
        obj_post_test: np.ndarray
    ):
    '''
    test that the given function performs a specific part of the seam carving algorithm in the correct manner

    parameters
    ----------
        fun: seam carving function to test
        obj_pre: object to process
        obj_post_test: correct result after the processing
    '''

    res = fun(obj_pre) # process the given matrix
    print(f'test:\n{obj_post_test}\nres:\n{res}')

    s = '' # initialize the message
    # check that the algorithm has been performed correctly and print a message according to the case
    if np.array_equal(res, obj_post_test):
        s = 'well done'
    else:
        s = 'not the right result'

    print(s)


def compute_grad(img: np.ndarray):
    '''
    compute the gradient according to a specific rule on the given image

    the gradient is calculated as the average of the absolute difference
    between the cell itself and the two horizontally adjacent cells

    parameters
    ----------
        img: image to compute the gradient

    returns
    ----------
        grad: gradient computed on the image
    '''

    n, m = img.shape
    
    grad = np.zeros((n, m)) # initialize the gradient
    # apply the gradient rule
    grad[:, 0] = np.abs(img[:, 0]-img[:, 1]) # replace the left column with the one to its right
    grad[:, 1:-1] = (np.abs(img[:, 1:-1]-img[:, 0:-2]) + np.abs(img[:, 1:-1]-img[:, 2:]))/2
    grad[:, -1] = np.abs(img[:, -1]-img[:, -2]) # replace the right column with the one to its left

    return grad


def get_seam(grad: np.ndarray):
    '''
    compute the seam on the given gradient of an image

    the seam is calculated as the minimum cost path to descend the gradient vertically

    parameters
    ----------
        grad: previously calculated gradient of an image

    returns
    ----------
        seam: seam computed on the gradient 
    '''

    n, m = grad.shape

    c = np.zeros((n, m)) # initialize the cost
    c[0] = grad[0] # set the first row of the seam to be equal to the gradient

    c_left, c_top, c_right = np.zeros((n, m)), np.zeros((n, m)), np.zeros((n, m)) # initialize the structures to keep track of the cost
    # initialize the structures to keep track of the minimum cost and the path associated
    c_min = np.zeros((n, m))
    whence = np.zeros((n, m))

    # set the cost to get to first row as a terminal value
    c_left[0], c_top[0], c_right[0] = np.inf, np.inf, np.inf
    c_min[0], whence[0] = np.inf, np.inf

    # loop from the first row onward
    for i in range(1, n):

        # compute the cost of moving down-left
        c_left[i, 0] = np.inf # set the cost to get to the left column as a terminal value 
        c_left[i, 1:] = c[i-1, :-1]

        c_top[i, :] = c[i-1, :] # compute the cost of moving down

        # compute the cost of moving down-right
        c_right[i, :-1] = c[i-1, 1:]
        c_right[i, -1] = np.inf # set the cost to get to the right column as a terminal value

        # find the minimum cost choosing among the three possible directions
        c_min[i] = np.min((c_left[i], c_top[i], c_right[i]), axis=0)
        c[i] = c_min[i]+grad[i] # update the cost

        # keep track of the movements done to get to each cell
        whence[i, c_min[i] == c_left[i]] = -1
        whence[i, c_min[i] == c_top[i]] = 0
        whence[i, c_min[i] == c_right[i]] = 1

    seam = np.zeros(n, dtype=int) # initialize the seam

    js = np.argmin(c[-1, :]) # set the minimum cost in the last row of the cost matrix as a starting point
    # loop over the rows in reversed order for the backward pass
    for i in reversed(range(n)):
        seam[i] = js
        js += whence[i, int(js)] # back-track the movement to the previous row

    return seam


def carve(
        img: np.ndarray,
        seam: np.ndarray
    ):
    '''
    remove the given seam from an image to carve it
    
    parameters
    ----------
        img: image to carve
        seam: less important seam to remove from the image
    
    returns
    ----------
        img_red: image reduced with the seam carving algorithm
    '''    

    n, m = img.shape

    img_red = np.zeros((n, m-1)) # initialize the reduced image
    # loop over the rows
    for i in range(n):
        js = seam[i] # get the position of the seam at the current row

        # remove the seam from the image
        img_red[i, :js] = img[i, :js]
        img_red[i, js:] = img[i, js+1:]

    return img_red


def get_seamcarve(img: np.ndarray):
    '''
    put all the functions together to apply the **seam carving algorithm** on the given image

    the algorithm reduces the size of an image discarding the vector
    containing the less meaningful pieces of information for the image itself

    parameters
    ----------
        img: image to reduce
    
    returns
    ----------
        img_red: image reduced with the seam carving algorithm
    '''

    grad = compute_grad(img) # compute the gradient
    seam = get_seam(grad) # get the seam
    img_red = carve(img, seam) # carve the image

    return img_red


test_fun(
    fun=get_seamcarve,
    obj_pre=img_test,
    obj_post_test=img_red_test
)
