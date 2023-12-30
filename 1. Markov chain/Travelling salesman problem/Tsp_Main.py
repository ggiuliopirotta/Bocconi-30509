from Tsp import *
from Tsp_Optimize import *


tsp = Tsp(n=100) # create the travelling salesman problem

greedy(
    tsp,
    iters=110,
    restarts=1,
    plot=True
) # optimize the problem with a greedy algorithm
