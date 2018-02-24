# PySwarms implements a grid search and random search technique to find the best parameters
# for your optimizer. Setting them up is easy. In this example,
# let's try using pyswarms.utils.search.RandomSearch to find ' \
#    'the optimal parameters for LocalBestPSO optimizer.
#
# Here, we input a range, enclosed in tuples, to
# define the space in which the parameters will be found.
# Thus, (1,5) pertains to a range from 1 to 5.

import numpy as np
import pyswarms as ps
from pyswarms.utils.search import RandomSearch
from pyswarms.utils.functions import single_obj as fx

#set-up choices for the parameters
options={
    'c1':(1,5),
    'c2':(6,10),
    'w':(2,5),
    'k':(11,15),
    'p':1
}

#create a randomSearch oject
# n_selection_iters is the number of iterations to run the searcher
# iters is the number of iterations to run the optimizer
g=RandomSearch(ps.single.LocalBestPSO,n_particles=40,dimensions=20,options=options,
               objective_func=fx.sphere_func,iters=100,n_selection_iters=10)

best_score,best_options=g.search()

print (best_score,best_options['c1'])