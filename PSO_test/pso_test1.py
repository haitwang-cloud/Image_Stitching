#Optimizing a sphere function
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx

#set-up hyperparameters
options={'c1':0.5,'c2':0.3,'w':0.9}

#call instance of PSO
optimizer=ps.single.GlobalBestPSO(n_particles=10,dimensions=2,options=options)

#perform optimization
best_cost,best_pos=optimizer.optimize(fx.sphere_func,iters=1000,verbose=3,print_step=25)
print(best_pos)