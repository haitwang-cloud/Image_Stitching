import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.environments import PlotEnvironment
import matplotlib.pyplot as plt

#set-up optimizer
options={'c1':0.5,'c2':0.3,'w':0.9}
optimizer=ps.single.GlobalBestPSO(n_particles=10,dimensions=3,options=options)

#Initalize the plot environment
plt_env=PlotEnvironment(optimizer,fx.sphere_func,1000)

#plot the cost
plt_env.plot_cost(figsize=(8,6))
plt.show()