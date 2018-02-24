import sys
sys.path.append('../')
import numpy as np
import seaborn as sns
import pandas as pd

#import pyswarms
import pyswarms as ps
from sklearn.datasets import make_classification
from sklearn import linear_model

X,y=make_classification(n_samples=100,n_features=15,n_classes=3,
                        n_informative=4,n_redundant=1,n_repeated=2,
                        random_state=2018)
#plot toy dataset per feature
df=pd.DataFrame(X)
df['labels']=pd.Series(y)

#create an classifier
classifier=linear_model.LinearRegression()

#define objective function
def f_per_particle(m,alpha):

    total_feature=15
    #get the subset of feature from binary mask
    if np.count_nonzero(m)==0:
        X_subset=X
    else:
        X_subset=X[:,m==1]
    #perform classification and store performance in P
    classifier.fit(X_subset,y)
    P=(classifier.predict(X_subset)==y).mean()
    # compute for objective function
    j=(alpha*(1.0-P)+(1.0-alpha)*(1-(X_subset.shape[1]/total_feature)))

    return j
def f(x,alpha=0.613):

    n_particles=x.shape[0]
    j=[f_per_particle(x[i],alpha) for i in range(n_particles)]
    return np.array(j)
#Initialize swarm,arbitrary
options={'c1':0.5,'c2':0.5,'w':0.9,'k':30,'p':2}

#call instance of PSO
dimensions=15
optimizer=ps.discrete.BinaryPSO(n_particles=50,dimensions=dimensions,options=options)
#Perform optimization
cost,pos=optimizer.optimize(f,print_step=100,iters=1000,verbose=2)
print(pos)


