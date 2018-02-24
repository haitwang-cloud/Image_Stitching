import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pyswarms as ps
data=load_iris()
X=data.data
y=data.target