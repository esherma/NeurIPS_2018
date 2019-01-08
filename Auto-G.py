from networkx.generators.random_graphs import random_regular_graph
import networkx as nx
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from Unit import unit
import pickle
import copy
from DataGen import generateData
from Estimation import getEstimates
import multiprocessing
from multiprocessing import Pool

nodeList = {}

#Parameters for the experiments for each variable
Cscale = [[1.5, 3], [6, 2], [.8, .8]]
Uscale = [[2.3, 1.1], [.9, 1.1], [2, 2]]
tauA = {'intcp': -1, 'C0': .5, 'C1': .2, 'C2': .25, 'U0': .3, 'U1': -.2, 'U2': .25}
tauM = {'intcp': -1, 'C0': -.3, 'C1': .4, 'C2': .1, 'A': 1, 'nborA': -.5, 'nborM': -1.5} #nborM 1.5
tauY = {'intcp': -.3, 'nborA' : -1, 'M': 3, 'C0': -.2, 'C1': .2, 'C2': -.05, 'U0': .1, 
        'U1': -.2, 'U2': .25}

#Load pre-created graph
with open('./400_3/graph400_3.pkl', 'rb') as fname:
    graph = pickle.load(fname)
    
    
#Calculate maximally independent set of Units in graph
Xi = []
lens = []
for i in range(50):
    Xi.append(nx.maximal_independent_set(graph))
    lens.append(len(Xi[-1]))
S_max = Xi[np.argmax(lens)]

#Calculate causal effects from graph
def doInference(i):
    print('starting', i)
    nodeList = generateData(graph)
    return getEstimates(nodeList, S_max)

#Use multiprocessing to perform (N) bootstrap samples (make it fast!)
with Pool(multiprocessing.cpu_count()) as pool:
    Effects = pool.map(doInference, np.arange(1000))

#Dump results to disk
with open('./400_3/Effects400_3.pkl', 'wb') as fname:
    pickle.dump(Effects, fname)
    

