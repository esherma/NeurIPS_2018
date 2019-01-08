from networkx.generators.random_graphs import random_regular_graph
import networkx as nx
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from Unit import unit
import pickle
import copy

def Gibbs(numIter, M, tauM, nodeList):
    for k in range(1, numIter+1):
        for i in range(len(nodeList)):
            sumM = tauM['intcp'] + (tauM['C0'] * nodeList[i].C[0]) + (tauM['C1'] * nodeList[i].C[1])
            sumM += (tauM['C2'] * nodeList[i].C[2]) + (tauM['A'] * nodeList[i].A)

            for nbor in nodeList[i].adj:
                sumM += (tauM['nborA'] * nodeList[nbor].A) + (tauM['nborM'] * M[k-1, nbor])

            Mi = np.random.binomial(1, expit(sumM))
            
            M[k,i] = Mi
        
    return M

def generateData(graph):
    nodeList = {}
    
    Cscale = [[1.5, 3], [6, 2], [.8, .8]]
    Uscale = [[2.3, 1.1], [.9, 1.1], [2, 2]]
    tauA = {'intcp': -1, 'C0': .5, 'C1': .2, 'C2': .25, 'U0': .3, 'U1': -.2, 'U2': .25}
    tauM = {'intcp': -1, 'C0': -.3, 'C1': .4, 'C2': .1, 'A': 1, 'nborA': -.5, 'nborM': -1.5} #nborM 1.5
    tauY = {'intcp': -.3, 'nborA' : -1, 'M': 3, 'C0': -.2, 'C1': .2, 'C2': -.05, 'U0': .1, 
            'U1': -.2, 'U2': .25}
    
    for node in graph.adj:
        nodeList[node] = unit(np.zeros(3), np.zeros(3), graph.adj[node])
        for i in range(len(Cscale)):
            nodeList[node].C[i] = np.random.beta(Cscale[i][0], Cscale[i][1])
            nodeList[node].U[i] = np.random.beta(Uscale[i][0], Uscale[i][1])
        sumA = tauA['intcp'] + (tauA['C0'] * nodeList[node].C[0]) + (tauA['C1'] * nodeList[node].C[1])
        sumA += (tauA['C2'] * nodeList[node].C[2]) + (tauA['U0'] * nodeList[node].U[0]) + (tauA['U1'] * nodeList[node].U[1])
        sumA += (tauA['U2'] * nodeList[node].U[2])
        nodeList[node].A = np.random.binomial(1, expit(sumA))
    
    M = np.random.binomial(1, .5, (1001, len(nodeList))) #Markov chain for M's
    M = Gibbs(1000, M, tauM, nodeList)
    M = M[1000]
    for node in nodeList:
        nodeList[node].M = M[node]
        sumY = tauY['intcp'] + (tauY['M'] * nodeList[node].M) + (tauY['C0'] * nodeList[node].C[0])
        sumY += (tauY['C1'] * nodeList[node].C[1]) + (tauY['C2'] * nodeList[node].C[2]) + (tauY['U0'] * nodeList[node].U[0])
        sumY += (tauY['U1'] * nodeList[node].U[1]) + (tauY['U2'] * nodeList[node].U[2])
        for nbor in nodeList[node].adj:
            sumY += (tauY['nborA'] * nodeList[nbor].A)
        nodeList[node].Y = np.random.binomial(1, expit(sumY))
        
    return nodeList
