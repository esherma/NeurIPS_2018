
# coding: utf-8

# In[1]:


from networkx.generators.random_graphs import random_regular_graph
import networkx as nx
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from Unit import unit
from sklearn.linear_model import LogisticRegression
import pickle
import copy
from DataGen import generateData


# In[2]:


def codingLikelihoodM(tauM, S_max, nodeList):
    total = 0
    for i in S_max:
        sumM = tauM[0] + (tauM[1] * nodeList[i].C[0]) + (tauM[2] * nodeList[i].C[1])
        sumM += (tauM[3] * nodeList[i].C[2]) + (tauM[4] * nodeList[i].A)

        for nbor in nodeList[i].adj:
            sumM += (tauM[5] * nodeList[nbor].A) + (tauM[6] * nodeList[nbor].M)

        if nodeList[i].M == 1:
            total += np.log(expit(sumM))
        else:
            total += np.log(1 - expit(sumM))

    return (-1 * total)

def pseudoLikelihoodM(tauM, nodeList):
    total = 0
    for i in nodeList:
        sumM = tauM[0] + (tauM[1] * nodeList[i].C[0]) + (tauM[2] * nodeList[i].C[1])
        sumM += (tauM[3] * nodeList[i].C[2]) + (tauM[4] * nodeList[i].A)

        for nbor in nodeList[i].adj:
            sumM += (tauM[5] * nodeList[nbor].A) + (tauM[6] * nodeList[nbor].M)

        if nodeList[i].M == 1:
            total += np.log(expit(sumM))
        else:
            total += np.log(1 - expit(sumM))

    return (-1 * total)


# In[3]:


def Gibbs(numIter, M, tauM, nodeList):    
        for k in range(1, numIter+1):
            for i in range(len(nodeList)):
                sumM = tauM['intcp'] + (tauM['C0'] * nodeList[i].C[0]) + (tauM['C1'] * nodeList[i].C[1])
                sumM += (tauM['C2'] * nodeList[i].C[2])
                sumM += (tauM['A'] * nodeList[i].A)

                
                for nbor in nodeList[i].adj:
                    sumM += (tauM['nborA'] * nodeList[nbor].A) + (tauM['nborM'] * M[k-1, nbor])

                Mi = np.random.binomial(1, expit(sumM))

                M[k,i] = Mi

        return M


# In[4]:


def doIntervention(nodeList, aVal):
    nodeListIntervention = copy.deepcopy(nodeList)
    for node in nodeListIntervention:
        nodeListIntervention[node].A = aVal
    return nodeListIntervention


# In[5]:


def getEstimates(nodeList, S_max):
    As = np.zeros(len(nodeList))
    nborAs = np.zeros((len(nodeList), len(nodeList[0].adj)))
    Cs = np.zeros((len(nodeList), 3))
    Ms = np.zeros(len(nodeList))
    Ys = np.zeros(len(nodeList))

    for node in nodeList:
        As[node] = nodeList[node].A
        idx = 0
        for nbor in nodeList[node].adj:
            nborAs[node, idx] = nodeList[nbor].A
            idx += 1
        Cs[node] = nodeList[node].C
        Ms[node] = nodeList[node].M
        Ys[node] = nodeList[node].Y

    clfA = LogisticRegression()
    clfA.fit(Cs, As)

    cov = np.concatenate((Cs, np.reshape(As, (As.shape[0], 1)), nborAs, np.reshape(Ms, (Ms.shape[0], 1))), axis=1)
    clfY = LogisticRegression()
    clfY.fit(cov, Ys)
    
    tauM = np.random.rand(7)
    codingM = minimize(codingLikelihoodM, tauM, (S_max, nodeList)).x
    codingM = {'intcp': codingM[0], 'C0': codingM[1], 'C1': codingM[2], 'C2': codingM[3], 'A': codingM[4],
               'nborA': codingM[5], 'nborM': codingM[6]}
    pseudoM = minimize(pseudoLikelihoodM, tauM, (nodeList)).x
    pseudoM = {'intcp': pseudoM[0], 'C0': pseudoM[1], 'C1': pseudoM[2], 'C2': pseudoM[3], 'A': pseudoM[4],
               'nborA': pseudoM[5], 'nborM': pseudoM[6]}
    
    A_i_vals = [0,1]
    Y_DE_cod = 0
    Y_SE_cod = 0
    Y_DE_pse = 0
    Y_SE_pse = 0
    
    
    for i in range(len(nodeList)):
        if (i % 100 == 0):
            print(i)
        Acov = np.array([nodeList[i].C[0], nodeList[i].C[1], nodeList[i].C[2]])
        Ahat = clfA.predict_proba(Acov.reshape(1,-1))
        #[p(0|C), p(1|C)]
        
        A1_Direct = doIntervention(nodeList, 0)
        A0_Direct = doIntervention(nodeList, 0)
        A1_Direct[i].A = 1
#         A0_Direct[i].A = 0 (redundant)
        
        A1_Spillover = doIntervention(nodeList, 1)
        A0_Spillover = doIntervention(nodeList, 0)
#         A1_Spillover[i].A = 1 (redundant)
        A0_Spillover[i].A = 1
            
        initMatDE1 = np.random.binomial(1, .5, (1051, len(nodeList)))
        initMatDE0 = np.random.binomial(1, .5, (1051, len(nodeList)))
        initMatSE1 = np.random.binomial(1, .5, (1051, len(nodeList)))
        initMatSE0 = np.random.binomial(1, .5, (1051, len(nodeList)))
        DirectCoding1 = Gibbs(1050, initMatDE1, codingM, A1_Direct)
        DirectCoding0 = Gibbs(1050, initMatDE0, codingM, A0_Direct)
        SpilloverCoding1 = Gibbs(1050, initMatSE1, codingM, A1_Spillover)
        SpilloverCoding0 = Gibbs(1050, initMatSE0, codingM, A0_Spillover)
        
        DirectCoding1 = DirectCoding1[1001:]
        DirectCoding1 = DirectCoding1[::10,]
        DirectCoding0 = DirectCoding0[1001:]
        DirectCoding0 = DirectCoding0[::10,]
        SpilloverCoding1 = SpilloverCoding1[1001:]
        SpilloverCoding1 = SpilloverCoding1[::10,]
        SpilloverCoding0 = SpilloverCoding0[1001:]
        SpilloverCoding0 = SpilloverCoding0[::10,]
        
        for j in range(5):
            for k in range(len(A_i_vals)):
                Y1covDE = np.array([nodeList[i].C[0], nodeList[i].C[1], nodeList[i].C[2], A_i_vals[k], 0, 0, 0, DirectCoding1[j, i]])
                Y1hatDE = clfY.predict_proba(Y1covDE.reshape(1,-1))
                Y0covDE = np.array([nodeList[i].C[0], nodeList[i].C[1], nodeList[i].C[2], A_i_vals[k], 0, 0, 0, DirectCoding0[j, i]])
                Y0hatDE = clfY.predict_proba(Y0covDE.reshape(1,-1))
                Y_DE_cod += (Y1hatDE[0][1] * Ahat[0][k] - Y0hatDE[0][1] * Ahat[0][k])
                
                Y1covSE = np.array([nodeList[i].C[0], nodeList[i].C[1], nodeList[i].C[2], A_i_vals[k], 1, 1, 1, SpilloverCoding1[j, i]])
                Y1hatSE = clfY.predict_proba(Y1covSE.reshape(1,-1))
                Y0covSE = np.array([nodeList[i].C[0], nodeList[i].C[1], nodeList[i].C[2], A_i_vals[k], 0, 0, 0, SpilloverCoding0[j, i]])
                Y0hatSE = clfY.predict_proba(Y0covSE.reshape(1,-1))
                Y_SE_cod += (Y1hatSE[0][1] * Ahat[0][k] - Y0hatSE[0][1] * Ahat[0][k])
                
        initMatDE1 = np.random.binomial(1, .5, (1051, len(nodeList)))
        initMatDE0 = np.random.binomial(1, .5, (1051, len(nodeList)))
        initMatSE1 = np.random.binomial(1, .5, (1051, len(nodeList)))
        initMatSE0 = np.random.binomial(1, .5, (1051, len(nodeList)))
        DirectPseudo1 = Gibbs(1050, initMatDE1, pseudoM, A1_Direct)
        DirectPseudo0 = Gibbs(1050, initMatDE0, pseudoM, A0_Direct)
        SpilloverPseudo1 = Gibbs(1050, initMatSE1, pseudoM, A1_Spillover)
        SpilloverPseudo0 = Gibbs(1050, initMatSE0, pseudoM, A0_Spillover)
        
        DirectPseudo1 = DirectPseudo1[1001:]
        DirectPseudo1 = DirectPseudo1[::10,]
        DirectPseudo0 = DirectPseudo0[1001:]
        DirectPseudo0 = DirectPseudo0[::10,]
        SpilloverPseudo1 = SpilloverPseudo1[1001:]
        SpilloverPseudo1 = SpilloverPseudo1[::10,]
        SpilloverPseudo0 = SpilloverPseudo0[1001:]
        SpilloverPseudo0 = SpilloverPseudo0[::10,]
        
        for j in range(5):
            for k in range(len(A_i_vals)):
                Y1covDE = np.array([nodeList[i].C[0], nodeList[i].C[1], nodeList[i].C[2], A_i_vals[k], 0, 0, 0, DirectPseudo1[j, i]])
                Y1hatDE = clfY.predict_proba(Y1covDE.reshape(1,-1))
                Y0covDE = np.array([nodeList[i].C[0], nodeList[i].C[1], nodeList[i].C[2], A_i_vals[k], 0, 0, 0, DirectPseudo0[j, i]])
                Y0hatDE = clfY.predict_proba(Y0covDE.reshape(1,-1))
                Y_DE_pse += (Y1hatDE[0][1] * Ahat[0][k] - Y0hatDE[0][1] * Ahat[0][k])
                
                Y1covSE = np.array([nodeList[i].C[0], nodeList[i].C[1], nodeList[i].C[2], A_i_vals[k], 1, 1, 1, SpilloverPseudo1[j, i]])
                Y1hatSE = clfY.predict_proba(Y1covSE.reshape(1,-1))
                Y0covSE = np.array([nodeList[i].C[0], nodeList[i].C[1], nodeList[i].C[2], A_i_vals[k], 0, 0, 0, SpilloverPseudo0[j, i]])
                Y0hatSE = clfY.predict_proba(Y0covSE.reshape(1,-1))
                Y_SE_pse += (Y1hatSE[0][1] * Ahat[0][k] - Y0hatSE[0][1] * Ahat[0][k])
    Y_DE_cod = Y_DE_cod / (5 * len(nodeList))
    Y_SE_cod = Y_SE_cod / (5 * len(nodeList))
    Y_DE_pse = Y_DE_pse / (5 * len(nodeList))
    Y_SE_pse = Y_SE_pse / (5 * len(nodeList))
    return Y_DE_cod, Y_SE_cod, Y_DE_pse, Y_SE_pse