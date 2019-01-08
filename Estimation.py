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

#Function to estimate the parameters of M via the coding likelihood function
#Inputs: tauM - parameters for M, S_max - listing of node IDs in maximal independent set,
#        nodeList - dictionary representation of Units in graph
#Output: likelihood value associated with tauM
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

#Function to estimate the parameters of M via the pseudo likelihood function
#Inputs: tauM - parameters for M, S_max - listing of node IDs in maximal independent set,
#        nodeList - dictionary representation of Units in graph
#Output: likelihood value associated with tauM
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

#Function for sampling from the joint distribution p(M1, M2, ...)
#Inputs: numIter - number of iterations to run the sampler, M - a dummy initialization matrix,
#        tauM - parameters for M, nodeList - dictionary representation of the nodes in the graph (elements are Units; see Unit.py)
#Output: M - a single sample from the joint p(M1, M2, ...)
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

#Function to perform an intervention in the network
#Inputs: nodeList - dictionary representation of Units in graph, aVal - interventional value to set for A for ALL units in network
#Output: nodeListIntervention - modified nodeList post-intervention (but not accounting for downstream effects)
def doIntervention(nodeList, aVal):
    nodeListIntervention = copy.deepcopy(nodeList)
    for node in nodeListIntervention:
        nodeListIntervention[node].A = aVal
    return nodeListIntervention

#Function to calculate causal effect estimates associated with interventions on A's in the network
#Inputs: nodeList - dictionary representation of Units in graph, S_max - listing of node IDs in maximal independent set
def getEstimates(nodeList, S_max):
    #create initialized matrices for units' variables
    As = np.zeros(len(nodeList))
    nborAs = np.zeros((len(nodeList), len(nodeList[0].adj)))
    Cs = np.zeros((len(nodeList), 3))
    Ms = np.zeros(len(nodeList))
    Ys = np.zeros(len(nodeList))

    #Load data into nice, neat arrays
    for node in nodeList:
        As[node] = nodeList[node].A
        idx = 0
        for nbor in nodeList[node].adj:
            nborAs[node, idx] = nodeList[nbor].A
            idx += 1
        Cs[node] = nodeList[node].C
        Ms[node] = nodeList[node].M
        Ys[node] = nodeList[node].Y

    #Fit P(A|C) model
    clfA = LogisticRegression()
    clfA.fit(Cs, As)

    #Fit P(Y|C, A, M) model
    cov = np.concatenate((Cs, np.reshape(As, (As.shape[0], 1)), nborAs, np.reshape(Ms, (Ms.shape[0], 1))), axis=1)
    clfY = LogisticRegression()
    clfY.fit(cov, Ys)
    
    #Fit tauM via coding and pseudo likelihood estimation
    tauM = np.random.rand(7)
    codingM = minimize(codingLikelihoodM, tauM, (S_max, nodeList)).x
    codingM = {'intcp': codingM[0], 'C0': codingM[1], 'C1': codingM[2], 'C2': codingM[3], 'A': codingM[4],
               'nborA': codingM[5], 'nborM': codingM[6]}
    pseudoM = minimize(pseudoLikelihoodM, tauM, (nodeList)).x
    pseudoM = {'intcp': pseudoM[0], 'C0': pseudoM[1], 'C1': pseudoM[2], 'C2': pseudoM[3], 'A': pseudoM[4],
               'nborA': pseudoM[5], 'nborM': pseudoM[6]}
    
    #[p(0|C), p(1|C)]
    A1_NetEffect = doIntervention(nodeList, 1)
    A0_NetEffect = doIntervention(nodeList, 0)

    #Run Gibbs to obtain M's for each intervention
    initMat1 = np.random.binomial(1, .5, (1051, len(nodeList)))
    initMat0 = np.random.binomial(1, .5, (1051, len(nodeList)))
    NetEffectCoding1 = Gibbs(1050, initMat1, codingM, A1_NetEffect)
    NetEffectCoding0 = Gibbs(1050, initMat0, codingM, A0_NetEffect)

    NetEffectCoding1 = NetEffectCoding1[1001:]
    NetEffectCoding1 = NetEffectCoding1[::10,]
    NetEffectCoding0 = NetEffectCoding0[1001:]
    NetEffectCoding0 = NetEffectCoding0[::10,]
    
    initMat1 = np.random.binomial(1, .5, (1051, len(nodeList)))
    initMat0 = np.random.binomial(1, .5, (1051, len(nodeList)))
    NetEffectPseudo1 = Gibbs(1050, initMat1, pseudoM, A1_NetEffect)
    NetEffectPseudo0 = Gibbs(1050, initMat0, pseudoM, A0_NetEffect)

    NetEffectPseudo1 = NetEffectPseudo1[1001:]
    NetEffectPseudo1 = NetEffectPseudo1[::10,]
    NetEffectPseudo0 = NetEffectPseudo0[1001:]
    NetEffectPseudo0 = NetEffectPseudo0[::10,]
    
    A_i_vals = [0,1]
    Y_cod = 0
    Y_pse = 0
    
    #Average over all nodes in graph
    for i in range(len(nodeList)):
        #Predict A given C
        Acov = np.array([nodeList[i].C[0], nodeList[i].C[1], nodeList[i].C[2]])
        Ahat = clfA.predict_proba(Acov.reshape(1,-1))
        Yi_cod = 0
        Yi_pse = 0
        
        #Predict Y 5 times for each intervention for coding likelihood to account for randomness in Gibbs process
        for j in range(5):
            for k in range(len(A_i_vals)):
                Y1_cod_cov = np.array([nodeList[i].C[0], nodeList[i].C[1], nodeList[i].C[2], A_i_vals[k], 1, 1, 1, NetEffectCoding1[j, i]])
                Y1_cod_hat = clfY.predict_proba(Y1_cod_cov.reshape(1,-1))
                Y0_cod_cov = np.array([nodeList[i].C[0], nodeList[i].C[1], nodeList[i].C[2], A_i_vals[k], 0, 0, 0, NetEffectCoding0[j, i]])
                Y0_cod_hat = clfY.predict_proba(Y0_cod_cov.reshape(1,-1))
                Yi_cod += (Y1_cod_hat[0][1] * Ahat[0][k] - Y0_cod_hat[0][1] * Ahat[0][k])
        #Average over the 5 Gibbs samples for Ms
        Y_cod += Yi_cod / 5

        #Predict Y 5 times for each intervention for pseudo likelihood to account for randomness in Gibbs process
        for j in range(5):
            for k in range(len(A_i_vals)):
                Y1_pse_cov = np.array([nodeList[i].C[0], nodeList[i].C[1], nodeList[i].C[2], A_i_vals[k], 1, 1, 1, NetEffectPseudo1[j, i]])
                Y1_pse_hat = clfY.predict_proba(Y1_pse_cov.reshape(1,-1))
                Y0_pse_cov = np.array([nodeList[i].C[0], nodeList[i].C[1], nodeList[i].C[2], A_i_vals[k], 0, 0, 0, NetEffectPseudo0[j, i]])
                Y0_pse_hat = clfY.predict_proba(Y0_pse_cov.reshape(1,-1))
                Yi_pse += (Y1_pse_hat[0][1] * Ahat[0][k] - Y0_pse_hat[0][1] * Ahat[0][k])
        #Average over the 5 Gibbs samples for Ms
        Y_pse += Yi_pse / 5

    #Average over all nodes in network
    Y_cod = Y_cod / len(nodeList)
    Y_pse = Y_pse / len(nodeList)
    return Y_cod, Y_pse
