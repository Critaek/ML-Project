import scipy.stats as st
import numpy

def Normalization(D):
    N = D.shape[1]

    ranks = []
    for i in range(D.shape[0]): #per ogni feature
        row = []
        for j in range(D.shape[1]): #per ogni sample
            r = rank(D[i,j], D, N, i)
            value = st.norm.ppf(r,0,1)
            row.append(value)
        ranks.append(row)
    
    mat = numpy.vstack(ranks)

    return mat


def NormDTE(DTR, DTE):
    #Normalizing DTE with ranking computed from DTR, for some reasons, it doesn't work, returns
    #very bad DCF on every model
    N = DTR.shape[1]

    ranks = []
    for i in range(DTR.shape[0]): #per ogni feature
        row = []
        for j in range(DTE.shape[1]): #per ogni sample
            r = rank(DTE[i,j], DTR, N, i)
            value = st.norm.ppf(r,0,1)
            row.append(value)
        ranks.append(row)
    
    mat = numpy.vstack(ranks)

    return mat

def rank(x, D, N, i): #Sample rank
    accum = 0
    for j in D[i, :]:
        accum += int(j<x)
    return (accum + 1) / (N + 2)