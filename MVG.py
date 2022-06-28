import numpy
import math

def mcol(v):
    return v.reshape(v.size, 1)

def vrow(vect):
    return vect.reshape(1, vect.size)

def logpdf_GAU_1Sample(x, mu, C):
    #C qua rappresenta la matrice delle covarianze chiamata sigma nelle slide
    inv = numpy.linalg.inv(C)
    sign, det = numpy.linalg.slogdet(C)
    M = x.shape[0]
    ret = -(M/2) * math.log(2*math.pi) - (0.5) * det - (0.5) * numpy.dot( (x-mu).T, numpy.dot(inv, (x-mu)))
    return ret.ravel()

def logpdf_GAU_ND(X, mu, C):
    Y = [logpdf_GAU_1Sample(X[:, i:i+1], mu, C) for i in range(X.shape[1])]
    return numpy.array(Y).ravel()

def numberOfCorrectLabreturn(Predictions, LTE):
    n = 0
    for i in range(0,Predictions.shape[0]):
        if Predictions[i] == LTE[i]:
            n += 1
            
    return n

def MultiV(DTR, LTR, DTE):
    mu0, C0 = meanAndCovMat(DTR[:, LTR == 0]) #Calcolo media e matrice delle covarianze per ogni classe
    mu1, C1 = meanAndCovMat(DTR[:, LTR == 1])
    S0 = logpdf_GAU_ND(DTE, mu0, C0)
    S1 = logpdf_GAU_ND(DTE, mu1, C1)
    LLRs = S1 - S0
    
    return LLRs

def Bayes(DTR, LTR, DTE):
    mu0, C0 = meanAndCovMat(DTR[:, LTR == 0]) #Calcolo media e matrice delle covarianze per ogni classe
    mu1, C1 = meanAndCovMat(DTR[:, LTR == 1])
    I = numpy.identity(C0.shape[0])
    C0Diag = C0 * I
    C1Diag = C1 * I
    S0 = logpdf_GAU_ND(DTE, mu0, C0Diag)
    S1 = logpdf_GAU_ND(DTE, mu1, C1Diag)
    LLRs = S1 - S0
    
    return LLRs

def Tied(DTR, LTR, DTE):
    mu0, C0 = meanAndCovMat(DTR[:, LTR == 0]) #Calcolo media e matrice delle covarianze per ogni classe
    mu1, C1 = meanAndCovMat(DTR[:, LTR == 1])
    N0 = DTR[:, LTR == 0].shape[1]            #Queste sarebbero le mie Nc, dove a c è sostituito il numero della classe
    N1 = DTR[:, LTR == 1].shape[1]
    N = DTR.shape[1]                          #Prendo la grandezza del mio traning set, quindi quanti sample contiene
    nC0 = N0*C0
    nC1 = N1*C1
    C = numpy.add(nC0, nC1)
    C = C/N

    S0 = logpdf_GAU_ND(DTE, mu0, C)
    S1 = logpdf_GAU_ND(DTE, mu1, C)
    LLRs = S1 - S0
    
    return LLRs

def meanAndCovMat(X):
    N = X.shape[1]
    mu = X.mean(1) #Calcolo la media nella direzione delle colonne, quindi da sinistra verso destra
    mu = mcol(mu)
    XC = X - mu
    C = (1/N) * numpy.dot( (XC), (XC).T )
    return mu, C