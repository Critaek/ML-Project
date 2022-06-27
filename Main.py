import scipy.stats as st
import numpy
import time
import Normalization as n
import DimReduction as dr
import Plot as plt
import Load as l
import MVG as mvg
import ModelEvaluation as me

from sklearn import svm

def mcol(v):
    return v.reshape(v.size, 1)

def vrow(vect):
    return vect.reshape(1, vect.size)

def kFold(D, L, K):
    numpy.random.seed(0)
    idx = numpy.random.permutation(D.shape[1])

    N = D.shape[1]
    M = round(N/K)

    LLRs = []

    for i in range(K): #K=3 -> 0,1,2
        idxTrain = numpy.concatenate([idx[0:i*M], idx[(i+1)*M:N]])
        idxTest = idx[i*M:(i+1)*M]
        DTR = D[:, idxTrain]
        LTR = L[idxTrain]
        DTE = D[:, idxTest]
        LTE = L[idxTest]
        Return = mvg.MultiV(DTR, LTR, DTE)
        LLRs.append(Return)

    LLRs = numpy.hstack(LLRs)

    return LLRs


if __name__ == "__main__":
    D, L = l.load()
    
    PCA9 = dr.PCA(D, L, 9)
    LLRs = kFold(PCA9, L, 10)

    #-------------------------------------------------------------------------------------------------#

    numpy.random.seed(0)
    idx = numpy.random.permutation(D.shape[1])

    L = L[idx]

    pi1 = 0.5
    pi0 = 1-pi1
    Cfn = 1
    Cfp = 1
    classPriors = numpy.array([pi1,pi0]) #[0.5, 0.5]
    minDCF = []

    pred = me.Predictions(pi1, Cfn, Cfp, LLRs)
    
    confusionMatrix = numpy.zeros((2, 2))

    for i in range(0,len(classPriors)):
        for j in range(0,len(classPriors)):
            confusionMatrix[i,j] = ((L == j) * (pred == i)).sum()

    (DCFu,FPRi,TPRi) = me.BiasRisk(pi1,Cfn,Cfp,confusionMatrix)
        
    minDummy = me.MinDummy(pi1,Cfn,Cfp)
    normalizedDCF = DCFu/minDummy
    
    print("DCF:", normalizedDCF)

    comm = sorted(LLRs)

    for score in comm:
        
        PredicionsByScore = me.PredicionsByScore(score, LLRs)
        wine_labels = L
        
        confusionMatrix = numpy.zeros((2, 2))

        for i in range(0,len(classPriors)):
            for j in range(0,len(classPriors)):
                confusionMatrix[i,j] = ((wine_labels == j) * (PredicionsByScore == i)).sum()

        (DCFu,FPRi,TPRi) = me.BiasRisk(pi1,Cfn,Cfp,confusionMatrix)
        
        minDummy = me.MinDummy(pi1,Cfn,Cfp)
        normalizedDCF = DCFu/minDummy
        minDCF.append(normalizedDCF)

    minDCF=min(minDCF)
    print("minDCF:" , minDCF)

    #-------------------------------------------------------------------------------------------------#



    
    #plt.HeatMapPearson(D, "Raw")
    #plt.plotHist(D, L, "Raw")

    #NormD = n.Normalization(D)
    #kFold(NormD, L, 5)

    #plt.HeatMapPearson(NormD, "Normalized")
    #plt.plotHist(NormD, L, "Normalized")

    