import numpy

import matplotlib.pyplot as plt

def PredicionsByScore(score, LLRs):
    pred = numpy.zeros(LLRs.shape)
    # HERE WE TRY TO OPTIMIZE THE THRESHOLD USING DIFFERENT VALUES FROM A SET OF TEST SCORES
    
    threshold=score
    for i in range(LLRs.size):
            if(LLRs[i]>threshold):
                pred[i] = 1
            else:
                pred[i] = 0
    return pred

#-------------------------------------------------------------------------------------------------#

def Predictions(pi1,Cfn,Cfp, LLRs):
    pi0 = 1-pi1
    pred = numpy.zeros(LLRs.shape)
    threshold = -numpy.log((pi1*Cfn)/((pi0*Cfp)))
    #WE USE PARTICULAR VALUE OF PI IN ORDER TO COMPUTE BYAS ERROR
    for i in range(LLRs.size):
            if(LLRs[i]>threshold):
                pred[i] = 1
            else:
                pred[i] = 0
    return pred

#-------------------------------------------------------------------------------------------------#

def ConfusionMatrix(pi1,Cfn,Cfp):
    pi0=1-pi1
    commediaLLRs = numpy.load('data/commedia_llr_infpar.npy')
    predMatrix = numpy.zeros(commediaLLRs.shape)
    threshold=-numpy.log((pi1*Cfn)/((pi0*Cfp)))
    for i in range(commediaLLRs.size):
            if(commediaLLRs[i]>threshold):
                predMatrix[i] = 1
            else:
                predMatrix[i] = 0
    return predMatrix

#-------------------------------------------------------------------------------------------------#

def BiasRisk(pi1,Cfn,Cfp,M):
    FNR = M[0][1]/(M[0][1]+M[1][1])
    FPR = M[1][0]/(M[0][0]+M[1][0])
    return (((pi1*Cfn*FNR)+(1-pi1)*Cfp*FPR),FPR,1-FNR)

#-------------------------------------------------------------------------------------------------#

def MinDummy(pi1,Cfn,Cfp):
    dummyAlwaysReject=pi1*Cfn
    dummyAlwaysAccept=(1-pi1)*Cfp
    if(dummyAlwaysReject < dummyAlwaysAccept):
        return dummyAlwaysReject
    else:
        return dummyAlwaysAccept

#-------------------------------------------------------------------------------------------------#

def printDCFs(D, L, pred, LLRs):
    numpy.random.seed(0)
    idx = numpy.random.permutation(D.shape[1])

    L = L[idx]

    pi1 = 0.5
    pi0 = 1-pi1
    Cfn = 1
    Cfp = 1
    classPriors = numpy.array([pi1,pi0]) #[0.5, 0.5]
    minDCF = []

    #normalizedDCF
    
    confusionMatrix = numpy.zeros((2, 2))

    for i in range(0,len(classPriors)):
        for j in range(0,len(classPriors)):
            confusionMatrix[i,j] = ((L == j) * (pred == i)).sum()

    (DCFu,FPRi,TPRi) = BiasRisk(pi1,Cfn,Cfp,confusionMatrix)
        
    minDummy = MinDummy(pi1,Cfn,Cfp)
    normalizedDCF = DCFu/minDummy
    
    print("DCF:", normalizedDCF)

    #minDCF
    comm = sorted(LLRs) #aggiungere -inf, inf

    for score in comm:
        
        Predicions_By_Score = PredicionsByScore(score, LLRs)
        wine_labels = L
        
        confusionMatrix = numpy.zeros((2, 2))

        for i in range(0,len(classPriors)):
            for j in range(0,len(classPriors)):
                confusionMatrix[i,j] = ((wine_labels == j) * (Predicions_By_Score == i)).sum()

        (DCFu,FPRi,TPRi) = BiasRisk(pi1,Cfn,Cfp,confusionMatrix)
        
        minDummy = MinDummy(pi1,Cfn,Cfp)
        normalizedDCF = DCFu/minDummy
        minDCF.append(normalizedDCF)

    minDCF=min(minDCF)
    print("minDCF:" , minDCF)
#-------------------------------------------------------------------------------------------------#