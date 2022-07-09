import numpy
import DimReduction as dr
import ScoreCalibration as sc
import math
import scipy.special
import ModelEvaluation as me

from ModelEvaluation import PredicionsByScore

def vcol(vect):
    return vect.reshape(vect.size, 1)

def vrow(vect):
    return vect.reshape(1, vect.size)

 #-----------------------------------------------------------------------------------#

def logpdf_GAU_ND_Opt(X, mu, C):
    inv = numpy.linalg.inv(C)
    sign, det = numpy.linalg.slogdet(C)
    M = X.shape[0]
    const = -(M/2) * math.log(2*math.pi) - (0.5) * det 
    
    Y = []
    for i in range(X.shape[1]):
        x = X[:, i:i+1]
        res = const - (0.5) * numpy.dot( (x-mu).T, numpy.dot(inv, (x-mu)))
        Y.append(res)

    return numpy.array(Y).ravel()

#-----------------------------------------------------------------------------------#

def GMM_ll_perSample(X, gmm):

    G = len(gmm)
    N = X.shape[1]
    S = numpy.zeros((G, N))
    for g in range(G):
        S[g, :] = logpdf_GAU_ND_Opt(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
    
    return scipy.special.logsumexp(S, axis = 0)

#-----------------------------------------------------------------------------------# 

def GMM_EM_Full(X, gmm, psi = 0.01):

    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]

    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ = numpy.zeros((G, N))
        for g in range(G): #numero componenti
            SJ[g, :] = logpdf_GAU_ND_Opt(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
        SM = scipy.special.logsumexp(SJ, axis = 0)
        llNew = SM.sum() / N
        P = numpy.exp(SJ - SM)
        gmmNew = []

        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()            
            F = (vrow(gamma)*X).sum(1)
            S = numpy.dot(X, (vrow(gamma)*X).T)
            w = Z / N
            mu = vcol(F / Z)
            Sigma = S/Z - numpy.dot(mu, mu.T)
            U, s, _ = numpy.linalg.svd(Sigma)
            s[s < psi] = psi
            Sigma = numpy.dot(U, vcol(s) * U.T)
            gmmNew.append((w, mu, Sigma))

        gmm = gmmNew

    return gmm

#-----------------------------------------------------------------------------------#

def GMM_EM_Diagonal(X, gmm, psi = 0.01):

    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]

    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ = numpy.zeros((G, N))
        for g in range(G): #numero componenti
            SJ[g, :] = logpdf_GAU_ND_Opt(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
        SM = scipy.special.logsumexp(SJ, axis = 0)
        llNew = SM.sum() / N
        P = numpy.exp(SJ - SM)
        gmmNew = []

        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()            
            F = (vrow(gamma)*X).sum(1)
            S = numpy.dot(X, (vrow(gamma)*X).T)
            w = Z / N
            mu = vcol(F / Z)
            Sigma = S/Z - numpy.dot(mu, mu.T)
            U, s, _ = numpy.linalg.svd(Sigma)
            s[s < psi] = psi
            Sigma = numpy.dot(U, vcol(s) * U.T)
            
            #Diagonalizzo
            Sigma = Sigma * numpy.eye(Sigma.shape[0])

            gmmNew.append((w, mu, Sigma))

        gmm = gmmNew

    return gmm

#-----------------------------------------------------------------------------------#

def GMM_EM_Tied(X, gmm, psi = 0.01):

    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]

    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ = numpy.zeros((G, N))
        for g in range(G): #numero componenti
            SJ[g, :] = logpdf_GAU_ND_Opt(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
        SM = scipy.special.logsumexp(SJ, axis = 0)
        llNew = SM.sum() / N
        P = numpy.exp(SJ - SM)
        gmmNew = []
        Z_List = []

        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()

            Z_List.append(Z)
            
            F = (vrow(gamma)*X).sum(1)
            S = numpy.dot(X, (vrow(gamma)*X).T)
            w = Z / N
            mu = vcol(F / Z)
            Sigma = S/Z - numpy.dot(mu, mu.T)
            U, s, _ = numpy.linalg.svd(Sigma)
            s[s < psi] = psi
            Sigma = numpy.dot(U, vcol(s) * U.T)

            gmmNew.append((w, mu, Sigma))

        #-----------------------Tied-------------------------#
        gmmTied = []
        sum = numpy.zeros(gmmNew[0][2].shape)

        for g in range(G):
            sum = sum + Z_List[g] * gmm[g][2]

        TiedSigma = sum / X.shape[1]

        for g in range(G):
            gmmTied.append((gmmNew[g][0], gmmNew[g][1], TiedSigma))

        gmm = gmmTied

    return gmm

#-----------------------------------------------------------------------------------#


def meanAndCovMat(X):
    N = X.shape[1]
    mu = X.mean(1) #calcolo la media nella direzione delle colonne, quindi da sinistra verso destra
    mu = vcol(mu)
    XC = X - mu
    C = (1/N) * numpy.dot( (XC), (XC).T )
    return mu, C

#-----------------------------------------------------------------------------------#

def GMM_LBG_Full(X, G, alpha = 0.1):
    mu, C = meanAndCovMat(X)
    gmms = []
    
    gmms.append((1.0, mu, C))
    
    gmms = GMM_EM_Full(X, gmms)

    for g in range(G): #G = 2 -> 0, 1
        newList = []
        for element in gmms:
            w = element[0] / 2
            mu = element[1]
            C = element[2]
            U, s, Vh = numpy.linalg.svd(C)
            d = U[:, 0:1] * s[0]**0.5 * alpha
            newList.append((w, mu + d, C))
            newList.append((w, mu - d, C))
        gmms = GMM_EM_Full(X, newList)  

    return gmms 

#-----------------------------------------------------------------------------------#

def GMM_LBG_Diagonal(X, G, alpha = 0.1):
    mu, C = meanAndCovMat(X)
    gmms = []
    
    gmms.append((1.0, mu, C))
    
    gmms = GMM_EM_Diagonal(X, gmms)

    for g in range(G): #G = 2 -> 0, 1
        newList = []
        for element in gmms:
            w = element[0] / 2
            mu = element[1]
            C = element[2]
            U, s, Vh = numpy.linalg.svd(C)
            d = U[:, 0:1] * s[0]**0.5 * alpha
            newList.append((w, mu + d, C))
            newList.append((w, mu - d, C))
        gmms = GMM_EM_Diagonal(X, newList)  

    return gmms 

#-----------------------------------------------------------------------------------#

def GMM_LBG_Tied(X, G, alpha = 0.1):
    mu, C = meanAndCovMat(X)
    gmms = []
    
    gmms.append((1.0, mu, C))
    
    gmms = GMM_EM_Tied(X, gmms)

    for g in range(G): #G = 2 -> 0, 1
        newList = []
        for element in gmms:
            w = element[0] / 2
            mu = element[1]
            C = element[2]
            U, s, Vh = numpy.linalg.svd(C)
            d = U[:, 0:1] * s[0]**0.5 * alpha
            newList.append((w, mu + d, C))
            newList.append((w, mu - d, C))
        gmms = GMM_EM_Tied(X, newList)  

    return gmms 

#-----------------------------------------------------------------------------------#

def GMM_Scores(DTE, gmm0, gmm1):
    Scores0 = GMM_ll_perSample(DTE, gmm0)
    Scores1 = GMM_ll_perSample(DTE, gmm1)
    
    Scores = Scores1 - Scores0
    Predictions = Scores > 0

    return Predictions, Scores

#-----------------------------------------------------------------------------------#


def kFold_GMM_Full(D, L, K, n): #K per fold
    numpy.random.seed(0)
    idx = numpy.random.permutation(D.shape[1])

    N = D.shape[1]
    M = round(N/K)

    LLRs = []
    Predictions = []

    for i in range(K): #K=3 -> 0,1,2
        idxTrain = numpy.concatenate([idx[0:i*M], idx[(i+1)*M:N]])
        idxTest = idx[i*M:(i+1)*M]
        DTR = D[:, idxTrain]
        LTR = L[idxTrain]
        DTE = D[:, idxTest]
        LTE = L[idxTest]
        DTR0 = DTR[LTR == 0] # bad wines
        DTR1 = DTR[LTR == 1] # good wines
        gmm0 = GMM_LBG_Full(DTR0, n) #n number of components
        gmm1 = GMM_LBG_Full(DTR1, n) #n number of components
        PredRet, LLRsRet = GMM_Scores(DTE, gmm0, gmm1)
 
        LLRs.append(LLRsRet)
        Predictions.append(PredRet)

    LLRs = numpy.hstack(LLRs)
    Predictions = numpy.hstack(Predictions)

    return Predictions, LLRs

def kFold_GMM_Diagonal(D, L, K, n): #K per fold
    numpy.random.seed(0)
    idx = numpy.random.permutation(D.shape[1])

    N = D.shape[1]
    M = round(N/K)

    LLRs = []
    Predictions = []

    for i in range(K): #K=3 -> 0,1,2
        idxTrain = numpy.concatenate([idx[0:i*M], idx[(i+1)*M:N]])
        idxTest = idx[i*M:(i+1)*M]
        DTR = D[:, idxTrain]
        LTR = L[idxTrain]
        DTE = D[:, idxTest]
        LTE = L[idxTest]
        DTR0 = DTR[LTR == 0] # bad wines
        DTR1 = DTR[LTR == 1] # good wines
        gmm0 = GMM_LBG_Diagonal(DTR0, n) #n number of components
        gmm1 = GMM_LBG_Diagonal(DTR1, n) #n number of components
        PredRet, LLRsRet = GMM_Scores(DTE, gmm0, gmm1)
 
        LLRs.append(LLRsRet)
        Predictions.append(PredRet)

    LLRs = numpy.hstack(LLRs)
    Predictions = numpy.hstack(Predictions)

    return Predictions, LLRs

def kFold_GMM_Tied(D, L, K, n): #K per fold
    numpy.random.seed(0)
    idx = numpy.random.permutation(D.shape[1])

    N = D.shape[1]
    M = round(N/K)

    LLRs = []
    Predictions = []

    for i in range(K): #K=3 -> 0,1,2
        idxTrain = numpy.concatenate([idx[0:i*M], idx[(i+1)*M:N]])
        idxTest = idx[i*M:(i+1)*M]
        DTR = D[:, idxTrain]
        LTR = L[idxTrain]
        DTE = D[:, idxTest]
        LTE = L[idxTest]
        DTR0 = DTR[LTR == 0] # bad wines
        DTR1 = DTR[LTR == 1] # good wines
        gmm0 = GMM_LBG_Tied(DTR0, n) #n number of components
        gmm1 = GMM_LBG_Tied(DTR1, n) #n number of components
        PredRet, LLRsRet = GMM_Scores(DTE, gmm0, gmm1)
 
        LLRs.append(LLRsRet)
        Predictions.append(PredRet)

    LLRs = numpy.hstack(LLRs)
    Predictions = numpy.hstack(Predictions)

    return Predictions, LLRs



def trainGMM_Full(D, L, NormD, nSet):
    prior_tilde_set = [0.1, 0.5, 0.9]
    i = 5

    for nComponents in nSet:
        PCA = dr.PCA(D, L, 5+i)
        Predictions, Scores = kFold_GMM_Full(PCA, L, 5, nComponents)
        #Still called LLRs in the printDCFs function, but they are scores with no probabilistic interpretation
        #We use the same function for every model
        for prior_tilde in prior_tilde_set: 
            CalibratedScores = sc.calibrate_scores(vrow(Scores), L, prior_tilde)
            CalibratedScores = CalibratedScores.reshape(CalibratedScores.shape[1])
            ActDCF, minDCF = me.printDCFs(D, L, Scores > 0, Scores, prior_tilde)
            print(prior_tilde, "| GMM Full | nComponents =", 2**nComponents, "| Raw | Uncalibrated | PCA =", 5+i,
                        "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))
            ActDCF, minDCF = me.printDCFs(D, L, CalibratedScores > 0, Scores, prior_tilde)
            print(prior_tilde, "| GMM Full | nComponents =", 2**nComponents, "| Raw | Calibrated | PCA =", 5+i,
                        "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))

    for nComponents in nSet:
        PCA = dr.PCA(NormD, L, 5+i)
        Predictions, Scores = kFold_GMM_Full(PCA, L, 5, nComponents)
        #Still called LLRs in the printDCFs function, but they are scores with no probabilistic interpretation
        #We use the same function for every model
        for prior_tilde in prior_tilde_set: 
            CalibratedScores = sc.calibrate_scores(vrow(Scores), L, prior_tilde)
            CalibratedScores = CalibratedScores.reshape(CalibratedScores.shape[1])
            ActDCF, minDCF = me.printDCFs(D, L, Scores > 0, Scores, prior_tilde)
            print(prior_tilde, "| GMM Full | nComponents =", 2**nComponents, "| Normalized | Uncalibrated | PCA =", 5+i,
                        "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))
            ActDCF, minDCF = me.printDCFs(D, L, CalibratedScores > 0, Scores, prior_tilde)
            print(prior_tilde, "| GMM Full | nComponents =", 2**nComponents, "| Normalized | Calibrated | PCA =", 5+i,
                        "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))


def trainGMM_Diagonal(D, L, NormD, nSet):
    prior_tilde_set = [0.1, 0.5, 0.9]
    i = 5

    for nComponents in nSet:
        PCA = dr.PCA(D, L, 5+i)
        Predictions, Scores = kFold_GMM_Diagonal(PCA, L, 5, nComponents)
        #Still called LLRs in the printDCFs function, but they are scores with no probabilistic interpretation
        #We use the same function for every model
        for prior_tilde in prior_tilde_set: 
            CalibratedScores = sc.calibrate_scores(vrow(Scores), L, prior_tilde)
            CalibratedScores = CalibratedScores.reshape(CalibratedScores.shape[1])
            ActDCF, minDCF = me.printDCFs(D, L, Scores > 0, Scores, prior_tilde)
            print(prior_tilde, "| GMM Diagonal | nComponents =", 2**nComponents, "| Raw | Uncalibrated | PCA =", 5+i,
                        "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))
            ActDCF, minDCF = me.printDCFs(D, L, CalibratedScores > 0, Scores, prior_tilde)
            print(prior_tilde, "| GMM Diagonal | nComponents =", 2**nComponents, "| Raw | Calibrated | PCA =", 5+i,
                        "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))

    for nComponents in nSet:
        PCA = dr.PCA(NormD, L, 5+i)
        Predictions, Scores = kFold_GMM_Diagonal(PCA, L, 5, nComponents)
        #Still called LLRs in the printDCFs function, but they are scores with no probabilistic interpretation
        #We use the same function for every model
        for prior_tilde in prior_tilde_set: 
            CalibratedScores = sc.calibrate_scores(vrow(Scores), L, prior_tilde)
            CalibratedScores = CalibratedScores.reshape(CalibratedScores.shape[1])
            ActDCF, minDCF = me.printDCFs(D, L, Scores > 0, Scores, prior_tilde)
            print(prior_tilde, "| GMM Diagonal | nComponents =", 2**nComponents, "| Normalized | Uncalibrated | PCA =", 5+i,
                        "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))
            ActDCF, minDCF = me.printDCFs(D, L, CalibratedScores > 0, Scores, prior_tilde)
            print(prior_tilde, "| GMM Diagonal | nComponents =", 2**nComponents, "| Normalized | Calibrated | PCA =", 5+i,
                        "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))

def trainGMM_Tied(D, L, NormD, nSet):
    prior_tilde_set = [0.1, 0.5, 0.9]
    i = 5

    for nComponents in nSet:
        PCA = dr.PCA(D, L, 5+i)
        Predictions, Scores = kFold_GMM_Tied(PCA, L, 5, nComponents)
        #Still called LLRs in the printDCFs function, but they are scores with no probabilistic interpretation
        #We use the same function for every model
        for prior_tilde in prior_tilde_set: 
            CalibratedScores = sc.calibrate_scores(vrow(Scores), L, prior_tilde)
            CalibratedScores = CalibratedScores.reshape(CalibratedScores.shape[1])
            ActDCF, minDCF = me.printDCFs(D, L, Scores > 0, Scores, prior_tilde)
            print(prior_tilde, "| GMM Tied | nComponents =", 2**nComponents, "| Raw | Uncalibrated | PCA =", 5+i,
                        "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))
            ActDCF, minDCF = me.printDCFs(D, L, CalibratedScores > 0, Scores, prior_tilde)
            print(prior_tilde, "| GMM Tied | nComponents =", 2**nComponents, "| Raw | Calibrated | PCA =", 5+i,
                        "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))

    for nComponents in nSet:
        PCA = dr.PCA(NormD, L, 5+i)
        Predictions, Scores = kFold_GMM_Tied(PCA, L, 5, nComponents)
        #Still called LLRs in the printDCFs function, but they are scores with no probabilistic interpretation
        #We use the same function for every model
        for prior_tilde in prior_tilde_set: 
            CalibratedScores = sc.calibrate_scores(vrow(Scores), L, prior_tilde)
            CalibratedScores = CalibratedScores.reshape(CalibratedScores.shape[1])
            ActDCF, minDCF = me.printDCFs(D, L, Scores > 0, Scores, prior_tilde)
            print(prior_tilde, "| GMM Tied | nComponents =", 2**nComponents, "| Normalized | Uncalibrated | PCA =", 5+i,
                        "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))
            ActDCF, minDCF = me.printDCFs(D, L, CalibratedScores > 0, Scores, prior_tilde)
            print(prior_tilde, "| GMM Tied | nComponents =", 2**nComponents, "| Normalized | Calibrated | PCA =", 5+i,
                        "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))
