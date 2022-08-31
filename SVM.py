import numpy
import scipy.optimize
import DimReduction as dr
import time
import ModelEvaluation as me
import ScoreCalibration as sc
import K_Fold

def mcol(vect):
    return vect.reshape(vect.size, 1)

def vrow(vect):
    return vect.reshape(1, vect.size)

def SVMLinear(DTR, LTR, DTE, LTE, K, C, prior_t):
    expandedD = numpy.vstack([DTR, K * numpy.ones(DTR.shape[1])])

    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1
    
    H = numpy.dot(expandedD.T, expandedD)
    H = mcol(Z) * vrow(Z) * H

    def JDual(alpha):
        Ha = numpy.dot(H, mcol(alpha))
        aHa = numpy.dot(vrow(alpha), Ha)
        a1 = alpha.sum()

        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)

    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad

    ##
        
    boundaries = numpy.empty(LTR.shape, dtype = 'f,f')
    Pi_T_Emp = LTR[LTR == 1].size / LTR.size
    Pi_F_Emp = LTR[LTR == 0].size / LTR.size
    Ct = C * prior_t / Pi_T_Emp
    Cf = C * (1 - prior_t) / Pi_F_Emp
    boundaries[LTR == 0] = (0, Cf)
    boundaries[LTR == 1] = (0, Ct)
        
    alphaStar, x, y = scipy.optimize.fmin_l_bfgs_b(LDual,
                                                   numpy.zeros(DTR.shape[1]),
                                                   bounds = boundaries,
                                                   factr = 1.0,
                                                   maxiter=5000,
                                                   maxfun=100000
                                                  )
    
    wStar = numpy.dot(expandedD, mcol(alphaStar) * mcol(Z))

    expandedDTE = numpy.vstack([DTE, K * numpy.ones(DTE.shape[1])])
    score = numpy.dot(wStar.T, expandedDTE)
    Predictions = score > 0

    return score[0]




def SVMPoly(DTR, LTR, DTE, LTE, K, C, d, c, prior_t):
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1
    
    epsilon = K**2

    product = numpy.dot(DTR.T, DTR)
    Kernel = (product + c)**d + epsilon
    H = mcol(Z) * vrow(Z) * Kernel

    def JDual(alpha):
        Ha = numpy.dot(H, mcol(alpha))
        aHa = numpy.dot(vrow(alpha), Ha)
        a1 = alpha.sum()

        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)

    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad

    ##
    
    boundaries = numpy.empty(LTR.shape, dtype = 'f,f')
    Pi_T_Emp = (LTR == 1).size / LTR.size
    Pi_F_Emp = (LTR == 0).size / LTR.size

    Ct = C * prior_t / Pi_T_Emp
    Cf = C * (1 - prior_t) / Pi_F_Emp
    boundaries[LTR == 0] = (0, Cf)
    boundaries[LTR == 1] = (0, Ct)
    alphaStar, x, y = scipy.optimize.fmin_l_bfgs_b(LDual,
                                                   numpy.zeros(DTR.shape[1]),
                                                   bounds = boundaries,
                                                   factr = 1.0,
                                                   maxiter=5000,
                                                   maxfun=100000
                                                  )
    
    scores = []
    for x_t in DTE.T:
        score = 0
        for i in range(DTR.shape[1]):
            Kernel = (numpy.dot(DTR.T[i].T, x_t) + c)**d + epsilon
            score += alphaStar[i] * Z[i] * Kernel
        scores.append(score)
    
    scores = numpy.hstack(scores)
     
    Predictions = scores > 0
    Predictions = numpy.hstack(Predictions)

    return scores




def SVM_RBF(DTR, LTR, DTE, LTE, K, C, gamma, prior_t):
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1
    
    epsilon = K**2

    Dist = numpy.zeros([DTR.shape[1], DTR.shape[1]])

    for i in range(DTR.shape[1]):
        for j in range(DTR.shape[1]):
            xi = DTR[:, i]
            xj = DTR[:, j]
            Dist[i, j] = numpy.linalg.norm(xi - xj)**2

    Kernel = numpy.exp(- gamma * Dist) + epsilon
    H = mcol(Z) * vrow(Z) * Kernel

    def JDual(alpha):
        Ha = numpy.dot(H, mcol(alpha))
        aHa = numpy.dot(vrow(alpha), Ha)
        a1 = alpha.sum()

        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)

    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad

    ##
    
    boundaries = numpy.empty(LTR.shape, dtype = 'f,f')
    Pi_T_Emp = (LTR == 1).size / LTR.size
    Pi_F_Emp = (LTR == 0).size / LTR.size

    Ct = C * prior_t / Pi_T_Emp
    Cf = C * (1 - prior_t) / Pi_F_Emp
    boundaries[LTR == 0] = (0, Cf)
    boundaries[LTR == 1] = (0, Ct)
    alphaStar, x, y = scipy.optimize.fmin_l_bfgs_b(LDual,
                                                   numpy.zeros(DTR.shape[1]),
                                                   bounds=boundaries,
                                                   factr = 1.0,
                                                   maxiter=5000,
                                                   maxfun=100000
                                                  )
    
    scores = []
    for x_t in DTE.T:
        score = 0
        for i in range(DTR.shape[1]):
            Dist = numpy.linalg.norm(DTR[:, i] - x_t)
            Kernel = numpy.exp(- gamma * Dist) + epsilon
            score += alphaStar[i] * Z[i] * Kernel
        scores.append(score)
    
    scores = numpy.hstack(scores)
     
    Predictions = scores > 0
    Predictions = numpy.hstack(Predictions)

    return scores


def kFoldLinear(loadFunction, KModel, C, prior_t, pca): #KModel è il K relativo al modello
    folds = loadFunction()

    LLRs = []

    for f in folds:
        DTR = f[0]
        LTR = f[1]
        DTE = f[2]
        LTE = f[3]
        P = dr.PCA_P(DTR, pca)
        DTR = numpy.dot(P.T, DTR)
        DTE = numpy.dot(P.T, DTE)
        LLRsRet = SVMLinear(DTR, LTR, DTE, LTE, KModel, C, prior_t)
        LLRs.append(LLRsRet)
    
    LLRs = numpy.hstack(LLRs)

    return LLRs

def trainSVMLinear(D, L, K_Set, C_Set, prior_t): #K relativo al modello, non k_fold
    prior_tilde_set = [0.1, 0.5, 0.9]
    pca = 8
    
    for K in K_Set:
        for C in C_Set:
            Scores = kFoldLinear(K_Fold.loadRawFolds, K, C, prior_t, pca)
            #Still called LLRs in the printDCFs function, but they are scores with no probabilistic interpretation
            #We use the same function for every model
            CalibratedScores, labels = sc.calibrate_scores(Scores, L, prior_t)
            for prior_tilde in prior_tilde_set:
                ActDCF, minDCF = me.printDCFs(D, L, Scores, prior_tilde) 
                print(prior_t, "|" ,prior_tilde, "| SVM Linear | K =", K, "| C =", C, "| Raw | Uncalibrated | PCA =", pca,
                      "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))
                ActDCF, minDCF = me.printDCFsNoShuffle(D, labels, CalibratedScores, prior_tilde) 
                print(prior_t, "|" ,prior_tilde, "| SVM Linear | K =", K, "| C =", C, "| Raw | Calibrated | PCA =", pca,
                      "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))
    
    for K in K_Set:
        for C in C_Set:
            Scores = kFoldLinear(K_Fold.loadNormFolds, K, C, prior_t, pca)
            CalibratedScores, labels = sc.calibrate_scores(Scores, L, prior_t)
            for prior_tilde in prior_tilde_set:
                ActDCF, minDCF = me.printDCFs(D, L, Scores, prior_tilde) 
                print(prior_t, "|" ,prior_tilde, "| SVM Linear | K =", K, "| C =", C, "| Normalized | Uncalibrated | PCA =", pca,
                      "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))
                ActDCF, minDCF = me.printDCFsNoShuffle(D, labels, CalibratedScores, prior_tilde)
                print(prior_t, "|" ,prior_tilde, "| SVM Linear | K =", K, "| C =", C, "| Normalized | Calibrated | PCA =", pca,
                      "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF)) 


def kFoldPoly(loadFunction, KModel, C, d, c, prior_t, pca): #KModel è il K relativo al modello
    folds = loadFunction()

    LLRs = []

    for f in folds:
        DTR = f[0]
        LTR = f[1]
        DTE = f[2]
        LTE = f[3]
        P = dr.PCA_P(DTR, pca)
        DTR = numpy.dot(P.T, DTR)
        DTE = numpy.dot(P.T, DTE)
        LLRsRet = SVMPoly(DTR, LTR, DTE, LTE, KModel, C, d, c, prior_t)
        LLRs.append(LLRsRet)
    
    LLRs = numpy.hstack(LLRs)

    return LLRs


def trainSVMPoly(D, L, K_Set, C_Set, d_Set, c_Set, prior_t): #K relativo al modello, non k_fold
    pca = 8
    prior_tilde_set = [0.1, 0.5, 0.9]

    for K in K_Set:
        for C in C_Set:
            for d in d_Set:
                for c in c_Set:
                    Scores = kFoldPoly(K_Fold.loadRawFolds, K, C, d, c, prior_t, pca)
                    #Still called LLRs in the printDCFs function, but they are scores with no probabilistic interpretation
                    #We use the same function for every model
                    CalibratedScores, labels = sc.calibrate_scores(Scores, L, prior_t)
                    for prior_tilde in prior_tilde_set: 
                        ActDCF, minDCF = me.printDCFs(D, L, Scores, prior_tilde) 
                        print(prior_t, "|" ,prior_tilde, "| SVM Poly | K =", K, "| C =", C, "| d =", d, "| c =", c, "| Raw | Uncalibrated | PCA =", pca,
                              "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))
                        ActDCF, minDCF = me.printDCFsNoShuffle(D, labels, CalibratedScores, prior_tilde) 
                        print(prior_t, "|" ,prior_tilde, "| SVM Poly | K =", K, "| C =", C, "| d =", d, "| c =", c, "| Raw | Calibrated | PCA =", pca,
                              "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF)) 

    
    for K in K_Set:
        for C in C_Set:
            for d in d_Set:
                for c in c_Set:
                    Scores = kFoldPoly(K_Fold.loadNormFolds, K, C, d, c, prior_t, pca)
                    CalibratedScores, labels = sc.calibrate_scores(Scores, L, prior_t)
                    for prior_tilde in prior_tilde_set: 
                        ActDCF, minDCF = me.printDCFs(D, L, Scores, prior_tilde)
                        print(prior_t, "|" ,prior_tilde, "| SVM Poly | K =", K, "| C =", C, "| d =", d, "| c =", c, "| Normalized | Uncalibrated | PCA =", pca,
                              "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))
                        ActDCF, minDCF = me.printDCFsNoShuffle(D, labels, CalibratedScores, prior_tilde)
                        print(prior_t, "|" ,prior_tilde, "| SVM Poly | K =", K, "| C =", C, "| d =", d, "| c =", c, "| Normalized | Calibrated | PCA =", pca,
                              "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))  
                        

def kFold_RBF(loadFunction, KModel, C, gamma, prior_t, pca): #KModel è il K relativo al modello
    folds = loadFunction()

    LLRs = []

    for f in folds:
        DTR = f[0]
        LTR = f[1]
        DTE = f[2]
        LTE = f[3]
        P = dr.PCA_P(DTR, pca)
        DTR = numpy.dot(P.T, DTR)
        DTE = numpy.dot(P.T, DTE)
        LLRsRet = SVM_RBF(DTR, LTR, DTE, LTE, KModel, C, gamma, prior_t)
        LLRs.append(LLRsRet)
    
    LLRs = numpy.hstack(LLRs)

    return LLRs

def trainSVM_RBF(D, L, K_Set, C_Set, gamma_Set, prior_t): #K relativo al modello, non k_fold
    prior_tilde_set = [0.1, 0.5, 0.9]
    pca = 8

    for K in K_Set:
        for C in C_Set:
            for gamma in gamma_Set:
                Scores = kFold_RBF(K_Fold.loadRawFolds, K, C, gamma, prior_t, pca)
                #Still called LLRs in the printDCFs function, but they are scores with no probabilistic interpretation
                #We use the same function for every model
                CalibratedScores, labels = sc.calibrate_scores(Scores, L, prior_t)
                for prior_tilde in prior_tilde_set: 
                    ActDCF, minDCF = me.printDCFs(D, L, Scores, prior_tilde) 
                    print(prior_t, "|", prior_tilde, "| SVM RBF | K =", K, "| C =", C, "| gamma =", gamma, "| Raw | Uncalibrated | PCA =", pca,
                              "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))
                    ActDCF, minDCF = me.printDCFsNoShuffle(D, labels, CalibratedScores, prior_tilde) 
                    print(prior_t, "|", prior_tilde, "| SVM RBF | K =", K, "| C =", C, "| gamma =", gamma, "| Raw | Calibrated | PCA =", pca,
                              "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))  

   
    for K in K_Set:
        for C in C_Set:
            for gamma in gamma_Set:
                Scores = kFold_RBF(K_Fold.loadNormFolds, K, C, gamma, prior_t, pca)
                CalibratedScores, labels = sc.calibrate_scores(Scores, L, prior_t)
                for prior_tilde in prior_tilde_set: 
                    ActDCF, minDCF = me.printDCFs(D, L, Scores, prior_tilde) 
                    print(prior_t, "|", prior_tilde, "| SVM RBF | K =", K, "| C =", C, "| gamma =", gamma, "| Normalized | Uncalibrated | PCA =", pca,
                              "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))
                    ActDCF, minDCF = me.printDCFsNoShuffle(D, labels, CalibratedScores, prior_tilde) 
                    print(prior_t, "|", prior_tilde, "| SVM RBF | K =", K, "| C =", C, "| gamma =", gamma, "| Normalized | Calibrated | PCA =", pca,
                              "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))