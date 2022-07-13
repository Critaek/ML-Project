import numpy
import scipy.optimize
import DimReduction as dr
import time
import ModelEvaluation as me
import ScoreCalibration as sc

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
                                                   maxiter=100000,
                                                   maxfun=100000
                                                  )
    
    wStar = numpy.dot(expandedD, mcol(alphaStar) * mcol(Z))

    expandedDTE = numpy.vstack([DTE, K * numpy.ones(DTE.shape[1])])
    score = numpy.dot(wStar.T, expandedDTE)
    Predictions = score > 0

    return Predictions[0], score[0]




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
                                                   maxiter=100000,
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

    return Predictions, scores




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
                                                   maxiter=100000,
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

    return Predictions, scores


def kFoldLinear(D, L, K, KModel, C, prior_t): #KModel è il K relativo al modello
    numpy.random.seed(0)
    idx = numpy.random.permutation(D.shape[1])

    N = D.shape[1]
    M = round(N/K)

    Scores = []
    Predictions = []

    for i in range(K): #K=5 -> 0,1,2,3,4
        idxTrain = numpy.concatenate([idx[0:i*M], idx[(i+1)*M:N]])
        idxTest = idx[i*M:(i+1)*M]
        DTR = D[:, idxTrain]
        LTR = L[idxTrain]
        DTE = D[:, idxTest]
        LTE = L[idxTest]
        PredRet, Scores_Ret = SVMLinear(DTR, LTR, DTE, LTE, KModel, C, prior_t)
        Scores.append(Scores_Ret)
        Predictions.append(PredRet)

    Scores = numpy.hstack(Scores)
    Predictions = numpy.hstack(Predictions)

    return Predictions, Scores

def trainSVMLinear(D, L, NormD, K_Set, C_Set, prior_t): #K relativo al modello, non k_fold
    i = 5
    prior_tilde_set = [0.1, 0.5, 0.9]

    for K in K_Set:
        for C in C_Set:
            PCA = dr.PCA(D, L, 5+i)
            Predictions, Scores = kFoldLinear(PCA, L, 5, K, C, prior_t)
            #Still called LLRs in the printDCFs function, but they are scores with no probabilistic interpretation
            #We use the same function for every model
            CalibratedScores = sc.calibrate_scores(vrow(Scores), L, prior_t)
            CalibratedScores = CalibratedScores.reshape(CalibratedScores.shape[1])
            for prior_tilde in prior_tilde_set:
                ActDCF, minDCF = me.printDCFs(D, L, Scores > 0, Scores, prior_tilde) 
                print(prior_t, "|" ,prior_tilde, "| SVM Linear | K =", K, "| C =", C, "| Raw | Uncalibrated | PCA =", 5+i,
                      "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))
                ActDCF, minDCF = me.printDCFs(D, L, CalibratedScores > 0, Scores, prior_tilde) 
                print(prior_t, "|" ,prior_tilde, "| SVM Linear | K =", K, "| C =", C, "| Raw | Calibrated | PCA =", 5+i,
                      "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))
                            

    for K in K_Set:
        for C in C_Set:
            PCA = dr.PCA(NormD, L, 5+i)
            Predictions, Scores = kFoldLinear(PCA, L, 5, K, C, prior_t)
            CalibratedScores = sc.calibrate_scores(vrow(Scores), L, prior_t)
            CalibratedScores = CalibratedScores.reshape(CalibratedScores.shape[1])
            for prior_tilde in prior_tilde_set:
                ActDCF, minDCF = me.printDCFs(D, L, Scores > 0, Scores, prior_tilde) 
                print(prior_t, "|" ,prior_tilde, "| SVM Linear | K =", K, "| C =", C, "| Normalized | Uncalibrated | PCA =", 5+i,
                      "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))
                ActDCF, minDCF = me.printDCFs(D, L, CalibratedScores > 0, Scores, prior_tilde) 
                print(prior_t, "|" ,prior_tilde, "| SVM Linear | K =", K, "| C =", C, "| Normalized | Calibrated | PCA =", 5+i,
                      "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF)) 
            

def kFoldPoly(D, L, K, KModel, C, d, c, prior_t): #KModel è il K relativo al modello
    numpy.random.seed(0)
    idx = numpy.random.permutation(D.shape[1])

    N = D.shape[1]
    M = round(N/K)

    Scores = []
    Predictions = []

    for i in range(K): #K=5 -> 0,1,2,3,4
        idxTrain = numpy.concatenate([idx[0:i*M], idx[(i+1)*M:N]])
        idxTest = idx[i*M:(i+1)*M]
        DTR = D[:, idxTrain]
        LTR = L[idxTrain]
        DTE = D[:, idxTest]
        LTE = L[idxTest]
        PredRet, Scores_Ret = SVMPoly(DTR, LTR, DTE, LTE, KModel, C, d, c,prior_t)
        Scores.append(Scores_Ret)
        Predictions.append(PredRet)

    Scores = numpy.hstack(Scores)
    Predictions = numpy.hstack(Predictions)

    return Predictions, Scores


def trainSVMPoly(D, L, NormD, K_Set, C_Set, d_Set, c_Set, prior_t): #K relativo al modello, non k_fold
    i = 5
    prior_tilde_set = [0.1, 0.5, 0.9]

    for K in K_Set:
        for C in C_Set:
            for d in d_Set:
                for c in c_Set:
                    PCA = dr.PCA(D, L, 5+i)
                    Predictions, Scores = kFoldPoly(PCA, L, 5, K, C, d, c, prior_t)
                    #Still called LLRs in the printDCFs function, but they are scores with no probabilistic interpretation
                    #We use the same function for every model
                    CalibratedScores = sc.calibrate_scores(vrow(Scores), L, prior_t)
                    CalibratedScores = CalibratedScores.reshape(CalibratedScores.shape[1])
                    for prior_tilde in prior_tilde_set: 
                        ActDCF, minDCF = me.printDCFs(D, L, Scores > 0, Scores, prior_tilde)
                        print(prior_t, "|" ,prior_tilde, "| SVM Poly | K =", K, "| C =", C, "| d =", d, "| c =", c, "| Raw | Uncalibrated | PCA =", 5+i,
                              "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))
                        ActDCF, minDCF = me.printDCFs(D, L, CalibratedScores > 0, Scores, prior_tilde)
                        print(prior_t, "|" ,prior_tilde, "| SVM Poly | K =", K, "| C =", C, "| d =", d, "| c =", c, "| Raw | Calibrated | PCA =", 5+i,
                              "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF)) 

    
    for K in K_Set:
        for C in C_Set:
            for d in d_Set:
                for c in c_Set:
                    PCA = dr.PCA(NormD, L, 5+i)
                    Predictions, Scores = kFoldPoly(PCA, L, 5, K, C, d, c, prior_t)
                    CalibratedScores = sc.calibrate_scores(vrow(Scores), L, prior_t)
                    CalibratedScores = CalibratedScores.reshape(CalibratedScores.shape[1])
                    for prior_tilde in prior_tilde_set: 
                        ActDCF, minDCF = me.printDCFs(D, L, Scores > 0, Scores, prior_tilde)
                        print(prior_t, "|" ,prior_tilde, "| SVM Poly | K =", K, "| C =", C, "| d =", d, "| c =", c, "| Normalized | Uncalibrated | PCA =", 5+i,
                              "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))
                        ActDCF, minDCF = me.printDCFs(D, L, CalibratedScores > 0, Scores, prior_tilde)
                        print(prior_t, "|" ,prior_tilde, "| SVM Poly | K =", K, "| C =", C, "| d =", d, "| c =", c, "| Normalized | Calibrated | PCA =", 5+i,
                              "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))  
                        

def kFold_RBF(D, L, K, KModel, C, gamma, prior_t): #KModel è il K relativo al modello
    numpy.random.seed(0)
    idx = numpy.random.permutation(D.shape[1])

    N = D.shape[1]
    M = round(N/K)

    Scores = []
    Predictions = []

    for i in range(K): #K=5 -> 0,1,2,3,4
        idxTrain = numpy.concatenate([idx[0:i*M], idx[(i+1)*M:N]])
        idxTest = idx[i*M:(i+1)*M]
        DTR = D[:, idxTrain]
        LTR = L[idxTrain]
        DTE = D[:, idxTest]
        LTE = L[idxTest]
        PredRet, Scores_Ret = SVM_RBF(DTR, LTR, DTE, LTE, KModel, C, gamma, prior_t)
        Scores.append(Scores_Ret)
        Predictions.append(PredRet)

    Scores = numpy.hstack(Scores)
    Predictions = numpy.hstack(Predictions)

    return Predictions, Scores

def trainSVM_RBF(D, L, NormD, K_Set, C_Set, gamma_Set, prior_t): #K relativo al modello, non k_fold
    prior_tilde_set = [0.1, 0.5, 0.9]
    i = 5

    for K in K_Set:
        for C in C_Set:
            for gamma in gamma_Set:
                PCA = dr.PCA(D, L, 5+i)
                Predictions, Scores = kFold_RBF(PCA, L, 5, K, C, gamma, prior_t)
                #Still called LLRs in the printDCFs function, but they are scores with no probabilistic interpretation
                #We use the same function for every model
                CalibratedScores = sc.calibrate_scores(vrow(Scores), L, prior_t)
                CalibratedScores = CalibratedScores.reshape(CalibratedScores.shape[1])
                for prior_tilde in prior_tilde_set: 
                    ActDCF, minDCF = me.printDCFs(D, L, Scores > 0, Scores, prior_tilde)
                    print(prior_t, "|", prior_tilde, "| SVM RBF | K =", K, "| C =", C, "| gamma =", gamma, "| Raw | Uncalibrated | PCA =", 5+i,
                              "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))
                    ActDCF, minDCF = me.printDCFs(D, L, CalibratedScores > 0, Scores, prior_tilde)
                    print(prior_t, "|", prior_tilde, "| SVM RBF | K =", K, "| C =", C, "| gamma =", gamma, "| Raw | Calibrated | PCA =", 5+i,
                              "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))  

   
    for K in K_Set:
        for C in C_Set:
            for gamma in gamma_Set:
                PCA = dr.PCA(NormD, L, 5+i)
                Predictions, Scores = kFold_RBF(PCA, L, 5, K, C, gamma, prior_t)
                CalibratedScores = sc.calibrate_scores(vrow(Scores), L, prior_t)
                CalibratedScores = CalibratedScores.reshape(CalibratedScores.shape[1])
                for prior_tilde in prior_tilde_set: 
                    ActDCF, minDCF = me.printDCFs(D, L, Scores > 0, Scores, prior_tilde)
                    print(prior_t, "|", prior_tilde, "| SVM RBF | K =", K, "| C =", C, "| gamma =", gamma, "| Normalized | Uncalibrated | PCA =", 5+i,
                              "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))
                    ActDCF, minDCF = me.printDCFs(D, L, CalibratedScores > 0, Scores, prior_tilde)
                    print(prior_t, "|", prior_tilde, "| SVM RBF | K =", K, "| C =", C, "| gamma =", gamma, "| Normalized | Calibrated | PCA =", 5+i,
                              "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))