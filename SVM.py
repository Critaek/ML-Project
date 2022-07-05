import numpy
import scipy.optimize
import DimReduction as dr
import time
import ModelEvaluation as me

def mcol(vect):
    return vect.reshape(vect.size, 1)

def vrow(vect):
    return vect.reshape(1, vect.size)

def SVMLinear(DTR, LTR, DTE, LTE, K, C):
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

    alphaStar, x, y = scipy.optimize.fmin_l_bfgs_b(LDual,
                                                   numpy.zeros(DTR.shape[1]),
                                                   bounds=[(0, C)] * DTR.shape[1],
                                                   factr = 1.0,
                                                   maxiter=100000,
                                                   maxfun=100000
                                                  )
    
    wStar = numpy.dot(expandedD, mcol(alphaStar) * mcol(Z))

    expandedDTE = numpy.vstack([DTE, K * numpy.ones(DTE.shape[1])])
    score = numpy.dot(wStar.T, expandedDTE)
    Predictions = score > 0

    return Predictions[0], score[0]




def SVMPoly(DTR, LTR, DTE, LTE, K, C, d, c):
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

    alphaStar, x, y = scipy.optimize.fmin_l_bfgs_b(LDual,
                                                   numpy.zeros(DTR.shape[1]),
                                                   bounds=[(0, C)] * DTR.shape[1],
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




def SVM_RBF(DTR, LTR, DTE, LTE, K, C, gamma):
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

    alphaStar, x, y = scipy.optimize.fmin_l_bfgs_b(LDual,
                                                   numpy.zeros(DTR.shape[1]),
                                                   bounds=[(0, C)] * DTR.shape[1],
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


def kFoldLinear(D, L, K, KModel, C): #KModel è il K relativo al modello
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
        PredRet, Scores_Ret = SVMLinear(DTR, LTR, DTE, LTE, KModel, C)
        Scores.append(Scores_Ret)
        Predictions.append(PredRet)

    Scores = numpy.hstack(Scores)
    Predictions = numpy.hstack(Predictions)

    return Predictions, Scores

def trainSVMLinear(D, L, NormD, K_Set, C_Set): #K relativo al modello, non k_fold
    
    print("Raw")
    for K in K_Set:
        print("K =", K)
        for C in C_Set:
            print("C =", C)
            for i in range(7):
                if(i==6):
                    print("----------------No PCA", "----------------")
                else:    
                    print("----------------PCA", 5+i, "----------------")
                PCA = dr.PCA(D, L, 5+i)
                start = time.time()
                Predictions, Scores = kFoldLinear(PCA, L, 5, K, C)
                end = time.time()
                #Still called LLRs in the printDCFs function, but they are scores with no probabilistic interpretation
                #We use the same function for every model
                me.printDCFs(D, L, Predictions, Scores) 
                print(end-start, "seconds\n")

    print("Normalized")
    for K in K_Set:
        print("K =", K)
        for C in C_Set:
            print("C =", C)
            for i in range(7):
                if(i==6):
                    print("----------------No PCA", "----------------")
                else:    
                    print("----------------PCA", 5+i, "----------------")
                PCA = dr.PCA(NormD, L, 5+i)
                start = time.time()
                Predictions, Scores = kFoldLinear(PCA, L, 5, K, C)
                end = time.time()
                me.printDCFs(D, L, Predictions, Scores)
                print(end-start, "seconds\n")
            

def kFoldPoly(D, L, K, KModel, C, d, c): #KModel è il K relativo al modello
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
        PredRet, Scores_Ret = SVMPoly(DTR, LTR, DTE, LTE, KModel, C, d, c)
        Scores.append(Scores_Ret)
        Predictions.append(PredRet)

    Scores = numpy.hstack(Scores)
    Predictions = numpy.hstack(Predictions)

    return Predictions, Scores


def trainSVMPoly(D, L, NormD, K_Set, C_Set, d_Set, c_Set): #K relativo al modello, non k_fold
    
    print("Raw")
    for K in K_Set:
        print("K =", K)
        for C in C_Set:
            print("C =", C)
            for d in d_Set:
                print("d =", d)
                for c in c_Set:
                    print("c =", c)
                    for i in range(7):
                        if(i==6):
                            print("----------------No PCA", "----------------")
                        else:    
                            print("----------------PCA", 5+i, "----------------")
                        PCA = dr.PCA(D, L, 5+i)
                        start = time.time()
                        Predictions, Scores = kFoldPoly(PCA, L, 5, K, C, d, c)
                        end = time.time()
                        #Still called LLRs in the printDCFs function, but they are scores with no probabilistic interpretation
                        #We use the same function for every model
                        me.printDCFs(D, L, Predictions, Scores) 
                        print(end-start, "seconds\n")

    print("Normalized")
    for K in K_Set:
        print("K =", K)
        for C in C_Set:
            print("C =", C)
            for d in d_Set:
                print("d =", d)
                for c in c_Set:
                    print("c =", c)
                    for i in range(7):
                        if(i==6):
                            print("----------------No PCA", "----------------")
                        else:    
                            print("----------------PCA", 5+i, "----------------")
                        PCA = dr.PCA(NormD, L, 5+i)
                        start = time.time()
                        Predictions, Scores = kFoldPoly(PCA, L, 5, K, C, d, c)
                        end = time.time()
                        me.printDCFs(D, L, Predictions, Scores)
                        print(end-start, "seconds\n")

def kFold_RBF(D, L, K, KModel, C, gamma): #KModel è il K relativo al modello
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
        PredRet, Scores_Ret = SVM_RBF(DTR, LTR, DTE, LTE, KModel, C, gamma)
        Scores.append(Scores_Ret)
        Predictions.append(PredRet)

    Scores = numpy.hstack(Scores)
    Predictions = numpy.hstack(Predictions)

    return Predictions, Scores

def trainSVM_RBF(D, L, NormD, K_Set, C_Set, gamma_Set): #K relativo al modello, non k_fold
    
    print("Raw")
    for K in K_Set:
        print("K =", K)
        for C in C_Set:
            print("C =", C)
            for gamma in gamma_Set:
                print("gamma =", gamma)
                for i in range(7):
                    if(i==6):
                        print("----------------No PCA", "----------------")
                    else:    
                        print("----------------PCA", 5+i, "----------------")
                    PCA = dr.PCA(D, L, 5+i)
                    start = time.time()
                    Predictions, Scores = kFold_RBF(PCA, L, 5, K, C, gamma)
                    end = time.time()
                    #Still called LLRs in the printDCFs function, but they are scores with no probabilistic interpretation
                    #We use the same function for every model
                    me.printDCFs(D, L, Predictions, Scores) 
                    print(end-start, "seconds\n")

    print("Normalized")
    for K in K_Set:
        print("K =", K)
        for C in C_Set:
            print("C =", C)
            for gamma in gamma_Set:
                print("gamma =", gamma)
                for i in range(7):
                    if(i==6):
                        print("----------------No PCA", "----------------")
                    else:    
                        print("----------------PCA", 5+i, "----------------")
                    PCA = dr.PCA(NormD, L, 5+i)
                    start = time.time()
                    Predictions, Scores = kFold_RBF(PCA, L, 5, K, C, gamma)
                    end = time.time()
                    me.printDCFs(D, L, Predictions, Scores)
                    print(end-start, "seconds\n")