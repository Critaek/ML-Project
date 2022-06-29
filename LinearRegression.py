from contextlib import ContextDecorator
import numpy
import scipy.optimize
import time
import DimReduction as dr
import ModelEvaluation as me

def mcol(v):
    return v.reshape(v.size, 1)

def logreg_obj(v, DTR, LTR, l, prior):
    w, b = mcol(v[0:-1]), v[-1]

    LTR0 = LTR[LTR == 0]
    LTR1 = LTR[LTR == 1]

    Z0 = LTR0 * 2.0 - 1.0
    Z1 = LTR1 * 2.0 - 1.0

    Z = LTR * 2.0 - 1.0

    S0 = numpy.dot(w.T, DTR[:, LTR == 0]) + b
    S1 = numpy.dot(w.T, DTR[:, LTR == 1]) + b
    
    NF = len(LTR0)
    NT = len(LTR1)
    
    cxeF = numpy.logaddexp(0, -Z0*S0).sum() * (1 - prior) / NF
    cxeT = numpy.logaddexp(0, -Z1*S1).sum() * prior / NT

    return l/2 * numpy.linalg.norm(w)**2 + cxeT + cxeF

def LinearRegression(DTR, LTR, DTE, l, prior):
    x0 = numpy.zeros(DTR.shape[0] + 1)
    x, f, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, args=(DTR, LTR, l, prior), approx_grad = True)
    w = x[0:DTR.shape[0]]
    b = x[-1]
    LLRs = numpy.dot(w.T, DTE) + b
    Predictions = LLRs > 0

    return Predictions, LLRs



def kFold(D, L, K, l, model):
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
        PredRet, LLRsRet = model(DTR, LTR, DTE, l, 0.5)
        LLRs.append(LLRsRet)
        Predictions.append(PredRet)

    LLRs = numpy.hstack(LLRs)
    Predictions = numpy.hstack(Predictions)

    return Predictions, LLRs



def tryRegression(model, D, L, NormD, lSet):
    for l in lSet:
        print("Lambda =", l)
        print("Raw")
        for i in range(7):
            if(i==6):
                print("----------------No PCA", "----------------")
            else:    
                print("----------------PCA", 5+i, "----------------")
            PCA = dr.PCA(D, L, 5+i)
            start = time.time()
            Predictions, LLRs = kFold(PCA, L, 5, l, model)
            end = time.time()
            me.printDCFs(D, L, Predictions, LLRs)
            print(end-start, "seconds\n")  

        print("\n")
        
        print("Normalized")
        for i in range(7):
            if(i==6):
                print("----------------No PCA", "----------------")
            else:    
                print("----------------PCA", 5+i, "----------------")
            PCA = dr.PCA(NormD, L, 5+i)
            start = time.time()
            Predictions, LLRs = kFold(PCA, L, 5, l, model)
            end = time.time()    
            me.printDCFs(D, L, Predictions, LLRs)
            print(end-start, "seconds\n") 