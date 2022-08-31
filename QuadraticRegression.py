from contextlib import ContextDecorator
import numpy
import scipy.optimize
import time
import DimReduction as dr
import ModelEvaluation as me
import Plot
import K_Fold

def mcol(v):
    return v.reshape(v.size, 1)

def expandFeature(dataset):
    def vecxxT(x):
        x = x[:, None]
        xxT = x.dot(x.T).reshape(x.size**2)
        return xxT
    expanded = numpy.apply_along_axis(vecxxT, 0, dataset)
    return numpy.vstack([expanded, dataset])


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

def QuadraticRegression(DTR, LTR, DTE, l, prior_t):
    x0 = numpy.zeros(DTR.shape[0] + 1)
    x, f, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, args=(DTR, LTR, l, prior_t), approx_grad = True)
    w = x[0:DTR.shape[0]]
    b = x[-1]
    scores = numpy.dot(w.T, DTE) + b

    return scores



def kFold(prior_t, loadFunction, l, pca):
    folds = loadFunction()

    LLRs = []
    Predictions = []

    for f in folds:
        DTR = f[0]
        LTR = f[1]
        DTE = f[2]
        LTE = f[3]
        P = dr.PCA_P(DTR, pca)
        DTR = numpy.dot(P.T, DTR)
        DTE = numpy.dot(P.T, DTE)
        LLRsRet = QuadraticRegression(DTR, LTR, DTE, l, prior_t)
        LLRs.append(LLRsRet)
    
    LLRs = numpy.hstack(LLRs)

    return LLRs



def trainQuadraticRegression(D, L, lSet, prior_t):
    prior_tilde_set = [0.1, 0.5, 0.9]

    for l in lSet:
        pca = 7
        LLRs = kFold(prior_t, K_Fold.loadRawFolds, l, pca)
        for prior_tilde in prior_tilde_set:
            ActDCF, minDCF = me.printDCFs(D, L, LLRs, prior_tilde)
            print(prior_t, "|" ,prior_tilde, "| Quadratic Regression | Lambda ={:.2e}".format(l), "| Raw | PCA =", pca, "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF)) 
        
        LLRs = kFold(prior_t, K_Fold.loadRawFolds, l, pca)
        for prior_tilde in prior_tilde_set:    
            ActDCF, minDCF = me.printDCFs(D, L, LLRs, prior_tilde)
            print(prior_t, "|" ,prior_tilde, "| Quadratic Regression | Lambda ={:.2e}".format(l), "| Normalized | PCA =", pca, "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))