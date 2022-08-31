from contextlib import ContextDecorator
import numpy
import scipy.optimize
import time
import DimReduction as dr
from Main import vrow
import ModelEvaluation as me
import ScoreCalibration as sc
import K_Fold

def mcol(v):
    return v.reshape(v.size, 1)

def logreg_obj(v, DTR, LTR, l, prior):
    w, b = mcol(v[0:-1]), v[-1]

    LTR0 = LTR[LTR == 0]
    LTR1 = LTR[LTR == 1]

    Z0 = LTR0 * 2.0 - 1.0
    Z1 = LTR1 * 2.0 - 1.0

    S0 = numpy.dot(w.T, DTR[:, LTR == 0]) + b
    S1 = numpy.dot(w.T, DTR[:, LTR == 1]) + b
    
    NF = len(LTR0)
    NT = len(LTR1)
    
    cxeF = numpy.logaddexp(0, -Z0*S0).sum() * (1 - prior) / NF
    cxeT = numpy.logaddexp(0, -Z1*S1).sum() * prior / NT

    return l/2 * numpy.linalg.norm(w)**2 + cxeT + cxeF

def LinearRegression(DTR, LTR, DTE, l, prior_t):
    x0 = numpy.zeros(DTR.shape[0] + 1)
    x, f, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, args=(DTR, LTR, l, prior_t), approx_grad = True)
    w = x[0:DTR.shape[0]]
    b = x[-1]
    scores = numpy.dot(w.T, DTE) + b

    return scores

def logreg_obj_w_b(v, score, labels, l):
    w, b = mcol(v[0:-1]), v[-1]
    Z = labels * 2.0 - 1.0
    score = vrow(score)
    S = numpy.dot(w.T, score) + b
    cxe = numpy.logaddexp(0, -Z*S).mean()
    return l/2 * numpy.linalg.norm(w)**2 + cxe

def LinearRegression_w_b(score, labels, l, prior_t):
    x0 = numpy.zeros(1 + 1)
    x, f, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj_w_b, x0, args=(score, labels, l), approx_grad = True,
                                            maxfun = 15000, factr=1.0)
    w = x[0:-1]
    b = x[-1]

    return w, b

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
        LLRsRet = LinearRegression(DTR, LTR, DTE, l, prior_t)
        LLRs.append(LLRsRet)
    
    LLRs = numpy.hstack(LLRs)

    return LLRs



def trainLinearRegression(D, L, lSet, prior_t):
    prior_tilde_set = [0.1, 0.5, 0.9]

    #print("result[0] = prior_t | result[1] = prior_tilde | result[2] = model_name | result[3] = pre-processing | result[4] = PCA | result[5] = ActDCF | result[6] = MinDCF")

    for l in lSet:
        pca = 8
        LLRs = kFold(prior_t, K_Fold.loadRawFolds, l, pca)
        #Score Calibration before estimating DCFs
        #CalibratedLLRs = sc.calibrate_scores(LLRs, L, prior_t)

        for prior_tilde in prior_tilde_set:
            ActDCF, minDCF = me.printDCFs(D, L, LLRs, prior_tilde)
            print(prior_t, "|", prior_tilde, "| Linear Regression | Lambda ={:.2e}".format(l), "| Raw | Uncalibrated | PCA =", pca, "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))
            #ActDCF, minDCF = me.printDCFs(D, L, CalibratedLLRs, prior_tilde)
            #print(prior_t, "|", prior_tilde, "| Linear Regression | Lambda ={:.2e}".format(l), "| Raw | Calibrated | PCA =", pca, "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))  
        

        LLRs = kFold(prior_t, K_Fold.loadNormFolds, l, pca)
        #Score Calibration before estimating DCFs
        #CalibratedLLRs = sc.calibrate_scores(LLRs, L, prior_t)
        for prior_tilde in prior_tilde_set:    
            ActDCF, minDCF = me.printDCFs(D, L, LLRs, prior_tilde)
            print(prior_t, "|", prior_tilde, "| Linear Regression | Lambda ={:.2e}".format(l), "| Normalized | Uncalibrated | PCA =", pca, "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))
            #ActDCF, minDCF = me.printDCFs(D, L, CalibratedLLRs, prior_tilde)
            #print(prior_t, "|", prior_tilde, "| Linear Regression | Lambda ={:.2e}".format(l), "| Normalized | Calibrated | PCA =", pca, "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))  