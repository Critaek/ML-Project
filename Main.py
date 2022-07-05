from asyncio.constants import DEBUG_STACK_DEPTH
import scipy.stats as st
import numpy
import time
import Normalization as n
import DimReduction as dr
import Plot as plt
import Load as l
import MVG as mvg
import ModelEvaluation as me
import LinearRegression as lr
import SVM

def mcol(v):
    return v.reshape(v.size, 1)

def vrow(vect):
    return vect.reshape(1, vect.size)

def split_db_2to1(D, L, seed = 0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    
    return (DTR, LTR), (DTE, LTE)

if __name__ == "__main__":
    start = time.time()

    D, L = l.load()
    NormD = n.Normalization(D)

    #Full-Covariance MVG
    print("Full-Cov MVG")
    mvg.trainMVG(mvg.MultiV, D, L, NormD)
    print("\n")

    print("Bayes-Cov MVG")
    mvg.trainMVG(mvg.Bayes, D, L, NormD)
    print("\n")

    print("Tied-Cov MVG")
    mvg.trainMVG(mvg.Tied, D, L, NormD)
    print("\n")

    print("Linear Regression")
    lSet = numpy.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
    lr.trainRegression(lr.LinearRegression, D, L, NormD, lSet)
    print("\n")

    print("SVM Linear")
    K_Set = numpy.array([0.0, 1.0, 10.0])
    C_Set = numpy.array([0.1, 1.0, 10.0])
    SVM.trainSVMLinear(D, L, NormD, K_Set, C_Set)
    print("\n")

    print("SVM Poly")
    K_Set = numpy.array([0.0, 1.0, 10.0])
    C_Set = numpy.array([0.5, 1.0])
    d_Set = numpy.array([2.0, 3.0])
    c_Set = numpy.array([0.0, 1.0])
    SVM.trainSVMPoly(D, L, NormD, K_Set, C_Set, d_Set, c_Set)
    print("\n")

    print("SVM RBF")
    K_Set = numpy.array([0.0, 1.0, 10.0])
    C_Set = numpy.array([0.5, 1.0])
    gamma_Set = numpy.array([1.0, 10.0])
    SVM.trainSVM_RBF(D, L, NormD, K_Set, C_Set, gamma_Set)
    

    end = time.time()
    print("Total time", end - start)

    #-------------------------------------------------------------------------------------------------#
    
    #plt.HeatMapPearson(D, "Raw")
    #plt.plotHist(D, L, "Raw")

    #plt.HeatMapPearson(NormD, "Normalized")
    #plt.plotHist(NormD, L, "Normalized")


