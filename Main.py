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
import QuadraticRegression as qr

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
    NormD = numpy.load("data/normalizedD.npy")
    
    #NormD = n.Normalization(D)
    #numpy.save("data/normalizedD.npy", NormD)


    different_Application=[0.1, 0.5 ,0.9]
    for different_prior in different_Application:
        #mvg.trainMVG(mvg.MultiV, D, L, NormD, "Full", different_prior)

        #mvg.trainMVG(mvg.Bayes, D, L, NormD, "Bayes", different_prior)

        #mvg.trainMVG(mvg.Tied, D, L, NormD, "Tied", different_prior)

        lSet = numpy.logspace(-5,2, num = 20) #20 values between 1e-5 and 1e2
        lr.trainLinearRegression(D, L, NormD, lSet, different_prior)
        qr.trainQuadraticRegression(D, L, NormD, lSet, different_prior)
        
        K_Set = numpy.array([0.0, 1.0, 10.0])
        C_Set = numpy.array([0.1, 1.0, 10.0])
        SVM.trainSVMLinear(D, L, NormD, K_Set, C_Set, different_prior)

        
        K_Set = numpy.array([0.0, 1.0, 10.0])
        C_Set = numpy.array([0.5, 1.0])
        d_Set = numpy.array([2.0, 3.0])
        c_Set = numpy.array([0.0, 1.0])
        SVM.trainSVMPoly(D, L, NormD, K_Set, C_Set, d_Set, c_Set, different_prior)

        
        K_Set = numpy.array([0.0, 1.0, 10.0])
        C_Set = numpy.array([0.5, 1.0])
        gamma_Set = numpy.array([1.0, 10.0])
        SVM.trainSVM_RBF(D, L, NormD, K_Set, C_Set, gamma_Set, different_prior)
    

    end = time.time()
    print("Total time", end - start)

    #-------------------------------------------------------------------------------------------------#
    
    #plt.HeatMapPearson(D, "Raw")
    #plt.plotHist(D, L, "Raw")

    #plt.HeatMapPearson(NormD, "Normalized")
    #plt.plotHist(NormD, L, "Normalized")


