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

def mcol(v):
    return v.reshape(v.size, 1)

def vrow(vect):
    return vect.reshape(1, vect.size)

if __name__ == "__main__":
    start = time.time()

    D, L = l.load()
    NormD = n.Normalization(D)

    #Full-Covariance MVG
    print("Full-Cov MVG")
    #mvg.tryMVG(mvg.MultiV, D, L, NormD)
    print("\n")
    print("Bayes-Cov MVG")
    #mvg.tryMVG(mvg.Bayes, D, L, NormD)
    print("\n")
    print("Tied-Cov MVG")
    #mvg.tryMVG(mvg.Tied, D, L, NormD)
    print("\n")
    print("Liner Regression")
    lSet = numpy.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    lr.tryRegression(lr.LinearRegression, D, L, NormD, lSet)

    end = time.time()
    print("Total time", end - start)

    #-------------------------------------------------------------------------------------------------#
    
    #plt.HeatMapPearson(D, "Raw")
    #plt.plotHist(D, L, "Raw")

    #plt.HeatMapPearson(NormD, "Normalized")
    #plt.plotHist(NormD, L, "Normalized")


