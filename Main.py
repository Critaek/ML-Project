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
import GMM
import ScoreCalibration as sc
import K_Fold

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

    raw = numpy.load("data/raw.npy",  allow_pickle=True)

    #K_Fold.saveRawFolds(D, L, 5)
    #K_Fold.saveNormFolds(D, L, 5)

    print(time.time() - start)
    
    different_Application=[0.1, 0.5 ,0.9]
    for different_prior in different_Application:
        #mvg.trainMVG(mvg.MultiV, D, L, "Full", different_prior)

        #mvg.trainMVG(mvg.Bayes, D, L, "Bayes", different_prior)

        #mvg.trainMVG(mvg.Tied, D, L, "Tied", different_prior)

        lSet = numpy.logspace(-5,2, num = 20) #20 values between 1e-5 and 1e2
        #lr.trainLinearRegression(D, L, lSet, different_prior)
        #qr.trainQuadraticRegression(D, L, lSet, different_prior)
        
        K_Set = numpy.array([0.0, 1.0, 10.0])
        C_Set = numpy.logspace(-2,0, num = 10)
        #SVM.trainSVMLinear(D, L, K_Set, C_Set, different_prior)

        
        K_Set = numpy.array([0.0, 1.0, 10.0])
        C_Set = numpy.logspace(-2,0, num = 10)
        d_Set = numpy.array([2.0, 3.0])
        c_Set = numpy.array([0.0, 1.0])
        #SVM.trainSVMPoly(D, L, K_Set, C_Set, d_Set, c_Set, different_prior)

        
        K_Set = numpy.array([0.0, 1.0, 10.0])
        C_Set = numpy.logspace(-2,0, num = 10)
        gamma_Set = numpy.logspace(-3,-1, num = 3)
        #SVM.trainSVM_RBF(D, L, K_Set, C_Set, gamma_Set, different_prior)
    
    nSet = numpy.array([6])
    #GMM.trainGMM_Full(D, L, nSet)
    #GMM.trainGMM_Diagonal(D, L, nSet)
    #GMM.trainGMM_Tied(D, L, nSet)

    
    #PredictionsPoly, scoresPoly = SVM.kFoldPoly(NormD, L, 5, 10.0, 0.01, 3.0, 1.0, 0.5)
    #numpy.save("data/SVMPoly_10_01_3_1.npy", scoresPoly)

    #PredictionsGMM, scoresGMM = GMM.kFold_GMM_Full(NormD, L, 5, 4)
    #numpy.save("data/GMMFull_4_Norm.npy", scoresGMM)

    #scoresGMM = numpy.load("data/GMMFull_4_Norm.npy")
    #scoresPoly = numpy.load("data/SVMPoly_10_01_3_1.npy")
    #PredictionsPoly = scoresPoly > 0
    #PredictionsGMM = scoresGMM > 0
    
    #x, ActPoly, MinPoly = me.BiasErrorPlot(L, PredictionsPoly, scoresPoly, 0.5)
    #x, ActGMM, MinGMM = me.BiasErrorPlot(L, PredictionsGMM, scoresGMM, 0.5)
    #plt.BiasErrorPlot(x, Act, Min)

    #CalibratedScoresPoly = sc.calibrate_scores(vrow(scoresPoly), L, 0.5)
    #CalibratedScoresGMM = sc.calibrate_scores(vrow(scoresGMM), L, 0.5)
    #CalibratedScoresPoly = CalibratedScoresPoly.reshape(CalibratedScoresPoly.shape[1])
    #CalibratedScoresGMM = CalibratedScoresGMM.reshape(CalibratedScoresGMM.shape[1])

    #legend = ["ActDCF Poly", "MinDCF Poly", "ActDCF GMM", "MinDCF GMM"]
    #plt.BiasErrorPlotCalUncal(x, ActPoly, MinPoly, ActGMM, MinGMM, legend)
    #x, ActPoly, MinPoly = me.BiasErrorPlot(L, CalibratedScoresPoly > 0, CalibratedScoresPoly, 0.5)
    #x, ActGMM, MinGMM = me.BiasErrorPlot(L, CalibratedScoresGMM > 0, CalibratedScoresGMM, 0.5)
    #legend = ["ActDCF Poly (Calibrated)", "MinDCF Poly", "ActDCF GMM (Calibrated)", "MinDCF GMM"]
    #plt.BiasErrorPlotCalUncal(x, ActPoly, MinPoly, ActGMM, MinGMM, legend)




    end = time.time()
    print("Total time", end - start)

    #-------------------------------------------------------------------------------------------------#
    
    #plt.HeatMapPearson(D, "Raw")
    #plt.plotHist(D, L, "Raw")

    #plt.HeatMapPearson(NormD, "Normalized")
    #plt.plotHist(NormD, L, "Normalized")

